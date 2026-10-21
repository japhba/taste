"""Per-rank conversion of DeepSeek-V4.1-Flash HF weights to the tensor-parallel layout of DeepSeek's reference inference code.

DeepSeek's inference/convert.py keeps all ranks in RAM (~600 GB) and reads every tensor; this does one rank at a time and
never reads another rank's experts. Renaming and sharding are copied verbatim from convert.py (imported from the checkpoint).

  python -m taste.ds_convert --hf DIR --out DIR --rank 0 [--mp 8]
"""
import argparse, importlib.util, json, os, shutil, time
from collections import defaultdict
import torch
from safetensors.torch import safe_open, save_file


def convert(HF, TP, rank, MP=8, expert_dtype="fp4"):
    spec = importlib.util.spec_from_file_location("ds_convert", f"{HF}/inference/convert.py")
    cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
    weight_map = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    n_experts, mtp_n_experts = cv.infer_num_experts(weight_map)
    by_file = defaultdict(list)
    for src, fn in weight_map.items(): by_file[fn].append(src)
    t, sd, i, mp = time.time(), {}, rank, MP
    for fn, srcs in sorted(by_file.items()):
        with safe_open(f"{HF}/{fn}", framework="pt", device="cpu") as f:
            for source_name in srcs:
                name = source_name.removeprefix("model.")
                if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"): continue
                name = name.replace("self_attn", "attn")
                if not name.startswith("vision."): name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")
                key = name.split(".")[-1] if any(x in name for x in ["hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_"]) else name.split(".")[-2]
                new_key, dim = cv.mapping.get(key, (key, None))
                name = name.replace(key, new_key)
                if "experts" in name and "shared_experts" not in name:
                    n_local = (mtp_n_experts if name.startswith("mtp.") else n_experts) // mp
                    if not i * n_local <= int(name.split(".")[-3]) < (i + 1) * n_local: continue  # not this rank's expert: never read
                param = f.get_tensor(source_name)
                if "experts" in name and "shared_experts" not in name: new_param = param
                elif ".engram.embed." in name:
                    shard = (param.size(0) + mp - 1) // mp
                    new_param = param[i * shard:(i + 1) * shard].contiguous()
                    if new_param.size(0) < shard:
                        new_param = torch.cat([new_param, param.new_full((shard - new_param.size(0), param.size(1)), 1 if name.endswith(".scale") else 0)])
                elif dim is not None:
                    assert param.size(dim) % mp == 0, (name, dim)
                    shard = param.size(dim) // mp
                    new_param = param.narrow(dim, i * shard, shard).contiguous()
                else: new_param = param
                sd[name] = new_param
    for name in list(sd):
        if name.endswith("wo_a.weight"):
            w, s = sd[name], sd.pop(name.replace("weight", "scale"))
            ob, ib = w.size(0) // s.size(0), w.size(1) // s.size(1)
            assert (ob, ib) in ((32, 32), (128, 128)), (name, w.shape, s.shape)
            sd[name] = (w.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * s[:, None, :, None].float()).flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and sd[name].dtype == torch.int8:
            if expert_dtype == "fp8":
                sn = name.replace("weight", "scale"); sd[name], sd[sn] = cv.cast_e2m1fn_to_e4m3fn(sd.pop(name), sd.pop(sn))
            else: sd[name] = sd[name].view(torch.float4_e2m1fn_x2)
    os.makedirs(TP, exist_ok=True)
    save_file(sd, f"{TP}/model{i}-mp{mp}.safetensors")
    if i == 0:
        for fn in ["tokenizer.json", "tokenizer_config.json"]: shutil.copyfile(f"{HF}/{fn}", f"{TP}/{fn}")
    gb = os.path.getsize(f"{TP}/model{i}-mp{mp}.safetensors") / 1e9
    print(f"rank {i}: {len(sd)} tensors, {gb:.1f} GB in {time.time() - t:.0f} s", flush=True)
    return gb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, required=True); ap.add_argument("--mp", type=int, default=8)
    a = ap.parse_args(); convert(a.hf, a.out, a.rank, a.mp)
