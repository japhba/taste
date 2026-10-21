"""DeepSeek-V4.1-Flash-1017 rater on DeepSeek's reference inference code (8-way tensor parallel): same prompts, digit readout
and output files as taste.rate, one item per forward pass (prefill only, no generation), hidden states kept.

  torchrun --nproc-per-node 8 -m taste.ds_rate --ckpt TP8_DIR --hf HF_DIR [--haiku-per-gen 50] [--datasets acx lapis ava]

Hidden state = the hyper-connection residual stream (hc_mult copies) averaged over copies, after every 4th block.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from taste.paths import DATA, RATINGS, HIDDEN
from taste.rate import HAIKU_Q, IMAGE_Q, ORIGIN_Q

NAME = "deepseek-v4.1-flash-1017"
KEEP = list(range(0, 41, 4))  # embedding output + after blocks 4, 8, ..., 40


@torch.inference_mode()
def forward_hidden(model, M, input_ids, images, token_types):
    """Transformer.forward (model.py) with the residual stream recorded and full-position logits."""
    image_mask = None if token_types is None else token_types >= 0
    engram_mask = None if image_mask is None else ~image_mask
    engram_hashes = model.engram_hash(input_ids, 0, engram_mask) if model.engram_hash is not None else None
    h = model.embed(input_ids)
    if images is not None: model.merge_image_embeddings(images, h)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
    states = {0: h.mean(2)}
    for i, layer in enumerate(model.layers):
        if layer.engram is not None: h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
        h, pre_mix = layer(h, 0, pre_mix, image_mask)
        if i + 1 in KEEP: states[i + 1] = h.mean(2)
    h = layer.hc_pre(h, pre_mix)
    return model.head(model.norm(h), full_logits=True)[0], states


def main(a):
    sys.path.insert(0, f"{a.hf}/inference"); sys.path.insert(0, f"{a.hf}/encoding")
    import model as M
    from encoding import encode_case
    from image_processor import prepare_vl_inputs
    from safetensors.torch import load_model
    from transformers import AutoTokenizer
    ws, rank, lr = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl"); torch.cuda.set_device(lr)
    torch.set_default_dtype(torch.bfloat16); torch.set_num_threads(8)
    args = M.ModelArgs(**json.load(open(f"{a.hf}/inference/config.json")))
    args.max_batch_size, args.max_seq_len, args.temperature = 1, 4096, 0.0
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    t = time.time()
    with torch.device("cuda"): model = M.Transformer(args, tok)
    load_model(model, f"{a.ckpt}/model{rank}-mp{ws}.safetensors")
    torch.set_default_device("cuda")
    if rank == 0: print(f"loaded in {time.time() - t:.0f} s", flush=True)
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]
    origin = [tok.encode(w, add_special_tokens=False)[0] for w in ("Human", "AI")]

    def encode(content):
        prompt, recs = encode_case({"messages": [{"role": "user", "content": content}]}, "chat")
        ids, types, images = prepare_vl_inputs(prompt, recs, tok, args)
        ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1).cuda()
        types = None if not recs else torch.as_tensor(types, dtype=torch.long).view(1, -1).cuda()
        return prompt, ids, types, ([images] if recs else None)  # model.forward takes one image list per sample

    for which in a.sets:
        items = pd.read_parquet(DATA / f"{which}_items.parquet")
        if which == "haiku" and a.haiku_per_gen: items = items.groupby("generator", group_keys=False).apply(lambda g: g.sample(min(len(g), a.haiku_per_gen), random_state=0))
        if which == "images": items = items[items.dataset.isin(a.datasets)]
        recs, last, mean = [], [], []
        t = time.time()
        for k, row in enumerate(items.itertuples()):
            if which == "haiku":
                prompt, ids, types, images = encode([{"type": "text", "text": HAIKU_Q.format(haiku=row.haiku)}])
                enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                assert enc.input_ids == ids[0].tolist(), "prompt tokenisation differs from prepare_vl_inputs"
                s = prompt.index(row.haiku); e = s + len(row.haiku)
                span = torch.tensor([o[0] < e and o[1] > s for o in enc.offset_mapping], device="cuda")
            else:
                prompt, ids, types, images = encode([{"type": "image_url", "image_url": {"url": str(DATA / row.path)}}, {"type": "text", "text": IMAGE_Q}])
                span = types[0] >= 0
            logits, states = forward_hidden(model, M, ids, images, types)
            p = logits[-1].float().softmax(-1)
            rec = {"item_id": row.item_id, **{f"p{d}": p[digits[d - 1]].item() for d in range(1, 10)}, "n_item_tokens": int(span.sum())}
            if which == "haiku":
                pos = span[1:].nonzero()[:, 0]  # predict token pos+1 from pos
                lp = logits[pos].float().log_softmax(-1).gather(1, ids[0, pos + 1][:, None])[:, 0]
                rec["logprob_mean"] = lp.mean().item()
            elif row.dataset == "acx":
                _, ids2, types2, images2 = encode([{"type": "image_url", "image_url": {"url": str(DATA / row.path)}}, {"type": "text", "text": ORIGIN_Q}])
                po = forward_hidden(model, M, ids2, images2, types2)[0][-1].float().softmax(-1)[origin]
                rec["p_human"] = (po[0] / po.sum()).item()
            else: rec["p_human"] = np.nan
            H = torch.stack([states[l][0] for l in KEEP])  # [L, T, d]
            last.append(H[:, -1].half().cpu().numpy()); mean.append(H[:, span].float().mean(1).half().cpu().numpy())
            recs.append(rec)
            if rank == 0 and k % 50 == 0: print(f"{which} {k}/{len(items)} {(time.time() - t) / (k + 1):.2f} s/item", flush=True)
        if rank == 0:
            df = pd.DataFrame(recs); P = df[[f"p{d}" for d in range(1, 10)]].to_numpy()
            df["digit_mass"] = P.sum(1); df["score"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, 10)
            (RATINGS / which).mkdir(exist_ok=True); (HIDDEN / which).mkdir(exist_ok=True)
            df.assign(rater=NAME).to_parquet(RATINGS / which / f"{NAME}.parquet")
            np.savez(HIDDEN / which / f"{NAME}.npz", item_id=df.item_id.to_numpy().astype(str), layers=np.array(KEEP), last=np.stack(last), mean=np.stack(mean))
            print(which, "done", len(df), "digit mass", df.digit_mass.describe()[["min", "50%"]].round(3).to_dict(), "score", df.score.describe()[["mean", "std"]].round(2).to_dict(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--hf", required=True)
    ap.add_argument("--sets", nargs="+", default=["haiku", "images"]); ap.add_argument("--haiku-per-gen", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=["acx", "lapis", "ava"])
    main(ap.parse_args())
