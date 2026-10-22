"""DeepSeek-V4.1-Flash-1017 rater on DeepSeek's reference inference code (8-way tensor parallel): same prompts, digit readout
and output files as taste.rate, one item per forward pass (prefill only, no generation), hidden states kept.

  torchrun --nproc-per-node 8 -m taste.ds_rate --ckpt TP8_DIR --hf HF_DIR [--plan acx lapis haiku ava] [--haiku-per-gen 50] [--max-minutes 40]
      [--persona editor|reader|lover|lay] [--human-all]

--persona puts a persona before the rating question (it never names a feature) and writes to ratings/<set>_<persona> and
hidden/<set>_<persona>; --human-all keeps every human haiku when --haiku-per-gen samples the model writers.

Parts are rated in `plan` order and saved every 100 items, and all ranks stop together once --max-minutes (counted from
process start, model loading included) is used up, so a budget-capped run keeps everything it finished.

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
PERSONA = {"editor": "Answer as yourself: the editor of a leading English-language haiku journal, such as Modern Haiku or Frogpond, who has read "
                     "and published haiku for decades.",
           "reader": "Answer as yourself: a regular reader of the r/haiku subreddit, who enjoys the haiku posted there and upvotes the ones you like.",
           "lover": "Answer as yourself: a person with a strong interest in art. You often visit museums and galleries, you read about art, "
                    "and you know art history well.",  # the art personas of taste.steer
           "lay": "Answer as yourself: a person with no interest in art. You never visit museums or galleries, you never read about art, "
                  "and you know nothing about art history."}


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

    t0 = time.time()
    haiku = pd.read_parquet(DATA / "haiku_items.parquet")
    if a.haiku_per_gen: haiku = haiku.groupby("generator", group_keys=False).apply(
        lambda g: g if a.human_all and g.name.startswith("human") else g.sample(min(len(g), a.haiku_per_gen), random_state=0))
    lead = PERSONA[a.persona] + "\n\n" if a.persona else ""
    sub = lambda which: which + (f"_{a.persona}" if a.persona else "")  # output folder
    images = pd.read_parquet(DATA / "images_items.parquet")
    def existing(which):
        """resume: what a previous (preempted / budget-capped) run of the same plan already saved"""
        r, h = RATINGS / sub(which) / f"{NAME}.parquet", HIDDEN / sub(which) / f"{NAME}.npz"
        if not (r.exists() and h.exists()): return dict(recs=[], last=[], mean=[])
        df, z = pd.read_parquet(r), np.load(h, allow_pickle=True)
        assert list(z["item_id"]) == df.item_id.tolist()
        return dict(recs=df.drop(columns=["digit_mass", "score", "rater", "persona"], errors="ignore").to_dict("records"), last=list(z["last"]), mean=list(z["mean"]))
    out = {w: existing(w) for w in ["haiku", "images"]}
    done = {w: {r["item_id"] for r in o["recs"]} for w, o in out.items()}
    if rank == 0: print("resuming with", {w: len(d) for w, d in done.items()}, flush=True)

    def save(which):
        o = out[which]
        if rank != 0 or not o["recs"]: return
        df = pd.DataFrame(o["recs"]); P = df[[f"p{d}" for d in range(1, 10)]].to_numpy()
        df["digit_mass"] = P.sum(1); df["score"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, 10)
        (RATINGS / sub(which)).mkdir(exist_ok=True); (HIDDEN / sub(which)).mkdir(exist_ok=True)
        df.assign(rater=NAME, persona=a.persona or "").to_parquet(RATINGS / sub(which) / f"{NAME}.parquet")
        np.savez(HIDDEN / sub(which) / f"{NAME}.npz", item_id=df.item_id.to_numpy().astype(str), layers=np.array(KEEP), last=np.stack(o["last"]), mean=np.stack(o["mean"]))
        print(f"saved {which}: {len(df)} items, digit mass min {df.digit_mass.min():.3f}, score {df.score.mean():.2f}±{df.score.std():.2f}", flush=True)

    stop = False
    for part in a.plan:
        which = "haiku" if part == "haiku" else "images"
        items = haiku if part == "haiku" else images[images.dataset == part]
        t = time.time()
        items = items[~items.item_id.isin(done[which])]
        for k, row in enumerate(items.itertuples()):
            flag = torch.tensor([float(time.time() - t0 > a.max_minutes * 60)], device="cuda"); dist.broadcast(flag, 0)
            if flag.item(): stop = True; break
            if which == "haiku":
                prompt, ids, types, imgs = encode([{"type": "text", "text": lead + HAIKU_Q.format(haiku=row.haiku)}])
                enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                assert enc.input_ids == ids[0].tolist(), "prompt tokenisation differs from prepare_vl_inputs"
                s_ = prompt.index(row.haiku); e_ = s_ + len(row.haiku)
                span = torch.tensor([o[0] < e_ and o[1] > s_ for o in enc.offset_mapping], device="cuda")
            else:
                prompt, ids, types, imgs = encode([{"type": "image_url", "image_url": {"url": str(DATA / row.path)}}, {"type": "text", "text": lead + IMAGE_Q}])
                span = types[0] >= 0
            logits, states = forward_hidden(model, M, ids, imgs, types)
            p = logits[-1].float().softmax(-1)
            rec = {"item_id": row.item_id, **{f"p{d}": p[digits[d - 1]].item() for d in range(1, 10)}, "n_item_tokens": int(span.sum())}
            if which == "haiku":
                pos = span[1:].nonzero()[:, 0]  # predict token pos+1 from pos
                lp = logits[pos].float().log_softmax(-1).gather(1, ids[0, pos + 1][:, None])[:, 0]
                rec["logprob_mean"] = lp.mean().item()
            elif part == "acx":
                _, ids2, types2, imgs2 = encode([{"type": "image_url", "image_url": {"url": str(DATA / row.path)}}, {"type": "text", "text": ORIGIN_Q}])
                po = forward_hidden(model, M, ids2, imgs2, types2)[0][-1].float().softmax(-1)[origin]
                rec["p_human"] = (po[0] / po.sum()).item()
            else: rec["p_human"] = np.nan
            H = torch.stack([states[l][0] for l in KEEP])  # [L, T, d]
            o = out[which]; o["last"].append(H[:, -1].half().cpu().numpy()); o["mean"].append(H[:, span].float().mean(1).half().cpu().numpy()); o["recs"].append(rec)
            if rank == 0 and k % 25 == 0: print(f"{part} {k}/{len(items)} {(time.time() - t) / (k + 1):.2f} s/item | {(time.time() - t0) / 60:.1f} min", flush=True)
            if k % 100 == 99: save(which)
        save(which)
        if stop: print("time budget reached", flush=True) if rank == 0 else None; break
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--hf", required=True)
    ap.add_argument("--plan", nargs="+", default=["acx", "lapis", "haiku", "ava"]); ap.add_argument("--haiku-per-gen", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=1e9); ap.add_argument("--persona", choices=list(PERSONA)); ap.add_argument("--human-all", action="store_true")
    main(ap.parse_args())
