"""Steering a judge's taste. People with more interest in art like expressionist and abstract works more, and Rococo, Baroque and
Romantic ones less, than people without (taste.lapis_taste). Does a model's taste move the same way
  (1) when told to answer as a person with or without an interest in art (the personas never mention styles), and
  (2) when the difference between those two prompts, read from its residual stream at the last prompt token and averaged over
      training artworks, is added to its activations (every position, one layer) while it rates with no persona at all?
Ratings are the expected digit of the usual 1-9 question (taste.rate), on held-out LAPIS artworks.

  python -m taste.steer qwen3.5-27b [--n-train 100 --n-test 400]
-> data/steer/<model>.parquet (item, condition, depth, alpha, p1..p9, score), data/steer/<model>_vectors.npz
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from taste.rate import Rater, IMAGE_Q
from taste.paths import RAW, DATA

PERSONA = {"lover": "Answer as yourself: a person with a strong interest in art. You often visit museums and galleries, you read about art, "
                    "and you know art history well.",
           "lay": "Answer as yourself: a person with no interest in art. You never visit museums or galleries, you never read about art, "
                  "and you know nothing about art history."}
ALPHAS = [-4, -2, -1, 1, 2, 4]
DEPTHS = [1 / 3, 1 / 2, 2 / 3]
LAPIS = RAW / "asmaerten__LAPIS"
OUT = DATA / "steer"


def prompt(persona=None):
    return IMAGE_Q if persona is None else PERSONA[persona] + "\n\n" + IMAGE_Q


def decoder_layers(model):
    n = model.config.get_text_config().num_hidden_layers
    return next(m for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n and "language_model" in name)


class Steer:
    """Adds a vector to the output of one decoder layer, i.e. to hidden_states[layer]."""
    def __init__(self, model): self.blocks, self.handle = decoder_layers(model), None

    def on(self, layer, vec):
        self.off()
        def hook(mod, inp, out):
            if isinstance(out, tuple): return (out[0] + vec.to(out[0].device, out[0].dtype),) + out[1:]
            return out + vec.to(out.device, out.dtype)
        self.handle = self.blocks[layer - 1].register_forward_hook(hook)

    def off(self):
        if self.handle: self.handle.remove(); self.handle = None


def rate(r, path, q, hidden=False):
    enc, span, _ = r.encode_images([path], q)
    return r.forward(enc, span, hidden=hidden)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n-train", type=int, default=100); ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--out", default=str(OUT)); a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    art = pd.concat([pd.read_csv(LAPIS / f"giaa/{s}.csv") for s in ["train", "validation", "test"]]).drop_duplicates("image_id")
    art = art.sample(a.n_train + a.n_test, random_state=0)
    train, test = art.iloc[:a.n_train], art.iloc[a.n_train:]
    r = Rater(a.model); steer = Steer(r.model)
    depth_layer = {d: min(r.layers[1:], key=lambda l: abs(l - d * r.layers[-1])) for d in DEPTHS}
    print("layers", r.layers, "steering at", depth_layer)

    vec_path = out / f"{a.model}_vectors.npz"
    if vec_path.exists(): V = np.load(vec_path)["V"]
    else:
        diffs = [rate(r, LAPIS / "images" / p, prompt("lover"), hidden=True)["last"][0].astype(np.float32) - rate(r, LAPIS / "images" / p, prompt("lay"), hidden=True)["last"][0].astype(np.float32)
                 for p in tqdm(train.path, desc="persona vectors")]
        V = np.mean(diffs, 0); np.savez(vec_path, V=V, layers=np.array(r.layers), image_id=train.image_id.values)
    print("vector norms by layer", dict(zip(r.layers, np.linalg.norm(V, axis=1).round(1))))

    out_path = out / f"{a.model}.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["image_id"])
    rows = [done] if len(done) else []
    for it in tqdm(test.itertuples(), total=len(test), desc="rate", mininterval=30):
        if it.image_id in set(done.image_id): continue
        path, recs = LAPIS / "images" / it.path, []
        for persona in [None, "lover", "lay"]:
            recs.append(dict(condition=persona or "none", depth=np.nan, alpha=0.0, p=rate(r, path, prompt(persona))["p_digits"][0]))
        for d, layer in depth_layer.items():
            for al in ALPHAS:
                steer.on(layer, torch.tensor(al * V[r.layers.index(layer)]))
                recs.append(dict(condition="steered", depth=d, alpha=al, p=rate(r, path, prompt())["p_digits"][0])); steer.off()
        df = pd.DataFrame(recs); P = np.stack(df.pop("p"))
        df[[f"p{k}" for k in range(1, 10)]] = P; df["digit_mass"] = P.sum(1); df["score"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, 10)
        rows.append(df.assign(image_id=it.image_id, path=it.path, model=a.model))
        if len(rows) % 25 == 0: pd.concat(rows, ignore_index=True).to_parquet(out_path)
    pd.concat(rows, ignore_index=True).to_parquet(out_path)


if __name__ == "__main__":
    main()
