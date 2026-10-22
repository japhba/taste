"""Steering the features of taste.taste_features: is a judge's weight on a feature causal? For each judge, its own direction for each
feature (top fifth minus bottom fifth of haiku by the feature, in the judge's stored mean-over-haiku-tokens representation at 1/3
depth) is added, times alpha, to the residual stream at the haiku's tokens at that depth while the judge rates the haiku.
At 2/3 depth the same intervention does nothing (the verdict has already read the haiku; Gemma's small models also share keys and
values across their later layers), and beyond |alpha| ~ 4 any push lowers the rating, so the effect is read from the part that
flips sign with alpha (taste.steer_report). Prediction: it follows the sign of the judge's observed weight, so a high/low-brow
feature pushes tiny and large judges in opposite directions.

  python -m taste.steer_features qwen3.5-0.8b qwen3.5-27b ... [--n 150] [--craft --features PC1 PC2 PC5] [--controls 3]
  --controls k: k control directions instead (difference between two random fifths of haiku, rescaled to the norm of the first
  feature's direction) -> data/steer<sfx>/control<k>_<judge>.parquet
-> data/steer/haiku_<judge>.parquet (item_id, feature, alpha, p1..p9, digit_mass, score); with --craft (the topic-removed features,
   steered on model haiku only) data/steer_craft/haiku_<judge>.parquet
"""
import argparse, gc
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from taste.rate import Rater
from taste.common import items, hidden
from taste.paths import DATA
from taste.steer import decoder_layers

FEATURES = ["PC1", "PC2", "PC4", "PC8"]
ALPHAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]
DEPTH = 1 / 3


class SpanSteer:
    """Adds a vector to hidden_states[layer] at the item's own tokens."""
    def __init__(self, model): self.blocks, self.handle = decoder_layers(model), None

    def on(self, layer, vec, span):
        self.off()
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vec.to(h.device, h.dtype) * span.to(h.device)[..., None].to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        self.handle = self.blocks[layer - 1].register_forward_hook(hook)

    def off(self):
        if self.handle: self.handle.remove(); self.handle = None


def directions(name, feats, index):
    """The judge's own direction for each feature at DEPTH: top fifth minus bottom fifth of haiku by the feature."""
    ids, layers, H = hidden("haiku", name, "mean")
    li = int(np.argmin(np.abs(layers - DEPTH * layers.max())))
    X = H[pd.Series(np.arange(len(ids)), index=ids).reindex(index).values, li].astype(np.float32)
    out = {}
    for f in feats.columns:
        q = feats[f].quantile([0.2, 0.8]); out[f] = X[(feats[f] >= q[0.8]).values].mean(0) - X[(feats[f] <= q[0.2]).values].mean(0)
    return int(layers[li]), out, float(np.linalg.norm(X, axis=1).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("judges", nargs="+"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--craft", action="store_true"); ap.add_argument("--features", nargs="+", default=FEATURES); ap.add_argument("--controls", type=int, default=0)
    a = ap.parse_args()
    sfx = "_craft" if a.craft else ""
    feats = pd.read_parquet(DATA / f"steer/haiku_features{sfx}.parquet")[a.features]  # oriented PC scores (taste.taste_features)
    it = items("haiku").reindex(feats.index)
    test = it.sample(a.n, random_state=0)
    (DATA / f"steer{sfx}").mkdir(exist_ok=True)
    if a.controls:  # random fifths: the same marginal as the first feature, no meaning
        rng = np.random.default_rng(1)
        feats = feats.assign(**{f"control{k + 1}": rng.permutation(feats[a.features[0]].values) for k in range(a.controls)})
    for name in a.judges:
        out = DATA / f"steer{sfx}" / (f"control{a.controls}_{name}.parquet" if a.controls else f"haiku_{name}.parquet")
        if out.exists(): continue
        layer, D, hnorm = directions(name, feats, it.index)
        if a.controls:
            D = {f"control{k + 1}": D[f"control{k + 1}"] * np.linalg.norm(D[a.features[0]]) / np.linalg.norm(D[f"control{k + 1}"]) for k in range(a.controls)}
        r = Rater(name); steer = SpanSteer(r.model)
        print(name, "steering at layer", layer, "mean |h|", round(hnorm, 1), {f: round(float(np.linalg.norm(v)), 2) for f, v in D.items()}, flush=True)
        recs = []
        for row in tqdm(test.itertuples(), total=len(test), desc=name, mininterval=30):
            enc, span, _ = r.encode_haiku([row.haiku])
            recs.append(dict(item_id=row.Index, feature="none", alpha=0.0, p=r.forward(enc, span, hidden=False)["p_digits"][0]))
            for f in D:
                for al in ALPHAS:
                    steer.on(layer, torch.tensor(al * D[f]), span)
                    recs.append(dict(item_id=row.Index, feature=f, alpha=al, p=r.forward(enc, span, hidden=False)["p_digits"][0])); steer.off()
        df = pd.DataFrame(recs); P = np.stack(df.pop("p"))
        df[[f"p{k}" for k in range(1, 10)]] = P; df["digit_mass"] = P.sum(1); df["score"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, 10)
        df.assign(judge=name, layer=layer).to_parquet(out)
        del r, steer; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
