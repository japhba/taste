"""The model of taste in features found without supervision. Features: the top principal components of Qwen3.5-27B's representation
of every haiku (mean over the haiku's tokens, at 2/3 depth), oriented so that +PC is what Qwen3.5-27B rates higher, named by Claude
from exemplars and checked on held-out forced choices (taste.pcs). Every judge's weight on every feature (Spearman of the PC with its
score) then sorts the features by who uses them:
  obvious        judges of every size weight it the same way
  non-obvious    large judges weight it, tiny ones do not
  high/low-brow  tiny and large judges weight it with opposite signs (+ end: what large judges like and tiny ones dislike)
Ground truth alongside: the writer contrasts (weak < open < frontier models) and human crowd (top r/haiku) < journal.
For the steering run (taste.steer_features), each local judge's own direction for each named feature: the difference between the
mean representation of the top and the bottom fifth of haiku by that feature, at the judge's 2/3-depth layer.

  python -m taste.taste_features [--craft]   (--craft: model haiku only, topic means removed from representations and scores, as in
                                               taste.pcs 'haiku-craft': how a haiku is written, not what it is about)
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from taste.common import ratings, items, hidden, save, record, fam, DIVERGING, HIDDEN
from taste.models import BY_NAME
from taste.paths import DATA
from sklearn.decomposition import PCA
from taste.pcs import basis, label_pc, N_PC
from taste.probes import LAYER_FRAC
from taste.gen_haiku import openrouter
from taste.taste_space import group_of

REF = "qwen3.5-27b"
TIERS = {"tiny": ["qwen3.5-0.8b", "qwen3.5-2b", "gemma-4-e2b"], "small": ["qwen3.5-4b", "qwen3.5-9b", "gemma-4-e4b"],
         "large": ["qwen3.5-27b", "gemma-4-26b-a4b", "gemma-4-31b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "claude-opus-5.5"]}
TRUTH = {"weak < open models": ("weak", "open"), "open < frontier models": ("open", "frontier"), "r/haiku top < journal": ("crowd", "journal")}
DIRS = DATA / "steer/haiku_directions.npz"


def classify(w):
    """w: weights by judge for one feature (oriented: + = what the reference judge likes)."""
    t = {k: w[v].mean() for k, v in TIERS.items()}
    if abs(t["large"]) < 0.12: return "weak"
    if np.sign(t["tiny"]) != np.sign(t["large"]) and abs(t["tiny"]) >= 0.1: return "high/low-brow"
    if abs(t["tiny"]) < 0.4 * abs(t["large"]): return "non-obvious"
    return "obvious"


def humans_on_craft(it, keep, demean):
    """Human haiku (no topic) placed on the topic-removed PCs: the same PCA as taste.pcs.basis, fit on topic-centred model haiku, applied to
    human haiku centred on the model haiku's mean."""
    ids, layers, H = hidden("haiku", REF, "mean")
    li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max()))); X = H[:, li].astype(np.float32)
    m = keep.reindex(ids).to_numpy(bool); Xm = X[m]
    Xc = Xm - pd.DataFrame(Xm).groupby(demean.reindex(ids[m]).to_numpy()).transform("mean").to_numpy()
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    pca = PCA(N_PC, random_state=0).fit((Xc - mu) / sd)
    hum = it.index[it.generator.str.startswith("human")].intersection(ids); pos = pd.Series(np.arange(len(ids)), index=ids)[hum].values
    return pd.DataFrame(pca.transform((X[pos] - Xm.mean(0) - mu) / sd), index=hum, columns=[f"PC{k + 1}" for k in range(N_PC)])


def directions(it, B, pcs):
    """Per local judge: top-fifth minus bottom-fifth mean representation for each feature, at the judge's 2/3-depth layer."""
    out = {}
    for r in [p.stem for p in (HIDDEN / "haiku").glob("*.npz")]:
        ids, layers, H = hidden("haiku", r, "mean")
        li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))
        X = pd.DataFrame(H[:, li].astype(np.float32), index=ids).reindex(it.index)
        out[f"{r}|layer"] = np.array(int(layers[li]))
        for k in pcs:
            q = B[k].quantile([0.2, 0.8])
            out[f"{r}|{k}"] = X[B[k] >= q[0.8]].mean().values - X[B[k] <= q[0.2]].mean().values
    return out


def main(craft):
    it = items("haiku"); S = ratings("haiku").reindex(it.index); g = group_of(it); variant, sfx = ("haiku-craft", "_craft") if craft else ("haiku", "")
    keep = demean = None
    if craft:
        keep = it.topic.notna(); it, S, g = it[keep], S[keep], g[keep]; demean = it.topic; S = S - S.groupby(it.topic).transform("mean")
    B, evr = basis("haiku", REF, keep, demean); B = B.reindex(it.index)
    Hu = humans_on_craft(items("haiku"), keep, demean) if craft else None
    W = pd.DataFrame({r: [stats.spearmanr(B[k], S[r], nan_policy="omit").statistic for k in B] for r in S.columns}, index=B.columns).T
    for k in B:  # +PC = what the reference judge rates higher (as in taste.pcs)
        sg = np.sign(W.loc[REF, k]); B[k] *= sg; W[k] *= sg
        if craft: Hu[k] *= sg
    T = pd.DataFrame({name: [stats.spearmanr(B.loc[g.isin([lo, hi]), k], (g[g.isin([lo, hi])] == hi)).statistic for k in B] for name, (lo, hi) in TRUTH.items() if g.isin([lo, hi]).any()}, index=B.columns).T
    if craft:  # people's ladder on the topic-removed features: top-voted r/haiku posts < journal haiku
        gen = items("haiku").generator.reindex(Hu.index); sel = gen.isin(["human-reddit-top", "human-journal"])
        T.loc["r/haiku top < journal (projected)"] = [stats.spearmanr(Hu.loc[sel, k], gen[sel] == "human-journal").statistic for k in B]
    cls = pd.Series({k: classify(W[k]) for k in B})
    tier = pd.DataFrame({t: W.loc[v].mean() for t, v in TIERS.items()})
    summary = tier.join(cls.rename("class")).join(T.T).join(pd.Series(evr, index=B.columns, name="var_explained"))
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    # name the strongest features of each class (cached names are reused)
    chosen = [k for c in ["obvious", "non-obvious", "high/low-brow"] for k in summary[summary["class"] == c].large.abs().sort_values(ascending=False).index[:3]]
    client = openrouter()
    labels = {k: label_pc(client, "haiku", it, B[k], f"{variant}/{REF}/{k}") for k in chosen}
    for k in chosen: print(k, cls[k], labels[k])
    print("labelling cost $", round(sum(l["cost"] for l in labels.values()), 3))
    D = directions(it, B, chosen); DIRS.parent.mkdir(exist_ok=True); np.savez(DIRS.with_name(DIRS.stem + sfx + ".npz"), **D)
    B[chosen].to_parquet(DATA / f"steer/haiku_features{sfx}.parquet")
    record("taste_features" + sfx, dict(summary=summary.reset_index(names="pc").to_dict("records"), weights=W.to_dict(), chosen=chosen, labels=labels))
    plot(W, T, cls, labels, chosen, sfx)


def plot(W, T, cls, labels, chosen, sfx):
    order = [r for t in TIERS.values() for r in t if r in W.index]
    rows = sorted(chosen, key=lambda k: ["obvious", "non-obvious", "high/low-brow"].index(cls[k]))
    M = pd.concat([W.loc[order, rows].T, T[rows].T], axis=1)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 2.6), layout="constrained")
    im = ax.imshow(M.values, cmap=DIVERGING, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(M.shape[1]), [c if c in TRUTH else c.replace("qwen3.5-", "Qwen ").replace("gemma-4-", "Gemma ").replace("claude-opus-5.5", "Claude").upper().replace("QWEN", "Qwen").replace("GEMMA", "Gemma").replace("CLAUDE", "Claude").replace("-A", "-A") for c in M.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [f"{cls[k]}: {labels[k]['low']}  →  {labels[k]['high']}" for k in rows])
    for x in [len(TIERS["tiny"]) - 0.5, len(TIERS["tiny"]) + len(TIERS["small"]) - 0.5, len(order) - 0.5]: ax.axvline(x, color="white", lw=4)
    for y in range(len(rows) - 1):
        if cls[rows[y]] != cls[rows[y + 1]]: ax.axhline(y + 0.5, color="white", lw=4)
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.7, label="ρ(feature, judge's score)\nground truth: ρ(feature, better writer)")
    ax.set_title("tiny judges  |  small judges  |  large judges  |  ground truth", loc="left")
    save(fig, "taste_features" + sfx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--craft", action="store_true"); main(ap.parse_args().craft)
