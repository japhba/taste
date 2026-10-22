"""Do a model's representations carry the obvious and the expert features of the model of taste, and do they separate good from bad?

Axes, defined by judges (each a held-out ridge readout from the model's own representation of a haiku, mean over its tokens):
  obvious axis   what the novices respond to: consensus of Qwen3.5-4B and Gemma-4-E4B
  expert axis    what the experts (27B and up, Claude) respond to beyond that (expert consensus orthogonal to the novice one)
The model whose representation is read is left out of both consensuses. Layer 0 (token embeddings) is the raw stimulus.
Ground truth, used only to evaluate: the writer. Weak writers (SmolLM2-135M/360M/1.7B, Qwen3.5-0.8B) < competent open models
(Qwen3.5 9B-397B) < frontier models (GPT-5.6, Gemini 3.8 Flash, DeepSeek V4.1 Flash, Claude Opus 5.5 / Fable 5.1); for human
haiku, the crowd's top-voted r/haiku posts < journal haiku. Reported: held-out AUC of each ground-truth contrast along each axis,
by model and depth, next to the AUC of the model's own spoken rating.

  python -m taste.taste_space
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from taste.common import ratings, items, hidden, save, record, fam, FAMILY_COLOR, HIDDEN
from taste.models import BY_NAME

GROUPS = {"weak": ["smollm2-135m", "smollm2-360m", "smollm2-1.7b", "qwen3.5-0.8b"],
          "open": ["qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"],
          "frontier": ["gpt-5.6-sol", "gemini-3.8-flash", "deepseek-v4.1-flash", "claude-opus-5.5", "claude-fable-5.1"],
          "crowd": ["human-reddit-top"], "journal": ["human-journal"]}
CONTRASTS = {"weak vs open models (obvious axis)": ("weak", "open", "obvious"), "open vs frontier models (expert axis)": ("open", "frontier", "expert"),
             "r/haiku top vs journal (expert axis)": ("crowd", "journal", "expert")}
NOVICES = ["qwen3.5-4b", "gemma-4-e4b"]
EXPERTS = ["qwen3.5-27b", "gemma-4-26b-a4b", "gemma-4-31b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "claude-opus-5.5"]
GROUP_COLOR = {"weak": "#c0302f", "open": "#e8c9a0", "frontier": "#1b8f65", "crowd": "#e8c9a0", "journal": "#1b8f65"}
RNG = np.random.default_rng(0)


def group_of(it):
    g = pd.Series(np.nan, it.index, dtype=object)
    for k, gens in GROUPS.items(): g[it.generator.isin(gens)] = k
    return g


def folds(it, k=5):
    topics = sorted(it.topic.dropna().unique())
    f = it.topic.map({t: i % k for i, t in enumerate(topics)})
    return f.fillna(pd.Series(RNG.integers(0, k, len(it)), index=it.index)).astype(int).values


def targets(Zc, without):
    """Novice consensus N and the experts' consensus orthogonal to it, E⊥, leaving out `without`."""
    N = Zc[[r for r in NOVICES if r != without]].mean(1).values
    E = Zc[[r for r in EXPERTS if r != without]].mean(1).values
    ok = np.isfinite(N) & np.isfinite(E)
    Ep = np.full(len(E), np.nan); A = np.c_[np.ones(ok.sum()), N[ok]]
    Ep[ok] = E[ok] - A @ np.linalg.lstsq(A, E[ok], rcond=None)[0]
    return N, Ep


def readout(X, y, fold):
    """Held-out ridge readout of y from X for every haiku (PCA to 64 dims and ridge fit on the training folds; NaN targets never trained on)."""
    out, has = np.empty(len(X)), np.isfinite(y)
    for k in np.unique(fold):
        tr, te = (fold != k) & has, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pca = PCA(64, random_state=0).fit((X[tr] - mu) / sd)
        out[te] = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(pca.transform((X[tr] - mu) / sd), y[tr]).predict(pca.transform((X[te] - mu) / sd))
    return out


def separation(g, axis_scores):
    """Held-out AUC of every ground-truth contrast along its axis."""
    out = {}
    for name, (lo, hi, ax) in CONTRASTS.items():
        sel = g.isin([lo, hi]).values
        out[name] = roc_auc_score((g == hi).values[sel], axis_scores[ax][sel])
    return out


def main():
    it = items("haiku"); g = group_of(it); fold = folds(it)
    S = ratings("haiku").reindex(it.index)
    Zc = (S - S.mean()) / S.std()
    local = [p.stem for p in (HIDDEN / "haiku").glob("*.npz")]
    rows = []
    for r in tqdm(sorted(local, key=lambda r: (fam(r), BY_NAME[r].params_b)), desc="models"):
        N, Ep = targets(Zc, r)
        ids, layers, H = hidden("haiku", r, "mean")
        pos = pd.Series(np.arange(len(ids)), index=ids).reindex(it.index).values
        for li, layer in enumerate(layers):
            X = H[pos, li].astype(np.float32)
            sep = separation(g, {"obvious": readout(X, N, fold), "expert": readout(X, Ep, fold)})
            rows += [dict(model=r, family=fam(r), params_b=BY_NAME[r].params_b, layer=int(layer), depth=layer / layers.max(), contrast=c, auc=a) for c, a in sep.items()]
    for r in S.columns:  # what each judge says, same contrasts
        for name, (lo, hi, _) in CONTRASTS.items():
            sel = g.isin([lo, hi]).values & S[r].notna().values
            rows.append(dict(model=r, family=fam(r), params_b=BY_NAME[r].params_b, layer=-1, depth=np.nan, contrast=name, auc=roc_auc_score((g == hi).values[sel], S[r].values[sel])))
    D = pd.DataFrame(rows)
    record("taste_space", D.to_dict("records"))
    pd.set_option("display.width", 250)
    print(D[D.layer >= 0].groupby(["model", "contrast"]).auc.agg(["first", "max"]).unstack().round(2).to_string())
    print(D[D.layer < 0].pivot(index="model", columns="contrast", values="auc").round(2).to_string())
    plot_curves(D)
    plot_planes(it, g, fold, Zc)


def plot_curves(D):
    """Held-out AUC vs depth for the novice and the expert, per contrast; and the best layer vs what the model says, vs size."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout="constrained")
    for ax, name in zip(axs, CONTRASTS):
        d = D[(D.contrast == name)]
        for f, dd in d[d.layer >= 0].groupby("family"):
            best = dd.groupby("model").agg(params_b=("params_b", "first"), auc=("auc", "max")).sort_values("params_b")
            emb = dd[dd.layer == 0].set_index("model").reindex(best.index)
            ax.plot(best.params_b, best.auc, color=FAMILY_COLOR[f], lw=2, marker="o", ms=6, mec="white", mew=1.2, label=f"{f.capitalize()}: best layer")
            ax.plot(best.params_b, emb.auc, color=FAMILY_COLOR[f], lw=1.5, ls=(0, (1, 1.5)), marker="s", ms=4, label=f"{f.capitalize()}: layer 0 (raw stimulus)")
        for f, dd in d[(d.layer < 0) & d.family.isin(["qwen", "gemma"])].groupby("family"):
            dd = dd.sort_values("params_b")
            ax.plot(dd.params_b, dd.auc, color=FAMILY_COLOR[f], lw=2, ls=(0, (4, 2)), marker="o", ms=6, mfc="white", mec=FAMILY_COLOR[f], mew=1.5, label=f"{f.capitalize()}: what it says")
        ax.axhline(0.5, color="#8a8984", lw=1); ax.set_xscale("log"); ax.set_title(name, loc="left"); ax.set_xlabel("model size (billion parameters)")
    axs[0].set_ylabel("held-out AUC"); axs[0].set_ylim(0.3, 1.0)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save(fig, "taste_space_auc")


def plot_planes(it, g, fold, Zc):
    """Raw stimulus (layer 0), novice and expert representations, each on its own held-out obvious / expert axes, coloured by the
    writer (ground truth)."""
    show = g.isin(["weak", "open", "frontier"]).values
    sub = np.flatnonzero(show); sub = RNG.choice(sub, min(1200, len(sub)), replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9), layout="constrained")
    for ax, (r, which, ttl) in zip(axs, [("qwen3.5-27b", 0, "raw stimulus x (token embeddings)"), ("qwen3.5-0.8b", 2 / 3, "novice: Qwen3.5-0.8B"),
                                         ("qwen3.5-27b", 2 / 3, "expert: Qwen3.5-27B")]):
        ids, layers, H = hidden("haiku", r, "mean")
        li = int(np.argmin(np.abs(layers - which * layers.max())))
        X = H[pd.Series(np.arange(len(ids)), index=ids).reindex(it.index).values, li].astype(np.float32)
        N, Ep = targets(Zc, r)
        a, b = readout(X, N, fold), readout(X, Ep, fold)
        sep = separation(g, {"obvious": a, "expert": b})
        a, b = (a - a[show].mean()) / a[show].std(), (b - b[show].mean()) / b[show].std()
        for grp in ["open", "weak", "frontier"]:
            m = sub[g.values[sub] == grp]
            ax.scatter(a[m], b[m], s=8, c=GROUP_COLOR[grp], linewidths=0, alpha=0.85, label={"weak": "weak writers", "open": "competent open models", "frontier": "frontier models"}[grp])
        ax.set_title(f"{ttl}\nweak|open AUC {sep[list(CONTRASTS)[0]]:.2f} · open|frontier AUC {sep[list(CONTRASTS)[1]]:.2f}", loc="left")
        ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
        ax.set_xlabel("obvious axis"); ax.set_ylabel("expert axis"); ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        lim = np.percentile(np.abs(np.r_[a[sub], b[sub]]), 99) * 1.1; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, markerscale=2)
    save(fig, "taste_space_planes")


if __name__ == "__main__":
    main()
