"""The model slide's sketch from data: the latent space of a high-brow expert (Qwen3.5-27B, top row) and of a low-brow novice
(Qwen3.5-0.8B, bottom row), each coloured by that judge's own ratings of the 4,000 model-written haiku. Columns as in the sketch:
  raw stimulus x        the first two principal components of the judge's token embeddings (layer 0), averaged over the haiku
  shared vs expert      all-haiku PC1 (needs no expertise) against all-haiku PC5 (needs expertise), from taste.taste_features
  shared vs brow        all-haiku PC1 against the topic-removed kitsch feature (craft PC2), as on the main slide
  shared vs subjective  all-haiku PC1 against what Qwen3.5-27B likes beyond another high-brow expert, DeepSeek-V4.1: the difference of
                        their z-scored ratings (topic means removed), read out of Qwen3.5-27B's representation, held out by topic. No
                        unsupervised feature splits two large judges: their weights on every top-20 component share a sign.
The features are Qwen3.5-27B's (its principal components, at 2/3 depth); in the novice's row the same coordinates are read out of the
novice's own representation (ridge, held out by topic, taste.taste_space.readout), which holds them almost as well. Arrow: the judge's
weights, the Spearman rho of its rating with each axis.

  python -m taste.sketch_data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from taste.common import ratings, items, hidden, save, record
from taste.paths import DATA
from taste.taste_space_images import QUALITY
from taste.taste_space import readout, folds
from taste.probes import LAYER_FRAC

EXPERT, NOVICE, OTHER = "qwen3.5-27b", "qwen3.5-0.8b", "deepseek-v4.1-flash-1017"
ROWS = {EXPERT: ("high-brow expert\nQwen3.5-27B", "#1c5cab"), NOVICE: ("low-brow novice\nQwen3.5-0.8B", "#d9822b")}
LABELS = [("$x_1$", "$x_2$")] + [(r"$\phi^{\mathrm{shared}}$", rf"$\phi^{{\mathrm{{{k}}}}}$") for k in ["expert", "brow", "subjective"]]
N_SHOW, LIM = 700, 2.6


def states(model, ids, depth):
    i0, layers, H = hidden("haiku", model, "mean")
    li = list(layers).index(0) if depth == 0 else int(np.argmin(np.abs(layers - depth * layers.max())))
    return pd.DataFrame(H[:, li].astype(np.float32), index=i0).loc[ids].values


def axes_like_sketch(ax, xl, yl):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_position("zero")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_box_aspect(1)
    ax.text(1, 0.52, xl, transform=ax.transAxes, ha="right", va="bottom"); ax.text(0.52, 1, yl, transform=ax.transAxes, ha="left", va="top")


def main():
    it = items("haiku"); S = ratings("haiku").reindex(it.index)
    B = pd.read_parquet(DATA / "steer/haiku_features.parquet"); Bc = pd.read_parquet(DATA / "steer/haiku_features_craft.parquet")
    ids = Bc.index; s = S.loc[ids]; fold = folds(it.loc[ids])
    sd = s - s.groupby(it.topic[ids]).transform("mean"); z = (sd - sd.mean()) / sd.std()
    subj = readout(states(EXPERT, ids, LAYER_FRAC), (z[EXPERT] - z[OTHER]).values, fold)
    feats = [B.PC1[ids].values, B.PC5[ids].values, Bc.PC2[ids].values, subj]  # Qwen3.5-27B's coordinates
    show = np.random.default_rng(0).choice(len(ids), N_SHOW, replace=False)
    fig, axs = plt.subplots(2, 4, figsize=(9.6, 4.5), layout="constrained")
    out = {}
    for row, (judge, (name, colour)) in zip(axs, ROWS.items()):
        X0 = states(judge, ids, 0); raw = PCA(2, random_state=0).fit_transform((X0 - X0.mean(0)) / (X0.std(0) + 1e-6))
        if judge == EXPERT: F = feats; fit = [1.0] * 4
        else:
            X = states(judge, ids, LAYER_FRAC); F = [readout(X, f, fold) for f in feats]  # the same features, in the novice's own states
            fit = [stats.spearmanr(a, b).statistic for a, b in zip(F, feats)]
        planes = [(raw[:, 0], raw[:, 1])] + [(F[0], F[k]) for k in (1, 2, 3)]
        pct = s[judge].rank(pct=True).values; order = show[np.argsort(np.abs(pct[show] - 0.5))]  # extremes drawn last
        w = []
        for ax, (a, b), (xl, yl) in zip(row, planes, LABELS):
            a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
            sc = ax.scatter(a[order], b[order], c=pct[order], cmap=QUALITY, vmin=0, vmax=1, s=6, linewidths=0)
            wk = (stats.spearmanr(a, s[judge]).statistic, stats.spearmanr(b, s[judge]).statistic); w.append(wk)
            if xl != "$x_1$":
                u = np.array(wk) / np.linalg.norm(wk) * 0.55 * LIM
                ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2, mutation_scale=14))
            axes_like_sketch(ax, xl, yl)
        row[0].annotate(name, (-0.06, 0.5), xycoords="axes fraction", ha="right", va="center")
        out[judge] = dict(weights=w, readout_rho=fit)
    fig.colorbar(sc, ax=axs, shrink=0.6, label="the judge's own rating (percentile)")
    out["subjective_heldout_rho"] = stats.spearmanr(subj, z[EXPERT] - z[OTHER]).statistic
    record("sketch_data", out)
    for j in ROWS: print(j, "weights", [tuple(round(x, 2) for x in v) for v in out[j]["weights"]], "readout rho", np.round(out[j]["readout_rho"], 2))
    save(fig, "latent_expert_novice")


if __name__ == "__main__":
    main()
