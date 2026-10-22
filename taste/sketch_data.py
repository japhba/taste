"""The model slide's sketch drawn from data: Qwen3.5-27B (a high-brow expert) as the subject, its representation of the model-written
haiku as the space, its own ratings as the colour. One panel per sketch panel:
  raw stimulus x        the first two principal components of the token embeddings (layer 0), averaged over the haiku
  shared vs expert      all-haiku PC1 (needs no expertise) against all-haiku PC5 (needs expertise), from taste.taste_features
  shared vs brow        all-haiku PC1 against the topic-removed kitsch feature (craft PC2), as on the main slide
  shared vs subjective  all-haiku PC1 against what Qwen3.5-27B likes beyond another high-brow expert, DeepSeek-V4.1: the difference of
                        their z-scored ratings (topic means removed), read out of Qwen3.5-27B's representation (2/3 depth), held out by
                        topic (taste.taste_space.readout). No unsupervised feature splits two large judges: their weights on every top-20
                        component share a sign and differ only in size (largest gap: craft PC6, Qwen3.5-27B 0.30 vs Claude 0.15).
Blue arrow: Qwen3.5-27B's weights (Spearman rho of its rating with each axis); dashed orange: Qwen3.5-0.8B's (a low-brow novice), and
DeepSeek-V4.1's in the subjective panel. One square PNG per panel (sketch_data_0..3), drawn like the sketch: axes through the origin, no ticks.

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
BLUE, ORANGE = "#1c5cab", "#d9822b"
N_SHOW = 700


def axes_like_sketch(ax, xl, yl, lim):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_position("zero")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_box_aspect(1)
    ax.text(1, 0.52, xl, transform=ax.transAxes, ha="right", va="bottom"); ax.text(0.52, 1, yl, transform=ax.transAxes, ha="left", va="top")


def arrow(ax, w, lim, color, dashed):
    u = np.asarray(w) / np.linalg.norm(w) * 0.55 * lim
    ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, ls=(0, (3, 2)) if dashed else "-", mutation_scale=14))


def main():
    it = items("haiku"); S = ratings("haiku").reindex(it.index)
    B = pd.read_parquet(DATA / "steer/haiku_features.parquet"); Bc = pd.read_parquet(DATA / "steer/haiku_features_craft.parquet")
    ids = Bc.index; s = S.loc[ids]
    ids0, layers, H = hidden("haiku", EXPERT, "mean"); X0 = pd.DataFrame(H[:, list(layers).index(0)].astype(np.float32), index=ids0).loc[ids].values
    raw = PCA(2, random_state=0).fit_transform((X0 - X0.mean(0)) / (X0.std(0) + 1e-6))
    X = pd.DataFrame(H[:, int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))].astype(np.float32), index=ids0).loc[ids].values
    sd = s - s.groupby(it.topic[ids]).transform("mean"); z = (sd - sd.mean()) / sd.std()
    subj = readout(X, (z[EXPERT] - z[OTHER]).values, folds(it.loc[ids]))
    planes = {"x": (raw[:, 0], raw[:, 1], "$x_1$", "$x_2$", None),
              "expert": (B.PC1[ids].values, B.PC5[ids].values, r"$\phi^{\mathrm{shared}}$", r"$\phi^{\mathrm{expert}}$", NOVICE),
              "brow": (B.PC1[ids].values, Bc.PC2[ids].values, r"$\phi^{\mathrm{shared}}$", r"$\phi^{\mathrm{brow}}$", NOVICE),
              "subjective": (B.PC1[ids].values, subj, r"$\phi^{\mathrm{shared}}$", r"$\phi^{\mathrm{subjective}}$", OTHER)}
    pct = s[EXPERT].rank(pct=True).values; show = np.random.default_rng(0).choice(len(ids), N_SHOW, replace=False)
    show = show[np.argsort(np.abs(pct[show] - 0.5))]  # extremes drawn last
    rho = lambda a, j: stats.spearmanr(a, s[j], nan_policy="omit").statistic
    out = {}
    for k, (name, (a, b, xl, yl, other)) in enumerate(planes.items()):
        a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std(); lim = 2.6
        w = {j: (rho(a, j), rho(b, j)) for j in [EXPERT, NOVICE, OTHER]}
        out[name] = dict(weights=w)
        fig, ax = plt.subplots(figsize=(2.0, 2.0), layout="constrained")
        ax.scatter(a[show], b[show], c=pct[show], cmap=QUALITY, vmin=0, vmax=1, s=7, linewidths=0)
        if other:
            arrow(ax, w[other], lim, ORANGE, True); arrow(ax, w[EXPERT], lim, BLUE, False)
        axes_like_sketch(ax, xl, yl, lim)
        save(fig, f"sketch_data_{k}")
    out["subjective"]["heldout_rho"] = stats.spearmanr(subj, z[EXPERT] - z[OTHER]).statistic
    record("sketch_data", out)
    for name, o in out.items(): print(name, {j: tuple(round(x, 2) for x in v) for j, v in o["weights"].items()})


if __name__ == "__main__":
    main()
