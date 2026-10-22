"""The model of taste in representation space, on images, where the raw stimulus is truly raw: pixels.

Do pixels separate good from bad images, and do the judges' obvious and expert axes, read out of each model's representation,
separate them better? Same axes as taste_space (novice consensus N; the experts' consensus beyond it, E⊥; the model read is left
out of both), fit within each image set. Ground truth: the human mean score (AVA photographs: ~200 DPChallenge voters each; LAPIS
paintings: the LAPIS raters). Reported: held-out AUC of the top vs bottom third by human score along each axis, read from pixels
(32x32 RGB), from simple image statistics, and from every model and layer (layer 0 is already past the vision encoder); next to
the AUC of each judge's own rating, and of the human score read out directly (how much of the quality a representation holds).

  python -m taste.taste_space_images
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from taste.common import ratings, items, hidden, save, record, fam, zscore, FAMILY_COLOR, HIDDEN, DATA
from taste.models import BY_NAME
from taste.taste_space import targets, readout

SETS = {"ava": "AVA photographs", "lapis": "LAPIS paintings"}
AXES = {"obvious": "obvious axis (novices)", "expert": "expert axis (beyond novices)", "human": "human score itself (supervised)"}
RAW = {"pixels": "pixels (raw stimulus)", "image statistics": "image statistics (colour, contrast, edges, ...)"}
PIX = DATA / "images_pixels.npz"
QUALITY = LinearSegmentedColormap.from_list("quality", ["#c0302f", "#d9d6cf", "#1b8f65"])
RNG = np.random.default_rng(0)


def pixels(it):
    """32x32 RGB thumbnails (centre crop), cached."""
    if not PIX.exists():
        X = np.stack([np.asarray(ImageOps.fit(Image.open(DATA / p).convert("RGB"), (32, 32), Image.LANCZOS), np.float32).ravel() / 255 for p in tqdm(it.path, desc="pixels")])
        np.savez(PIX, item_id=it.index.values.astype(object), X=X)
    z = np.load(PIX, allow_pickle=True)
    return pd.DataFrame(z["X"], index=z["item_id"]).reindex(it.index).values


def high_low(gt, score):
    """AUC of the top vs the bottom third by human score."""
    lo, hi = gt <= np.quantile(gt, 1 / 3), gt >= np.quantile(gt, 2 / 3)
    return roc_auc_score(hi[lo | hi], score[lo | hi])


def representations(it):
    """name -> (layer, depth, items x features) for the raw stimulus and every model layer."""
    stats = pd.read_parquet(DATA / "images_features.parquet").set_index("item_id").reindex(it.index).values
    out = {"pixels": [(0, 0.0, pixels(it))], "image statistics": [(0, 0.0, stats)]}
    for r in sorted((p.stem for p in (HIDDEN / "images").glob("*.npz")), key=lambda r: (fam(r), BY_NAME[r].params_b)):
        ids, layers, H = hidden("images", r, "mean")
        pos = pd.Series(np.arange(len(ids)), index=ids).reindex(it.index).values
        out[r] = [(int(l), l / layers.max(), H[pos, li].astype(np.float32)) for li, l in enumerate(layers)]
    return out


def main():
    it = items("images"); S = ratings("images").reindex(it.index); reps = representations(it)
    rows = []
    for ds in SETS:
        m = (it.dataset == ds).values; gt = it.human_score.values[m]; fold = RNG.integers(0, 5, m.sum()); Zc = zscore(S[m])
        for r, layers in tqdm(reps.items(), desc=ds):
            N, Ep = targets(Zc, r)
            for layer, depth, X in layers:
                for ax, y in [("obvious", N), ("expert", Ep), ("human", gt)]:
                    rows.append(dict(set=ds, model=r, family=fam(r) if r in BY_NAME else "raw", params_b=BY_NAME[r].params_b if r in BY_NAME else np.nan,
                                     layer=layer, depth=depth, axis=ax, auc=high_low(gt, readout(X[m], y, fold))))
        for r in S.columns[S[m].notna().any().values]:  # what each judge says
            s = S[r].values[m]; ok = np.isfinite(s)
            rows.append(dict(set=ds, model=r, family=fam(r), params_b=BY_NAME[r].params_b, layer=-1, depth=np.nan, axis="says", auc=high_low(gt[ok], s[ok])))
    D = pd.DataFrame(rows)
    record("taste_space_images", D.to_dict("records"))
    pd.set_option("display.width", 250)
    print(D[D.layer >= 0].groupby(["set", "model", "axis"]).auc.agg(["first", "max"]).unstack().round(2).to_string())
    print(D[D.layer < 0].pivot(index="model", columns="set", values="auc").round(2).to_string())
    plot_curves(D)
    plot_planes(it, S, reps)


def plot_curves(D):
    """Per image set and axis: best layer and layer 0 vs model size, pixels and image statistics as lines, what each judge says."""
    fig, axs = plt.subplots(len(SETS), len(AXES), figsize=(12, 6.6), sharey=True, layout="constrained")
    for row, ds in zip(axs, SETS):
        for ax, a in zip(row, AXES):
            d = D[(D.set == ds) & (D.axis == a)]
            for f, dd in d[d.family.isin(["qwen", "gemma"])].groupby("family"):
                best = dd.groupby("model").agg(params_b=("params_b", "first"), auc=("auc", "max")).sort_values("params_b")
                l0 = dd[dd.layer == 0].set_index("model").reindex(best.index)
                ax.plot(best.params_b, best.auc, color=FAMILY_COLOR[f], lw=2, marker="o", ms=6, mec="white", mew=1.2, label=f"{f.capitalize()}: best layer")
                ax.plot(best.params_b, l0.auc, color=FAMILY_COLOR[f], lw=1.5, ls=(0, (1, 1.5)), marker="s", ms=4, label=f"{f.capitalize()}: layer 0 (after the vision encoder)")
            says = D[(D.set == ds) & (D.axis == "says") & D.family.isin(["qwen", "gemma"])]
            for f, dd in says.groupby("family"):
                dd = dd.sort_values("params_b")
                ax.plot(dd.params_b, dd.auc, color=FAMILY_COLOR[f], lw=2, ls=(0, (4, 2)), marker="o", ms=6, mfc="white", mec=FAMILY_COLOR[f], mew=1.5, label=f"{f.capitalize()}: what it says")
            for raw, ls in zip(RAW, ["-", ":"]):
                ax.axhline(d[d.model == raw].auc.iloc[0], color="#3d3d3a", lw=1.5, ls=ls, label=RAW[raw])
            ax.axhline(0.5, color="#8a8984", lw=1); ax.set_xscale("log")
            if ds == list(SETS)[0]: ax.set_title(AXES[a], loc="left")
        row[0].set_ylabel(f"{SETS[ds]}\ntop vs bottom third, held-out AUC")
    for ax in axs[-1]: ax.set_xlabel("model size (billion parameters)")
    axs[0, 0].set_ylim(0.4, 1.0)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    save(fig, "taste_space_images_auc")


def plot_planes(it, S, reps):
    """Pixels, a novice and an expert representation, each on its own held-out obvious / expert axes, coloured by the human score."""
    show = [("pixels", 0, "pixels (raw stimulus)"), ("qwen3.5-0.8b", 2 / 3, "novice: Qwen3.5-0.8B"), ("qwen3.5-27b", 2 / 3, "expert: Qwen3.5-27B")]
    fig, axs = plt.subplots(len(SETS), len(show), figsize=(11, 7.6), layout="constrained")
    for row, ds in zip(axs, SETS):
        m = (it.dataset == ds).values; gt = it.human_score.values[m]; fold = RNG.integers(0, 5, m.sum()); Zc = zscore(S[m])
        pct = pd.Series(gt).rank(pct=True).values; order = np.argsort(np.abs(pct - 0.5))  # extremes drawn last
        for ax, (r, which, ttl) in zip(row, show):
            layers = reps[r]; layer, _, X = layers[int(np.argmin([abs(d - which) for _, d, _ in layers]))]
            N, Ep = targets(Zc, r)
            a, b = readout(X[m], N, fold), readout(X[m], Ep, fold)
            auc_a, auc_b = high_low(gt, a), high_low(gt, b)
            a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
            sc = ax.scatter(a[order], b[order], s=10, c=pct[order], cmap=QUALITY, vmin=0, vmax=1, linewidths=0)
            ax.set_title(f"{SETS[ds]}\n{ttl}\nAUC: obvious {auc_a:.2f}, expert {auc_b:.2f}", loc="left")
            ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
            ax.set_xlabel("obvious axis"); ax.set_ylabel("expert axis"); ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
            lim = np.percentile(np.abs(np.r_[a, b]), 99) * 1.1; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.colorbar(sc, ax=axs, shrink=0.5, label="human score (percentile within the set)")
    save(fig, "taste_space_images_planes")


if __name__ == "__main__":
    import sys, json
    from taste.common import RESULTS
    if "--plot" in sys.argv: plot_curves(pd.DataFrame(json.load(open(RESULTS))["taste_space_images"]))
    else: main()
