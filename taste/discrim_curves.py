"""Human taste on x, each judge's discriminability on y: does telling better from worse improve with judge size at every level?
Haiku: the human ladder (tweets < r/haiku ignored < r/haiku top 5% < journal); at each step, the AUC of the judge's rating for the better tier.
Photos: AVA, in fifths by the human mean score (about 200 photographers per photo); at each step, the AUC between adjacent fifths.
Each point sits between the two rungs it compares. One curve per Qwen3.5 judge (lighter = smaller), DeepSeek-V4.1-Flash (763B), Claude Opus 5.5 (haiku only), and DeepSeek told to answer as a
haiku-journal editor (taste.ds_rate --persona editor). Above each tier or fifth, a representative example: the haiku whose mean
percentile over all judges is closest to its tier's mean, the photo closest to its fifth's median human score.

Companion: each judge's AUC averaged over the three steps of the haiku ladder against its size (95% bootstrap interval, resampling
haiku within each tier).

  python -m taste.discrim_curves
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from taste.common import ratings, items, save, record, ANCHOR, FAMILY_COLOR
from taste.paths import DATA, RATINGS
from taste.findings import auc, boot_auc, PROFANITY

DS = "deepseek-v4.1-flash-1017"
QWEN = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"]
RAMP = ["#c6dbf7", "#9ec5f0", "#6ea8e8", "#3987e5", "#1c5cab", "#134a8e", "#0d366b"]
SIZE = {"qwen3.5-0.8b": "0.8B", "qwen3.5-2b": "2B", "qwen3.5-4b": "4B", "qwen3.5-9b": "9B", "qwen3.5-27b": "27B", "qwen3.5-122b-a10b": "122B", "qwen3.5-397b-a17b": "397B"}
TIERS = {"human-twaiku": "found in a tweet", "human-reddit-low": "r/haiku, ignored", "human-reddit-top": "r/haiku, top 5%", "human-journal": "haiku journal"}
DS_COLOR, CLAUDE_COLOR = "#e87ba4", FAMILY_COLOR["frontier"]


def curves(scores, groups, order):
    """AUC per adjacent step for each judge (columns of `scores`), with a bootstrap 95% interval."""
    rows = []
    for k, (lo, hi) in enumerate(zip(order[:-1], order[1:])):
        for j in scores.columns:
            w, b = scores[j][groups == lo].dropna().values, scores[j][groups == hi].dropna().values
            if len(w) and len(b): rows.append(dict(step=k, judge=j, auc=auc(w, b), ci=list(boot_auc(w, b, 500))))
    return pd.DataFrame(rows)


def draw(ax, R, judges, n_tiers):
    ax.axhspan(0.25, 0.5, color="#f0efec", lw=0, zorder=0); ax.axhline(0.5, color="#8a8984", lw=1)
    x = np.arange(n_tiers - 1) + 0.5  # each point between the two rungs it compares
    for j, col in zip(QWEN, RAMP):
        if j in judges: ax.plot(x, R[R.judge == j].set_index("step").auc.reindex(range(len(x))), color=col, lw=2.2, marker="o", ms=6, mec="white", mew=1, label=f"Qwen3.5-{SIZE[j]}")
    if DS in judges: ax.plot(x, R[R.judge == DS].set_index("step").auc.reindex(range(len(x))), color=DS_COLOR, lw=2.6, marker="D", ms=7, mec="white", mew=1, label="DeepSeek-V4.1 (763B)")
    if ANCHOR in judges: ax.plot(x, R[R.judge == ANCHOR].set_index("step").auc.reindex(range(len(x))), color=CLAUDE_COLOR, lw=2, marker="*", ms=11, mec="white", mew=0.8, label="Claude Opus 5.5")
    if "editor" in judges: ax.plot(x, R[R.judge == "editor"].set_index("step").auc.reindex(range(len(x))), color=DS_COLOR, lw=2, ls=(0, (3, 2)), marker="D", ms=7, mfc="white", mec=DS_COLOR, mew=1.6, label="DeepSeek-V4.1, told it is a journal editor")
    ax.set_xlim(-0.5, n_tiers - 0.5); ax.set_ylim(0.25, 1.0); ax.set_ylabel("judge prefers the better one (AUC)")


def haiku(ax):
    it = items("haiku"); S = ratings("haiku").reindex(it.index)
    hum = it[it.generator.isin(TIERS)]
    ed = pd.read_parquet(RATINGS / "haiku_editor" / f"{DS}.parquet").set_index("item_id").score
    sc = S.loc[hum.index, QWEN + [DS, ANCHOR]].assign(editor=ed.reindex(hum.index))
    R = curves(sc, hum.generator, list(TIERS)); draw(ax, R, list(sc.columns), len(TIERS))
    P = S.loc[hum.index].rank(pct=True).mean(axis=1)  # mean percentile over all judges, within the human haiku
    for i, (g, name) in enumerate(TIERS.items()):
        short = hum.haiku.map(lambda h: max(len(l) for l in h.split("\n")) <= 24)
        cand = hum[(hum.generator == g) & ~hum.haiku.str.contains(PROFANITY, flags=re.I) & short & (hum.haiku.str.count("\n") == 2)]
        ex = (P[cand.index] - P[hum.index[hum.generator == g]].mean()).abs().idxmin()
        ax.annotate(hum.haiku[ex], (i, 1.02), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f7f6f3", ec="#d6d4cf"), annotation_clip=False)
    ax.set_xticks(np.arange(len(TIERS)), list(TIERS.values()))
    ax.set_xlabel("human taste: haiku from the bottom to the top of the ladder (each point: the two rungs on either side)")
    return R


def photos(ax):
    it = items("images"); ava = it[it.dataset == "ava"]; S = ratings("images").reindex(ava.index)
    fifth = pd.qcut(ava.human_score, 5, labels=False)
    judges = [j for j in QWEN + [DS] if j in S.columns]
    R = curves(S[judges], fifth, list(range(5))); draw(ax, R, judges, 5)
    for k in range(5):
        g = ava[fifth == k]; ex = (g.human_score - g.human_score.median()).abs().idxmin()
        im = Image.open(DATA / ava.path[ex]).convert("RGB"); im = im.resize((round(im.width * 100 / im.height), 100))  # all the same height
        ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=0.5), (k, 1.03), xycoords=("data", "axes fraction"), box_alignment=(0.5, 0), frameon=False, annotation_clip=False)
        ax.add_artist(ab)
    lo = ava.human_score.groupby(fifth).agg(["min", "max"])
    ax.set_xticks(range(5), [f"{a:.1f}–{b:.1f}" for a, b in lo.values])
    ax.set_xlabel("human taste: AVA photos in fifths by their mean score, 1–10 (each point: the two fifths on either side)")
    return R


GEMMA = ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-26b-a4b", "gemma-4-31b"]


def by_size(n_boot=300):
    """Mean AUC over the human ladder's steps per judge, with a bootstrap interval (haiku resampled within each tier)."""
    from taste.models import BY_NAME
    it = items("haiku"); S = ratings("haiku").reindex(it.index); hum = it[it.generator.isin(TIERS)]
    ed = pd.read_parquet(RATINGS / "haiku_editor" / f"{DS}.parquet").set_index("item_id").score
    sc = S.loc[hum.index, QWEN + GEMMA + [DS, ANCHOR]].assign(editor=ed.reindex(hum.index)); g = hum.generator.values; order = list(TIERS)
    rng = np.random.default_rng(0)
    def mean_auc(idx):
        out = {}
        for j in sc.columns:
            v = sc[j].values[idx]; gg = g[idx]
            out[j] = np.mean([auc(v[(gg == lo) & np.isfinite(v)], v[(gg == hi) & np.isfinite(v)]) for lo, hi in zip(order[:-1], order[1:])])
        return pd.Series(out)
    base = mean_auc(np.arange(len(g)))
    groups = [np.flatnonzero(g == t) for t in order]
    boots = pd.DataFrame([mean_auc(np.concatenate([rng.choice(ix, len(ix)) for ix in groups])) for _ in range(n_boot)])
    D = pd.DataFrame({"auc": base, "lo": boots.quantile(0.025), "hi": boots.quantile(0.975)})
    D["size"] = [BY_NAME[j].params_b if j in BY_NAME else np.nan for j in D.index]
    return D


CLAUDE_GUESS = 2000  # Claude's size is not public; ~2T is the figure people float online, drawn as a guess


def plot_by_size(D):
    """One line through every judge sorted by size (Claude at a guessed ~2T); markers keep the family; the editor persona off the line."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
    ax.axhline(0.5, color="#8a8984", lw=1)
    D = D.copy(); D.loc[ANCHOR, "size"] = CLAUDE_GUESS
    line = D.drop(index="editor").sort_values("size")
    ax.plot(line["size"], line.auc, color="#3d3d3a", lw=2, zorder=2)
    style = {**{j: ("o", "#1c5cab", "Qwen3.5") for j in QWEN}, **{j: ("o", FAMILY_COLOR["gemma"], "Gemma 4") for j in GEMMA},
             DS: ("D", DS_COLOR, "DeepSeek-V4.1 (763B)"), ANCHOR: ("*", CLAUDE_COLOR, "Claude Opus 5.5 (size guessed, ~2T)")}
    seen = set()
    for j, r in line.iterrows():
        m, c, lab = style[j]
        ax.errorbar([r["size"]], [r.auc], yerr=[[r.auc - r.lo], [r.hi - r.auc]], color=c, marker=m, ms=12 if m == "*" else 7, mec="white", mew=1, capsize=2, ls="none", zorder=3,
                    label=None if lab in seen else lab); seen.add(lab)
    r = D.loc["editor"]
    ax.errorbar([D.loc[DS, "size"]], [r.auc], yerr=[[r.auc - r.lo], [r.hi - r.auc]], color=DS_COLOR, marker="D", ms=7, mfc="white", mew=1.6, capsize=2, ls="none", zorder=3,
                label="DeepSeek-V4.1, told it is a journal editor")
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100, 1000], ["1B", "10B", "100B", "1T"]); ax.minorticks_off()
    ax.set_xlabel("judge size (parameters)"); ax.set_ylabel("prefers the better haiku, mean over the ladder (AUC)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "discrim_by_size")


def main():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.15, 1]))
    Rh, Rp = haiku(axs[0]), photos(axs[1])
    axs[1].set_ylabel("")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    record("discrim_curves", dict(haiku=Rh.to_dict("records"), photos=Rp.to_dict("records")))
    pd.set_option("display.width", 200)
    for name, R in [("haiku", Rh), ("photos", Rp)]: print(name); print(R.pivot(index="judge", columns="step", values="auc").round(2).to_string())
    save(fig, "discrim_curves")
    fig, ax = plt.subplots(figsize=(10, 6.4), layout="constrained")  # haiku alone, for a slide
    haiku(ax); fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save(fig, "discrim_haiku")
    D = by_size(); record("discrim_by_size", D.reset_index(names="judge").to_dict("records")); print(D.round(3).to_string()); plot_by_size(D)


if __name__ == "__main__":
    main()
