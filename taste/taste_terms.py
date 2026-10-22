"""The model of taste, one panel per term, from saved results (no new computation):
  needs no expertise     a feature judges of every size weight the same way (all-haiku PC1 of Qwen3.5-27B, taste.taste_features)
  needs expertise        open vs frontier writers: AUC read out of each model's representation (best layer) vs what it says (taste.taste_space)
  high-brow vs low-brow  the topic-removed kitsch feature (craft PC2): each judge's weight, and the steering effect (taste.steer_features)
  subjective     share of one vote that is the voter's own stable taste: people (LAPIS) and LLM judges (taste.taste_model)
  random         share of one vote that does not replicate: people's repeat ratings and LLM judges' vote-to-vote noise

  python -m taste.taste_terms
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taste.common import save, fam, FAMILY_COLOR, RESULTS, ANCHOR
from taste.models import BY_NAME

CLAUDE_X = 1600
SIZE = lambda j: CLAUDE_X if j == ANCHOR else BY_NAME[j].params_b
PEOPLE = FAMILY_COLOR["human"]
LABEL = {"qwen": "Qwen judges", "gemma": "Gemma judges"}


DS = "deepseek-v4.1-flash-1017"


def by_size(ax, s, **kw):
    """Qwen and Gemma lines against size, DeepSeek as a diamond, Claude as a star right of a divider."""
    for f in ["qwen", "gemma"]:
        d = s[[j for j in s.index if fam(j) == f]]; d = d.reindex(sorted(d.index, key=SIZE))
        ax.plot([SIZE(j) for j in d.index], d.values, color=FAMILY_COLOR[f], marker="o", lw=2, label=LABEL[f], **kw)
    if DS in s.index: ax.plot([SIZE(DS)], [s[DS]], marker="D", ms=8, color="#e87ba4", ls="none", label="DeepSeek-V4.1-Flash", mfc="white" if kw.get("ls") else "#e87ba4")
    if ANCHOR in s.index: ax.plot([CLAUDE_X], [s[ANCHOR]], marker="*", ms=13, color=FAMILY_COLOR["frontier"], ls="none", label="Claude Opus 5.5")
    ax.set_xscale("log"); ax.axvline(900, color="#d6d4cf", lw=1); ax.set_xlabel("judge size (billion parameters)")


def main():
    R = json.loads(RESULTS.read_text())
    W, Wc = pd.DataFrame(R["taste_features"]["weights"]), pd.DataFrame(R["taste_features_craft"]["weights"])
    lab, labc = R["taste_features"]["labels"], R["taste_features_craft"]["labels"]
    Hs = pd.DataFrame(R["taste_space"]); ef = "open vs frontier models (expert axis)"
    rep = Hs[(Hs.layer >= 0) & (Hs.contrast == ef)].groupby("model").auc.max(); says = Hs[(Hs.layer < 0) & (Hs.contrast == ef)].set_index("model").auc
    E = pd.DataFrame(R["steering"]["haiku_craft"]["effects"]).query("feature == 'PC2'").set_index("judge")
    tm = pd.DataFrame(R["taste_model"]["shares"]).set_index("judge")
    L = pd.DataFrame(R["lapis_taste"]["shares"]).set_index(["group", "part"]).share
    parts = L.index.get_level_values(1).unique()

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), layout="constrained")
    ax = axs[0, 0]; by_size(ax, W["PC1"].drop(index=[j for j in W.index if fam(j) not in ("qwen", "gemma") and j not in (ANCHOR, DS)]))
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_ylabel("weight: ρ(feature, judge's rating)")
    ax.set_title(f"needs no expertise: every judge rewards it\n“{lab['PC1']['low']}”\n→ “{lab['PC1']['high']}”", loc="left")
    ax.legend(frameon=False)
    ax = axs[0, 1]
    by_size(ax, rep.drop(index=[j for j in rep.index if fam(j) not in ("qwen", "gemma") and j != DS]), ls=(0, (1, 1.5)))
    by_size(ax, says.drop(index=[j for j in says.index if fam(j) not in ("qwen", "gemma") and j not in (ANCHOR, DS)]))
    ax.axhline(0.5, color="#8a8984", lw=0.8); ax.set_ylabel("open vs frontier writers, AUC")
    h = [plt.Line2D([], [], color="#3d3d3a", ls=(0, (1, 1.5)), lw=2, label="read out of the representation"), plt.Line2D([], [], color="#3d3d3a", lw=2, label="what the judge says")]
    ax.legend(handles=h, frameon=False, loc="lower right"); ax.set_title("needs expertise: open vs frontier writers\nevery model represents it,\nonly large judges use it", loc="left")
    ax = axs[0, 2]; by_size(ax, Wc["PC2"].drop(index=[j for j in Wc.index if fam(j) not in ("qwen", "gemma") and j not in (ANCHOR, DS)]))
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_ylabel("weight: ρ(feature, judge's rating)")
    ax.set_title(f"high-brow vs low-brow: small and large judges disagree\n“{labc['PC2']['low']}”\n→ “{labc['PC2']['high']}”", loc="left")
    ax = axs[1, 0]
    C = pd.DataFrame(R["steering"]["controls_craft"]).groupby("judge").effect.agg(["min", "max"])
    ax.vlines([SIZE(j) for j in C.index], C["min"], C["max"], color="#c9c7c0", lw=6, label="random directions, same size")
    by_size(ax, E.effect); ax.axhline(0, color="#8a8984", lw=0.8)
    ax.legend(handles=[h for h in ax.get_legend_handles_labels()[0] if h.get_label().startswith("random")], frameon=False, loc="lower right")  # colours as in the first panel
    ax.set_ylabel("rating change per unit of steering"); ax.set_title("high-brow vs low-brow, causally: steering each judge\nalong its own direction for that feature", loc="left")
    for ax, part, people_part, ttl in [(axs[1, 1], "subjective: its own", parts[2], "subjective: stable, but the voter's own"),
                                       (axs[1, 2], "random (vote to vote)", parts[3], "random: does not replicate")]:
        s = tm[part].reindex(sorted(tm.index, key=lambda j: (fam(j) != "qwen", BY_NAME[j].params_b)))
        names = [j.replace("deepseek-v4.1-flash-1017", "DeepSeek 763B").replace("qwen3.5-", "Qwen ").replace("gemma-4-", "Gemma ").replace("-a10b", "").replace("-a17b", "").replace("-a4b", "").replace("b", "B").replace("eB", "E").replace("e2B", "E2B").replace("e4B", "E4B") for j in s.index]
        y = np.arange(len(s))
        ax.barh(y, s.values, color=[FAMILY_COLOR[fam(j)] for j in s.index], height=0.7)
        ax.barh([len(s) + 0.6, len(s) + 1.6], [L[("lay", people_part)], L[("lover", people_part)]], color=PEOPLE, height=0.7)
        ax.set_yticks(list(y) + [len(s) + 0.6, len(s) + 1.6], names + ["people: lay vote", "people: art lover's vote"])
        ax.set_xlim(0, 1); ax.set_xlabel("share of one vote's variance"); ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    save(fig, "taste_terms")


def kitsch():
    """Each judge's weight on the topic-removed kitsch feature (craft PC2) against its size, for the high-brow vs low-brow slide."""
    R = json.loads(RESULTS.read_text()); W = pd.Series(R["taste_features_craft"]["weights"]["PC2"]); lab = R["taste_features_craft"]["labels"]["PC2"]
    DS = "deepseek-v4.1-flash-1017"
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    ax.axhspan(-0.45, 0, color="#f6e3e2", zorder=0); ax.axhline(0, color="#8a8984", lw=0.8)
    by_size(ax, W.drop(index=[j for j in W.index if fam(j) not in ("qwen", "gemma") and j != ANCHOR]))
    if DS in W.index:
        ax.plot([BY_NAME[DS].params_b], [W[DS]], marker="D", ms=9, color="#e87ba4", ls="none", label="DeepSeek-V4.1-Flash")
    ax.set_ylim(-0.45, 0.5); ax.set_ylabel("judge's weight on the feature (ρ)")
    ax.text(0.03, 0.04, f"prefers “{lab['low']}”", transform=ax.transAxes, color="#8f2a29")
    ax.text(0.03, 0.93, f"prefers “{lab['high']}”", transform=ax.transAxes, color="#14684a")
    ax.legend(frameon=False, loc="center right")
    save(fig, "kitsch_by_size")


def pooled(n_boot=300):
    """The three expertise terms on one scale for the main deck: how often a judge rates a haiku from one end above one from the other
    (AUC, 95% bootstrap interval), every judge pooled into one line against size (Claude at a guessed ~2T). Shared: top vs bottom fifth of
    the all-haiku PC1. Expert: frontier vs open models' haiku. Brow: top vs bottom fifth of the topic-removed kitsch feature (craft PC2),
    ratings with topic means removed. Above and below each panel the two ends, each by one haiku: for the features, a clear one from that
    end's fifth (the brow pair as on the high-brow slide); for the writers, an open and a frontier haiku on one topic, each closest to its
    group's mean percentile over all judges."""
    import re
    from taste.common import ratings, items, record
    from taste.paths import DATA
    from taste.findings import auc, boot_auc, PROFANITY
    from taste.taste_space import group_of
    from taste.discrim_curves import QWEN, GEMMA, CLAUDE_GUESS
    it = items("haiku"); S = ratings("haiku").reindex(it.index); J = QWEN + GEMMA + [DS, ANCHOR]
    short = it.haiku.map(lambda h: h.count("\n") == 2 and max(map(len, h.split("\n"))) <= 24) & ~it.haiku.str.contains(PROFANITY, flags=re.I)
    def ends(f, sc, starts):
        lo, hi = f.index[f <= f.quantile(0.2)], f.index[f >= f.quantile(0.8)]
        return lo, hi, sc, [next(i for i in ix if it.haiku[i].startswith(st)) for ix, st in zip((lo, hi), starts)]  # fails if not in that fifth
    B = pd.read_parquet(DATA / "steer/haiku_features.parquet").PC1; Bc = pd.read_parquet(DATA / "steer/haiku_features_craft.parquet").PC2
    Sd = S.loc[Bc.index] - S.loc[Bc.index].groupby(it.topic[Bc.index]).transform("mean")
    g = group_of(it); P = S.rank(pct=True).mean(axis=1); op, fr = it.index[(g == "open") & short], it.index[(g == "frontier") & short]
    pairs = [(t, (P[op[it.topic[op] == t]] - P[g == "open"].mean()).abs(), (P[fr[it.topic[fr] == t]] - P[g == "frontier"].mean()).abs()) for t in it.topic[op].unique()]
    t, a, b = min((p for p in pairs if len(p[1]) and len(p[2])), key=lambda p: p[1].min() + p[2].min())
    TERMS = {"needs no expertise": ends(B, S, ["So who's going to?", "Pale mist veils the lake"]),
             "needs expertise": (it.index[g == "open"], it.index[g == "frontier"], S, [a.idxmin(), b.idxmin()]),
             "high-brow vs low-brow": ends(Bc, Sd, ["Bright sounds fill the air\nJoyful echoes", "Empty chair still warm"])}
    rows = []
    for name, (lo, hi, sc, ex) in TERMS.items():
        for j in J:
            w, bb = sc.loc[lo, j].dropna().values, sc.loc[hi, j].dropna().values
            rows.append(dict(term=name, judge=j, size=CLAUDE_GUESS if j == ANCHOR else BY_NAME[j].params_b, auc=auc(w, bb), ci=list(boot_auc(w, bb, n_boot))))
    D = pd.DataFrame(rows); X = {name: dict(bottom=it.haiku[ex[0]], top=it.haiku[ex[1]]) for name, (_, _, _, ex) in TERMS.items()}
    record("taste_terms_pooled", dict(auc=D.to_dict("records"), examples=X, expert_topic=t))
    print(D.pivot(index="judge", columns="term", values="auc").round(2).to_string()); print(X)

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 6.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.27, wspace=0.12)
    box = dict(boxstyle="round,pad=0.5", fc="#f7f6f3", ec="#d6d4cf")
    for ax, (name, d) in zip(axs, D.groupby("term", sort=False)):
        d = d.sort_values("size"); ci = np.array(d.ci.tolist())
        ax.axhspan(0.2, 0.5, color="#f0efec", lw=0, zorder=0); ax.axhline(0.5, color="#8a8984", lw=1)
        ax.errorbar(d["size"], d.auc, yerr=[d.auc - ci[:, 0], ci[:, 1] - d.auc], color="black", lw=2, marker="o", ms=6, mec="white", mew=1, capsize=2)
        ax.set_xscale("log"); ax.set_xticks([1, 10, 100, 1000], ["1B", "10B", "100B", "1T"]); ax.minorticks_off(); ax.set_xlim(0.5, 4000); ax.set_ylim(0.2, 1.0)
        ax.set_xlabel("judge size (parameters)")
        ax.annotate(X[name]["top"], (0.5, 1.04), xycoords="axes fraction", ha="center", va="bottom", bbox=box, annotation_clip=False)
        ax.annotate(X[name]["bottom"], (0.5, -0.2), xycoords="axes fraction", ha="center", va="top", bbox=box, annotation_clip=False)
        ax.annotate(name, (0.5, 1.33), xycoords="axes fraction", ha="center", va="bottom", fontweight="bold", annotation_clip=False)
    axs[0].set_ylabel("prefers the top haiku over the bottom one (AUC)")
    save(fig, "taste_terms_pooled")


if __name__ == "__main__":
    import sys
    if "--kitsch" in sys.argv: kitsch()
    elif "--pooled" in sys.argv: pooled()
    else: main(); kitsch(); pooled()
