"""Results of the steering runs.
Haiku (taste.steer_features): for every judge and feature, the steering effect is the part of the rating change that flips sign with
alpha, (score(+a) - score(-a)) / 2a averaged over a in {1, 2, 4} and over haiku (bootstrap CI over haiku); large pushes lower
every rating whatever their sign, and this cancels that. Shown against the judge's observed weight on the feature
(taste.taste_features), plus the dose-response of the clearest high/low-brow feature for tiny and large judges.
Art (taste.steer, Qwen3.5-27B on held-out LAPIS artworks): for every condition (no persona, the art-lover and the lay persona,
the persona difference added at three depths times alpha), the model's weight on the human high-brow axis: the regression of its
z-scored ratings on the style's high-brow score (art lovers minus lay people, averaged over all ratings of that style) with the
artwork's overall human consensus as a covariate; conditions whose digit mass drops below 0.9 are left out.

  python -m taste.steer_report
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taste.common import save, record, RESULTS
from taste.paths import DATA
from taste.taste_features import TIERS

RNG = np.random.default_rng(0)
TIER_COLOR = {"tiny": "#c0302f", "small": "#e8a33a", "large": "#1b8f65"}
SHORT = lambda j: j.replace("qwen3.5-", "Qwen ").replace("gemma-4-", "Gemma ")
z = lambda x: (x - x.mean()) / x.std()


def art():
    d = pd.read_parquet(DATA / "steer/qwen3.5-27b.parquet")
    A = pd.read_parquet(DATA / "lapis_highbrow.parquet")
    A["style_highbrow"] = A["style"].map(A.groupby("style").highbrow.mean())
    d = d.join(A[["style_highbrow", "consensus"]], on="image_id")
    rows = []
    for (c, dep, al), g in d.groupby(["condition", d.depth.fillna(-1), "alpha"]):
        if g.digit_mass.mean() < 0.9: continue
        X = np.c_[np.ones(len(g)), z(g.consensus), z(g.style_highbrow)]; y = z(g.score.values)
        b = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][2] for i in [RNG.integers(0, len(g), len(g)) for _ in range(500)]]
        rows.append(dict(condition=c, depth=dep, alpha=al, mean_score=g.score.mean(), w_highbrow=np.linalg.lstsq(X, y, rcond=None)[0][2],
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    R = pd.DataFrame(rows); print(R.round(3).to_string())
    return R


def haiku(sfx=""):
    W = pd.DataFrame(json.load(open(RESULTS))["taste_features" + sfx]["weights"])  # judge x PC
    rows, curves = [], []
    for p in sorted(q for q in (DATA / f"steer{sfx}").glob("haiku_*.parquet") if not q.stem.startswith("haiku_features")):
        d = pd.read_parquet(p); j = d.judge.iloc[0]
        base = d[d.feature == "none"].set_index("item_id").score
        for f, g in d[d.feature != "none"].groupby("feature"):
            s = g.pivot(index="item_id", columns="alpha", values="score")
            anti = pd.concat([(s[a] - s[-a]) / (2 * a) for a in [1.0, 2.0, 4.0]], axis=1).mean(axis=1)
            bs = [anti.sample(frac=1, replace=True, random_state=k).mean() for k in range(1000)]
            tier = next(t for t, v in TIERS.items() if j in v)
            rows.append(dict(judge=j, feature=f, effect=anti.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), observed=W.loc[j, f],
                             digit_mass=g.digit_mass.mean(), tier=tier))
            c = (s.sub(base.reindex(s.index), axis=0)).mean(); c[0.0] = 0.0
            curves += [dict(judge=j, feature=f, tier=tier, alpha=a, delta=v) for a, v in c.sort_index().items()]
    R, C = pd.DataFrame(rows), pd.DataFrame(curves); print(R.round(3).to_string())
    return R, C


def controls(sfx="_craft"):
    """Control directions (random fifths, rescaled to the feature direction's norm; taste.steer_features --controls): the same effect measure."""
    rows = []
    files = pd.Series({p: int(p.stem.split("_")[0][len("control"):]) for p in (DATA / f"steer{sfx}").glob("control*_*.parquet")})
    files = files.groupby([p.stem.split("_", 1)[1] for p in files.index]).idxmax()  # per judge, the run with the most controls
    for p in files.values:
        d = pd.read_parquet(p)
        for f, g in d[d.feature != "none"].groupby("feature"):
            s = g.pivot(index="item_id", columns="alpha", values="score")
            anti = pd.concat([(s[a] - s[-a]) / (2 * a) for a in [1.0, 2.0, 4.0]], axis=1).mean(axis=1)
            rows.append(dict(judge=d.judge.iloc[0], control=f, effect=anti.mean(), abs_effect=abs(anti.mean())))
    C = pd.DataFrame(rows)
    if len(C): print(C.round(3).to_string())
    return C


MARKERS = ["o", "s", "D", "^", "v"]


def main():
    out = {}
    variants = [s for s in ["", "_craft"] if any(q for q in (DATA / f"steer{s}").glob("haiku_*.parquet") if not q.stem.startswith("haiku_features"))]
    fig, axs = plt.subplots(len(variants), 2, figsize=(13, 4.6 * len(variants)), layout="constrained", squeeze=False)
    for row, sfx in zip(axs, variants):
        R, C = haiku(sfx); out["haiku" + sfx] = dict(effects=R.to_dict("records"), curves=C.to_dict("records"))
        labels = json.load(open(RESULTS))["taste_features" + sfx]["labels"]
        ax = row[0]
        feats = sorted(R.feature.unique(), key=lambda k: int(k[2:]))
        for (t, f), g in R.groupby(["tier", "feature"]):
            ax.errorbar(g.observed, g.effect, yerr=[g.effect - g.lo, g.hi - g.effect], color=TIER_COLOR[t], marker=MARKERS[feats.index(f)], ms=7, ls="none", capsize=2)
        ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
        h = [plt.Line2D([], [], color=c, marker="o", ls="none", label=f"{t} judges") for t, c in TIER_COLOR.items()]
        h += [plt.Line2D([], [], color="#6b6a66", marker=MARKERS[i], ls="none", label=f"{f}: {labels[f]['low']} → {labels[f]['high']}") for i, f in enumerate(feats)]
        ax.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1)
        ax.set_xlabel("observed weight on the feature (Spearman ρ with the judge's score)"); ax.set_ylabel("rating change per unit of steering")
        agree = (np.sign(R.effect) == np.sign(R.observed)) & ((R.lo > 0) | (R.hi < 0))
        ax.set_title(f"{'topic-removed features' if sfx else 'features'}: {agree.sum()} of {len(R)} effects significant with the observed sign", loc="left")
        ax = row[1]
        f = R.assign(gap=R.groupby("feature")["observed"].transform(lambda w: w.max() - w.min())).sort_values("gap").feature.iloc[-1]  # the feature judges disagree on most
        for j, g in C[C.feature == f].groupby("judge"):
            ax.plot(g.alpha, g.delta, color=TIER_COLOR[g.tier.iloc[0]], marker="o", label=SHORT(j))
        ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
        ax.set_xlabel(f"steering α along {f}:  −  {labels[f]['low']}   |   +  {labels[f]['high']}"); ax.set_ylabel("rating change (1–9 scale)")
        ax.set_title(f"{f}, the feature the judges disagree on most", loc="left"); ax.legend(frameon=False, ncol=2)
    save(fig, "steering_haiku")
    if (DATA / "steer/qwen3.5-27b.parquet").exists():
        R = art(); out["art"] = R.to_dict("records")
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", gridspec_kw=dict(width_ratios=[1, 2.4]), sharey=True)
        P = R[R.condition.isin(["lay", "none", "lover"])].set_index("condition").loc[["lay", "none", "lover"]]
        a0.errorbar(range(3), P.w_highbrow, yerr=[P.w_highbrow - P.lo, P.hi - P.w_highbrow], color="#3d3d3a", marker="o", ms=7, capsize=3, ls="none")
        a0.set_xticks(range(3), ["no interest\nin art", "no persona", "art lover"]); a0.set_xlim(-0.5, 2.5)
        a0.axhline(0, color="#8a8984", lw=0.8); a0.set_ylabel("weight on the art lovers' high-brow axis"); a0.set_title("prompted persona", loc="left")
        for dep, c in zip(sorted(R[R.condition == "steered"].depth.unique()), ["#9ec5f0", "#2a78d6", "#123f75"]):
            st = pd.concat([R[(R.condition == "steered") & (R.depth == dep)], R[R.condition == "none"]]).sort_values("alpha")
            a1.errorbar(st.alpha, st.w_highbrow, yerr=[st.w_highbrow - st.lo, st.hi - st.w_highbrow], color=c, marker="o", capsize=3, label=f"at {dep:.0%} depth")
        a1.axhline(0, color="#8a8984", lw=0.8); a1.set_xticks([-4, -2, -1, 0, 1, 2, 4])
        a1.set_xlabel("steering α along the persona difference (− no interest in art, + art lover)"); a1.set_title("Qwen3.5-27B, no persona, steered", loc="left"); a1.legend(frameon=False)
        save(fig, "steering_art")
    C = controls()
    if len(C):
        K = pd.DataFrame(out["haiku_craft"]["effects"]).query("feature == 'PC2'").set_index("judge")
        js = [j for t in TIERS.values() for j in t if j in set(C.judge)]
        rank = {j: float((C[C.judge == j].effect.abs() >= abs(K.loc[j, "effect"])).mean()) for j in js}  # share of controls at least as large
        out["controls_craft"] = C.to_dict("records"); out["controls_craft_share_as_large"] = rank
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for x, j in enumerate(js):
            c = C[C.judge == j].effect.values
            ax.scatter(x + RNG.uniform(-0.12, 0.12, len(c)), c, s=18, color="#b9b7b0", zorder=2, label="random directions, same norm" if x == 0 else None)
            t = next(t for t, v in TIERS.items() if j in v)
            ax.scatter([x], [K.loc[j, "effect"]], s=80, color=TIER_COLOR[t], edgecolor="white", zorder=3, label=f"kitsch feature, {t} judge" if t not in [next(tt for tt, v in TIERS.items() if jj in v) for jj in js[:x]] else None)
        ax.axhline(0, color="#8a8984", lw=0.8); ax.set_xticks(range(len(js)), [SHORT(j) for j in js], rotation=30, ha="right")
        ax.set_ylabel("rating change per unit of steering"); ax.set_title("steering along the kitsch feature vs along random directions", loc="left"); ax.legend(frameon=False)
        save(fig, "steering_controls")
    record("steering", out)


if __name__ == "__main__":
    main()
