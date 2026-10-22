"""Test the talk's model of taste on the haiku ratings:

    value_θ(x) = x·φ^obvious + x·φ^non-obvious + x·φ_θ^subject + ξ

Variance of one judge's single vote (judges with a 1-9 distribution; Claude returns a digit only) is split into
  obvious       what the novices agree on (consensus of Qwen3.5-4B and Gemma-4-E4B)
  non-obvious   what the experts (27B and up, Claude) agree on beyond that
  subjective θ  the judge's remaining persistent preference shared with its own family (its "school")
  own           the judge's remaining persistent preference shared with nobody
  random ξ      vote-to-vote noise: the variance of the judge's own 1-9 distribution for that haiku
Consensuses always leave the judge out; a permutation control (consensus shuffled across haiku) must give ~0.
The picture: x = a haiku's representation (64 PCs each of Qwen3.5-27B's and Gemma-4-31B's mean hidden state at ⅔ depth);
the obvious / non-obvious directions are ridge fits of the two consensuses on x, cross-fitted over topic folds.

  python -m taste.taste_model
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from taste.common import ratings, items, hidden, save, record, rater_order, fam, FAMILY_COLOR, ANCHOR, RATINGS, DATA
from taste.models import BY_NAME

NOVICES = ["qwen3.5-4b", "gemma-4-e4b"]
EXPERTS = ["qwen3.5-27b", "gemma-4-26b-a4b", "gemma-4-31b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", ANCHOR]
FEATURE_MODELS = ["qwen3.5-27b", "gemma-4-31b"]
ALPHAS = np.logspace(-1, 5, 13)
PARTS = ["obvious", "non-obvious", "subjective: its school (family)", "subjective: its own", "random (vote to vote)"]
LABEL = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-27b": "Qwen3.5-27B",
         "qwen3.5-122b-a10b": "Qwen3.5-122B", "qwen3.5-397b-a17b": "Qwen3.5-397B", "gemma-4-e2b": "Gemma-4-E2B", "gemma-4-e4b": "Gemma-4-E4B",
         "gemma-4-26b-a4b": "Gemma-4-26B", "gemma-4-31b": "Gemma-4-31B", "deepseek-v4.1-flash-1017": "DeepSeek-V4.1-Flash"}
TINY = ["qwen3.5-0.8b", "qwen3.5-2b", "gemma-4-e2b"]  # not counted in a family's school: their taste runs against everyone else's
PART_COLOR = ["#1c5cab", "#6da7ec", "#eb6834", "#f4b598", "#c3c2b7"]  # blue pair = shared value, orange pair = the judge's own, grey = noise
RNG = np.random.default_rng(0)
# where each writer group sits in the plane (centroids): label, text offset (points), alignment
GROUPS = {"smollm": ("SmolLM", (6, -3), "left"), "open": ("Qwen / Gemma", (6, -10), "left"), "human-twaiku": ("tweets", (6, 3), "left"),
          "human-journal": ("journal", (6, -3), "left"), "claude": ("Claude", (6, 3), "left")}


def features(it):
    blocks = []
    for r in FEATURE_MODELS:
        ids, layers, H = hidden("haiku", r, "mean")
        li = int(np.argmin(np.abs(layers - 2 / 3 * layers.max())))
        X = pd.DataFrame(H[:, li].astype(np.float32), index=ids).reindex(it.index).values
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        blocks.append(PCA(64, random_state=0).fit_transform(X))
    X = np.hstack(blocks)
    return (X - X.mean(0)) / X.std(0)


def folds(it, k=5):
    topics = sorted(it.topic.dropna().unique())
    f = it.topic.map({t: i % k for i, t in enumerate(topics)})
    return f.fillna(pd.Series(RNG.integers(0, k, len(it)), index=it.index)).astype(int).values  # human haiku have no topic: random folds


def crossfit(X, y, fold):
    """Out-of-fold ridge predictions of y from X for every haiku; haiku without a target (NaN) are only predicted, never trained on."""
    out, has = np.empty(len(y)), np.isfinite(y)
    for k in np.unique(fold):
        tr = (fold != k) & has
        out[fold == k] = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[fold == k])
    return out


def r2(y, *cols):
    A = np.c_[np.ones(len(y)), *cols]
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - res.var() / y.var(), res


def axes(X, Zc, fold, without=(), shuffle=False):
    """Out-of-fold obvious and non-obvious coordinates of every haiku, from consensuses that leave out `without`."""
    N = Zc[[r for r in NOVICES if r not in without]].mean(1).values
    E = Zc[[r for r in EXPERTS if r not in without]].mean(1).values
    if shuffle: p = RNG.permutation(len(N)); N, E = N[p], E[p]
    n_hat = crossfit(X, N, fold)
    has = np.isfinite(E)
    e_res = np.full(len(E), np.nan); e_res[has] = r2(E[has], n_hat[has])[1]
    e_hat = crossfit(X, e_res, fold)
    return n_hat, e_hat


def consensus(Zc, without=(), shuffle=False):
    """Leave-out novice consensus N and the experts' consensus beyond it, E⊥ (per haiku)."""
    N = Zc[[r for r in NOVICES if r not in without]].mean(1)
    E = Zc[[r for r in EXPERTS if r not in without]].mean(1)
    if shuffle: p = RNG.permutation(len(N)); N, E = pd.Series(N.values[p], N.index), pd.Series(E.values[p], E.index)
    ok = N.notna() & E.notna()
    E_perp = pd.Series(np.nan, E.index); E_perp[ok] = r2(E[ok].values, N[ok].values)[1]
    return N, E_perp


def decompose(m, v, N, E_perp, u_school):
    """Split one judge's single-vote variance into the five parts (fractions summing to 1)."""
    var_m = m.var()
    r_obv, _ = r2(m, N)
    r_both, u = r2(m, N, E_perp)
    r_school = r2(u, u_school)[0]
    parts = np.array([r_obv * var_m, (r_both - r_obv) * var_m, r_school * u.var(), (1 - r_school) * u.var(), v.mean()])
    return parts / parts.sum()


def main():
    it = items("haiku")
    S = ratings("haiku").reindex(it.index)
    Zc = (S - S.mean()) / S.std()
    judges = [r for r in rater_order(S.columns) if r != ANCHOR]  # Claude returns a digit only: no vote distribution
    mv = {}
    for j in judges:
        P = pd.read_parquet(RATINGS / "haiku" / f"{j}.parquet").set_index("item_id").reindex(it.index)
        p = P[[f"p{k}" for k in range(1, 10)]].values; p = p / p.sum(1, keepdims=True); k = np.arange(1, 10)
        mv[j] = (pd.Series(p @ k, it.index), pd.Series(p @ k ** 2 - (p @ k) ** 2, it.index))
    # persistent residual of every judge after the two shared parts (each against consensuses without it)
    U = {}
    for j in judges:
        N, Ep = consensus(Zc, (j,)); m = mv[j][0]; ok = m.notna() & N.notna() & Ep.notna()
        U[j] = pd.Series(np.nan, it.index); U[j][ok] = r2(m[ok].values, N[ok].values, Ep[ok].values)[1]
    U = pd.DataFrame(U); U = (U - U.mean()) / U.std()
    rows, ctrl = [], []
    for j in judges:
        school = [r for r in judges if fam(r) == fam(j) and r != j and r not in TINY]
        m, v = mv[j]
        for shuffle, out in [(False, rows), (True, ctrl)]:
            N, Ep = consensus(Zc, (j,), shuffle)
            us = U[school].mean(1) if school else pd.Series(0.0, it.index)  # no family members (DeepSeek): no school, its share is 0
            ok = m.notna() & N.notna() & Ep.notna() & us.notna()
            fr = decompose(m[ok].values, v[ok].values, N[ok].values, Ep[ok].values, RNG.permutation(us[ok].to_numpy(copy=True)) if shuffle else us[ok].values)
            out.append(dict(judge=j, params_b=BY_NAME[j].params_b, family=fam(j), **dict(zip(PARTS, fr))))
        print(j, {k_: round(v_, 3) for k_, v_ in rows[-1].items() if k_ in PARTS}, flush=True)
    D, C = pd.DataFrame(rows).set_index("judge"), pd.DataFrame(ctrl).set_index("judge")
    record("taste_model", dict(shares=D.reset_index().to_dict("records"), permutation_control=C.reset_index().to_dict("records"), novices=NOVICES, experts=EXPERTS))
    print("permutation control (should be ~0 for the first three parts):\n", C[PARTS[:3]].round(3).to_string())
    plot_shares(D)
    X, fold = features(it), folds(it)
    plot_plane(it, X, fold, Zc, D)


def plot_shares(D):
    fig, ax = plt.subplots(figsize=(9, 3.1), layout="constrained")
    y = np.arange(len(D)); left = np.zeros(len(D))
    for part, col in zip(PARTS, PART_COLOR):
        ax.barh(y, D[part], left=left, color=col, height=0.7, edgecolor="white", linewidth=1, label=part); left += D[part].values
    ax.set_yticks(y, [f"{j}" for j in D.index]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("share of a single vote's variance")
    fig.legend(loc="outside lower center", ncol=3, frameon=False)
    save(fig, "taste_model_shares")


def plane(it, X, fold, Zc, nov, exp_):
    n_hat, e_hat = axes(X, Zc, fold, without=(nov, exp_))
    return (n_hat - n_hat.mean()) / n_hat.std(), (e_hat - e_hat.mean()) / e_hat.std(), PCA(2, random_state=0).fit_transform(X)


def draw_plane(axs, it, Zc, n_hat, e_hat, raw, nov, exp_):
    """The sketch on slide 6 with data: haiku in the (obvious, non-obvious) plane, coloured by a novice's and an expert's rating."""
    sub = RNG.choice(len(it), 900, replace=False)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["#c0302f", "#f0efec", "#1b8f65"])
    for ax, (xx, yy, j, ttl, xl, yl) in zip(axs, [(raw[:, 0], raw[:, 1], exp_, "the haiku x", "x₁", "x₂"),
                                                 (n_hat, e_hat, nov, f"novice: {LABEL[nov]}", "obvious", "non-obvious"),
                                                 (n_hat, e_hat, exp_, f"expert: {LABEL[exp_]}", "obvious", "non-obvious")]):
        z = Zc[j].values
        ax.scatter(xx[sub], yy[sub], c=z[sub], cmap=cmap, vmin=-2, vmax=2, s=7, linewidths=0)
        ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
        if xl == "obvious":
            A = np.c_[np.ones(len(z)), xx, yy]; ok = ~np.isnan(z); b = np.linalg.lstsq(A[ok], z[ok], rcond=None)[0][1:]
            b = b / np.linalg.norm(b) * 1.8
            ax.annotate("", xy=(b[0], b[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=2))
            for g, (lab, off, ha) in GROUPS.items():
                sel = (it.generator.str.startswith(g) if g != "open" else it.generator.str.match(r"^(qwen|gemma)")).values
                ax.plot(xx[sel].mean(), yy[sel].mean(), "o", ms=6, mfc="none", mec="#0b0b0b", mew=1.4)
                ax.annotate(lab, (xx[sel].mean(), yy[sel].mean()), xytext=off, textcoords="offset points", ha=ha, color="#0b0b0b")
        ax.set_title(ttl, loc="left"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)


def draw_compass(ax, Zc, n_hat, e_hat):
    """Every judge's taste as a direction in the (obvious, non-obvious) plane; length = how well the two axes explain its ratings."""
    show = {"qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-397b-a17b", "gemma-4-31b", ANCHOR}
    ax.axhline(0, color="#8a8984", lw=0.8); ax.axvline(0, color="#8a8984", lw=0.8)
    for j in Zc.columns:
        z = Zc[j].values; ok = ~np.isnan(z)
        b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), n_hat[ok], e_hat[ok]], z[ok], rcond=None)[0][1:]
        R = np.sqrt(r2(z[ok], n_hat[ok], e_hat[ok])[0]); b = b / np.linalg.norm(b) * R
        col = FAMILY_COLOR[fam(j)]
        ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2 if j in show else 1, alpha=1 if j in show else 0.45))
        if j in show: ax.annotate(LABEL.get(j, "Claude"), b, xytext=(4, 2), textcoords="offset points", color="#0b0b0b")
    ax.set_xlim(-0.25, 1.05); ax.set_ylim(-0.75, 0.75); ax.set_box_aspect(1)
    ax.set_title("each judge's direction", loc="left"); ax.set_xlabel("obvious"); ax.set_ylabel("non-obvious"); ax.set_xticks([]); ax.set_yticks([])


def draw_shares(ax, D):
    parts = {"obvious": D["obvious"], "non-obvious": D["non-obvious"], "subjective θ": D[PARTS[2]] + D[PARTS[3]], "random ξ": D[PARTS[4]]}
    y = np.arange(len(D)); left = np.zeros(len(D))
    for (lab, val), col in zip(parts.items(), ["#1c5cab", "#86b6ef", "#eb6834", "#c3c2b7"]):
        ax.barh(y, val, left=left, color=col, height=0.72, edgecolor="white", linewidth=1, label=lab); left += val.values
    ax.set_yticks(y, [LABEL[j] for j in D.index]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("share of the variance of one vote")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)


def plot_shares(D):
    fig, ax = plt.subplots(figsize=(9, 3.1), layout="constrained")
    draw_shares(ax, D)
    save(fig, "taste_model_shares")


def plot_plane(it, X, fold, Zc, D):
    nov, exp_ = "qwen3.5-4b", "qwen3.5-397b-a17b"
    n_hat, e_hat, raw = plane(it, X, fold, Zc, nov, exp_)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
    draw_plane(axs[:3], it, Zc, n_hat, e_hat, raw, nov, exp_)
    draw_compass(axs[3], Zc, n_hat, e_hat)
    save(fig, "taste_model_plane")


if __name__ == "__main__":
    main()
