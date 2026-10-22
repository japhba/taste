"""The model of taste on real people: 281k individual ratings of 11,723 artworks by 555 people (LAPIS), with each person's art
interest (VAIAK questionnaire) and 614 repeated ratings (same person, same artwork, rated twice).

A single vote, z-scored within its rater, splits by covariances (no model fitting):
  obvious          what art lovers and lay people share: the covariance of two raters from different groups on the same artwork
  group taste      what a group shares beyond that: within-group minus cross-group covariance. For art lovers this is high-brow
                   taste, for lay people low-brow taste
  own taste        stable but nobody else's: the repeat-rating covariance minus the within-group covariance
  random           what does not replicate when the same person rates the same artwork again: 1 minus the repeat covariance
Groups are the top and bottom third by art interest. Control: the same split with art interest shuffled across raters.
Then the high-brow axis itself: per artwork, art lovers' mean minus lay mean (from half the raters), and every other rater's weight
on it, which is reliable across halves of their own ratings and grows with their art interest. Finally an unsupervised check: a
low-rank factorisation of the raters x artworks matrix, whose second factor is read against art interest and style.

  python -m taste.lapis_taste
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from taste.common import save, record
from taste.paths import RAW

PIAA = RAW / "asmaerten__LAPIS/piaa/piaa.csv"
PARTS = ["obvious", "group taste (high- or low-brow)", "own taste", "random (vote to vote)"]
PART_COLOR = ["#2a78d6", "#1b8f65", "#e8a33a", "#b9b7b0"]
RNG = np.random.default_rng(0)


def load():
    d = pd.read_csv(PIAA, keep_default_na=False, na_values=[""]).rename(columns={"Art Interest VAIAK": "vaiak", "4_styles": "style4", "2_styles": "style2"})
    d = d[d.groupby("participant_id").rating.transform("size") >= 100]
    d["z"] = d.groupby("participant_id").rating.transform(lambda r: (r - r.mean()) / r.std())
    first = d.groupby(["participant_id", "image_id"]).cumcount() == 0
    rep = d[~first].merge(d[first], on=["participant_id", "image_id"], suffixes=("", "_first"))  # repeated ratings
    return d[first].reset_index(drop=True), rep


def pair_cov(v, img, grp, a, b):
    """Mean product of two different raters' z on the same artwork, raters from group a and group b."""
    za, zb = np.where(grp == a, v, 0.0), np.where(grp == b, v, 0.0)
    sa, sb = np.bincount(img, za), np.bincount(img, zb)
    na, nb = np.bincount(img, grp == a), np.bincount(img, grp == b)
    if a == b: return ((sa ** 2 - np.bincount(img, za ** 2)).sum() / 2) / ((na * (na - 1)).sum() / 2)
    return (sa * sb).sum() / (na * nb).sum()


def split(d, rep, group_of):
    """Shares of one vote for each group, from the pairwise covariances; `group_of` maps participant -> 'lay' | 'lover' | other."""
    grp = d.participant_id.map(group_of).values; img = pd.factorize(d.image_id)[0]
    c = {(a, b): pair_cov(d.z.values, img, grp, a, b) for a, b in [("lay", "lay"), ("lover", "lover"), ("lay", "lover")]}
    rg = rep.participant_id.map(group_of).values
    out = {}
    for g in ["lay", "lover"]:
        tt = (rep.z.values * rep.z_first.values)[rg == g].mean()  # repeat-rating covariance
        out[g] = dict(zip(PARTS, [c[("lay", "lover")], c[(g, g)] - c[("lay", "lover")], tt - c[(g, g)], 1 - tt]))
    return out


def highbrow_axis(d, raters):
    """Per artwork: art lovers' mean z minus lay people's mean z, using only `raters`."""
    s = d[d.participant_id.isin(raters)]
    m = s.groupby(["image_id", "group"]).z.mean().unstack()
    return (m.lover - m.lay).rename("highbrow"), s.groupby("image_id").z.mean().rename("consensus")


def weights(d, axis, consensus):
    """Each rater's weight on the high-brow axis, controlling for the consensus, on two halves of their own ratings."""
    s = d.join(axis, on="image_id").join(consensus, on="image_id").dropna(subset=["highbrow", "consensus"])
    s["half"] = s.groupby("participant_id").cumcount() % 2
    out = {}
    for (p, h), g in s.groupby(["participant_id", "half"]):
        X = np.c_[np.ones(len(g)), (g.highbrow - axis.mean()) / axis.std(), (g.consensus - consensus.mean()) / consensus.std()]
        out[(p, h)] = np.linalg.lstsq(X, g.z.values, rcond=None)[0][1]
    return pd.Series(out).unstack()


def factorise(d, k=4, lam=5.0, iters=30):
    """Low-rank factorisation z_pi ~ a_p . b_i (rater and artwork intercepts included), by alternating ridge; 10% of votes held out."""
    p, P = pd.factorize(d.participant_id); i, I = pd.factorize(d.image_id); z = d.z.values
    test = RNG.random(len(z)) < 0.1
    A, B = RNG.normal(0, 0.1, (len(P), k + 2)), RNG.normal(0, 0.1, (len(I), k + 2))
    A[:, 0] = 1; B[:, 1] = 1  # column 0 of B: artwork intercept; column 1 of A: rater intercept
    tr = ~test; rows_p = [np.flatnonzero(tr & (p == u)) for u in range(len(P))]; rows_i = [np.flatnonzero(tr & (i == u)) for u in range(len(I))]
    for _ in tqdm(range(iters), desc=f"factorise k={k}", leave=False):
        for u, r in enumerate(rows_p):
            X = B[i[r]]; free = [j for j in range(k + 2) if j != 0]; y = z[r] - X[:, 0]
            A[u, free] = np.linalg.solve(X[:, free].T @ X[:, free] + lam * np.eye(len(free)), X[:, free].T @ y)
        for u, r in enumerate(rows_i):
            X = A[p[r]]; free = [j for j in range(k + 2) if j != 1]; y = z[r] - X[:, 1]
            B[u, free] = np.linalg.solve(X[:, free].T @ X[:, free] + lam * np.eye(len(free)), X[:, free].T @ y)
    pred = (A[p] * B[i]).sum(1)
    r2 = lambda m: 1 - ((z[m] - pred[m]) ** 2).mean() / z[m].var()
    return pd.DataFrame(A[:, 2:], index=P), pd.DataFrame(B[:, 2:], index=I), r2(tr), r2(test)


def main():
    d, rep = load()
    rater = d.groupby("participant_id").agg(vaiak=("vaiak", "first"), n=("z", "size"))
    rater = rater[rater.vaiak.notna()]
    lo, hi = rater.vaiak.quantile([1 / 3, 2 / 3])
    group_of = pd.Series(np.where(rater.vaiak <= lo, "lay", np.where(rater.vaiak >= hi, "lover", "middle")), index=rater.index)
    d["group"] = d.participant_id.map(group_of)
    print(f"{d.participant_id.nunique()} raters, {len(d)} votes, {len(rep)} repeats; art interest thirds at {lo:.2f} / {hi:.2f}")
    print("repeats by group:", rep.participant_id.map(group_of).value_counts().to_dict(), "raters:", rep.groupby(rep.participant_id.map(group_of)).participant_id.nunique().to_dict())
    shares = split(d, rep, group_of)
    boot = []
    imgs = d.image_id.unique()
    for _ in tqdm(range(200), desc="bootstrap artworks"):
        pick = pd.Series(RNG.choice(imgs, len(imgs)))
        db = d.set_index("image_id").loc[pick.values].reset_index()
        db["image_id"] = np.repeat(np.arange(len(pick)), d.groupby("image_id").size().reindex(pick.values).values)
        boot.append(split(db, rep.sample(frac=1, replace=True, random_state=int(RNG.integers(1 << 31))), group_of))
    shuffled = split(d, rep, pd.Series(RNG.permutation(group_of.values), index=group_of.index))
    T = pd.DataFrame({(g, k): dict(share=shares[g][p], lo=np.percentile([b[g][p] for b in boot], 2.5), hi=np.percentile([b[g][p] for b in boot], 97.5),
                                   shuffled=shuffled[g][p]) for g in ["lay", "lover"] for k, p in enumerate(PARTS)}).T
    T.index = pd.MultiIndex.from_tuples([(g, PARTS[k]) for g, k in T.index])
    print(T.round(3).to_string())

    # the high-brow axis from half the raters, every other rater's weight on it
    half = pd.Series(RNG.random(len(rater)) < 0.5, index=rater.index)
    W = []
    for a, b in [(half, ~half), (~half, half)]:
        axis, cons = highbrow_axis(d, rater.index[a])
        w = weights(d[d.participant_id.isin(rater.index[b])], axis, cons); w["vaiak"] = rater.vaiak; W.append(w)
    W = pd.concat(W).dropna()
    rel = stats.pearsonr(W[0], W[1]).statistic
    rho = stats.spearmanr(W[[0, 1]].mean(axis=1), W.vaiak).statistic
    print(f"weight on the high-brow axis: split-half reliability {rel:.2f} across {len(W)} raters, Spearman with art interest {rho:.2f}")
    axis, cons = highbrow_axis(d, rater.index)
    meta = d.groupby("image_id")[["path", "style", "style2", "style4", "genre", "artist"]].first()
    A = meta.join(axis).join(cons).join(d.groupby("image_id").size().rename("n"))
    by_style = A.groupby("style").agg(highbrow=("highbrow", "mean"), consensus=("consensus", "mean"), n=("n", "size")).query("n >= 100").sort_values("highbrow")
    print(by_style.round(2).to_string())

    # unsupervised: low-rank factorisation
    fac = {}
    for k in [1, 2, 3, 4, 6]:
        Ra, Rb, r2tr, r2te = factorise(d, k); fac[k] = (Ra, Rb, r2tr, r2te); print(f"k={k}: R2 train {r2tr:.3f}, held-out {r2te:.3f}")
    Ra, Rb = fac[4][0], fac[4][1]
    fstats = []
    for j in range(Ra.shape[1]):
        fstats.append(dict(factor=j, rho_art_interest=stats.spearmanr(Ra[j].reindex(rater.index), rater.vaiak, nan_policy="omit").statistic,
                           rho_highbrow=stats.spearmanr(Rb[j], axis.reindex(Rb.index), nan_policy="omit").statistic,
                           abstract_minus_figurative=Rb[j][meta.style2.reindex(Rb.index).eq("ABSTRACT").values].mean() - Rb[j][meta.style2.reindex(Rb.index).eq("FIGURATIVE").values].mean()))
    print(pd.DataFrame(fstats).round(2).to_string())
    record("lapis_taste", dict(shares=T.reset_index().rename(columns={"level_0": "group", "level_1": "part"}).to_dict("records"), weight_reliability=rel, weight_rho_art_interest=rho,
                               by_style=by_style.reset_index().to_dict("records"), factors=fstats, factor_r2={k: v[2:] for k, v in fac.items()}))
    A.to_parquet(RAW.parent / "data/lapis_highbrow.parquet")
    T.to_parquet(RAW.parent / "data/lapis_taste_shares.parquet"); W.columns = W.columns.astype(str); W.to_parquet(RAW.parent / "data/lapis_taste_weights.parquet")
    by_style.to_parquet(RAW.parent / "data/lapis_taste_by_style.parquet")
    W.columns = [0, 1, "vaiak"]; plot(T, W, by_style, rel, rho)


def plot(T, W, by_style, rel, rho):
    """Shares of one vote per group. A group's shared part is drawn as what a vote shares with another member of its group: everyone's
    part (cross-group) plus the group's own extra; art lovers share less with each other than with lay people, so theirs is drawn
    at their within-group level and the deficit is noted."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1, 1.2]))
    ax = axs[0]
    for y, g in enumerate(["lover", "lay"]):
        sh = dict(zip(PARTS, T.loc[g, "share"].values)); within = sh[PARTS[0]] + sh[PARTS[1]]
        parts = [min(sh[PARTS[0]], within), max(sh[PARTS[1]], 0), sh[PARTS[2]], sh[PARTS[3]]]
        left = 0
        for k, v in enumerate(parts):
            ax.barh(y, v, left=left, color=PART_COLOR[k], edgecolor="white", linewidth=2, label=PARTS[k] if y == 1 else None); left += v
        if sh[PARTS[1]] < 0: ax.annotate(f"art lovers agree with each other ({within:.2f})\nless than with lay people ({sh[PARTS[0]]:.2f})", (within, y), xytext=(0.3, y + 0.42),
                                         textcoords="data", arrowprops=dict(arrowstyle="-", color="#6b6a66"), color="#3d3d3a", va="center")
    ax.set_yticks([0, 1], ["art lover's vote", "lay vote"]); ax.set_xlim(0, 1); ax.set_xlabel("share of one vote's variance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    ax = axs[1]
    w = W[[0, 1]].mean(axis=1)
    ax.scatter(W.vaiak, w, s=10, c="#3d3d3a", linewidths=0, alpha=0.6)
    b = np.polyfit(W.vaiak, w, 1); x = np.linspace(W.vaiak.min(), W.vaiak.max(), 2); ax.plot(x, np.polyval(b, x), color="#1b8f65", lw=2)
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_xlabel("art interest (VAIAK)"); ax.set_ylabel("weight on the high-brow axis")
    ax.set_title(f"raters' weights: split-half r = {rel:.2f}, ρ = {rho:.2f}", loc="left")
    ax = axs[2]
    s = by_style.sort_values("highbrow"); s = pd.concat([s.head(7), s.tail(7)])  # the 7 most low-brow and the 7 most high-brow styles
    ax.barh(np.arange(len(s)), s.highbrow, color=np.where(s.highbrow > 0, "#1b8f65", "#c0302f"), height=0.7)
    ax.set_yticks(np.arange(len(s)), [x.replace("_", " ") for x in s.index]); ax.axvline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("art lovers minus lay people (z)"); ax.set_title("high-brow and low-brow styles", loc="left")
    save(fig, "lapis_taste")


if __name__ == "__main__":
    import sys, json
    from taste.common import RESULTS
    if "--plot" in sys.argv:  # redraw from the saved results
        L = json.load(open(RESULTS))["lapis_taste"]; W = pd.read_parquet(RAW.parent / "data/lapis_taste_weights.parquet"); W.columns = [0, 1, "vaiak"]
        plot(pd.read_parquet(RAW.parent / "data/lapis_taste_shares.parquet"), W, pd.read_parquet(RAW.parent / "data/lapis_taste_by_style.parquet"), L["weight_reliability"], L["weight_rho_art_interest"])
    else: main()
