"""Images: agreement with human ground truth per dataset vs rater size; ACX human-vs-AI preference and detection; human vs model
shared-taste (ICC).

  python -m taste.analysis_images
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from taste.common import ratings, zscore, items, fam, rater_order, save, record, FAMILY_COLOR, FAMILY_MARKER
from taste.paths import RAW, DATA
from taste.models import BY_NAME
from taste.analysis_haiku import icc_consistency


def human_icc_from_moments(mean, sd, n):
    """ICC(1) from per-image rating mean/sd/count (one-way random effects): shared share of a single human rating's variance."""
    n0 = n.mean(); msb = n0 * mean.var(ddof=1); msw = (sd ** 2).mean()
    return (msb - msw) / (msb + (n0 - 1) * msw)


def main():
    it = items("images")
    S = ratings("images"); S = S[rater_order(S.columns)]
    full = [r for r in S.columns if S[r].notna().all()]  # raters that saw every dataset

    # ---- 1. agreement with human ground truth, per dataset, vs rater size
    rows = []
    for ds, d in it.groupby("dataset"):
        for r in S.columns:
            s = S.loc[d.index, r]
            if s.notna().sum() < 10: continue
            rows.append(dict(dataset=ds, rater=r, family=fam(r), params_b=BY_NAME[r].params_b, rho=stats.spearmanr(s, d.human_score, nan_policy="omit").statistic))
    A = pd.DataFrame(rows); record("images_gt_agreement", A.to_dict("records"))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, ds, ttl in zip(axs, ["ava", "lapis", "acx"], ["AVA photos: mean of ~200 votes", "LAPIS art: mean of ~20 raters", "ACX: share of 9k 'favourite' votes"]):
        d = A[A.dataset == ds]
        for f, g in d[d.family.isin(["qwen", "gemma"])].groupby("family"):
            ax.plot(g.params_b, g.rho, marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], lw=2, ms=7, label=f)
        for _, g in d[~d.family.isin(["qwen", "gemma"])].iterrows():
            ax.axhline(g.rho, color=FAMILY_COLOR[g.family] if g.family in FAMILY_COLOR else "#52514e", ls=":", lw=1.5)
            ax.annotate(g.rater, (1, g.rho), xycoords=("axes fraction", "data"), ha="right", va="bottom", color="#52514e")
        ax.axhline(0, color="#8a8984", lw=1); ax.set_xscale("log"); ax.set_xlabel("rater parameters (B)"); ax.set_title(ttl)
    axs[0].set_ylabel("Spearman ρ with human ratings"); axs[0].legend(loc="upper left")
    save(fig, "images_gt_agreement")

    # ---- 2. ACX: do models prefer AI art more than people do? can they tell?
    acx = it[it.dataset == "acx"]
    ai = acx.source == "ai"
    rank = lambda s: s.rank(pct=True)
    pref = {"humans (favourite share)": rank(acx.human_score)[ai].mean() - rank(acx.human_score)[~ai].mean()}
    # detection = accuracy of the human/AI call. humans: per respondent (median) and crowd majority; local raters: P('Human') > 0.5;
    # API raters: the 'origin' answer from taste.acx_recognition
    resp = pd.read_parquet(DATA / "acx_responses.parquet")
    truth = dict(zip(acx.title, np.where(ai, "AI", "Human")))
    guess = resp[[f"guess::{t}" for t in acx.title]]
    per_person = (guess.values == np.array([truth[t] for t in acx.title])[None]).sum(1) / (guess.isin(["AI", "Human"])).sum(1).clip(lower=1)
    answered = guess.isin(["AI", "Human"]).sum(1) >= 40
    acxi = pd.read_parquet(DATA / "acx_images.parquet").set_index(acx.index)
    det = {"humans: median respondent": float(np.median(per_person[answered])), "humans: crowd majority": float(((acxi.p_judged_human > 0.5) == ~ai).mean())}
    ph = ratings("images", "p_human")
    for r in S.columns:
        s_ = S.loc[acx.index, r]
        if s_.notna().all(): pref[r] = rank(s_)[ai].mean() - rank(s_)[~ai].mean()
        if r in ph and ph.loc[acx.index, r].notna().all(): det[r] = float(((ph.loc[acx.index, r] > 0.5) == ~ai).mean())
        if (DATA / f"acx_recognition_{r}.parquet").exists(): det[r] = float(pd.read_parquet(DATA / f"acx_recognition_{r}.parquet").origin_correct.mean())
    record("acx_ai_preference", pref); record("acx_detection_accuracy", det)
    # does the preference for human art survive on human works the rater could NOT name? (taste.acx_recognition)
    rec_rows = []
    for r in S.columns:
        f = DATA / f"acx_recognition_{r}.parquet"
        if not f.exists(): continue
        rc = pd.read_parquet(f).reindex(acx.index)
        g = np.where(ai, "AI image", np.where(rc.identified, "human, artist named", "human, not named"))
        t = pd.DataFrame({"rater": rank(S.loc[acx.index, r]), "ACX crowd": rank(acx.human_score), "group": g})
        for grp, d in t.groupby("group"): rec_rows.append(dict(rater=r, group=grp, n=len(d), rater_pct_rank=d.rater.mean(), crowd_pct_rank=d["ACX crowd"].mean()))
    record("acx_recognition", rec_rows)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.95})
    for ax, dct, ttl, ref in [(axs[0], pref, "Preference for AI art:\nmean pct-rank(AI) − pct-rank(human)", 0),
                              (axs[1], det, "Telling AI from human art:\naccuracy of the human/AI call", 0.5)]:
        names = list(dct); vals = [dct[n] for n in names]
        cols = [FAMILY_COLOR[fam(n)] if n in BY_NAME else FAMILY_COLOR["human"] for n in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.7); ax.axvline(ref, color="#0b0b0b", lw=1)
        ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.set_title(ttl)
    save(fig, "acx_preference_detection")

    # ---- 3. shared taste: model panel ICC vs human ICC (single-rater reliability)
    lap = pd.read_csv(RAW / "asmaerten__LAPIS/giaa/test.csv")
    ava = pq.read_table(RAW / "trojblue__AVA-aesthetics-10pct-min50-10bins/data/test-00000-of-00002.parquet", columns=["rating_counts"]).to_pandas()
    rc = np.stack(ava.rating_counts.values).astype(float); k = np.arange(1, 11)
    n = rc.sum(1); mu = rc @ k / n; sd = np.sqrt(rc @ k ** 2 / n - mu ** 2)
    icc = {"lapis": dict(humans=human_icc_from_moments(lap.mean_response, lap.std_response, lap["number of raters"]),
                         models=icc_consistency(zscore(S.loc[it.dataset == "lapis", full]))),
           "ava": dict(humans=human_icc_from_moments(pd.Series(mu), pd.Series(sd), pd.Series(n)),
                       models=icc_consistency(zscore(S.loc[it.dataset == "ava", full])))}
    record("images_icc", icc); print(icc)


if __name__ == "__main__":
    main()
