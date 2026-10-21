"""Haiku: quality ladder, judge-size scaling, shared vs private taste, eye-of-the-beholder (rater x generator) matrices.

  python -m taste.analysis_haiku
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from taste.common import (ratings, zscore, items, fam, rater_order, generator_order, save, record, FAMILY_COLOR, FAMILY_MARKER,
                          DIVERGING, ANCHOR, RATINGS)
from taste.models import BY_NAME

RNG = np.random.default_rng(0)


def icc_consistency(Z):
    """ICC(C,1) for an items x raters matrix (two-way, consistency): share of variance that is shared across raters."""
    Z = Z.dropna(); n, k = Z.shape
    ms_items = k * Z.mean(1).var(ddof=1)
    resid = Z.sub(Z.mean(1), axis=0).sub(Z.mean(0), axis=1) + Z.values.mean()
    ms_resid = (resid ** 2).values.sum() / ((n - 1) * (k - 1))
    return (ms_items - ms_resid) / (ms_items + (k - 1) * ms_resid)


def interaction(Z, gen):
    """rater x generator mean z, double-centred: what is left after 'how good is this generator' and 'how lenient is this rater'."""
    M = Z.groupby(gen).mean().T  # raters x generators
    return M, M.sub(M.mean(1), axis=0).sub(M.mean(0), axis=1) + M.values.mean()


def family_contrast(I, fams=("qwen", "gemma")):
    """(own-family cells) - (cross-family cells) of the double-centred matrix; > 0 = families prefer their own family's haiku."""
    rf = I.index.map(fam); gf = I.columns.map(fam)
    cell = lambda a, b: I.loc[rf == a, gf == b].values.mean()
    a, b = fams
    return (cell(a, a) + cell(b, b) - cell(a, b) - cell(b, a)) / 2


def self_pref(I):
    s = [I.loc[r, r] for r in I.index if r in I.columns]
    return float(np.mean(s)), len(s)


def main():
    it = items("haiku")
    S = ratings("haiku")
    S = S[rater_order(S.columns)]
    Z = zscore(S)
    gen = it.generator.reindex(Z.index)
    model_items = gen.map(fam) != "human"

    # ---- 1. quality ladder: consensus (open raters) and anchor judge, per generator
    open_raters = [r for r in Z.columns if r != ANCHOR]
    cons = Z[open_raters].mean(1)
    order = generator_order(gen.unique())
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, (col, lab, mk) in enumerate([(cons, "open-model consensus (mean z)", "o"), (Z[ANCHOR], f"anchor judge: {ANCHOR}", "x")]):
        g = col.groupby(gen)
        m, se = g.mean().reindex(order), (g.std() / np.sqrt(g.size())).reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * 0.25
        ax.errorbar(x, m, 1.96 * se, fmt=mk, color="#0b0b0b" if j == 0 else "#52514e", ms=6, capsize=0, lw=1, label=lab)
    for i, g in enumerate(order): ax.axvspan(i - 0.5, i + 0.5, color=FAMILY_COLOR[fam(g)], alpha=0.08, lw=0)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right"); ax.set_ylabel("rating (z within rater)")
    ax.set_title("Haiku quality ladder: generators ordered by family and size (shading = family)"); ax.legend(loc="upper left")
    save(fig, "haiku_quality_ladder")

    # ---- 2. judge scaling: does a rater's score track generator size / the anchor / the human ladder?
    known = gen.map(lambda g: g in BY_NAME and not np.isnan(BY_NAME[g].params_b))
    logp = np.log10(gen[known].map(lambda g: BY_NAME[g].params_b))
    hum = gen.isin(["human-journal", "human-twaiku"])
    rows = []
    for r in S.columns:
        rows.append(dict(rater=r, family=fam(r), params_b=BY_NAME[r].params_b,
                         rho_gen_size=stats.spearmanr(S.loc[known, r], logp, nan_policy="omit").statistic,
                         rho_anchor=stats.spearmanr(S[r], S[ANCHOR], nan_policy="omit").statistic if r != ANCHOR else np.nan,
                         auc_journal_vs_twaiku=stats.mannwhitneyu(S.loc[gen == "human-journal", r], S.loc[gen == "human-twaiku", r]).statistic / ((gen == "human-journal").sum() * (gen == "human-twaiku").sum())))
    sc = pd.DataFrame(rows)
    record("haiku_judge_scaling", sc.to_dict("records"))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (col, lab) in zip(axs, [("rho_gen_size", "Spearman(score, log generator size)"), ("rho_anchor", f"Spearman(score, {ANCHOR})"),
                                     ("auc_journal_vs_twaiku", "AUC: journal haiku > tweet haiku")]):
        for f, d in sc[sc.family.isin(["qwen", "gemma"])].groupby("family"):
            ax.plot(d.params_b, d[col], marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], lw=2, ms=7, label=f)
        ax.set_xscale("log"); ax.set_xlabel("rater parameters (B)"); ax.set_title(lab)
        if col == "auc_journal_vs_twaiku": ax.axhline(0.5, color="#8a8984", lw=1, ls="--")
        else: ax.axhline(0, color="#8a8984", lw=1)
    axs[0].legend()
    save(fig, "haiku_judge_scaling")

    # ---- 3. shared vs private taste across the model panel
    record("haiku_icc", dict(all=icc_consistency(Z), qwen=icc_consistency(Z[[r for r in Z if fam(r) == "qwen"]]),
                             gemma=icc_consistency(Z[[r for r in Z if fam(r) == "gemma"]])))
    C = S.corr("spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1); fig.colorbar(im, ax=ax, label="Spearman ρ across haiku")
    ax.set_xticks(range(len(C)), C.columns, rotation=60, ha="right"); ax.set_yticks(range(len(C)), C.index)
    ax.grid(False); ax.set_title("Do raters agree on haiku?")
    save(fig, "haiku_rater_agreement")

    # ---- 4. eye of the beholder: rater x generator interaction (model generators only)
    Zm, gm = Z[model_items], gen[model_items]
    M, I = interaction(Zm, gm)
    I = I[generator_order(I.columns)]
    stat = dict(family_contrast=family_contrast(I), self_pref=self_pref(I))
    # 95% CI: bootstrap items within each generator, recompute the contrast
    boot = []
    for _ in tqdm(range(500), desc="bootstrap family contrast"):
        idx = gm.groupby(gm).sample(frac=1, replace=True, random_state=int(RNG.integers(1e9))).index
        boot.append(family_contrast(interaction(Zm.loc[idx], gm.loc[idx])[1]))
    stat["family_contrast_ci95"] = np.percentile(boot, [2.5, 97.5]).tolist()
    # familiarity control: residualise each local rater's z on its own log p(haiku) and length
    lp = ratings("haiku", "logprob_mean"); nt = ratings("haiku", "n_item_tokens")
    Zr = Zm.copy()
    for r in lp.columns:
        ok = Zm[r].notna() & lp.loc[Zm.index, r].notna()
        X = np.c_[np.ones(ok.sum()), lp.loc[Zm.index[ok], r], np.log(nt.loc[Zm.index[ok], r])]
        Zr.loc[ok, r] = Zm.loc[ok, r] - X @ np.linalg.lstsq(X, Zm.loc[ok, r].values, rcond=None)[0]
        stat.setdefault("rho_score_familiarity", {})[r] = stats.spearmanr(S.loc[Zm.index, r], lp.loc[Zm.index, r], nan_policy="omit").statistic
    Ir = interaction(Zr, gm)[1][I.columns]
    stat["family_contrast_familiarity_controlled"] = family_contrast(Ir)
    stat["self_pref_familiarity_controlled"] = self_pref(Ir)
    # assortative taste: does a rater's leftover preference track generator size, and does that depend on the rater's own size?
    gsize = pd.Series({g: np.log10(BY_NAME[g].params_b) for g in I.columns if g in BY_NAME and not np.isnan(BY_NAME[g].params_b)})
    assort = pd.DataFrame([dict(rater=r, family=fam(r), params_b=BY_NAME[r].params_b, rho_interaction_vs_gen_size=stats.spearmanr(I.loc[r, gsize.index], gsize).statistic) for r in I.index])
    assort = assort[assort.params_b.notna()]
    stat["assortative"] = assort.to_dict("records")
    stat["assortative_rho_across_raters"] = stats.spearmanr(np.log10(assort.params_b), assort.rho_interaction_vs_gen_size).statistic
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for f, d in assort.groupby("family"):
        ax.plot(d.params_b, d.rho_interaction_vs_gen_size, marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], lw=2, ms=7, label=f)
    ax.axhline(0, color="#8a8984", lw=1); ax.set_xscale("log"); ax.set_xlabel("rater parameters (B)"); ax.set_ylabel("ρ(leftover preference, log generator size)")
    ax.set_title("Assortative taste: small judges prefer small writers"); ax.legend()
    save(fig, "haiku_assortative")
    record("haiku_beholder", stat)
    for mat, nm, ttl in [(I, "haiku_beholder", "rater × generator interaction (z), main effects removed"),
                         (Ir, "haiku_beholder_familiarity_controlled", "same, after regressing out rater's log p(haiku) + length")]:
        fig, ax = plt.subplots(figsize=(11, 5))
        v = np.nanmax(np.abs(mat.values))
        im = ax.imshow(mat, cmap=DIVERGING, vmin=-v, vmax=v); fig.colorbar(im, ax=ax, label="interaction (z)")
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=60, ha="right"); ax.set_yticks(range(mat.shape[0]), mat.index); ax.grid(False)
        for i, r in enumerate(mat.index):
            if r in mat.columns: ax.add_patch(plt.Rectangle((list(mat.columns).index(r) - 0.5, i - 0.5), 1, 1, fill=False, ec="#0b0b0b", lw=1.5))
        ax.set_title(f"Eye of the beholder — {ttl}\nboxed = self-rating; family contrast = {family_contrast(mat):+.3f}")
        save(fig, nm)
    print(stat)


if __name__ == "__main__":
    main()
