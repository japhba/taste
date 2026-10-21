"""Linear probes on hidden states, evaluated only on far-OOD splits, against the model's own verbal (digit) rating.

  python -m taste.probes

Headline layer is fixed in advance (LAYER_FRAC of depth); per-layer curves are reported for the main splits, never used to pick.
Confound control: target and features residualised on confounds with an OLS fit on the training split only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import RidgeCV
from tqdm.auto import tqdm
from taste.common import ratings, zscore, items, fam, rater_order, hidden, save, record, FAMILY_COLOR, ANCHOR, HIDDEN
from taste.models import BY_NAME
from taste.paths import DATA

LAYER_FRAC = 2 / 3
ALPHAS = np.logspace(0, 6, 13)


def residualise(Ztr, Zte, Ctr, Cte):
    Ctr1, Cte1 = np.c_[np.ones(len(Ctr)), Ctr], np.c_[np.ones(len(Cte)), Cte]
    B = np.linalg.lstsq(Ctr1, Ztr, rcond=None)[0]
    return Ztr - Ctr1 @ B, Zte - Cte1 @ B


def probe(Xtr, ytr, Xte, Ctr=None, Cte=None):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if Ctr is not None:
        Xtr, Xte = residualise(Xtr, Xte, Ctr, Cte)
        ytr = residualise(ytr[:, None], ytr[:1, None], Ctr, Cte[:1])[0][:, 0]
    return RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte)


def rho(a, b):
    return stats.spearmanr(a, b, nan_policy="omit").statistic


def haiku_splits(it):
    g = it.generator; f = g.map(fam)
    open_ = f.isin(["smollm", "qwen", "gemma"]); model = f != "human"
    known = g.map(lambda x: x in BY_NAME and not np.isnan(BY_NAME[x].params_b))
    topics = sorted(it.topic.dropna().unique()); fold = it.topic.map({t: i % 5 for i, t in enumerate(topics)})
    yield "topic 5-fold (model haiku)", "anchor", [(model & (fold != k), model & (fold == k)) for k in range(5)]
    yield "open models → frontier + human", "anchor", [(open_, ~open_)]
    yield "model haiku → human haiku", "anchor", [(model, ~model)]
    yield "size: Qwen → Gemma", "size", [(known & (f != "gemma"), known & (f == "gemma"))]
    yield "size: Gemma → Qwen", "size", [(known & (f != "qwen"), known & (f == "qwen"))]
    yield "own rating: open → frontier + human", "self", [(open_, ~open_)]


def image_splits(it):
    d = it.dataset
    yield "AVA photos → LAPIS art", "human", [(d == "ava", d == "lapis")]
    yield "LAPIS art → AVA photos", "human", [(d == "lapis", d == "ava")]
    yield "AVA + LAPIS → ACX", "human", [(d != "acx", d == "acx")]
    yield "own rating: AVA → LAPIS", "self", [(d == "ava", d == "lapis")]


def evaluate(which, rater, pool, layer_idx, it, S, targets, splits, confounds):
    ids, layers, H = hidden(which, rater, pool)
    X = H[:, layer_idx].astype(np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids); have = it.index.isin(ids)
    rows = []
    for split, tgt, folds in splits(it):
        y = targets[tgt] if tgt != "self" else S[rater]
        folds = [(tr[tr & y.notna() & have].index, te[te & y.notna() & have].index) for tr, te in folds]  # raters may cover only part of the items
        if any(len(tr) == 0 or len(te) == 0 for tr, te in folds): continue  # e.g. a rater that never saw AVA has no AVA split
        for ctl in [False, True]:
            pred, truth, verbal = [], [], []
            for tr, te in folds:
                C = confounds.loc[:, rater] if ctl else None
                p = probe(X[pos[tr]], y[tr].values, X[pos[te]], *(None, None) if C is None else (C.loc[tr].values, C.loc[te].values))
                pred.append(p); truth.append(y[te].values); verbal.append(S.loc[te, rater].values)
            pred, truth, verbal = map(np.concatenate, (pred, truth, verbal))
            rows.append(dict(set=which, rater=rater, pool=pool, layer=int(layers[layer_idx]), split=split, target=tgt, confound_ctl=ctl,
                             probe=rho(pred, truth), verbal=rho(verbal, truth) if tgt != "self" else np.nan, n_test=len(truth)))
    return rows


def main():
    out = []
    for which in ["haiku", "images"]:
        it = items(which); S = ratings(which).reindex(it.index)
        local = rater_order([p.stem for p in (HIDDEN / which).glob("*.npz")])
        if which == "haiku":
            targets = {"anchor": zscore(S[ANCHOR]), "size": np.log10(it.generator.map(lambda g: BY_NAME[g].params_b if g in BY_NAME else np.nan))}
            lp, nt = ratings(which, "logprob_mean"), ratings(which, "n_item_tokens")
            confounds = pd.concat({r: pd.DataFrame({"lp": lp[r], "len": np.log(nt[r])}) for r in local}, axis=1).reindex(it.index)
            splits = haiku_splits
        else:
            targets = {"human": it.groupby("dataset").human_score.transform(lambda x: (x - x.mean()) / x.std())}
            nt = ratings(which, "n_item_tokens")
            aspect = np.log(pd.read_parquet(DATA / "images_features.parquet").set_index("item_id").f_aspect)
            confounds = pd.concat({r: pd.DataFrame({"len": np.log(nt[r]), "aspect": aspect}) for r in local}, axis=1).reindex(it.index)
            splits = image_splits
        for r in tqdm(local, desc=f"probes {which}"):
            layers = hidden(which, r)[1]
            li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))
            for pool in ["mean", "last"]:
                out += evaluate(which, r, pool, li, it, S, targets, splits, confounds)
    df = pd.DataFrame(out)
    df.to_parquet(HIDDEN.parent / "probe_results.parquet")
    record("probes", df.to_dict("records"))
    print(df[~df.confound_ctl & (df.pool == "mean")].pivot_table(index="rater", columns="split", values=["probe", "verbal"]).round(2).to_string())
    plot(df)
    cross_modal()


def cross_modal():
    """Is there one quality direction shared by text and images? Train on one modality, test on the other (same model, layer, pool);
    each modality standardised on its own statistics."""
    hi, ii = items("haiku"), items("images")
    yh = zscore(ratings("haiku")[ANCHOR]).reindex(hi.index)
    yi = ii.groupby("dataset").human_score.transform(lambda x: (x - x.mean()) / x.std())
    rows = []
    for r in tqdm(rater_order([p.stem for p in (HIDDEN / "haiku").glob("*.npz") if (HIDDEN / "images" / p.name).exists()]), desc="cross-modal"):
        for pool in ["mean", "last"]:
            idh, layers, Hh = hidden("haiku", r, pool); idi, _, Hi = hidden("images", r, pool)
            li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))
            Xh, Xi = Hh[:, li].astype(np.float32), Hi[:, li].astype(np.float32)
            Xh, Xi = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-6), (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6)
            ph = pd.Series(probe(Xh, yh.loc[idh].values, Xi), index=idi)  # haiku -> images
            for ds in ["ava", "lapis", "acx"]:
                m = ii.loc[idi].dataset.values == ds
                rows.append(dict(rater=r, pool=pool, direction="haiku → images", test=ds, rho=rho(ph[m], yi.loc[idi][m])))
            tr = ii.loc[idi].dataset.values != "acx"
            pi = probe(Xi[tr], yi.loc[idi][tr].values, Xh)  # images -> haiku
            rows.append(dict(rater=r, pool=pool, direction="images → haiku", test="haiku (anchor)", rho=rho(pi, yh.loc[idh].values)))
    df = pd.DataFrame(rows); record("cross_modal", df.to_dict("records")); print(df.pivot_table(index="rater", columns=["pool", "test"], values="rho").round(2).to_string())
    return df


def plot(df):
    for which in ["haiku", "images"]:
        d = df[(df.set == which) & (df.pool == "mean") & (df.target != "self")]
        splits = list(dict.fromkeys(d.split))
        raters = rater_order(d.rater.unique())
        fig, axs = plt.subplots(1, len(splits), figsize=(3.6 * len(splits), 4), sharey=True)
        for ax, sp in zip(np.atleast_1d(axs), splits):
            for j, (ctl, ls) in enumerate([(False, "-"), (True, ":")]):
                e = d[(d.split == sp) & (d.confound_ctl == ctl)].set_index("rater").reindex(raters)
                x = np.arange(len(raters))
                ax.plot(x, e.probe, ls, marker="o", color="#0b0b0b", label="probe" + (" (confounds removed)" if ctl else ""))
                if not ctl: ax.plot(x, e.verbal, ls, marker="x", color="#e34948", label="verbal rating")
            for i, r in enumerate(raters): ax.axvspan(i - 0.5, i + 0.5, color=FAMILY_COLOR[fam(r)], alpha=0.08, lw=0)
            ax.axhline(0, color="#8a8984", lw=1); ax.set_xticks(range(len(raters)), raters, rotation=70, ha="right"); ax.set_title(sp)
        np.atleast_1d(axs)[0].set_ylabel("Spearman ρ on OOD test set"); np.atleast_1d(axs)[0].legend(loc="lower right")
        save(fig, f"probes_{which}")


if __name__ == "__main__":
    main()
