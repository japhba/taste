"""DeepSeek-V4.1-Flash under personas (taste.ds_rate --persona): the haiku-journal editor and the r/haiku reader on all human haiku plus
25 haiku per model writer; the art lover and the person with no interest in art on the LAPIS paintings.
  Verdicts: per condition, AUC of each human tier step (tweet < r/haiku <=1 upvote < r/haiku top < journal), open < frontier writers,
  the weight on the kitsch axis (Qwen3.5-27B's craft PC2) and on DeepSeek's own craft PCs; for paintings, the weight on the art
  lovers' high-brow axis (style level, consensus as covariate, as in taste.steer_report).
  Perception vs weighting: DeepSeek's states at the item's own tokens with and without the persona (the persona precedes the item, so it
  can change them), compared at every depth by the correlation over items of their projection on DeepSeek's craft PCs; and the
  states at the answer position, where the verdict is read.

  python -m taste.ds_persona
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from taste.common import items, hidden, save, record, RESULTS
from taste.paths import DATA, RATINGS, HIDDEN
from taste.taste_space import group_of

DS = "deepseek-v4.1-flash-1017"
LADDER = [("human-twaiku", "human-reddit-low"), ("human-reddit-low", "human-reddit-top"), ("human-reddit-top", "human-journal")]
HAIKU_P = {"none": "haiku", "editor": "haiku_editor", "reader": "haiku_reader"}
ART_P = {"none": "images", "lover": "images_lover", "lay": "images_lay"}
z = lambda x: (x - x.mean()) / x.std()


def scores(folder):
    return pd.read_parquet(RATINGS / folder / f"{DS}.parquet").set_index("item_id").score


def haiku():
    it = items("haiku"); g = group_of(it)
    R = {p: scores(f) for p, f in HAIKU_P.items() if (RATINGS / f / f"{DS}.parquet").exists()}
    common = sorted(set.intersection(*[set(s.index) for s in R.values()]))
    kitsch = pd.read_parquet(DATA / "steer/haiku_features_craft.parquet")["PC2"]
    rows = []
    for p, s in R.items():
        s = s[common]; r = dict(persona=p, n=len(s), mean=s.mean())
        for lo, hi in LADDER:
            sel = it.loc[common, "generator"].isin([lo, hi])
            r[f"{lo.split('-', 1)[1]} < {hi.split('-', 1)[1]}"] = roc_auc_score(it.loc[s.index[sel], "generator"] == hi, s[sel])
        gg = g[common]; sel = gg.isin(["open", "frontier"]) & ~it.loc[common, "generator"].str.startswith("deepseek")
        r["open < frontier"] = roc_auc_score(gg[sel] == "frontier", s[sel])
        k = kitsch.reindex(common).dropna(); r["kitsch weight"] = stats.spearmanr(k, s[k.index]).statistic
        rows.append(r)
    return pd.DataFrame(rows).set_index("persona"), common


def art():
    A = pd.read_parquet(DATA / "lapis_highbrow.parquet"); A["style_highbrow"] = A["style"].map(A.groupby("style").highbrow.mean())
    rows = []
    for p, f in ART_P.items():
        if not (RATINGS / f / f"{DS}.parquet").exists(): continue
        s = scores(f); s = s[s.index.str.startswith("lapis_")]; s.index = s.index.str.replace("lapis_", "").astype(int)
        d = A.join(s.rename("score"), how="inner")
        X = np.c_[np.ones(len(d)), z(d.consensus), z(d.style_highbrow)]
        rows.append(dict(persona=p, n=len(d), mean=d.score.mean(), w_highbrow=np.linalg.lstsq(X, z(d.score).values, rcond=None)[0][2],
                         r_human=stats.spearmanr(d.score, d.consensus).statistic))
    return pd.DataFrame(rows).set_index("persona")


def perception(common, which="haiku", personas=HAIKU_P):
    """Per depth: correlation over items between the item-token states (projected on DeepSeek's craft PCs of the plain prompt) with and
    without each persona, and the same for the answer-position states projected on the plain readout of the rating."""
    ids0, layers, H0 = hidden(personas["none"].split("_")[0] if which == "haiku" else "images", DS, "mean")
    _, _, L0 = hidden(personas["none"], DS, "last")
    pos0 = pd.Series(np.arange(len(ids0)), index=ids0)
    rows = []
    for p, f in personas.items():
        if p == "none" or not (HIDDEN / f / f"{DS}.npz").exists(): continue
        ids1, _, H1 = hidden(f, DS, "mean"); _, _, L1 = hidden(f, DS, "last")
        both = [i for i in ids1 if i in pos0.index and i in common]; a, b = pos0[both].values, pd.Series(np.arange(len(ids1)), index=ids1)[both].values
        for li, l in enumerate(layers):
            X0, X1 = H0[a, li].astype(np.float32), H1[b, li].astype(np.float32)
            mu, U = X0.mean(0), np.linalg.svd(X0 - X0.mean(0), full_matrices=False)[2][:20]  # the plain prompt's top 20 directions
            P0, P1 = (X0 - mu) @ U.T, (X1 - mu) @ U.T
            same = np.mean([stats.pearsonr(P0[:, k], P1[:, k]).statistic for k in range(20)])
            Y0, Y1 = L0[a, li].astype(np.float32), L1[b, li].astype(np.float32)
            mu, V = Y0.mean(0), np.linalg.svd(Y0 - Y0.mean(0), full_matrices=False)[2][:20]
            Q0, Q1 = (Y0 - mu) @ V.T, (Y1 - mu) @ V.T
            last = np.mean([stats.pearsonr(Q0[:, k], Q1[:, k]).statistic for k in range(20)])
            rows.append(dict(persona=p, layer=int(l), depth=l / layers.max(), item_tokens=same, answer_position=last))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    Hk, common = haiku(); print(Hk.round(3).to_string())
    A = art(); print(A.round(3).to_string())
    Pp = perception(common); print(Pp.round(3).to_string())
    record("ds_persona", dict(haiku=Hk.reset_index().to_dict("records"), art=A.reset_index().to_dict("records"), perception=Pp.to_dict("records")))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    ax = axs[0]; cols = [c for c in Hk.columns if "<" in c and c != "kitsch weight"]
    x = np.arange(len(cols))
    for i, (p, r) in enumerate(Hk.iterrows()):
        ax.bar(x + (i - 1) * 0.26, r[cols].values, width=0.26, color={"none": "#8a8984", "editor": "#1b8f65", "reader": "#e8a33a"}[p], label={"none": "no persona", "editor": "haiku-journal editor", "reader": "r/haiku reader"}[p])
    ax.axhline(0.5, color="#3d3d3a", lw=0.8); ax.set_xticks(x, cols, rotation=20, ha="right"); ax.set_ylim(0.3, 1); ax.set_ylabel("AUC of DeepSeek's rating")
    ax.set_title("DeepSeek-V4.1-Flash under personas: haiku", loc="left"); ax.legend(frameon=False, loc="upper left")
    ax = axs[1]
    for p, d in Pp.groupby("persona"):
        c = {"editor": "#1b8f65", "reader": "#e8a33a"}[p]
        ax.plot(d.depth, d.item_tokens, color=c, lw=2, marker="o", label=f"{p}: at the haiku's tokens")
        ax.plot(d.depth, d.answer_position, color=c, lw=2, ls=(0, (3, 2)), marker="s", label=f"{p}: at the answer position")
    ax.set_ylim(0, 1.02); ax.set_xlabel("depth"); ax.set_ylabel("same as without persona (mean r over top 20 PCs)")
    ax.set_title("does the persona change perception or the verdict?", loc="left"); ax.legend(frameon=False, loc="lower left")
    ax = axs[2]
    if len(A):
        ax.bar(range(len(A)), A.w_highbrow, color=[{"none": "#8a8984", "lover": "#1b8f65", "lay": "#c0302f"}[p] for p in A.index])
        ax.set_xticks(range(len(A)), [{"none": "no persona", "lover": "art lover", "lay": "no interest in art"}[p] for p in A.index]); ax.axhline(0, color="#3d3d3a", lw=0.8)
        ax.set_ylabel("weight on the art lovers' high-brow axis"); ax.set_title("paintings (LAPIS)", loc="left")
    save(fig, "ds_persona")
    slide(Hk, Pp)


def slide(Hk, Pp):
    """Two panels for the deck: the tier steps under each persona (Claude for reference), and where the editor persona acts."""
    from taste.common import ratings, ANCHOR
    it = items("haiku"); c = ratings("haiku")[ANCHOR].reindex(it.index); sel = it.generator.isin(["human-reddit-top", "human-journal"]) & c.notna()
    claude = roc_auc_score(it.generator[sel] == "human-journal", c[sel])
    steps = {"twaiku < reddit-low": "tweets <\nignored r/haiku", "reddit-low < reddit-top": "ignored <\ntop r/haiku", "reddit-top < journal": "top r/haiku <\njournal", "open < frontier": "open <\nfrontier models"}
    col = {"none": "#8a8984", "reader": "#e8a33a", "editor": "#1b8f65"}; lab = {"none": "no persona", "reader": "as an r/haiku reader", "editor": "as a haiku-journal editor"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]; x = np.arange(len(steps))
    for i, p in enumerate(["none", "reader", "editor"]):
        ax.bar(x + (i - 1) * 0.26, Hk.loc[p, list(steps)].values, width=0.26, color=col[p], label=lab[p])
    ax.plot([2 - 0.45, 2 + 0.45], [claude] * 2, color="#6250d6", lw=2.5, label="Claude Opus 5.5 (no persona)")
    ax.axhline(0.5, color="#3d3d3a", lw=0.8); ax.set_xticks(x, list(steps.values())); ax.set_ylim(0.3, 1); ax.set_ylabel("DeepSeek-V4.1 rates the second higher (AUC)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax = axs[1]; d = Pp[Pp.persona == "editor"]
    ax.plot(d.depth, d.item_tokens, color=col["editor"], lw=2.5, marker="o", label="the haiku's own tokens")
    ax.plot(d.depth, d.answer_position, color=col["editor"], lw=2, ls=(0, (3, 2)), marker="s", label="the answer position")
    ax.set_ylim(0, 1.05); ax.set_xlabel("depth in DeepSeek-V4.1"); ax.set_ylabel("same as without the persona (r)")
    ax.set_title("editor persona: where does it change the model?", loc="left"); ax.legend(frameon=False, loc="lower left")
    save(fig, "ds_persona_slide")


if __name__ == "__main__":
    main()
