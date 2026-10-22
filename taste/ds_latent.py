"""DeepSeek-V4.1-Flash (763B) from the inside: its latent states for every haiku (mean over the haiku's tokens, residual stream after
every 4th block, hyper-connection copies averaged; taste.ds_rate).
  1. PCA of its representation of model haiku with topic means removed (as taste.taste_features --craft), oriented so that +PC is what
     DeepSeek rates higher; every judge's weight on every PC; the writer contrasts; human haiku projected in (r/haiku top < journal).
  2. How much of Qwen3.5-27B's kitsch axis (craft PC2) lies in DeepSeek's space: the best single DeepSeek PC and the held-out
     readout from its full representation at every depth.
  3. Probes at every depth (ridge on the top 64 PCs, held out by topic): its own rating, Claude's rating, the kitsch axis.
Claude names DeepSeek's PCs that matter most (cached as haiku-craft/deepseek-v4.1-flash-1017/PCk, validated as in taste.pcs).

  python -m taste.ds_latent
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from taste.common import ratings, items, hidden, save, record, fam, ANCHOR, DIVERGING, RESULTS
from taste.paths import DATA
from taste.pcs import basis, label_pc
from taste.gen_haiku import openrouter
from taste.taste_space import group_of, folds, readout
from taste.taste_features import TIERS, TRUTH, classify, humans_on_craft

DS = "deepseek-v4.1-flash-1017"
COMPARE = "qwen3.5-27b"


def main():
    it = items("haiku"); S = ratings("haiku").reindex(it.index); g = group_of(it)
    keep = it.topic.notna(); itm, Sm, gm = it[keep], S[keep], g[keep]; demean = itm.topic
    Sm = Sm - Sm.groupby(itm.topic).transform("mean")
    B, evr = basis("haiku", DS, keep, demean); B = B.reindex(itm.index)
    Hu = humans_on_craft(it, keep, demean, ref=DS)
    W = pd.DataFrame({r: [stats.spearmanr(B[k], Sm[r], nan_policy="omit").statistic for k in B] for r in Sm.columns}, index=B.columns).T
    for k in B: sg = np.sign(W.loc[DS, k]); B[k] *= sg; W[k] *= sg; Hu[k] *= sg
    T = pd.DataFrame({name: [stats.spearmanr(B.loc[gm.isin([lo, hi]), k], gm[gm.isin([lo, hi])] == hi).statistic for k in B]
                      for name, (lo, hi) in TRUTH.items() if gm.isin([lo, hi]).any()}, index=B.columns).T
    gen = it.generator.reindex(Hu.index); sel = gen.isin(["human-reddit-top", "human-journal"])
    T.loc["r/haiku top < journal (projected)"] = [stats.spearmanr(Hu.loc[sel, k], gen[sel] == "human-journal").statistic for k in B]
    tier = pd.DataFrame({t: W.loc[[j for j in v if j in W.index]].mean() for t, v in TIERS.items()})
    summary = tier.join(pd.Series({k: classify(W[k]) for k in B}, name="class")).join(W.loc[[DS, ANCHOR]].T.add_prefix("ρ ")).join(T.T).join(pd.Series(evr, index=B.columns, name="var"))
    # Qwen3.5-27B's kitsch axis inside DeepSeek
    Q = pd.read_parquet(DATA / "steer/haiku_features_craft.parquet").reindex(itm.index)["PC2"]
    rho_pc = pd.Series({k: stats.spearmanr(B[k], Q).statistic for k in B})
    r2_top20 = 1 - np.var(Q - np.c_[np.ones(len(B)), B.values] @ np.linalg.lstsq(np.c_[np.ones(len(B)), B.values], Q.values, rcond=None)[0]) / np.var(Q)
    summary["ρ with Qwen-27B kitsch axis"] = rho_pc
    pd.set_option("display.width", 250); print(summary.round(2).to_string())
    print(f"Qwen-27B kitsch axis: best DeepSeek PC {rho_pc.abs().idxmax()} (ρ {rho_pc[rho_pc.abs().idxmax()]:.2f}); variance explained by DeepSeek's top 20 PCs {r2_top20:.2f}")
    # probes by depth, held out by topic (model haiku, topic-centred targets)
    fold = folds(itm)
    targets = {"its own rating": Sm[DS].values, "Claude's rating": Sm[ANCHOR].values, "Qwen-27B kitsch axis": Q.values}
    rows = []
    for r in [DS, COMPARE]:
        ids, layers, H = hidden("haiku", r, "mean")
        pos = pd.Series(np.arange(len(ids)), index=ids).reindex(itm.index).values
        for li, l in enumerate(layers):
            X = H[pos, li].astype(np.float32); X = X - pd.DataFrame(X).groupby(itm.topic.values).transform("mean").values
            for name, y in targets.items():
                pred = readout(X, y.astype(float), fold); ok = np.isfinite(y)
                rows.append(dict(model=r, layer=int(l), depth=l / layers.max(), target=name, rho=stats.spearmanr(pred[ok], y[ok]).statistic))
    P = pd.DataFrame(rows); print(P.pivot_table(index=["model", "layer"], columns="target", values="rho").round(2).to_string())
    verbal = {"Claude's rating": stats.spearmanr(Sm[DS], Sm[ANCHOR], nan_policy="omit").statistic, "Qwen-27B kitsch axis": stats.spearmanr(Sm[DS], Q, nan_policy="omit").statistic}
    print("DeepSeek's rating itself: ρ with Claude", round(verbal["Claude's rating"], 2), "| with the kitsch axis", round(verbal["Qwen-27B kitsch axis"], 2))
    # name the PCs DeepSeek's rating leans on most, and any high-brow vs low-brow ones
    chosen = list(dict.fromkeys(summary["ρ " + DS].abs().sort_values(ascending=False).index[:4].tolist() + summary[summary["class"] == "high-brow vs low-brow"].large.abs().sort_values(ascending=False).index[:2].tolist()))
    client = openrouter()
    labels = {k: label_pc(client, "haiku", itm, B[k], f"haiku-craft/{DS}/{k}") for k in chosen}
    for k in chosen: print(k, summary.loc[k, "class"], {x: labels[k][x] for x in ["low", "high", "label_accuracy"]})
    print("labelling cost $", round(sum(l["cost"] for l in labels.values()), 3))
    record("ds_latent", dict(summary=summary.reset_index(names="pc").to_dict("records"), probes=P.to_dict("records"), verbal=verbal, labels=labels, chosen=chosen,
                             kitsch_best_pc=rho_pc.abs().idxmax(), kitsch_rho=float(rho_pc[rho_pc.abs().idxmax()]), kitsch_r2_top20=float(r2_top20)))
    plot(W, T, summary, labels, chosen, P, verbal)


def plot(W, T, summary, labels, chosen, P, verbal):
    order = [r for t in TIERS.values() for r in t if r in W.index] + [DS]
    M = pd.concat([W.loc[order, chosen].T, T[chosen].T], axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(16, 0.6 * len(chosen) + 3.2), layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    im = ax.imshow(M.values, cmap=DIVERGING, vmin=-0.6, vmax=0.6, aspect="auto"); ax.grid(False)
    name = lambda c: c if c in T.index else c.replace(ANCHOR, "Claude").replace(DS, "DeepSeek 763B").replace("qwen3.5-", "Qwen ").replace("gemma-4-", "Gemma ").replace("-a10b", "").replace("-a17b", "").replace("-a4b", "").replace("b", "B").replace("e2B", "E2B").replace("e4B", "E4B")
    ax.set_xticks(range(M.shape[1]), [name(c) for c in M.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(chosen)), [f"{labels[k]['low']} → {labels[k]['high']}" for k in chosen])
    for x in [len(TIERS["tiny"]) - 0.5, len(TIERS["tiny"]) + len(TIERS["small"]) - 0.5, len(order) - 1.5, len(order) - 0.5]: ax.axvline(x, color="white", lw=4)
    fig.colorbar(im, ax=ax, shrink=0.7, label="ρ(feature, judge's score)\nground truth: ρ(feature, better writer)")
    ax.set_title("DeepSeek-V4.1-Flash's own PCs (topic removed):\nwho rewards them", loc="left")
    ax = axs[1]
    for (m, t), d in P.groupby(["model", "target"]):
        ax.plot(d.depth, d.rho, color={"its own rating": "#2a78d6", "Claude's rating": "#6250d6", "Qwen-27B kitsch axis": "#1b8f65"}[t], lw=2 if m == DS else 1.2,
                ls="-" if m == DS else (0, (2, 2)), marker="o" if m == DS else None, label=f"{t} ({'DeepSeek' if m == DS else 'Qwen-27B'})")
    ax.axhline(verbal["Claude's rating"], color="#6250d6", lw=0.8, ls=":")
    ax.set_xlabel("depth"); ax.set_ylabel("held-out ρ of the readout"); ax.set_title("probes on the latent states, by depth", loc="left"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "ds_latent")


if __name__ == "__main__":
    main()
