"""The model of taste, one panel per term, from saved results (no new computation):
  obvious        a feature judges of every size weight the same way (all-haiku PC1 of Qwen3.5-27B, taste.taste_features)
  non-obvious    open vs frontier writers: AUC read out of each model's representation (best layer) vs what it says (taste.taste_space)
  high/low-brow  the topic-removed kitsch feature (craft PC2): each judge's weight, and the steering effect (taste.steer_features)
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


def by_size(ax, s, **kw):
    """Qwen and Gemma lines against size, Claude as a star right of a divider."""
    for f in ["qwen", "gemma"]:
        d = s[[j for j in s.index if fam(j) == f]]; d = d.reindex(sorted(d.index, key=SIZE))
        ax.plot([SIZE(j) for j in d.index], d.values, color=FAMILY_COLOR[f], marker="o", lw=2, label=LABEL[f], **kw)
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
    ax = axs[0, 0]; by_size(ax, W["PC1"].drop(index=[j for j in W.index if fam(j) not in ("qwen", "gemma") and j != ANCHOR]))
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_ylabel("weight: ρ(feature, judge's rating)")
    ax.set_title(f"obvious: every judge rewards it\n“{lab['PC1']['low']}”\n→ “{lab['PC1']['high']}”", loc="left")
    ax.legend(frameon=False)
    ax = axs[0, 1]
    by_size(ax, rep.drop(index=[j for j in rep.index if fam(j) not in ("qwen", "gemma")]), ls=(0, (1, 1.5)))
    by_size(ax, says.drop(index=[j for j in says.index if fam(j) not in ("qwen", "gemma") and j != ANCHOR]))
    ax.axhline(0.5, color="#8a8984", lw=0.8); ax.set_ylabel("open vs frontier writers, AUC")
    h = [plt.Line2D([], [], color="#3d3d3a", ls=(0, (1, 1.5)), lw=2, label="read out of the representation"), plt.Line2D([], [], color="#3d3d3a", lw=2, label="what the judge says")]
    ax.legend(handles=h, frameon=False, loc="lower right"); ax.set_title("non-obvious: open vs frontier writers\nevery model represents it,\nonly large judges use it", loc="left")
    ax = axs[0, 2]; by_size(ax, Wc["PC2"].drop(index=[j for j in Wc.index if fam(j) not in ("qwen", "gemma") and j != ANCHOR]))
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_ylabel("weight: ρ(feature, judge's rating)")
    ax.set_title(f"high/low-brow: small and large judges disagree\n“{labc['PC2']['low']}”\n→ “{labc['PC2']['high']}”", loc="left")
    ax = axs[1, 0]; by_size(ax, E.effect); ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_ylabel("rating change per unit of steering"); ax.set_title("high/low-brow, causally: steering each judge\nalong its own direction for that feature", loc="left")
    for ax, part, people_part, ttl in [(axs[1, 1], "subjective: its own", parts[2], "subjective: stable, but the voter's own"),
                                       (axs[1, 2], "random (vote to vote)", parts[3], "random: does not replicate")]:
        s = tm[part].reindex(sorted(tm.index, key=lambda j: (fam(j) != "qwen", BY_NAME[j].params_b)))
        names = [j.replace("qwen3.5-", "Qwen ").replace("gemma-4-", "Gemma ").replace("-a10b", "").replace("-a17b", "").replace("-a4b", "") for j in s.index]
        y = np.arange(len(s))
        ax.barh(y, s.values, color=[FAMILY_COLOR[fam(j)] for j in s.index], height=0.7)
        ax.barh([len(s) + 0.6, len(s) + 1.6], [L[("lay", people_part)], L[("lover", people_part)]], color=PEOPLE, height=0.7)
        ax.set_yticks(list(y) + [len(s) + 0.6, len(s) + 1.6], names + ["people: lay vote", "people: art lover's vote"])
        ax.set_xlim(0, 1); ax.set_xlabel("share of one vote's variance"); ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    save(fig, "taste_terms")


if __name__ == "__main__":
    main()
