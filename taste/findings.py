"""Five headline findings, one slide each (claim, a small chart, examples), for the top of the report and the slide export.

  python -m taste.findings   # charts + numbers only; report.py calls build() for the HTML
Examples are picked by fixed rules (stated in each slide's footer), not by hand.
"""
import base64, html, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from taste.common import ratings, items, fam, save, record, ANCHOR, FAMILY_COLOR, ROOT, RESULTS, DATA
from taste.models import BY_NAME
from taste.probes import probe, hidden, LAYER_FRAC
from taste.report_examples import pct, chip, haiku_card, img_card
from taste.analysis_haiku import interaction

RNG = np.random.default_rng(0)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
# judges shown on the resolution chart: Qwen ladder as an ordinal blue ramp (steps 250/400/550/700), Claude in the frontier slot
JUDGES = {"qwen3.5-0.8b": ("Qwen3.5-0.8B", "#86b6ef"), "qwen3.5-4b": ("Qwen3.5-4B", "#3987e5"), "qwen3.5-27b": ("Qwen3.5-27B", "#1c5cab"),
          "qwen3.5-397b-a17b": ("Qwen3.5-397B", "#0d366b"), ANCHOR: ("Claude Opus 5.5", FAMILY_COLOR["frontier"])}
NAIVE = {"575": ("naive: 5-7-5 syllable count", (0, (4, 2))), "fluency": ("naive: smooth, predictable wording", (0, (1, 1.5)))}
MODEL_RUNGS = {"≤0.4B": ["smollm2-135m", "smollm2-360m"], "0.8–1.7B": ["smollm2-1.7b", "qwen3.5-0.8b"], "2–4B": ["qwen3.5-2b", "qwen3.5-4b"],
               "9–397B": ["qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"],
               "frontier": ["gpt-5.6-sol", "gemini-3.8-flash", "deepseek-v4.1-flash"]}  # Claude-written haiku left out: the Claude judge would rate its own family
HUMAN_RUNGS = {"tweet": ["human-twaiku"], "r/haiku ≤1 upvote": ["human-reddit-low"], "r/haiku top 5%": ["human-reddit-top"], "journal": ["human-journal"]}
STEP_LABEL = {("≤0.4B", "0.8–1.7B"): "≤0.4B\nvs ~1B", ("0.8–1.7B", "2–4B"): "~1B\nvs 2–4B", ("2–4B", "9–397B"): "2–4B\nvs 9–397B",
              ("9–397B", "frontier"): "9–397B\nvs frontier", ("tweet", "r/haiku ≤1 upvote"): "tweet\nvs ≤1 vote",
              ("r/haiku ≤1 upvote", "r/haiku top 5%"): "≤1 vote\nvs top 5%", ("r/haiku top 5%", "journal"): "top 5%\nvs journal"}
EDITORS = STEP_LABEL[("r/haiku top 5%", "journal")]


def auc(worse, better):
    """P(score of a random `better` item > a random `worse` one), ties 1/2; computed from ranks."""
    r = stats.rankdata(np.r_[worse, better]); nb = len(better)
    return (r[len(worse):].sum() - nb * (nb + 1) / 2) / (len(worse) * nb)


def boot_auc(worse, better, n=1000):
    return np.percentile([auc(RNG.choice(worse, len(worse)), RNG.choice(better, len(better))) for _ in range(n)], [2.5, 97.5])


def pair_html(label, a, b):
    return f'<div class="pair"><div class="pair-label">{label}</div><div class="pair-cards">{a}<span class="vs">vs</span>{b}</div></div>'


def slide(n, tag, title, sub, chart, examples, foot):
    return (f'<div class="slide finding"><div class="kicker">Finding {n} of 5 · {tag}</div><h3>{html.escape(title)}</h3><p class="sub">{sub}</p>'
            f'<div class="fgrid"><div class="fchart"><img src="{chart}" alt=""></div><div class="fex">{examples}</div></div><p class="foot">{foot}</p></div>')


def png_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------- 1. resolution at the top
def resolution(it, S, P):
    F = pd.read_parquet(DATA / "haiku_features.parquet").set_index("item_id").reindex(it.index)
    X = pd.concat([S[list(JUDGES)], pd.DataFrame({"575": -F.f_575_deviation, "fluency": ratings("haiku", "logprob_mean")["qwen3.5-27b"].reindex(it.index)})], axis=1)
    rows = []
    for panel, rungs in [("model", MODEL_RUNGS), ("human", HUMAN_RUNGS)]:
        names = list(rungs)
        for lo, hi in zip(names[:-1], names[1:]):
            a, b = it.generator.isin(rungs[lo]), it.generator.isin(rungs[hi])
            for c in X.columns:
                w, bt = X.loc[a, c].dropna().values, X.loc[b, c].dropna().values
                rows.append(dict(panel=panel, step=STEP_LABEL[(lo, hi)], judge=c, auc=auc(w, bt), ci=list(boot_auc(w, bt))))
    R = pd.DataFrame(rows)
    record("finding_resolution", R.to_dict("records"))

    plt.rcParams["savefig.dpi"] = 200
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 4.5), sharey=True, gridspec_kw={"width_ratios": [4, 3]}, layout="constrained")
    for ax, panel, ttl in zip(axs, ["model", "human"], ["Haiku by models\nbetter = more capable writer", "Haiku by people (r/haiku)\nbetter = more upvotes, then a journal"]):
        d = R[R.panel == panel]; steps = list(dict.fromkeys(d.step)); x = np.arange(len(steps))
        ax.axhspan(0, 0.5, color="#f0efec", lw=0, zorder=0); ax.axhline(0.5, color=MUTED, lw=1)
        for c, (lab, ls) in NAIVE.items():
            ax.plot(x, d[d.judge == c].set_index("step").auc.reindex(steps), color=MUTED, ls=ls, lw=2, label=lab)
        for c, (lab, col) in JUDGES.items():
            ax.plot(x, d[d.judge == c].set_index("step").auc.reindex(steps), color=col, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=lab)
        ax.set_xticks(x, steps); ax.set_title(ttl, loc="left"); ax.set_xlim(-0.3, len(steps) - 0.7)
    axs[0].set_ylim(0, 1); axs[0].set_ylabel("judge prefers the better haiku (AUC)")
    axs[0].text(-0.2, 0.47, "chance", color=INK2, va="top"); axs[0].text(-0.2, 0.03, "prefers the worse one", color=INK2, va="bottom")
    h, l = axs[0].get_legend_handles_labels(); order = [2, 3, 4, 5, 6, 0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=4, frameon=False, handlelength=2.6)
    p = save(fig, "finding_resolution")

    # examples: the pair at the editors' step and the pair at the open-vs-frontier step that best show the average pattern
    small, large = P["qwen3.5-0.8b"], P[["qwen3.5-27b", "qwen3.5-397b-a17b", ANCHOR]]
    ok575 = F.f_575_deviation == 0
    def best_pair(worse, better, same_topic):
        w, b = it.index[worse], it.index[better]
        M = np.minimum((large.loc[b].values[None] - large.loc[w].values[:, None]).min(2), small.loc[w].values[:, None] - small.loc[b].values[None])
        if same_topic: M = np.where(it.topic[w].to_numpy()[:, None] == it.topic[b].to_numpy()[None], M, -np.inf)
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        return w[i], b[j]
    g = it.generator
    pairs = [("crowd's pick vs editors' pick", best_pair(g.isin(HUMAN_RUNGS["r/haiku top 5%"]) & ok575, g.isin(HUMAN_RUNGS["journal"]) & ~ok575, False)),
             ("open model vs frontier model, same prompt", best_pair(g.isin(MODEL_RUNGS["9–397B"]), g.isin(MODEL_RUNGS["frontier"]), True))]
    chips = lambda i: [chip("Qwen 0.8B", P.loc[i, "qwen3.5-0.8b"]), chip("Qwen 397B", P.loc[i, "qwen3.5-397b-a17b"]), chip("Claude", P.loc[i, ANCHOR]),
                       f'<span class="chip">{"5-7-5" if F.f_575_deviation[i] == 0 else "not 5-7-5"}</span>']
    ex = "".join(pair_html(lab + (f" · “{it.topic[w]}”" if isinstance(it.topic[w], str) else ""), haiku_card(it.haiku[w], g[w], chips(w)), haiku_card(it.haiku[b], g[b], chips(b))) for lab, (w, b) in pairs)
    d = R.set_index(["step", "judge"]).auc
    sub = (f"Judges from 4B up, and a naive 5-7-5 check, tell broken haiku from passable ones. Between the crowd's favourites and journal haiku, the naive checks "
           f"pick the crowd's (5-7-5 rule: AUC {d[EDITORS, '575']:.2f}), and so does Qwen3.5-0.8B ({d[EDITORS, 'qwen3.5-0.8b']:.2f}). Larger judges lean towards the "
           f"editors (Qwen3.5-27B {d[EDITORS, 'qwen3.5-27b']:.2f}, 397B {d[EDITORS, 'qwen3.5-397b-a17b']:.2f}), Claude most clearly ({d[EDITORS, ANCHOR]:.2f}).")
    ci = max(np.diff(c)[0] / 2 for c in R.ci)
    foot = (f"AUC = chance that the judge scores a random haiku from the better group above one from the worse group. Groups: 200 haiku per model, 120 per human tier; "
            f"95% bootstrap intervals are at most ±{ci:.2f}. Fluency = mean log-probability of the words under Qwen3.5-27B. Examples: the pair where the large judges "
            f"and Qwen3.5-0.8B disagree most in the typical direction. Chips: percentile within each judge (100 = its favourite of 4,480). Claude may have seen some journal haiku in training.")
    return slide(1, "Haiku", "Telling good from great takes scale; a naive reading gets it backwards", sub, png_uri(p), ex, foot)


# ---------------------------------------------------------------- 2. the same craft axis in two model families
def craft_axis(it, S, P):
    pcs = json.loads(RESULTS.read_text())["pcs"]
    q, gk = pcs["haiku-craft/qwen3.5-27b"], pcs["haiku-craft/gemma-4-31b"]
    sq = pd.read_parquet(ROOT / "pc_scores/haiku-craft__qwen3.5-27b.parquet")
    sg = pd.read_parquet(ROOT / "pc_scores/haiku-craft__gemma-4-31b.parquet")
    kq = q["top"][0]
    kg = max(sg.columns, key=lambda k: abs(stats.spearmanr(sq[kq], sg.loc[sq.index, k]).statistic))  # Gemma's PC that matches Qwen's top craft axis
    rho_qg = stats.spearmanr(sq[kq], sg.loc[sq.index, kg]).statistic
    sg_k = sg[kg] * np.sign(rho_qg)
    judges = [r for r in S.columns if r != ANCHOR]
    rows = [dict(judge=r, family=fam(r), params_b=BY_NAME[r].params_b, rho_qwen_axis=q["rater_pc_rho"][kq][r], rho_gemma_axis=np.sign(rho_qg) * gk["rater_pc_rho"][kg][r]) for r in judges]
    D = pd.DataFrame(rows).sort_values("params_b")
    record("finding_craft_axis", dict(qwen_pc=kq, gemma_pc=kg, rho_between_axes=rho_qg, judges=D.to_dict("records"), claude_qwen_axis=q["rater_pc_rho"][kq][ANCHOR]))

    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")
    ax.axhline(0, color=MUTED, lw=1)
    for f, d in D.groupby("family"):
        ax.plot(d.params_b, d.rho_qwen_axis, color=FAMILY_COLOR[f], lw=2, marker="o", ms=6, mec="white", mew=1.2, label=f"{f.capitalize()} judges · axis found in Qwen3.5-27B")
        ax.plot(d.params_b, d.rho_gemma_axis, color=FAMILY_COLOR[f], lw=2, ls=(0, (4, 2)), marker="s", ms=5, mec="white", mew=1, label=f"{f.capitalize()} judges · axis found in Gemma-4-31B")
    ax.axhline(q["rater_pc_rho"][kq][ANCHOR], color=FAMILY_COLOR["frontier"], lw=1.5, ls=(0, (1, 1.5)))
    ax.text(0.75, q["rater_pc_rho"][kq][ANCHOR] + 0.02, "Claude Opus 5.5", color=INK2, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("judge size (billion parameters)"); ax.set_ylabel("ρ between judge's rating and the axis")
    ax.text(0.98, 0.03, "below 0: judge prefers the cliché end", color=INK2, ha="right", va="bottom", transform=ax.transAxes)
    fig.legend(loc="outside lower center", ncol=2, frameon=False, handlelength=2.6)
    p = save(fig, "finding_craft_axis")

    # examples: per topic, the haiku at each end of the (topic-centred) axis on which both models' axes agree; the two topics with the widest such spread
    model = it.index[it.family != "human"].intersection(sq.index)
    both = (sq.loc[model, kq].rank(pct=True) + sg_k.loc[model].rank(pct=True)) / 2
    both = both[~it.loc[both.index, "generator"].str.startswith("smollm")]  # keep the cliché end free of broken text
    spread = both.groupby(it.topic[both.index]).agg(lambda s: s.max() - s.min())
    chips = lambda i: [chip("Qwen 0.8B", P.loc[i, "qwen3.5-0.8b"]), chip("Qwen 27B", P.loc[i, "qwen3.5-27b"]), chip("Claude", P.loc[i, ANCHOR])]
    ex = ""
    for t in spread.nlargest(2).index:
        s = both[it.topic[both.index] == t]
        ex += pair_html(f"“{t}”: cliché end vs concrete end", haiku_card(it.haiku[s.idxmin()], it.generator[s.idxmin()], chips(s.idxmin())),
                        haiku_card(it.haiku[s.idxmax()], it.generator[s.idxmax()], chips(s.idxmax())))
    lq, lg = q["labels"][kq], gk["labels"][kg]
    sub = (f"With each topic's average removed, the main direction in Qwen3.5-27B's representation of a haiku runs from “{html.escape(lq['low'])}” to “{html.escape(lq['high'])}”. "
           f"Gemma-4-31B, trained separately, has the same direction (ρ = {rho_qg:.2f} over 4,000 haiku). Large judges of both families reward the concrete end; "
           f"Qwen3.5-0.8B and 2B prefer the clichés.")
    foot = (f"Axes: {kq} of Qwen3.5-27B and {kg} of Gemma-4-31B (mean over the haiku's tokens at ⅔ depth, topic means removed). Names written by {ANCHOR} from 12 items per end; "
            f"given only the names it picks the higher of 20 held-out pairs {lq['label_accuracy']:.0%} of the time. Examples: the two topics where both axes spread widest.")
    return slide(2, "Haiku · inside the models", "Two model families learned the same taste axis: concrete images over clichés", sub, png_uri(p), ex, foot)


# ---------------------------------------------------------------- 3. small models know more than they say
def know_more(it, S, P):
    pr = pd.DataFrame(json.loads(RESULTS.read_text())["probes"])
    d = pr[(pr.set == "haiku") & (pr.pool == "mean") & (pr.split == "open models → frontier + human")]
    D = d.pivot(index="rater", columns="confound_ctl", values="probe").rename(columns={False: "probe", True: "probe_ctl"}).join(d[~d.confound_ctl].set_index("rater").verbal)
    D["params_b"] = D.index.map(lambda r: BY_NAME[r].params_b); D["family"] = D.index.map(fam); D = D.sort_values("params_b")
    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")
    ax.axhline(0, color=MUTED, lw=1)
    for f, g in D.groupby("family"):
        c = FAMILY_COLOR[f]
        ax.fill_between(g.params_b, g.verbal, g.probe, color=c, alpha=0.12, lw=0)
        ax.plot(g.params_b, g.probe, color=c, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=f"{f.capitalize()}: read from its hidden states")
        ax.plot(g.params_b, g.verbal, color=c, lw=2, ls=(0, (4, 2)), marker="o", ms=6, mfc="white", mec=c, mew=1.5, label=f"{f.capitalize()}: what it says (its rating)")
    ax.set_xscale("log"); ax.set_xlabel("judge size (billion parameters)"); ax.set_ylabel("ρ with Claude's rating on unseen haiku")
    fig.legend(loc="outside lower center", ncol=2, frameon=False, handlelength=2.6)
    p = save(fig, "finding_know_more")

    # examples: refit Qwen3.5-0.8B's probe and show the test items where its words and its hidden states disagree most
    r = "qwen3.5-0.8b"
    ids, layers, H = hidden("haiku", r, "mean"); li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))
    X = H[:, li].astype(np.float32); pos = pd.Series(np.arange(len(ids)), index=ids)
    y = (S[ANCHOR] - S[ANCHOR].mean()) / S[ANCHOR].std()
    open_ = it.generator.map(fam).isin(["smollm", "qwen", "gemma"])
    tr, te = it.index[open_ & y.notna()], it.index[~open_ & y.notna() & S[r].notna()]
    pred = pd.Series(probe(X[pos[tr]], y[tr].values, X[pos[te]]), index=te)
    says, thinks, claude = S.loc[te, r].rank(pct=True) * 100, pred.rank(pct=True) * 100, S.loc[te, ANCHOR].rank(pct=True, method="average") * 100
    gap = (thinks - says)[~it.loc[te, "haiku"].duplicated()]
    c = claude[gap.index] - 50
    agree = gap[(np.sign(gap) == np.sign(c)) & (c.abs() >= 25)]  # Claude sides clearly with the hidden states
    chips = lambda i: [chip("says", says[i]), chip("hidden states", thinks[i]), chip("Claude", claude[i])]
    lo, hi_ = agree.nsmallest(2).index, agree.nlargest(2).index
    ex = (pair_html("Qwen3.5-0.8B rates it high, its hidden states low", *[haiku_card(it.haiku[i], it.generator[i], chips(i)) for i in lo])
          + pair_html("Qwen3.5-0.8B rates it low, its hidden states high", *[haiku_card(it.haiku[i], it.generator[i], chips(i)) for i in hi_]))
    record("finding_know_more", D.reset_index().to_dict("records"))
    s = D.loc[r]; b = D.loc["qwen3.5-27b"]
    sub = (f"A linear probe trained on haiku from open models and tested on frontier and human haiku. Qwen3.5-0.8B's own ratings run against Claude's "
           f"(ρ = {s.verbal:+.2f}), yet Claude's verdict is linearly readable from its hidden states (ρ = {s.probe:+.2f}; {s.probe_ctl:+.2f} with word "
           f"predictability and length regressed out). The gap closes with size: Qwen3.5-27B {b.probe:+.2f} vs {b.verbal:+.2f}.")
    foot = ("Probe: ridge regression on the mean hidden state at ⅔ depth (fixed in advance), trained on 3,400 open-model haiku, tested on 1,480 frontier and human haiku. "
            "Chips: percentiles among the test haiku. Examples: the largest gaps between the rating and the probe among haiku where Claude clearly agrees with the probe.")
    return slide(3, "Haiku · inside the models", "Small models know more than they say", sub, png_uri(p), ex, foot)


# ---------------------------------------------------------------- 4. no family loyalty
QW = ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b"]
GM = ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-26b-a4b", "gemma-4-31b"]


def loyalty(it, S, P):
    g = it.generator; q, gm = g.isin(QW), g.isin(GM)
    judges = [r for r in S.columns if fam(r) in ("qwen", "gemma")]
    rows = []
    for r in judges:
        a, b = P.loc[q, r].dropna().values, P.loc[gm, r].dropna().values
        bs = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(1000)]
        rows.append(dict(judge=r, family=fam(r), params_b=BY_NAME[r].params_b, pref_qwen=a.mean() - b.mean(), ci=list(np.percentile(bs, [2.5, 97.5]))))
    D = pd.DataFrame(rows).sort_values("params_b")
    record("finding_loyalty", D.to_dict("records"))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    for f, d in D.groupby("family"):
        c = FAMILY_COLOR[f]; ci = np.array(d.ci.tolist())
        ax.errorbar(d.params_b * (1.04 if f == "gemma" else 1), d.pref_qwen, yerr=[d.pref_qwen - ci[:, 0], ci[:, 1] - d.pref_qwen], color=c, lw=2, marker="o", ms=6,
                    mec="white", mew=1.2, capsize=0, elinewidth=1, label=f"{f.capitalize()} judges")
    ax.set_xscale("log"); ax.set_xlabel("judge size (billion parameters)"); ax.set_ylabel("prefers Qwen's haiku over Gemma's\n(percentile points)")
    lim = np.abs(ax.get_ylim()).max(); ax.set_ylim(-lim, lim)
    ax.text(0.02, 0.98, "Qwen's haiku better", color=INK2, va="top", transform=ax.transAxes); ax.text(0.98, 0.02, "Gemma's haiku better", color=INK2, ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(loc="upper right", frameon=False)
    p = save(fig, "finding_loyalty")

    # example: the topic where Qwen3.5-27B and Gemma-4-31B, judging each other's and their own haiku, agree most strongly that the Qwen one is better
    jq, jg = "qwen3.5-27b", "gemma-4-31b"
    pairs = [(a, b) for t in it.topic.dropna().unique() for a in it.index[(g == jq) & (it.topic == t)] for b in it.index[(g == jg) & (it.topic == t)]]
    margin = lambda a, b: min(P.loc[a, jq] - P.loc[b, jq], P.loc[a, jg] - P.loc[b, jg])
    a, b = max(pairs, key=lambda ab: margin(*ab))
    chips = lambda i: [chip("Qwen 27B judge", P.loc[i, jq]), chip("Gemma 31B judge", P.loc[i, jg]), chip("Claude", P.loc[i, ANCHOR])]
    a2, b2 = max(pairs, key=lambda ab: min(P.loc[ab[1], jq] - P.loc[ab[0], jq], P.loc[ab[1], jg] - P.loc[ab[0], jg]))
    ex = (pair_html(f"“{it.topic[a]}”: both judges prefer Qwen's haiku, Gemma-4-31B included", haiku_card(it.haiku[a], jq, chips(a)), haiku_card(it.haiku[b], jg, chips(b)))
          + pair_html(f"“{it.topic[a2]}”: both judges prefer Gemma's haiku, Qwen3.5-27B included", haiku_card(it.haiku[a2], jq, chips(a2)), haiku_card(it.haiku[b2], jg, chips(b2))))
    beh = json.loads(RESULTS.read_text())["haiku_beholder"]
    dd = D.set_index("judge").pref_qwen
    sub = (f"If judges favoured their own family, the blue points would sit above zero and the orange ones below. Instead, which family's haiku a judge prefers "
           f"depends on its size: Qwen3.5-0.8B prefers Gemma's ({dd['qwen3.5-0.8b']:+.1f} points), Gemma-4-31B prefers Qwen's ({dd['gemma-4-31b']:+.1f}). "
           f"Across all judges the own-family bonus is {beh['family_contrast']:+.3f} z (95% CI {beh['family_contrast_ci95'][0]:+.3f} to {beh['family_contrast_ci95'][1]:+.3f}).")
    foot = ("Writers matched in size: Qwen3.5 2B–35B (1,000 haiku) vs Gemma 4 E2B–31B (800 haiku). Bars: 95% bootstrap intervals. Examples: the same-prompt pairs where the two judges agree most "
            "strongly, one in each direction. Chips: percentile within each judge.")
    return slide(4, "Haiku · eye of the beholder", "No family loyalty: what a judge prefers depends on its size, not its maker", sub, png_uri(p), ex, foot)


# ---------------------------------------------------------------- 5. ACX: Claude spots and dislikes AI art, but recognises the human art
def acx(Si, ii):
    R = json.loads(RESULTS.read_text()); pref, det = R["acx_ai_preference"], R["acx_detection_accuracy"]
    rec = pd.DataFrame(R["acx_recognition"]).set_index(["rater", "group"])
    who = [("People: median respondent", det["humans: median respondent"], pref["humans (favourite share)"], FAMILY_COLOR["human"]),
           ("People: majority vote", det["humans: crowd majority"], np.nan, FAMILY_COLOR["human"])]
    for r, lab in [("qwen3.5-397b-a17b", "Qwen3.5-397B"), ("gemma-4-31b", "Gemma-4-31B")]:
        who.append((lab, det[r], pref[r], FAMILY_COLOR[fam(r)]))
    who.append(("Claude Opus 5.5", det[ANCHOR], pref[ANCHOR], FAMILY_COLOR["frontier"]))
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.2), sharey=True)
    y = np.arange(len(who))
    axs[0].barh(y, [w[1] for w in who], color=[w[3] for w in who], height=0.62); axs[0].axvline(0.5, color=MUTED, lw=1)
    axs[0].set_xlim(0, 1); axs[0].set_title("tells AI from human art", loc="left"); axs[0].set_xlabel("accuracy (50 images)")
    axs[0].set_yticks(y, [w[0] for w in who]); axs[0].invert_yaxis()
    axs[1].barh(y, [0 if np.isnan(w[2]) else w[2] for w in who], color=[w[3] for w in who], height=0.62); axs[1].axvline(0, color=MUTED, lw=1)
    axs[1].set_title("prefers AI art", loc="left"); axs[1].set_xlabel("rank of AI − human images")
    axs[1].text(0.01, 1, "n/a", color=INK2, va="center")
    for ax in axs:
        for i, w in enumerate(who):
            v = w[1] if ax is axs[0] else w[2]
            if not np.isnan(v): ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.0%}" if ax is axs[0] else f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", color=INK)
    axs[1].set_xlim(-0.55, 0.4)
    p = save(fig, "finding_acx")
    acx_ = ii[ii.dataset == "acx"]; Pa = Si.loc[acx_.index].rank(pct=True) * 100; hum = acx_.human_score.rank(pct=True) * 100
    nai = lambda idx: int((acx_.loc[idx, "source"] == "ai").sum())
    achips = lambda i: [chip("people", hum[i]), chip("Claude", Pa.loc[i, ANCHOR])]
    row = lambda t, idx: f'<div class="row-label">{html.escape(t)}: {nai(idx)} of 5 are AI</div><div class="img-grid">{"".join(img_card(next(acx_.loc[[i]].itertuples()), achips(i), acx_.loc[i, "title"]) for i in idx)}</div>'
    ex = row("People's 5 favourites", hum.nlargest(5).index) + row("Claude's 5 favourites", Pa[ANCHOR].nlargest(5).index)
    c = rec.loc[ANCHOR]
    sub = (f"In Scott Alexander's AI Art Turing Test, people and open models are near chance at telling AI from human art and do not mind AI art. Claude Opus 5.5 "
           f"gets {det[ANCHOR]:.0%} right and ranks AI images far lower. But it also names the artist of {int(c.loc['human, artist named', 'n'])} of the "
           f"{int(c.loc['human, artist named', 'n'] + c.loc['human, not named', 'n'])} human works, and those are the ones it ranks highest "
           f"(mean percentile {c.loc['human, artist named', 'rater_pct_rank']:.0%} vs {c.loc['human, not named', 'rater_pct_rank']:.0%}; crowd: "
           f"{c.loc['human, artist named', 'crowd_pct_rank']:.0%} vs {c.loc['human, not named', 'crowd_pct_rank']:.0%}). Taste or recognition?")
    foot = ("People: 11k ACX respondents' per-image guesses and 9,172 'favourite' votes. Badge = Scott Alexander's answer key (public since November 2024). "
            "Chips: percentile among the 50 images. Claude rates in whole digits, so it has many ties.")
    return slide(5, "Images", "Claude spots AI art and dislikes it, but it also recognises the famous human works", sub, png_uri(p), ex, foot)


CSS = """
.finding{width:min(1280px,calc(100vw - 32px));aspect-ratio:16/9;position:relative;left:50%;transform:translateX(-50%);display:flex;flex-direction:column;padding:24px 32px 14px}
.finding .kicker{font-size:.78rem;font-weight:650;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.finding h3{font-size:1.6rem;line-height:1.2;margin-bottom:6px}.finding .sub{font-size:.95rem;line-height:1.4;margin-bottom:12px}
.fgrid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:22px;flex:1;align-items:center}
.fchart img{width:100%;display:block;background:#fff;border-radius:8px}
.fex{display:flex;flex-direction:column;gap:12px}
.pair-label{font-size:.82rem;font-weight:650;color:var(--ink2);margin:0 0 5px}
.pair-cards{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:6px;align-items:stretch}
.pair-cards .vs{align-self:center;font-size:.75rem;color:var(--muted)}
.finding .hk{padding:8px 11px;gap:5px}.finding .poem{font-size:.95rem;line-height:1.4}.finding .chip{font-size:.72rem}
.finding .img-grid{gap:6px}.finding .im img{height:80px}.finding .im .t{font-size:.72rem}.finding .row-label{margin:2px 0 4px}
.finding .foot{margin-top:8px}
@media (max-width:820px){.finding{aspect-ratio:auto;left:0;transform:none;width:auto}.fgrid{grid-template-columns:1fr}}
"""


def build():
    it = items("haiku"); S = ratings("haiku").reindex(it.index); P = pct(S)
    ii = items("images"); Si = ratings("images").reindex(ii.index)
    return [resolution(it, S, P), craft_axis(it, S, P), know_more(it, S, P), loyalty(it, S, P), acx(Si, ii)], CSS


if __name__ == "__main__":
    build()
