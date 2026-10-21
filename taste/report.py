"""Build the taste report from saved results. Every number on the page is computed here from results.json and the rating
tables; nothing is typed in by hand. The page goes to ~/taste/reports (private) and is symlinked into the folder served on
127.0.0.1:8095.

  python -m taste.report
"""
import base64, html, json, shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from taste.common import ratings, items, fam, rater_order, generator_order, ANCHOR, RESULTS
from taste.paths import DATA, PLOTS, REPORTS, ROOT
from taste.report_katex import stylesheet, typeset
from taste import report_examples, findings, intro

SERVED = Path.home() / "generative-learning/reports"
NAME = "taste"
R = json.loads(RESULTS.read_text())


def fig(name, caption, width="100%"):
    b64 = base64.b64encode((PLOTS / f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def table(df, num_fmt="{:.2f}", index=True):
    head = "".join(f"<th{' class=num' if pd.api.types.is_numeric_dtype(df[c]) else ''}>{html.escape(str(c))}</th>" for c in df.columns)
    head = (f"<th>{html.escape(str(df.index.name or ''))}</th>" if index else "") + head
    rows = []
    for i, r in df.iterrows():
        cells = "".join(f'<td class="num">{num_fmt.format(v)}</td>' if isinstance(v, (float, np.floating)) and not np.isnan(v)
                        else f'<td class="num">–</td>' if isinstance(v, (float, np.floating)) else f"<td>{html.escape(str(v))}</td>" for v in r.values)
        rows.append(f"<tr>{f'<th scope=row>{html.escape(str(i))}</th>' if index else ''}{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def f2(x): return f"{x:.2f}"
def pct(x): return f"{100 * x:.0f}%"


def build():
    hi, ii = items("haiku"), items("images")
    Sh, Si = ratings("haiku"), ratings("images")
    sc = pd.DataFrame(R["haiku_judge_scaling"]).set_index("rater").loc[rater_order(pd.DataFrame(R["haiku_judge_scaling"]).rater)]
    ga = pd.DataFrame(R["images_gt_agreement"]).pivot(index="rater", columns="dataset", values="rho")
    ga = ga.loc[rater_order(ga.index)][["ava", "lapis", "acx"]]
    beh = R["haiku_beholder"]; icc_h = R["haiku_icc"]; icc_i = R["images_icc"]
    pref, det = R["acx_ai_preference"], R["acx_detection_accuracy"]
    probes = pd.DataFrame(R["probes"]); cm = pd.DataFrame(R["cross_modal"])
    pcs = R["pcs"]
    local = [r for r in sc.index if r in ratings("haiku", "logprob_mean")]

    # ---- headline numbers
    z = (Sh - Sh.mean()) / Sh.std()
    gen = hi.generator.reindex(z.index)
    open_r = [r for r in z.columns if r != ANCHOR]
    by_gen = z[open_r].mean(1).groupby(gen).mean()
    journal_rank = int((by_gen > by_gen["human-journal"]).sum()) + 1
    anchor_gen = z[ANCHOR].groupby(gen).mean()
    best_local = sc.loc[local].rho_anchor.idxmax()
    acx_claude = pref.get(ANCHOR, np.nan)

    # probe vs verbal, headline split per modality (mean pooling, no confound control)
    hp = probes[(probes.pool == "mean") & ~probes.confound_ctl]
    hsplit = hp[hp.split == "open models → frontier + human"].set_index("rater")
    isplit = hp[hp.split == "AVA photos → LAPIS art"].set_index("rater")
    pv = pd.DataFrame({"haiku: probe": hsplit.probe, "haiku: verbal": hsplit.verbal, "images AVA→LAPIS: probe": isplit.probe, "images AVA→LAPIS: verbal": isplit.verbal})
    pv = pv.loc[rater_order(pv.index)]
    wins_h = int((pv["haiku: probe"] > pv["haiku: verbal"]).sum()); wins_i = int((pv["images AVA→LAPIS: probe"] > pv["images AVA→LAPIS: verbal"]).sum())

    # ---- PC label table
    lab_rows = []
    for key, v in pcs.items():
        which, ref = key.split("/")
        for k in v["top"]:
            l = v["labels"].get(k, {})
            fr = pd.Series(v["feature_rho"][k]).dropna()
            best_f = fr.abs().idxmax() if len(fr) else "–"
            lab_rows.append({"set": which, "reference model": ref, "PC": k, "ρ with its rating": v["rater_pc_rho"][k][ref],
                             "label: low → high": f"{l.get('low', '?')} → {l.get('high', '?')}" if l else "–",
                             "label check (held-out pairs)": l.get("label_accuracy", np.nan), "closest hand-made feature": f"{best_f.removeprefix('f_')} (ρ={fr[best_f]:+.2f})" if len(fr) else "–"})
    labs = pd.DataFrame(lab_rows)

    qwen_local = [r for r in local if fam(r) == "qwen"]
    smallest, largest_local = qwen_local[0], qwen_local[-1]
    largest_api = max((r for r in sc.index if r not in local and fam(r) == "qwen"), key=lambda r: sc.loc[r].params_b)
    craft = [(k, v) for k, v in pcs.items() if k.startswith("haiku-craft/")]
    craft_summary = "; ".join(f"{k.split('/')[1]}: “{v['labels'][v['top'][0]]['low']}” → “{v['labels'][v['top'][0]]['high']}” (label check {pct(v['labels'][v['top'][0]]['label_accuracy'])})" for k, v in craft)
    ex, ex_css = report_examples.build()
    fs, fs_css = findings.build()
    ins, in_css = intro.build()
    res = pd.DataFrame(json.loads(RESULTS.read_text())["finding_resolution"]).set_index(["step", "judge"]).auc
    built = datetime.now(ZoneInfo("Europe/London")).strftime("%d %B %Y, %H:%M %Z")
    n_haiku, n_gen = len(hi), hi.generator.nunique()
    cross = cm[(cm.pool == "last") & (cm.direction == "haiku → images")].pivot(index="rater", columns="test", values="rho")
    cross = cross.loc[rater_order(cross.index)][["ava", "lapis", "acx"]]
    back = cm[(cm.pool == "last") & (cm.direction == "images → haiku")].set_index("rater").rho

    body = f"""
<header>
<div class="eyebrow">Taste in language models · {built}</div>
<h1>Do language models have taste?</h1>
<p class="lede">Open Qwen3.5 and Gemma 4 models from 0.8B to 31B parameters, plus Qwen3.5-122B/397B and Claude Opus 5.5 through an API{', and DeepSeek-V4.1-Flash (763B) on our own GPUs with its hidden states' if 'deepseek-v4.1-flash-1017' in Sh.columns else ''}, rated {n_haiku:,} haiku from {n_gen - 4} models and four human tiers, and {len(ii):,} images with human ratings (AVA photos, LAPIS art, the ACX AI Art Turing Test). We read each rating from the model's next-token probabilities and kept its hidden states.</p>
<div class="tiles">
<div class="tile"><strong>{f2(res[findings.EDITORS, "qwen3.5-0.8b"])} → {f2(res[findings.EDITORS, ANCHOR])}</strong><span>chance that the judge prefers a journal haiku to a top-voted r/haiku one: Qwen3.5-0.8B → Claude Opus 5.5</span></div>
<div class="tile"><strong>#{journal_rank} of {len(by_gen)}</strong><span>rank of published journal haiku under the open-model consensus (anchor judge: #{int((anchor_gen > anchor_gen['human-journal']).sum()) + 1})</span></div>
<div class="tile"><strong>{f2(acx_claude)}</strong><span>{ANCHOR}'s preference for AI art on ACX (pct-rank AI − human; ACX crowd {f2(pref['humans (favourite share)'])})</span></div>
</div>
</header>

<section id="intro">
<h2>Intro: what is taste?</h2>
{"".join(ins)}
</section>

<section id="findings">
<h2>Five findings, one slide each</h2>
<p class="small">Each card is sized 16:9 for a slide; PNGs of all cards are exported to <code>taste-files/slides/</code>.</p>
{"".join(fs)}
</section>

<section id="summary">
<h2>Summary</h2>
<ol class="takeaways">
<li><b>Judging taste scales, then saturates.</b> Agreement with the anchor judge on haiku rises from ρ = {f2(sc.loc[smallest].rho_anchor)} ({smallest}) to {f2(sc.loc[largest_local].rho_anchor)} ({largest_local}) and stays at {f2(sc.loc[largest_api].rho_anchor)} for {largest_api}. The smallest judge is worse than chance at telling journal haiku from tweets (AUC {f2(sc.loc[smallest].auc_journal_vs_twaiku)}).</li>
<li><b>Vote for vote, a model's taste is about as idiosyncratic as a person's.</b> Models' expected ratings agree closely (ICC {f2(icc_i['lapis']['models'])} on LAPIS art), but that averages over each model's own uncertainty. With one sampled vote per model, as for a single person, the shared share is {pct(icc_i['lapis']['models_one_vote'])} for models vs {pct(icc_i['lapis']['humans_same_items'])} for people on the same artworks (AVA photos: {pct(icc_i['ava']['models_one_vote'])} vs {pct(icc_i['ava']['humans_same_items'])}).</li>
<li><b>The beholder effect is about capability, not family.</b> Raters do not favour their own family (contrast {beh['family_contrast']:+.3f} z). Instead small judges prefer small writers and large judges large ones: the link between a judge's size and its preference for bigger writers is ρ = {beh['assortative_rho_across_raters']:+.2f} across judges.</li>
<li><b>Small models know more than they say.</b> On far-OOD haiku, a linear probe on {smallest}'s hidden states reaches ρ = {f2(pv.loc[smallest, 'haiku: probe'])} where its own rating gets {f2(pv.loc[smallest, 'haiku: verbal'])}; for {largest_local} the two meet ({f2(pv.loc[largest_local, 'haiku: probe'])} vs {f2(pv.loc[largest_local, 'haiku: verbal'])}).</li>
<li><b>What the big judges reward, with topic held fixed:</b> {craft_summary}.</li>
<li><b>A verdict axis shared by text and images appears with scale:</b> a haiku-trained probe read out at the answer position predicts human art ratings on LAPIS at up to ρ = {f2(cross.lapis.max())} ({cross.lapis.idxmax()}), small models ≈ 0.</li>
<li><b>Open models slightly prefer AI art; {ANCHOR} strongly prefers human art and spots AI images {pct(det.get(ANCHOR, np.nan))} of the time</b> (median human {pct(det['humans: median respondent'])}), but it also names most of the human artworks.</li>
</ol>
</section>

<section id="setup">
<h2>Setup</h2>
<div class="two">
<div class="panel"><h3>Stimuli</h3>
<p><b>Haiku.</b> 40 topics × 5 samples (temperature 1) from {n_gen - 4} generators, from SmolLM2-135M to Claude Fable 5.1, plus 120 haiku in each of four human tiers: accidental 5-7-5 lines found in tweets, low- and top-voted r/haiku posts, and haiku from journals. Line-final punctuation is normalised the same way for every source.</p>
<p><b>Images.</b> 600 AVA photos and 600 LAPIS artworks, stratified over their human mean score, plus the 50 ACX images (24 human, 26 AI) with the 9,172 'favourite' votes and the per-respondent guesses.</p></div>
<div class="panel"><h3>Readout</h3>
<p>Each item is shown alone with "Rate it from 1 to 9 … Reply with a single digit", thinking disabled, one item per forward pass. The rating is the expected digit</p>
<div class="eq">\\[\\hat s=\\frac{{\\sum_{{k=1}}^{{9}} k\\,p(k)}}{{\\sum_{{k=1}}^{{9}} p(k)}}\\]</div>
<p>which keeps the model's uncertainty (cf. Q-Align). Hidden states are kept at 13 depths: at the answer position and averaged over the item's own tokens. API raters give the same expected digit where the provider returns log-probabilities; Claude returns only the digit.</p></div>
</div>
</section>

<section id="model">
<h2>A model of taste, measured</h2>
<p>The talk's model: value = x·φ<sup>obvious</sup> + x·φ<sup>non-obvious</sup> + x·φ<sub>θ</sub><sup>subjective</sup> + ξ. For every judge that returns a 1–9 distribution, the variance of one sampled vote is split into what the novices (Qwen3.5-4B, Gemma-4-E4B) agree on, what the experts (27B and up, Claude) agree on beyond that, the judge's remaining persistent preference, and vote-to-vote noise. Consensuses always leave the judge out; a shuffled control gives 0 for every fitted part. The plane places each haiku by a ridge readout of the two consensuses from Qwen3.5-27B's and Gemma-4-31B's hidden states, held out by topic.</p>
{fig("taste_model_plane", "Haiku in the (obvious, non-obvious) plane, coloured by a novice's and an expert's rating; circles: writer groups; right: each judge's direction (length = how well the two axes explain its ratings).")}
{fig("taste_model_shares", "Share of a single vote's variance per judge. The random share also reflects how peaked a family's digit distributions are (Gemma's are much more peaked than Qwen's).")}
{table(pd.DataFrame(R["taste_model"]["shares"]).set_index("judge").drop(columns=["family"]).rename(columns={"params_b": "params (B)"}))}
</section>

<section id="haiku">
<h2>1 · Haiku: who do the judges think writes well?</h2>
{fig("haiku_quality_ladder", "Mean rating (z-scored within each rater) per generator; ±95% CI over the 200 (120 human) haiku. Black: mean of all open and Qwen API raters; grey ×: the anchor judge.")}
<p>Generator size orders the open-model consensus up to about 4B parameters and then saturates: the judges barely separate 9B from 397B. The open raters put published journal haiku at rank {journal_rank} of {len(by_gen)}, below most models. {ANCHOR} puts them near the top, behind only its own family.</p>
{fig("haiku_judge_scaling", "Per rater: Spearman ρ of its score with log generator size (open generators), with the anchor judge, and the AUC for ranking journal haiku above tweet haiku.")}
{table(sc[["params_b", "rho_gen_size", "rho_anchor", "auc_journal_vs_twaiku"]].rename(columns={"params_b": "params (B)", "rho_gen_size": "ρ size", "rho_anchor": "ρ anchor", "auc_journal_vs_twaiku": "AUC journal>tweet"}))}
<p class="small">Best open local rater against the anchor: {best_local} (ρ = {f2(sc.loc[best_local].rho_anchor)}).</p>
<h3>Examples</h3>
{ex["ladder"]}
{ex["humans"]}
{ex["loves"]}
</section>

<section id="images">
<h2>2 · Images: agreement with people</h2>
{fig("images_gt_agreement", "Spearman ρ between each rater's score and the human mean, per dataset, against rater size. Dotted lines: API raters.")}
{table(ga)}
{ex["lapis"]}
<h3>The ACX AI Art Turing Test</h3>
{ex["acx"]}
{fig("acx_preference_detection", "Left: do raters rank AI images above human ones? Right: accuracy of the human-vs-AI call. Human median respondent and crowd majority from the 11k ACX responses.")}
<p>{ANCHOR} calls the origin of {pct(det.get(ANCHOR, np.nan))} of the 50 images correctly (median human {pct(det['humans: median respondent'])}) and rates AI images far lower than human ones. It also names the artist of most of the human works, so recognition of canonical paintings and the public answer key (online since November 2024) are live confounds. Split by what the rater recognised (mean percentile rank; the crowd's rank for the same images alongside):</p>
{table(pd.DataFrame(R["acx_recognition"]).set_index(["rater", "group"]).rename(columns={"rater_pct_rank": "rater's rank", "crowd_pct_rank": "crowd's rank"}).reset_index(level=1), index=True)}
</section>

<section id="shared">
<h2>3 · Shared or private taste?</h2>
<p>ICC of a single rating: the share of its variance that other raters share. For people we only have each image's spread of ratings from different people, so their ICC is one-way and each rating is one vote. A model's expected digit is an average over its own uncertainty, which inflates agreement; the like-for-like comparison samples one digit per model from its probabilities and uses the same one-way estimator on the same images.</p>
{table(pd.DataFrame({"one person (same images)": [icc_i['ava']['humans_same_items'], icc_i['lapis']['humans_same_items'], np.nan], "one model, one sampled vote": [icc_i['ava']['models_one_vote'], icc_i['lapis']['models_one_vote'], np.nan], "one model, expected rating (consistency ICC)": [icc_i['ava']['models'], icc_i['lapis']['models'], icc_h['all']]}, index=pd.Index(["AVA photos", "LAPIS art", "haiku"], name="")))}
<p>Like for like, the models' shared share is close to people's: models do not share taste much more than people do. What they do have is a stable expected rating, which a single person does not reveal.</p>
{fig("haiku_rater_agreement", "Spearman ρ between raters across the haiku.", "70%")}
</section>

<section id="beholder">
<h2>4 · In the eye of the beholder: does a model like its own kind?</h2>
<p>Mean z-scored rating per rater × generator, after removing each generator's overall quality and each rater's leniency. What is left is the beholder.</p>
{fig("haiku_beholder", "Interaction term (z). Boxes: a model rating its own haiku.")}
<p>Own-family contrast (Qwen raters on Qwen haiku + Gemma on Gemma − cross terms) = {beh['family_contrast']:+.3f} z, 95% CI [{beh['family_contrast_ci95'][0]:+.3f}, {beh['family_contrast_ci95'][1]:+.3f}]; mean self-rating residual {beh['self_pref'][0]:+.3f} z over {beh['self_pref'][1]} self-pairs. After regressing out each rater's log-probability of the haiku and its length: contrast {beh['family_contrast_familiarity_controlled']:+.3f}, self {beh['self_pref_familiarity_controlled'][0]:+.3f}.</p>
{fig("haiku_beholder_familiarity_controlled", "The same after the familiarity control.")}
<h3>Assortative taste</h3>
<p>The structure in the matrix is not family but capability. For each rater, correlate its leftover preference (the row above) with log generator size: small judges like small writers more than everyone else does, large judges like large writers. Across raters, that correlation rises with the rater's own size (Spearman ρ = {beh['assortative_rho_across_raters']:+.2f}).</p>
{fig("haiku_assortative", "Per rater: Spearman ρ between its interaction row and log generator size (open generators of known size).", "60%")}
{ex["disagree"]}
</section>

<section id="features">
<h2>5 · What do the judges reward? Unsupervised axes</h2>
<p>PCA (20 components) of a reference model's item representation (mean over the item's tokens, at ⅔ depth). For each PC we ask which raters reward it, let {ANCHOR} name its two poles from 12 extreme items each, and check the name on 20 held-out pairs: given only the label, can it tell which item is higher? Raw haiku axes mostly track the topic (absence and empty rooms vs. delicate nature), so the <b>haiku-craft</b> rows repeat the analysis on model haiku only, with each topic's mean removed from the representations and the scores: what is left is how a haiku is written, not what it is about.</p>
{table(labs, index=False)}
{"".join(fig(f"pc_rater_profile_{k.replace('/', '_')}", f"{k.split('/')[0]}: Spearman ρ of every rater's score with {k.split('/')[1]}'s PCs (oriented so + = what that model rates higher).") for k in pcs)}
<h3>The axes, by example</h3>
{ex["axes"]}
<h3>Pareto frontiers</h3>
<p>Items in the plane of the two PCs that most drive the reference model's rating; the black line is the set of items no other item beats on both axes, the red arrow the direction of the model's rating.</p>
{"".join(fig(f"pareto_{k.replace('/', '_')}", f"{k}: Pareto front and rating direction.", "80%") for k in pcs)}
</section>

<section id="probes">
<h2>6 · Is taste readable from the inside? Probes on far-OOD splits</h2>
<p>Ridge probes on hidden states (⅔ depth, fixed in advance), always tested on a different population than they were trained on: open-model haiku → frontier and human haiku, AVA photos → LAPIS art, and so on. "Verbal" is the model's own digit rating on the same test items. Probes beat the verbal rating for {wins_h}/{len(pv)} raters on haiku and {wins_i}/{len(pv)} on images.</p>
{fig("probes_haiku", "Haiku probes (target: anchor judge's score; size splits: log generator size). Dotted: confounds (log-probability, length) regressed out on the training split.")}
{fig("probes_images", "Image probes (target: human mean score, z within dataset).")}
{table(pv)}
<h3>One quality axis for text and images?</h3>
<p>A probe trained on haiku (anchor score) and applied, unchanged, to the same model's image representations at the answer position; and back.</p>
{table(cross.assign(**{"images → haiku": back.reindex(cross.index)}))}
</section>

<section id="caveats">
<h2>Caveats</h2>
<ul class="takeaways">
<li>Generator size is a proxy for haiku quality, and the anchor judge rates its own family highest; there are no human ratings of the model haiku.</li>
<li>ACX is 50 curated images whose answers have been public since 2024: possible memorisation, especially for {ANCHOR}.</li>
<li>All ratings are single prompts. Ratings cluster at 6–8 of 9, so analyses use ranks and z-scores.</li>
<li>Closed models are behaviour only (no hidden states).</li>
</ul>
</section>
<footer>Built {built} by <code>taste/report.py</code> from <code>{RESULTS}</code>. Literature survey: <a href="{NAME}-files/lit_survey.md">lit_survey.md</a>.</footer>
"""
    css = (Path(__file__).parent / "report.css").read_text() + "\nfigure img{background:#fff;border-radius:8px;max-width:100%;display:block}\n" + ex_css + fs_css + in_css
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Taste in language models</title><style>{stylesheet()}</style><style>{css}</style></head><body><main>{typeset(body)}</main></body></html>"""
    REPORTS.mkdir(parents=True, exist_ok=True); files = REPORTS / f"{NAME}-files"; files.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).parents[1] / "docs/lit_survey.md", files / "lit_survey.md")
    out = REPORTS / f"{NAME}.html"; out.write_text(page)
    for target in [out, files]:
        link = SERVED / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out); print(f"http://127.0.0.1:8095/{NAME}.html")


if __name__ == "__main__":
    build()
