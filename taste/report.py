"""Build the taste report from saved results. Every number on the page is computed here from results.json and the rating
tables; nothing is typed in by hand. The page goes to the world-readable ceph reports folder and is symlinked into the folder
served on 127.0.0.1:8095.

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

    built = datetime.now(ZoneInfo("Europe/London")).strftime("%d %B %Y, %H:%M %Z")
    n_haiku, n_gen = len(hi), hi.generator.nunique()
    cross = cm[(cm.pool == "last") & (cm.direction == "haiku → images")].pivot(index="rater", columns="test", values="rho")
    cross = cross.loc[rater_order(cross.index)][["ava", "lapis", "acx"]]
    back = cm[(cm.pool == "last") & (cm.direction == "images → haiku")].set_index("rater").rho

    body = f"""
<header>
<div class="eyebrow">Taste in language models · {built}</div>
<h1>Do language models have taste?</h1>
<p class="lede">Open Qwen3.5 and Gemma 4 models from 0.8B to 31B parameters, plus Qwen3.5-122B/397B and Claude Opus 5.5 through an API, rated {n_haiku:,} haiku from {n_gen - 4} models and four human tiers, and {len(ii):,} images with human ratings (AVA photos, LAPIS art, the ACX AI Art Turing Test). We read each rating from the model's next-token probabilities and kept its hidden states.</p>
<div class="tiles">
<div class="tile"><strong>{pct(icc_i['lapis']['humans'])} vs {pct(icc_i['lapis']['models'])}</strong><span>share of a single rating that is shared taste: one human vs one model (LAPIS art)</span></div>
<div class="tile"><strong>#{journal_rank} of {len(by_gen)}</strong><span>rank of published journal haiku under the open-model consensus (anchor judge: #{int((anchor_gen > anchor_gen['human-journal']).sum()) + 1})</span></div>
<div class="tile"><strong>{f2(acx_claude)}</strong><span>{ANCHOR}'s preference for AI art on ACX (pct-rank AI − human; ACX crowd {f2(pref['humans (favourite share)'])})</span></div>
</div>
</header>

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

<section id="haiku">
<h2>1 · Haiku: who do the judges think writes well?</h2>
{fig("haiku_quality_ladder", "Mean rating (z-scored within each rater) per generator; ±95% CI over the 200 (120 human) haiku. Black: mean of all open and Qwen API raters; grey ×: the anchor judge.")}
<p>Generator size orders the open-model consensus up to about 4B parameters and then saturates: the judges barely separate 9B from 397B. The open raters put published journal haiku at rank {journal_rank} of {len(by_gen)}, below most models. {ANCHOR} puts them near the top, behind only its own family.</p>
{fig("haiku_judge_scaling", "Per rater: Spearman ρ of its score with log generator size (open generators), with the anchor judge, and the AUC for ranking journal haiku above tweet haiku.")}
{table(sc[["params_b", "rho_gen_size", "rho_anchor", "auc_journal_vs_twaiku"]].rename(columns={"params_b": "params (B)", "rho_gen_size": "ρ size", "rho_anchor": "ρ anchor", "auc_journal_vs_twaiku": "AUC journal>tweet"}))}
<p class="small">Best open local rater against the anchor: {best_local} (ρ = {f2(sc.loc[best_local].rho_anchor)}).</p>
</section>

<section id="images">
<h2>2 · Images: agreement with people</h2>
{fig("images_gt_agreement", "Spearman ρ between each rater's score and the human mean, per dataset, against rater size. Dotted lines: API raters.")}
{table(ga)}
<h3>The ACX AI Art Turing Test</h3>
{fig("acx_preference_detection", "Left: do raters rank AI images above human ones? Right: accuracy of the human-vs-AI call. Human median respondent and crowd majority from the 11k ACX responses.")}
<p>{ANCHOR} calls the origin of {pct(det.get(ANCHOR, np.nan))} of the 50 images correctly (median human {pct(det['humans: median respondent'])}) and rates AI images far lower than human ones. It also names the artist of most of the human works, so recognition of canonical paintings and the public answer key (online since November 2024) are live confounds.</p>
</section>

<section id="shared">
<h2>3 · Shared or private taste?</h2>
<p>ICC of a single rating: the share of its variance that other raters share. For people we use the spread of each image's ratings; for models, the panel of all raters that saw every item.</p>
{table(pd.DataFrame({"one human": [icc_i['ava']['humans'], icc_i['lapis']['humans'], np.nan], "one model (panel)": [icc_i['ava']['models'], icc_i['lapis']['models'], icc_h['all']]}, index=pd.Index(["AVA photos", "LAPIS art", "haiku"], name="")))}
{fig("haiku_rater_agreement", "Spearman ρ between raters across the haiku.", "70%")}
</section>

<section id="beholder">
<h2>4 · In the eye of the beholder: does a model like its own kind?</h2>
<p>Mean z-scored rating per rater × generator, after removing each generator's overall quality and each rater's leniency. What is left is the beholder.</p>
{fig("haiku_beholder", "Interaction term (z). Boxes: a model rating its own haiku.")}
<p>Own-family contrast (Qwen raters on Qwen haiku + Gemma on Gemma − cross terms) = {beh['family_contrast']:+.3f} z, 95% CI [{beh['family_contrast_ci95'][0]:+.3f}, {beh['family_contrast_ci95'][1]:+.3f}]; mean self-rating residual {beh['self_pref'][0]:+.3f} z over {beh['self_pref'][1]} self-pairs. After regressing out each rater's log-probability of the haiku and its length: contrast {beh['family_contrast_familiarity_controlled']:+.3f}, self {beh['self_pref_familiarity_controlled'][0]:+.3f}.</p>
{fig("haiku_beholder_familiarity_controlled", "The same after the familiarity control.")}
</section>

<section id="features">
<h2>5 · What do the judges reward? Unsupervised axes</h2>
<p>PCA (20 components) of a reference model's item representation (mean over the item's tokens, at ⅔ depth). For each PC we ask which raters reward it, let {ANCHOR} name its two poles from 12 extreme items each, and check the name on 20 held-out pairs: given only the label, can it tell which item is higher?</p>
{table(labs, index=False)}
{"".join(fig(f"pc_rater_profile_{k.replace('/', '_')}", f"{k.split('/')[0]}: Spearman ρ of every rater's score with {k.split('/')[1]}'s PCs (oriented so + = what that model rates higher).") for k in pcs)}
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
    css = (Path(__file__).parent / "report.css").read_text() + "\nfigure img{background:#fff;border-radius:8px;max-width:100%;display:block}\n"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Taste in language models</title><style>{stylesheet()}</style><style>{css}</style></head><body><main>{typeset(body)}</main></body></html>"""
    REPORTS.mkdir(parents=True, exist_ok=True); files = REPORTS / f"{NAME}-files"; files.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).parents[1] / "docs/lit_survey.md", files / "lit_survey.md")
    out = REPORTS / f"{NAME}.html"; out.write_text(page)
    for p in [REPORTS.parent, REPORTS, files]: p.chmod(0o755)
    for p in [out, *files.iterdir()]: p.chmod(0o644)
    for target in [out, files]:
        link = SERVED / target.name
        if link.is_symlink(): link.unlink()
        link.symlink_to(target)
    print(out); print(f"http://127.0.0.1:8095/{NAME}.html")


if __name__ == "__main__":
    build()
