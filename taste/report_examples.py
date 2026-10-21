"""Example panels for the taste report: self-contained cards (title, one-line takeaway, haiku / thumbnails with scores) sized to
be screenshotted straight into slides. Scores are percentiles within each judge (100 = that judge's favourite item).
"""
import base64, html, io, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from taste.common import ratings, items, fam, FAMILY_COLOR, ANCHOR, ROOT
from taste.models import BY_NAME
from taste.paths import DATA

SMALL = ["qwen3.5-0.8b", "qwen3.5-2b", "gemma-4-e2b"]
LARGE = ["qwen3.5-27b", "gemma-4-31b", "qwen3.5-397b-a17b"]
LADDER = ["smollm2-135m", "smollm2-1.7b", "qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-27b", "qwen3.5-397b-a17b", "gemma-4-e2b", "gemma-4-31b",
          "gemini-3.8-flash", "gpt-5.6-sol", "claude-opus-5.5", "claude-fable-5.1"]
HUMAN_TIERS = {"human-twaiku": "accidental 5-7-5 in a tweet", "human-reddit-low": "r/haiku, ≤1 upvote", "human-reddit-top": "r/haiku, top 5% upvotes", "human-journal": "published in a haiku journal"}
CSS = """
.slide{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px 16px;margin:20px 0}
.slide h3{margin:0 0 2px;font-size:1.15rem}.slide .sub{color:var(--ink2);font-size:.9rem;margin:0 0 12px}
.slide .foot{color:var(--muted);font-size:.78rem;margin:10px 0 0}
.hk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:10px}
.hk{border:1px solid var(--border);border-radius:10px;padding:10px 12px;background:var(--bg);display:flex;flex-direction:column;gap:7px}
.poem{font-family:Georgia,"Times New Roman",serif;font-style:italic;white-space:pre-line;line-height:1.45;font-size:1rem;flex:1}
.who{font-size:.8rem;font-weight:650;display:flex;align-items:center;gap:6px;color:var(--ink)}
.who i{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:.74rem;padding:1px 7px;border-radius:999px;background:var(--grid);color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap}
.chip.hi{background:#d6e6fa}.chip.lo{background:#f8ddd9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .chip.hi{background:#1d3553}:root:not([data-theme="light"]) .chip.lo{background:#4a2420}}
:root[data-theme="dark"] .chip.hi{background:#1d3553}:root[data-theme="dark"] .chip.lo{background:#4a2420}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.cols h4{margin:0 0 8px;font-size:.92rem;color:var(--ink2)}
.cols .hk{margin-bottom:8px}
.img-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
@media (max-width:720px){.img-grid{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}}
.im{border:1px solid var(--border);border-radius:10px;padding:6px;background:var(--bg);display:flex;flex-direction:column;gap:5px}
.im img{width:100%;height:130px;object-fit:cover;border-radius:6px;display:block;background:#fff}
.im .t{font-size:.76rem;line-height:1.25;color:var(--ink)}
.badge{font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px;color:#fff;align-self:flex-start}
.badge.ai{background:#6250d6}.badge.human{background:#1b8f65}
.row-label{font-size:.85rem;font-weight:650;margin:12px 0 6px;color:var(--ink)}
.axis{display:flex;align-items:center;gap:10px;font-size:.9rem;margin:0 0 10px;flex-wrap:wrap}
.axis b{font-weight:650}.axis .arrow{color:var(--muted)}
"""


def pct(series):
    return series.rank(pct=True) * 100


def chip(label, v):
    cls = "hi" if v >= 66 else "lo" if v <= 33 else ""
    return f'<span class="chip {cls}">{html.escape(label)} {v:.0f}</span>'


def who(name):
    color = FAMILY_COLOR[fam(name)] if name in BY_NAME or name.startswith("human") else "#8a8984"
    label = HUMAN_TIERS.get(name, name)
    return f'<div class="who"><i style="background:{color}"></i>{html.escape(label)}</div>'


def haiku_card(text, name, chips):
    return f'<div class="hk">{who(name)}<div class="poem">{html.escape(text)}</div><div class="chips">{"".join(chips)}</div></div>'


def thumb(path, side=320):
    im = Image.open(DATA / path).convert("RGB"); im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def img_card(row, chips, title):
    badge = f'<span class="badge {row.source}">{"AI" if row.source == "ai" else "Human"}</span>' if row.dataset == "acx" else ""
    t = title
    return f'<div class="im"><img src="{thumb(row.path)}" alt="">{badge}<div class="t">{html.escape(str(t))}</div><div class="chips">{"".join(chips)}</div></div>'


def slide(title, sub, body, foot="Numbers are percentiles within each judge: 100 = the item that judge rated highest."):
    return f'<div class="slide"><h3>{html.escape(title)}</h3><p class="sub">{sub}</p>{body}<p class="foot">{foot}</p></div>'


def build():
    from taste.paths import RAW
    style = pd.read_csv(RAW / "asmaerten__LAPIS/giaa/test.csv").set_index("path")["style"].str.replace("_", " ")
    tile_title = lambda row: row.title if row.dataset == "acx" else style[row.title] if row.dataset == "lapis" else "AVA photo"
    hi = items("haiku"); S = ratings("haiku").reindex(hi.index)
    P = pct(S)
    small, large, claude = P[SMALL].mean(1), P[[r for r in LARGE if r in P]].mean(1), P[ANCHOR]
    hchips = lambda i: [chip("small judges", small[i]), chip("large judges", large[i]), chip("Claude", claude[i])]
    out = {}

    # 1. one topic, many writers: the topic whose large-judge scores best follow generator size
    known = hi.generator.isin(LADDER) & hi.topic.notna()
    size = hi.generator.map(lambda g: BY_NAME[g].params_b if g in BY_NAME else np.nan)
    rho_t = {t: stats.spearmanr(size[(hi.topic == t) & size.notna()], large[(hi.topic == t) & size.notna()]).statistic for t in hi.topic.dropna().unique()}
    topic = max(rho_t, key=rho_t.get)
    cards = []
    for g in LADDER:
        d = hi[known & (hi.topic == topic) & (hi.generator == g)]
        i = (large[d.index] - large[d.index].median()).abs().idxmin()  # the generator's typical (median) haiku on this topic
        cards.append(haiku_card(hi.loc[i, "haiku"], g, hchips(i)))
    out["ladder"] = slide(f"One prompt, twelve writers: “a haiku about {topic}”",
                          "Each model's most typical of its 5 samples (median score). Small models ramble or break the form; the frontier writes compressed, concrete images.",
                          f'<div class="hk-grid">{"".join(cards)}</div>')

    # 2. human tiers
    cards = []
    for tier in HUMAN_TIERS:
        d = hi[hi.generator == tier]
        for i in d.index[np.argsort(claude[d.index].values)][[len(d) // 2, int(len(d) * 0.9)]]:  # median and 90th-pct haiku by Claude
            cards.append(haiku_card(hi.loc[i, "haiku"], tier, hchips(i)))
    out["humans"] = slide("Human haiku: Claude rewards the journal poems, small open judges don't",
                          "Two per tier (Claude's median and 90th-percentile pick). Journal haiku break 5-7-5 and are oblique; small judges rank them like tweets.",
                          f'<div class="hk-grid">{"".join(cards)}</div>')

    # 3. loves / hates of three judges
    cols = []
    for r, lab in [("qwen3.5-0.8b", "Qwen3.5-0.8B"), ("qwen3.5-27b", "Qwen3.5-27B"), (ANCHOR, "Claude Opus 5.5")]:
        s = S[r].dropna(); s = s[~hi.loc[s.index, "haiku"].duplicated()]
        top, bot = s.nlargest(3).index, s[hi.loc[s.index, "family"] != "human"].nsmallest(3).index
        col = f"<h4>{lab} loves</h4>" + "".join(haiku_card(hi.loc[i, "haiku"], hi.loc[i, "generator"], [f'<span class="chip">score {S.loc[i, r]:.1f}/9</span>']) for i in top)
        col += f"<h4>{lab} hates</h4>" + "".join(haiku_card(hi.loc[i, "haiku"], hi.loc[i, "generator"], [f'<span class="chip">score {S.loc[i, r]:.1f}/9</span>']) for i in bot)
        cols.append(f"<div>{col}</div>")
    out["loves"] = slide("What each judge loves and hates", "The 3 highest-rated haiku (of 4,480) and the 3 lowest-rated model-written ones, for a small open judge, a large open judge and Claude.",
                         f'<div class="cols">{"".join(cols)}</div>', foot="Score = expected digit on the 1–9 scale, read from the judge's next-token probabilities.")

    # 4. disagreements between small and large judges
    gap = (small - large).dropna(); gap = gap[~hi.loc[gap.index, "haiku"].duplicated()]
    out["disagree"] = slide("Where small and large judges disagree most",
                            "Small judges (Qwen3.5-0.8B/2B, Gemma-4-E2B) vs large ones (Qwen3.5-27B/397B, Gemma-4-31B): weak judges like weak, wordy writing.",
                            '<div class="cols"><div><h4>small judges ≫ large judges</h4>' + "".join(haiku_card(hi.loc[i, "haiku"], hi.loc[i, "generator"], hchips(i)) for i in gap.nlargest(4).index)
                            + '</div><div><h4>large judges ≫ small judges</h4>' + "".join(haiku_card(hi.loc[i, "haiku"], hi.loc[i, "generator"], hchips(i)) for i in gap.nsmallest(4).index) + "</div></div>")

    # 5. labelled axes with their extreme items
    res = json.loads((ROOT / "results.json").read_text())["pcs"]
    ii = items("images"); Si = ratings("images").reindex(ii.index)
    axes = []
    for key in [k for k in res if k.startswith("haiku-craft/")] + [k for k in res if k.startswith("images/")] + [k for k in res if k.startswith("haiku/")]:
        variant, ref = key.split("/")
        sc = pd.read_parquet(ROOT / "pc_scores" / f"{variant}__{ref}.parquet")
        for k in res[key]["top"][:2 if not key.startswith("haiku/") else 1]:
            lab = res[key]["labels"][k]; rho = res[key]["rater_pc_rho"][k][ref]
            v = sc[k].dropna()
            head = (f'<div class="axis"><b>{html.escape(lab["low"])}</b><span class="arrow">⟶</span><b>{html.escape(lab["high"])}</b>'
                    f'<span class="chip">ρ with {ref}\'s rating {rho:+.2f}</span><span class="chip">label check {lab["label_accuracy"]:.0%}</span></div>')
            if variant.startswith("haiku"):
                v = v[~hi.loc[v.index, "haiku"].duplicated()]
                spread = lambda ser: ser[~(hi.loc[ser.index, "generator"] + "|" + hi.loc[ser.index, "topic"].fillna(ser.index.to_series())).duplicated()]  # one per model x prompt
                lo_idx, hi_idx = spread(v.sort_values()).index[:3], spread(v.sort_values(ascending=False)).index[:3]
                pole = lambda idx: "".join(haiku_card(hi.loc[i, "haiku"], hi.loc[i, "generator"], [chip(f"{ref}", P.loc[i, ref])]) for i in idx)
                body = head + f'<div class="cols"><div><h4>low end</h4>{pole(lo_idx)}</div><div><h4>high end</h4>{pole(hi_idx)}</div></div>'
            else:
                Pi = Si.groupby(ii.dataset).rank(pct=True) * 100
                pole = lambda idx: "".join((lambda row: img_card(row, [chip(ref, Pi.loc[i, ref])], tile_title(row)))(next(ii.loc[[i]].itertuples())) for i in idx)
                body = head + f'<div class="row-label">low end</div><div class="img-grid">{pole(v.nsmallest(5).index)}</div><div class="row-label">high end</div><div class="img-grid">{pole(v.nlargest(5).index)}</div>'
            what = "with topic held fixed" if variant == "haiku-craft" else "images" if variant == "images" else "raw haiku (topic included)"
            axes.append(slide(f"A taste axis inside {ref} ({what})", html.escape(lab["description"]), body,
                              foot=f"PC {k[2:]} of {ref}'s internal representation (⅔ depth). Named by {ANCHOR} from 12 items per end; the label check is its accuracy at picking the higher item in 20 held-out pairs."))
    out["axes"] = "".join(axes)

    # 6. ACX galleries
    acx = ii[ii.dataset == "acx"]; Pa = Si.loc[acx.index].rank(pct=True) * 100; hum = acx.human_score.rank(pct=True) * 100
    achips = lambda i: [chip("people", hum[i]), chip("Claude", Pa.loc[i, ANCHOR]), chip("Qwen-397B", Pa.loc[i, "qwen3.5-397b-a17b"])]
    nai = lambda idx: f"{int((acx.loc[idx, 'source'] == 'ai').sum())} of {len(idx)} are AI"
    row_html = lambda t, idx: f'<div class="row-label">{html.escape(t)} — {nai(idx)}</div><div class="img-grid">{"".join(img_card(next(acx.loc[[i]].itertuples()), achips(i), acx.loc[i, "title"]) for i in idx)}</div>'
    foot = "Percentiles within the 50 ACX images. Badge = ground truth from Scott Alexander's answer key. Claude rates in whole digits, so it has many ties."
    hp, cp, qp, cb = hum.nlargest(5).index, Pa[ANCHOR].nlargest(5).index, Pa["qwen3.5-397b-a17b"].nlargest(5).index, Pa[ANCHOR].nsmallest(5).index
    out["acx"] = (slide("ACX AI Art Turing Test: people vs Claude", f"People's top 5: {nai(hp)}. Claude Opus 5.5's top 5: {nai(cp)}.",
                        row_html("People's favourites (share of 9,172 ‘favourite’ votes)", hp) + row_html("Claude Opus 5.5's favourites", cp), foot=foot)
                  + slide("ACX AI Art Turing Test: an open model vs Claude's dislikes", f"Qwen3.5-397B's top 5: {nai(qp)}. Claude's bottom 5: {nai(cb)}.",
                          row_html("Qwen3.5-397B's favourites", qp) + row_html("Claude Opus 5.5's least favourite", cb), foot=foot))

    # 7. LAPIS: where the best open judge and people disagree
    lap = ii[ii.dataset == "lapis"]; ref = "gemma-4-31b"
    pl = Si.loc[lap.index, ref].rank(pct=True) * 100; hl = lap.human_score.rank(pct=True) * 100; d = (pl - hl)
    lchips = lambda i: [chip("people", hl[i]), chip("Gemma-4-31B", pl[i])]
    body = (f'<div class="row-label">Gemma-4-31B ≫ people</div><div class="img-grid">{"".join(img_card(next(lap.loc[[i]].itertuples()), lchips(i), style[lap.loc[i, "title"]]) for i in d.nlargest(5).index)}</div>'
            f'<div class="row-label">people ≫ Gemma-4-31B</div><div class="img-grid">{"".join(img_card(next(lap.loc[[i]].itertuples()), lchips(i), style[lap.loc[i, "title"]]) for i in d.nsmallest(5).index)}</div>')
    out["lapis"] = slide("Art: where the best open judge and people part ways", "LAPIS artworks (mean of ~20 raters each) with the largest gap between Gemma-4-31B's rank and the crowd's.", body)
    return out, CSS
