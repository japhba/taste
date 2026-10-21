"""Unsupervised taste axes: PCA of a reference model's item representations; which PCs carry each rater's verdict; LLM auto-labels
validated on held-out forced choices; correlation with hand-crafted features; Pareto frontier in the two most rating-relevant PCs.

  python -m taste.pcs [--no-label]
"""
import argparse, base64, io, json, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from taste.common import ratings, items, fam, rater_order, hidden, save, record, FAMILY_COLOR, FAMILY_MARKER, DIVERGING, ROOT
from taste.paths import DATA
from taste.probes import LAYER_FRAC
from taste.gen_haiku import openrouter

N_PC, N_LABEL, N_EXEMPLAR, N_PAIRS = 20, 4, 12, 20
LABELER = "anthropic/claude-opus-5.5"
RNG = np.random.default_rng(0)
DATASET_COLOR = {"acx-ai": "#6250d6", "acx-human": "#1baf7a", "lapis": "#eb6834", "ava": "#2a78d6"}


def basis(which, rater, keep=None, demean_by=None):
    """PCA of the item representation; `keep` restricts the items, `demean_by` removes group (e.g. topic) means first."""
    ids, layers, H = hidden(which, rater, "mean")
    li = int(np.argmin(np.abs(layers - LAYER_FRAC * layers.max())))
    X = H[:, li].astype(np.float32)
    if keep is not None: m = keep.reindex(ids).to_numpy(bool); ids, X = ids[m], X[m]
    if demean_by is not None: X = X - pd.DataFrame(X).groupby(demean_by.reindex(ids).to_numpy()).transform("mean").to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pca = PCA(N_PC, random_state=0).fit(X)
    return pd.DataFrame(pca.transform(X), index=ids, columns=[f"PC{k + 1}" for k in range(N_PC)]), pca.explained_variance_ratio_


def group(which, it):
    if which == "haiku": return it.generator.map(fam)
    return it.dataset.where(it.dataset != "acx", "acx-" + it.source)


# ---------- auto-labelling
def thumb(path, side):
    im = Image.open(DATA / path).convert("RGB"); im.thumbnail((side, side)); return im


def sheet(paths, side=192, cols=4):
    rows = -(-len(paths) // cols); S = Image.new("RGB", (cols * side, rows * side), "white")
    for i, p in enumerate(paths): S.paste(thumb(p, side), ((i % cols) * side, (i // cols) * side))
    return S


def img_part(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}


def ask(client, content):
    r = client.chat.completions.create(model=LABELER, messages=[{"role": "user", "content": content}], temperature=0.0, max_tokens=600,
                                       extra_body={"reasoning": {"effort": "low"}, "usage": {"include": True}})
    return r.choices[0].message.content.strip(), r.usage.model_extra.get("cost", 0.0)


LABEL_CACHE = DATA / "pc_labels.json"


def label_pc(client, which, it, pc, cache_key):
    """pc: Series (oriented). Exemplars from one random half of the items, validation pairs from the other half.
    The name is cached in LABEL_CACHE (keyed by set/reference/PC) so re-validation does not pay for re-labelling."""
    cache = json.loads(LABEL_CACHE.read_text()) if LABEL_CACHE.exists() else {}
    half = pd.Series(RNG.random(len(pc)) < 0.5, index=pc.index)
    ex, va = pc[half], pc[~half]
    hi, lo = ex.nlargest(N_EXEMPLAR).index, ex.nsmallest(N_EXEMPLAR).index
    kind = "haiku" if which == "haiku" else "images"
    intro = (f"Below are two groups of {kind} from opposite ends of one direction in a neural network's internal representation. "
             "Describe what distinguishes HIGH from LOW. Reply with JSON only: {\"low\": \"<=6 words\", \"high\": \"<=6 words\", \"description\": \"one sentence\"}.")
    if which == "haiku":
        fmt = lambda idx: "\n\n".join(it.loc[i, "haiku"] for i in idx)
        content = f"{intro}\n\nHIGH:\n{fmt(hi)}\n\nLOW:\n{fmt(lo)}"
    else:
        content = [{"type": "text", "text": intro + "\nFirst image: HIGH group. Second image: LOW group."},
                   img_part(sheet(it.loc[hi, "path"])), img_part(sheet(it.loc[lo, "path"]))]
    if "label_accuracy" in cache.get(cache_key, {}): return cache[cache_key] | {"cost": 0.0}  # already named and validated
    if cache_key in cache: lab, cost = cache[cache_key], 0.0
    else:
        txt, cost = ask(client, content)
        lab = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        cache[cache_key] = lab; LABEL_CACHE.write_text(json.dumps(cache, indent=1))
    q = va.quantile([0.2, 0.8]); top, bot = va[va >= q[0.8]].index, va[va <= q[0.2]].index
    pairs = [(RNG.choice(top), RNG.choice(bot), RNG.random() < 0.5) for _ in range(N_PAIRS)]

    def judge(p):
        h, l, h_first = p; a, b = (h, l) if h_first else (l, h)
        lead = f"An axis runs from LOW = \"{lab['low']}\" to HIGH = \"{lab['high']}\" ({lab['description']}). Which of these two {kind} is closer to the HIGH end? Answer with one letter: A or B."
        if which == "haiku": c = f"{lead}\n\nA:\n{it.loc[a, 'haiku']}\n\nB:\n{it.loc[b, 'haiku']}"
        else: c = [{"type": "text", "text": lead + " First image = A, second image = B."}, img_part(thumb(it.loc[a, "path"], 384)), img_part(thumb(it.loc[b, "path"], 384))]
        ans, cst = ask(client, c)
        m = re.search(r"\b([AB])\b", ans)  # answers come as 'A', '**B**', 'Image A', ...
        return (m is not None and m.group(1) == ("A" if h_first else "B")), cst
    with ThreadPoolExecutor(8) as ex_: res = list(ex_.map(judge, pairs))
    lab = dict(**lab, label_accuracy=float(np.mean([r[0] for r in res])))
    cache[cache_key] = lab; LABEL_CACHE.write_text(json.dumps(cache, indent=1))
    return lab | {"cost": cost + sum(r[1] for r in res)}


# ---------- plots
def pareto_mask(P):
    """non-dominated points when maximising both columns."""
    order = np.argsort(-P[:, 0]); best = -np.inf; m = np.zeros(len(P), bool)
    for i in order:
        if P[i, 1] > best: m[i] = True; best = P[i, 1]
    return m


def plot_pareto(which, ref, B, top, labels, score, grp):
    a, b = top[:2]
    P = B[[a, b]].to_numpy(); m = pareto_mask(P)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = FAMILY_COLOR if which.startswith("haiku") else DATASET_COLOR
    for g in grp.unique():
        s = (grp == g).to_numpy()
        ax.scatter(P[s, 0], P[s, 1], s=10, alpha=0.35, color=colors[g], marker=FAMILY_MARKER.get(g, "o"), label=f"{g} ({int((s & m).sum())} on front)", lw=0)
    f = np.argsort(P[m, 0]); ax.plot(P[m][f, 0], P[m][f, 1], color="#0b0b0b", lw=1.5, marker="o", ms=4, label="Pareto front")
    coef = np.linalg.lstsq(np.c_[np.ones(len(P)), P], score.to_numpy(), rcond=None)[0][1:]
    c = P.mean(0); d = coef / np.linalg.norm(coef) * 0.25 * np.ptp(P, 0).mean()
    ax.annotate("", c + d, c, arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=2.5)); ax.text(*(c + d * 1.1), f"{ref}'s rating ↑", color="#e34948")
    lab = lambda k: f"{k}: {labels[k]['low']} → {labels[k]['high']}" if k in labels else k
    ax.set_xlabel(lab(a)); ax.set_ylabel(lab(b)); ax.legend(loc="lower left"); ax.set_title(f"{which}: {ref}'s two most rating-relevant PCs")
    save(fig, f"pareto_{which}_{ref}")


def main(do_label):
    client = openrouter() if do_label else None
    out = {}
    # 'haiku-craft': model haiku only, topic means removed from representations and scores (what is left is how, not what)
    for variant in ["haiku", "images", "haiku-craft"]:
        which = variant.split("-")[0]
        it = items(which); S = ratings(which).reindex(it.index); grp = group(which, it)
        keep = demean = None
        if variant == "haiku-craft":
            keep = it.topic.notna(); it, S, grp = it[keep], S[keep], grp[keep]; demean = it.topic
            S = S - S.groupby(it.topic).transform("mean")
        feats = pd.read_parquet(DATA / f"{which}_features.parquet").set_index("item_id").reindex(it.index)
        local = rater_order([p.stem for p in (ROOT / "hidden" / which).glob("*.npz")])
        refs = [max((r for r in local if fam(r) == f), key=lambda r: local.index(r)) for f in ["qwen", "gemma"]]
        for ref in refs:
            B, evr = basis(which, ref, keep, demean); B = B.reindex(it.index)
            R = pd.DataFrame({r: [stats.spearmanr(B[k], S[r], nan_policy="omit").statistic for k in B] for r in S.columns}, index=B.columns).T
            top = R.loc[ref].abs().sort_values(ascending=False).index[:N_LABEL].tolist()
            for k in B: B[k] *= np.sign(R.loc[ref, k]); R[k] *= np.sign(R.loc[ref, k])  # +PC = what ref rates higher
            F = pd.DataFrame({k: [stats.spearmanr(B[k], feats[f], nan_policy="omit").statistic for f in feats] for k in top}, index=feats.columns)
            labels = {k: label_pc(client, which, it, B[k], f"{variant}/{ref}/{k}") for k in top} if do_label else {}
            (ROOT / "pc_scores").mkdir(exist_ok=True)
            B[top].to_parquet(ROOT / "pc_scores" / f"{variant}__{ref}.parquet")  # oriented per-item scores, for the example panels
            out[f"{variant}/{ref}"] = dict(evr=evr.tolist(), top=top, rater_pc_rho=R[top].to_dict(), feature_rho=F.to_dict(), labels=labels)
            fig, ax = plt.subplots(figsize=(8, 5))
            Rt = R.loc[rater_order(R.index)]
            im = ax.imshow(Rt.values, cmap=DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto"); fig.colorbar(im, ax=ax, label="Spearman ρ(PC, rater score)")
            ax.set_xticks(range(N_PC), [f"{k}*" if k in top else k for k in R.columns], rotation=90); ax.set_yticks(range(len(Rt)), Rt.index); ax.grid(False)
            ax.set_title(f"{variant}: which of {ref}'s PCs does each rater reward? (* = labelled)")
            save(fig, f"pc_rater_profile_{variant}_{ref}")
            plot_pareto(variant, ref, B, top, labels, S[ref], grp)
            print(variant, ref, "top PCs", top, {k: labels.get(k) for k in top})
    record("pcs", out)
    if do_label: print("labelling cost $", sum(l["cost"] for v in out.values() for l in v["labels"].values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-label", action="store_true")
    main(not ap.parse_args().no_label)
