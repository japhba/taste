"""Hand-crafted, interpretable features used to validate auto-labelled PCs.

  python -m taste.features -> data/haiku_features.parquet, data/images_features.parquet
haiku: 5-7-5 deviation, syllables, words, concreteness (Brysbaert et al. 2014 norms), word rarity (wordfreq Zipf), type-token ratio,
       dash/juxtaposition, first-person rate.
images: mirror symmetry (L-R correlation of grey image), colourfulness (Hasler & Suesstrunk 2003), edge density, box-counting fractal
       dimension of the edge map, luminance entropy, RMS contrast, mean saturation, brightness, aspect ratio.
"""
import re
import numpy as np
import pandas as pd
import pronouncing
from PIL import Image
from scipy import ndimage
from tqdm.auto import tqdm
from wordfreq import zipf_frequency
from taste.paths import RAW, DATA

CONC = pd.read_csv(RAW / "concreteness/Concreteness_ratings_Brysbaert_et_al_BRM.txt", sep="\t").set_index("Word")["Conc.M"].to_dict()
WORD = re.compile(r"[a-z']+")


def syllables(w):
    ph = pronouncing.phones_for_word(w)
    return pronouncing.syllable_count(ph[0]) if ph else max(1, len(re.findall(r"[aeiouy]+", w)))


def haiku_features(h):
    lines = h.lower().split("\n")
    words = [WORD.findall(l) for l in lines]
    syl = [sum(syllables(w) for w in ws) for ws in words]
    flat = [w for ws in words for w in ws]
    target = [5, 7, 5]
    dev = sum(abs(s - t) for s, t in zip(syl, target)) + 5 * abs(len(lines) - 3)
    conc = [CONC[w] for w in flat if w in CONC]
    return dict(f_575_deviation=dev, f_syllables=sum(syl), f_words=len(flat), f_concreteness=np.mean(conc) if conc else np.nan,
                f_zipf=np.mean([zipf_frequency(w, "en") for w in flat]) if flat else np.nan, f_ttr=len(set(flat)) / max(1, len(flat)),
                f_dash=int(bool(re.search(r"[—–-]", h))), f_first_person=sum(w in {"i", "me", "my", "mine", "we", "us", "our"} for w in flat) / max(1, len(flat)))


def image_features(path):
    im = Image.open(DATA / path).convert("RGB")
    a = np.asarray(im, dtype=np.float64)
    g = a @ [0.299, 0.587, 0.114]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255
    edges = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1)); e = edges > np.percentile(edges, 90)
    hist = np.histogram(g, 256, (0, 256))[0] / g.size
    sizes = 2 ** np.arange(1, int(np.log2(min(e.shape))) - 1)
    counts = [np.sum(e[: e.shape[0] // s * s, : e.shape[1] // s * s].reshape(e.shape[0] // s, s, -1, s).any((1, 3))) for s in sizes]
    return dict(f_symmetry_lr=np.corrcoef(g.ravel(), g[:, ::-1].ravel())[0, 1], f_colourfulness=np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()),
                f_edge_density=float(edges.mean()), f_fractal_dim=-np.polyfit(np.log(sizes), np.log(counts), 1)[0],
                f_lum_entropy=-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])), f_rms_contrast=g.std(), f_saturation=hsv[..., 1].mean(),
                f_brightness=hsv[..., 2].mean(), f_aspect=im.width / im.height)


if __name__ == "__main__":
    h = pd.read_parquet(DATA / "haiku_items.parquet")
    pd.DataFrame([haiku_features(x) for x in tqdm(h.haiku, desc="haiku feats")]).assign(item_id=h.item_id.values).to_parquet(DATA / "haiku_features.parquet")
    im = pd.read_parquet(DATA / "images_items.parquet")
    pd.DataFrame([image_features(p) for p in tqdm(im.path, desc="image feats")]).assign(item_id=im.item_id.values).to_parquet(DATA / "images_features.parquet")
