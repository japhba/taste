"""Assemble the rated item sets.

  python -m taste.items images  -> data/images_items.parquet  (ACX 50 + LAPIS art + AVA photos, resized copies in data/images/)
  python -m taste.items haiku   -> data/haiku_items.parquet  (all generators + human tiers)
"""
import io, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm
from taste.paths import RAW, DATA
from taste.models import BY_NAME

N_LAPIS, N_AVA, MAX_SIDE = 600, 600, 768
IMG_DIR = DATA / "images"; IMG_DIR.mkdir(exist_ok=True)


def stratified(df, col, n, bins=10, seed=0):
    q = pd.qcut(df[col].rank(method="first"), bins, labels=False)
    return df.groupby(q, group_keys=False).apply(lambda g: g.sample(n // bins, random_state=seed))


def store(img: Image.Image, item_id: str) -> str:
    img = img.convert("RGB"); img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    p = IMG_DIR / f"{item_id}.jpg"; img.save(p, quality=95)
    return str(p.relative_to(DATA))  # relative to DATA, so the item table moves with the data dir


def images():
    rows = []
    acx = pd.read_parquet(DATA / "acx_images.parquet")
    for r in acx.itertuples():
        rows.append(dict(item_id=f"acx_{r.acx_idx:02d}", dataset="acx", source=r.truth, title=r.title, human_score=r.fav_share, path=store(Image.open(r.path), f"acx_{r.acx_idx:02d}")))
    lap = pd.read_csv(RAW / "asmaerten__LAPIS/giaa/test.csv").rename(columns={"4_styles": "style4", "2_styles": "style2"})
    lap = stratified(lap, "mean_response", N_LAPIS)
    for r in tqdm(lap.itertuples(), total=len(lap), desc="lapis"):
        iid = f"lapis_{r.image_id}"
        rows.append(dict(item_id=iid, dataset="lapis", source="human", title=r.path, human_score=r.mean_response, style=r.style2, style4=r.style4, genre=r.genre,
                         path=store(Image.open(RAW / "asmaerten__LAPIS/images" / r.path), iid)))
    f = pq.read_table(RAW / "trojblue__AVA-aesthetics-10pct-min50-10bins/data/test-00000-of-00002.parquet").to_pandas()
    ava = stratified(f.drop(columns="image").assign(row=np.arange(len(f))), "mean_score", N_AVA)
    for r in tqdm(ava.itertuples(), total=len(ava), desc="ava"):
        iid = f"ava_{r.image_id}"
        rows.append(dict(item_id=iid, dataset="ava", source="human-photo", title=r.image_id, human_score=r.mean_score,
                         path=store(Image.open(io.BytesIO(f.image.iloc[r.row]["bytes"])), iid)))
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "images_items.parquet")
    print(df.groupby("dataset").human_score.describe())


def haiku():
    gens = pd.concat([pd.read_parquet(p) for p in sorted((DATA / "haiku_gen").glob("*.parquet"))])
    df = pd.concat([gens, pd.read_parquet(DATA / "haiku_human.parquet")], ignore_index=True)
    df = df[df.haiku.str.strip() != ""].reset_index(drop=True)
    df["item_id"] = [f"h{i:05d}" for i in range(len(df))]
    df["params_b"] = df.generator.map(lambda g: BY_NAME[g].params_b if g in BY_NAME else np.nan)
    df["n_lines"] = df.haiku.str.count("\n") + 1
    df.to_parquet(DATA / "haiku_items.parquet")
    print(df.groupby("generator").agg(n=("haiku", "size"), lines=("n_lines", "mean")).to_string())


if __name__ == "__main__":
    {"images": images, "haiku": haiku}[sys.argv[1]]()
