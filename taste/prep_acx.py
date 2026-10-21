"""ACX AI Art Turing Test (Nov 2024): 50 images + per-respondent survey -> data/acx_images.parquet, data/acx_responses.parquet.

Images and answer key come from the results post (each image follows a heading "N: Title" and a paragraph "Human./AI. ...").
Per-image human signal: share of the 9k 'favorite' votes, share judging it human, and favorite share split by attitude to AI art.
"""
import io, re, requests
from PIL import Image
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from taste.paths import RAW, DATA

ACX = RAW / "acx"
IMG = ACX / "images"; IMG.mkdir(parents=True, exist_ok=True)

s = BeautifulSoup((ACX / "results_post.html").read_text(), "html.parser")
body = s.find("div", class_="available-content")
rows = []
for im in body.find_all("img"):
    head = im.find_previous(["p", "h1", "h2", "h3", "h4"]).get_text().strip()
    m = re.match(r"^(\d+): (.+)$", head)
    if not m: continue
    idx, title = int(m.group(1)), m.group(2).strip()
    expl = im.find_next(["p"]).get_text().strip()
    truth = "human" if expl.startswith("Human") else "ai" if expl.startswith("AI") else None
    assert truth, expl
    rows.append(dict(acx_idx=idx, title=title, truth=truth, explanation=expl, url=re.sub(r"w_\d+", "w_1456", im["src"])))
img = pd.DataFrame(rows).sort_values("acx_idx").reset_index(drop=True)
assert len(img) == 50, len(img)

for r in tqdm(img.itertuples(), total=len(img), desc="download acx"):
    p = IMG / f"{r.acx_idx:02d}.jpg"
    if not p.exists():
        resp = requests.get(r.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60); resp.raise_for_status()
        Image.open(io.BytesIO(resp.content)).convert("RGB").save(p, quality=95)
img["path"] = [str(IMG / f"{i:02d}.jpg") for i in img.acx_idx]

df = pd.read_excel(ACX / "aituringpublic.xlsx", header=0).iloc[3:]  # first 3 rows are the scoring legend
titles = img.title.tolist()
assert all(t in df.columns for t in titles), [t for t in titles if t not in df.columns]
fav = df["Which picture is your favorite?"].replace({"Minaret Ship": "Minaret Boat"})  # favorite-question spelling
opinion = df["What is your opinion of AI art on an artistic level?"]
img["fav_votes"] = [int((fav == t).sum()) for t in titles]
img["fav_share"] = img.fav_votes / img.fav_votes.sum()
img["p_judged_human"] = [float((df[t] == "Human").sum() / df[t].notna().sum()) for t in titles]
for lo, hi, name in [(1, 2, "ai_skeptic"), (4, 5, "ai_fan")]:
    sub = fav[(opinion >= lo) & (opinion <= hi)]
    img[f"fav_share_{name}"] = [float((sub == t).sum() / sub.notna().sum()) for t in titles]
img.drop(columns=["url"]).to_parquet(DATA / "acx_images.parquet")

resp = df[titles + ["Which picture is your favorite?", "What is your opinion of AI art on an artistic level?",
                    "What is your level of familiarity with human art?", "What is your level of familiarity with AI art?"]].copy()
resp.columns = [f"guess::{t}" for t in titles] + ["favorite", "ai_art_opinion", "familiarity_human_art", "familiarity_ai_art"]
resp["favorite"] = fav.values
resp.astype(str).to_parquet(DATA / "acx_responses.parquet")
print(img[["acx_idx", "title", "truth", "fav_votes", "p_judged_human"]].to_string())
