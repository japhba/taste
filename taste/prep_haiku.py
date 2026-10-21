"""Topics + human haiku tiers -> data/haiku_topics.json, data/haiku_human.parquet.

Human tiers (all English): 'human-journal' (haiku_data_1/2 in statworx/haiku: published modern haiku),
'human-reddit-top' / 'human-reddit-low' (r/haiku by upvotes), 'human-twaiku' (accidental 5-7-5 found in tweets).
"""
import json, re
import pandas as pd
from taste.paths import RAW, DATA

TOPICS = [
    "autumn rain", "a city at night", "first snow", "grief", "a cat sleeping", "the ocean", "an old temple", "morning coffee",
    "a broken umbrella", "cherry blossoms", "a hospital waiting room", "fireflies", "a train station", "loneliness",
    "a child's laughter", "the moon over a field", "winter crows", "a dying houseplant", "summer heat", "a frog and a pond",
    "a smartphone's glow", "an abandoned house", "fog on a lake", "a mother's hands", "fallen leaves", "a thunderstorm",
    "the last day of summer", "a lighthouse", "insomnia", "a spider web with dew", "a busy kitchen", "mountain silence",
    "a funeral", "wind chimes", "a snail", "new year's morning", "a kite", "rush hour", "an empty playground", "starlight",
]
N_PER_TIER = 120


def normalize(lines: list[str]) -> str:
    """Same surface normalisation for every source: strip whitespace/markdown/quotes and line-final . , ; : (keep ? ! and dashes)."""
    out = []
    for l in lines:
        l = re.sub(r"[*_#`\"“”]", "", l).strip()
        l = re.sub(r"[.,;:]+$", "", l).strip()
        if l: out.append(l)
    return "\n".join(out)


def clean_generation(raw: str) -> str:
    """Model output -> haiku text: drop preambles/titles/commentary, keep the first 3 content lines."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    skip = re.compile(r"^(here('s| is)|sure|certainly|okay|title|haiku( about)?\b.*:|\(?\d+[-–]\d+[-–]\d+\)?$|---|#)", re.I)
    lines = [l for l in lines if not skip.match(re.sub(r"[*_#\"]", "", l).strip())]
    if len(lines) > 3: lines = [l for l in lines if not re.fullmatch(r"\*\*[^*]+\*\*|#+ .+|\"[^\"]+\"", l)]  # drop titles
    return normalize(lines[:3])


if __name__ == "__main__":
    (DATA / "haiku_topics.json").write_text(json.dumps(TOPICS, indent=1))
    s = pd.read_parquet(RAW / "statworx__haiku/train.parquet")
    r = pd.read_csv(RAW / "huanggab__reddit_haiku/merged_with_keywords.csv")
    r = r[r.processed_title.str.count("/") == 2]
    tiers = {
        "human-journal": s[s.source.isin(["haiku_data_1", "haiku_data_2"])].text.str.split(" / "),
        "human-twaiku": s[s.source == "twaiku"].text.str.split(" / "),
        "human-reddit-top": r[r.ups >= r.ups.quantile(0.95)].processed_title.str.split("/"),
        "human-reddit-low": r[r.ups <= 1].processed_title.str.split("/"),
    }
    rows = []
    for tier, ser in tiers.items():
        ser = ser[ser.str.len() == 3].sample(N_PER_TIER, random_state=0)
        rows += [dict(generator=tier, family="human", topic=None, sample=i, raw=" / ".join(x), haiku=normalize(x)) for i, x in enumerate(ser)]
    df = pd.DataFrame(rows).drop_duplicates("haiku")
    df.to_parquet(DATA / "haiku_human.parquet")
    print(df.generator.value_counts().to_dict())
    for t in tiers: print(t, "|", df[df.generator == t].haiku.head(3).str.replace("\n", " / ").tolist())
