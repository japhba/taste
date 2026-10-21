"""Behaviour-only raters via OpenRouter (same prompts as taste.rate; temperature 0; digit distribution from top_logprobs when the
provider returns them, else the sampled digit).

  python -m taste.rate_api <rater> haiku|images [--datasets acx lapis]
"""
import argparse, base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from taste.paths import DATA, RATINGS
from taste.models import BY_NAME
from taste.rate import HAIKU_Q, IMAGE_Q
from taste.gen_haiku import openrouter, REASONING


def content(which, row):
    if which == "haiku": return HAIKU_Q.format(haiku=row.haiku)
    b64 = base64.b64encode((DATA / row.path).read_bytes()).decode()
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": IMAGE_Q}]


def main(name, which, datasets):
    client, mid = openrouter(), BY_NAME[name].api
    items = pd.read_parquet(DATA / f"{which}_items.parquet")
    if datasets: items = items[items.dataset.isin(datasets)]
    out = RATINGS / which / f"{name}.parquet"; out.parent.mkdir(exist_ok=True)

    def call(row):
        r = client.chat.completions.create(model=mid, messages=[{"role": "user", "content": content(which, row)}], temperature=0.0, max_tokens=2000,
                                           logprobs=True, top_logprobs=5,  # some providers cap at 5; the digits carry ~all the mass
                                           extra_body={"reasoning": REASONING.get(mid, {"enabled": False}), "usage": {"include": True},
                                                       # route only to providers that return logprobs (Anthropic never does: digit only)
                                                       **({} if mid.startswith("anthropic/") else {"provider": {"require_parameters": True}})})
        ans = (r.choices[0].message.content or "").strip()
        p = np.zeros(9)
        lp = r.choices[0].logprobs
        if lp and lp.content:
            for t in lp.content[0].top_logprobs:
                if t.token.strip() in [str(k) for k in range(1, 10)]: p[int(t.token.strip()) - 1] += np.exp(t.logprob)
        digit = next((int(c) for c in ans if c in "123456789"), np.nan)
        score = (p / p.sum()) @ np.arange(1, 10) if p.sum() > 0 else digit
        return dict(item_id=row.item_id, answer=ans, digit=digit, score=score, digit_mass=p.sum(), cost=r.usage.model_extra.get("cost", 0.0),
                    **{f"p{k}": p[k - 1] for k in range(1, 10)})

    with ThreadPoolExecutor(16) as ex: rows = list(tqdm(ex.map(call, items.itertuples()), total=len(items), desc=f"{name} {which}"))
    df = pd.DataFrame(rows).assign(rater=name)
    df.to_parquet(out)
    print(f"{name} {which}: ${df.cost.sum():.2f}, logprobs for {(df.digit_mass > 0).mean():.0%}, unparsed {df.score.isna().sum()}, score mean {df.score.mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("rater"); ap.add_argument("which"); ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args(); main(a.rater, a.which, a.datasets)
