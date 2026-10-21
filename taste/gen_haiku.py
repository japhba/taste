"""Generate haiku: 40 topics x N_SAMPLES per generator -> data/haiku_gen/<generator>.parquet.

  python -m taste.gen_haiku local qwen3.5-0.8b gemma-4-e2b ...   (vLLM, thinking disabled)
  python -m taste.gen_haiku api claude-opus-5.5 ...              (OpenRouter, reasoning off/minimal)
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm.auto import tqdm
from taste.paths import DATA
from taste.models import BY_NAME
from taste.prep_haiku import clean_generation

N_SAMPLES = 5
PROMPT = "Write a haiku about {topic}. Reply with only the haiku: three lines, no title, no commentary."
TOPICS = json.loads((DATA / "haiku_topics.json").read_text())
OUT = DATA / "haiku_gen"; OUT.mkdir(exist_ok=True)
REASONING = {m: {"effort": "low"} for m in ["openai/gpt-5.6-sol", "google/gemini-3.8-flash", "anthropic/claude-opus-5.5", "anthropic/claude-fable-5.1"]}  # mandatory reasoning; others disabled


def save(name, rows):
    m = BY_NAME[name]
    df = pd.DataFrame(rows).assign(generator=name, family=m.family, haiku=lambda d: d.raw.map(clean_generation))
    df.to_parquet(OUT / f"{name}.parquet")
    print(name, len(df), "empty:", int((df.haiku.str.count("\n") < 2).sum()), "| e.g.", df.haiku.head(3).str.replace("\n", " / ").tolist())


def run_local(names):
    from vllm import LLM, SamplingParams
    for name in tqdm(names, desc="local generators"):
        if (OUT / f"{name}.parquet").exists(): continue
        m = BY_NAME[name]
        kw = dict(language_model_only=True) if m.vision else {}
        llm = LLM(model=m.hf, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.95, enforce_eager=True, seed=0, **kw)
        sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=0.95, max_tokens=150, seed=0)
        outs = llm.chat([[{"role": "user", "content": PROMPT.format(topic=t)}] for t in TOPICS], sp,
                        chat_template_kwargs={"enable_thinking": False})
        save(name, [dict(topic=t, sample=i, raw=c.text) for t, o in zip(TOPICS, outs) for i, c in enumerate(o.outputs)])
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()


def openrouter():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=8, timeout=180)


def run_api(names):
    client = openrouter()
    for name in names:
        if (OUT / f"{name}.parquet").exists(): continue
        mid = BY_NAME[name].api
        reasoning = REASONING.get(mid, {"enabled": False})

        def call(job):
            t, i = job
            r = client.chat.completions.create(model=mid, messages=[{"role": "user", "content": PROMPT.format(topic=t)}], temperature=1.0,
                                               max_tokens=2000, extra_body={"reasoning": reasoning, "usage": {"include": True}, "seed": i})
            return dict(topic=t, sample=i, raw=r.choices[0].message.content or "", cost=r.usage.model_extra.get("cost", 0.0))
        jobs = [(t, i) for t in TOPICS for i in range(N_SAMPLES)]
        with ThreadPoolExecutor(16) as ex: rows = list(tqdm(ex.map(call, jobs), total=len(jobs), desc=name))
        print(f"{name}: ${sum(r['cost'] for r in rows):.3f}")
        save(name, rows)


if __name__ == "__main__":
    {"local": run_local, "api": run_api}[sys.argv[1]](sys.argv[2:])
