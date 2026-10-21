"""Shared loading + plotting helpers for the analyses."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from taste.paths import DATA, RATINGS, HIDDEN, PLOTS, ROOT
from taste.models import BY_NAME

# validated categorical slots (dataviz reference palette, light mode); identity also carried by marker shape
FAMILY_COLOR = {"qwen": "#2a78d6", "gemma": "#eb6834", "human": "#1baf7a", "frontier": "#6250d6", "smollm": "#8a8984"}
FAMILY_MARKER = {"qwen": "o", "gemma": "s", "human": "D", "frontier": "^", "smollm": "v"}
FRONTIER = {"claude", "gpt", "gemini", "deepseek"}
ANCHOR = "claude-opus-5.5"
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#f0efec", "#c0302f"])
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
RESULTS = ROOT / "results.json"


def fam(name):
    if name in BY_NAME:
        f = BY_NAME[name].family
        return "frontier" if f in FRONTIER else f
    return "human"


def rater_order(names):
    key = lambda n: ({"qwen": 0, "gemma": 1, "frontier": 2}.get(fam(n), 3), BY_NAME[n].params_b if n in BY_NAME and not np.isnan(BY_NAME[n].params_b) else 1e4)
    return sorted(names, key=key)


def generator_order(names):
    fam_rank = {"smollm": 0, "qwen": 1, "gemma": 2, "frontier": 3, "human": 4}
    human_rank = {"human-twaiku": 0, "human-reddit-low": 1, "human-reddit-top": 2, "human-journal": 3}
    key = lambda n: (fam_rank[fam(n)], BY_NAME[n].params_b if n in BY_NAME and not np.isnan(BY_NAME[n].params_b) else human_rank.get(n, 1e4))
    return sorted(names, key=key)


def ratings(which, col="score"):
    """items x raters matrix of `col` (raters whose files lack `col`, e.g. API raters for logprob_mean, are left out)."""
    dfs = {p.stem: pd.read_parquet(p).set_index("item_id") for p in sorted((RATINGS / which).glob("*.parquet"))}
    return pd.DataFrame({k: d[col] for k, d in dfs.items() if col in d})


def ratings_long(which):
    return pd.concat([pd.read_parquet(p) for p in sorted((RATINGS / which).glob("*.parquet"))], ignore_index=True)


def zscore(df):
    return (df - df.mean()) / df.std()


def hidden(which, rater, pool="mean"):
    z = np.load(HIDDEN / which / f"{rater}.npz", allow_pickle=True)  # item_id saved as an object array (our own files)
    return pd.Index(z["item_id"]), z["layers"], z[pool]


def save(fig, name):
    p = PLOTS / f"{name}.png"; fig.savefig(p); plt.close(fig)
    print(p)
    return p


def record(key, value):
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[key] = value
    RESULTS.write_text(json.dumps(res, indent=1, default=float))


def items(which):
    return pd.read_parquet(DATA / f"{which}_items.parquet").set_index("item_id")
