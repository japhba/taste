from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

ROOT = Path(os.environ["TASTE_ROOT"]) if "TASTE_ROOT" in os.environ else Path(os.environ["CACHE_DIR"]) / "taste"  # ceph; TASTE_ROOT on rented boxes
RAW, DATA, RATINGS, HIDDEN, PLOTS = (ROOT / d for d in ["raw", "data", "ratings", "hidden", "plots"])
for d in (RAW, DATA, RATINGS, HIDDEN, PLOTS): d.mkdir(parents=True, exist_ok=True)
REPORTS = Path(__file__).resolve().parents[1] / "reports"  # private (NFS home), symlinked into the 8095-served dir
