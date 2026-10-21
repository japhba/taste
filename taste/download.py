"""Pre-fetch all local model weights into HF_HUB_CACHE (ceph), in parallel. Run on a compute node (fast NIC)."""
import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
from taste.models import MODELS
import taste.paths  # noqa: F401  (loads ~/.env)

assert os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1"
repos = [m.hf for m in MODELS if m.hf]
with ThreadPoolExecutor(4) as ex:
    for r, p in zip(repos, ex.map(lambda r: snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py"]), repos)):
        print(r, p, flush=True)
