"""DeepSeek-V4.1-Flash-1017 (763B, ~510 GB) into a Modal Volume, then converted to the 8-way tensor-parallel layout of
DeepSeek's reference inference code.

  uv tool run --from modal modal run modal_app/deepseek_weights.py            # download + convert
  uv tool run --from modal modal run modal_app/deepseek_weights.py --step convert

Download: the 48 shards are bin-packed into groups and fetched by parallel CPU containers inside Modal's network
(hf_xet high-performance mode) straight into the Volume, so nothing passes through u435d.
Convert: DeepSeek's convert.py keeps all 8 ranks in RAM (~600 GB); here each rank runs in its own container and reads
only its own experts.
"""
import modal

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
HF, TP, MP = "/vol/hf", "/vol/tp8", 8
GROUP_GB = 40

app = modal.App("taste-deepseek-weights")
vol = modal.Volume.from_name("taste-deepseek-v41-flash", create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub==1.33.0", "hf-xet==1.7.0", "safetensors==0.8.0", "tqdm", "numpy")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
         .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}))


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=16384, timeout=3 * 3600)
def fetch(files: list[str]) -> float:
    import os, time
    from huggingface_hub import hf_hub_download
    t, n = time.time(), 0
    for f in files:
        n += os.path.getsize(hf_hub_download(REPO, f, local_dir=HF))  # straight into the Volume
    vol.commit()
    print(f"{len(files)} files, {n / 1e9:.1f} GB in {time.time() - t:.0f} s ({n / 1e9 / (time.time() - t):.2f} GB/s)")
    return n / 1e9


@app.function(image=image.add_local_file("taste/ds_convert.py", "/root/ds_convert.py"), volumes={"/vol": vol}, cpu=16.0, memory=192 * 1024, timeout=3 * 3600)
def convert(rank: int) -> float:
    import ds_convert
    vol.reload(); gb = ds_convert.convert(HF, TP, rank, MP); vol.commit()
    return gb


@app.local_entrypoint()
def main(step: str = "all"):
    from huggingface_hub import HfApi
    if step in ("all", "download"):
        files = [(s.rfilename, s.size or 0) for s in HfApi().model_info(REPO, files_metadata=True).siblings]
        big = sorted([f for f in files if f[0].endswith(".safetensors")], key=lambda f: -f[1])
        groups, sizes = [[]], [0.0]
        for name, size in big:  # first-fit decreasing into ~GROUP_GB bins
            j = next((k for k, s in enumerate(sizes) if s + size / 1e9 <= GROUP_GB), None)
            if j is None: groups.append([]); sizes.append(0.0); j = len(groups) - 1
            groups[j].append(name); sizes[j] += size / 1e9
        groups[-1] += [f for f, _ in files if not f.endswith(".safetensors")]
        print(f"{len(big)} shards, {sum(s for _, s in big) / 1e9:.0f} GB in {len(groups)} parallel groups")
        print(f"downloaded {sum(fetch.map(groups)):.0f} GB")
    if step in ("all", "convert"):
        print(f"converted {sum(convert.map(range(MP))):.0f} GB")
