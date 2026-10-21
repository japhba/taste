"""Rate haiku + images with DeepSeek-V4.1-Flash-1017 on 8xH100 (Modal), hidden states kept. Weights: deepseek_weights.py.

  uv tool run --from modal modal run modal_app/deepseek_rate.py --max-minutes 36

Budget: 8xH100 is ~$32/h, so the run is capped twice: taste.ds_rate stops all ranks after --max-minutes (loading included)
and saves every 100 items; the function's hard timeout kills the container a few minutes later regardless.
Outputs land in the Volume under /taste/{ratings,hidden}; fetch them with `modal volume get`.
"""
import modal

app = modal.App("taste-deepseek-rate")
vol = modal.Volume.from_name("taste-deepseek-v41-flash", version=2)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .pip_install("torch==2.10.0", "transformers==5.5.4", "tokenizers", "safetensors==0.8.0", "numpy", "sympy", "Pillow",
                      "tilelang==0.1.12", "tqdm", "pandas", "pyarrow", "python-dotenv")
         .env({"TASTE_ROOT": "/vol/taste", "HF_HUB_OFFLINE": "1", "CUDA_HOME": "/usr/local/cuda"})
         .add_local_dir("taste", "/root/taste"))


@app.function(image=image, gpu="H100:8", volumes={"/vol": vol}, cpu=16.0, memory=128 * 1024, timeout=44 * 60)
def rate(max_minutes: float, plan: list[str], haiku_per_gen: int):
    import subprocess
    vol.reload()
    cmd = ["torchrun", "--nproc-per-node", "8", "-m", "taste.ds_rate", "--ckpt", "/vol/tp8", "--hf", "/vol/hf",
           "--max-minutes", str(max_minutes), "--haiku-per-gen", str(haiku_per_gen), "--plan", *plan]
    try: subprocess.run(cmd, cwd="/root", check=True)
    finally: vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=20 * 60)
def selftest():
    """1 GPU, cheap: DeepSeek's own kernel self-test (small random model, real fp8/fp4 kernels) + taste.ds_rate.forward_hidden."""
    import subprocess, sys
    vol.reload()
    subprocess.run([sys.executable, "model.py"], cwd="/vol/hf/inference", check=True)
    sys.path.insert(0, "/vol/hf/inference"); sys.path.insert(0, "/root")
    import torch, model as M
    from taste.ds_rate import forward_hidden, KEEP
    torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
    args = M.ModelArgs(); KEEP[:] = [0, args.n_layers]
    m = M.Transformer(args)
    logits, states = forward_hidden(m, M, torch.randint(0, args.vocab_size, (1, 64)), None, None)
    print("forward_hidden ok:", tuple(logits.shape), {k: tuple(v.shape) for k, v in states.items()})


@app.local_entrypoint()
def main(max_minutes: float = 36, haiku_per_gen: int = 50, plan: str = "acx,lapis,haiku,ava", test: bool = False):
    if test: selftest.remote(); return
    rate.remote(max_minutes, plan.split(","), haiku_per_gen)
