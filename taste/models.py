"""Generators and raters. params_b = total parameters (B), active_b = active per token (MoE)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class M:
    name: str
    family: str
    params_b: float
    active_b: float
    hf: str | None = None  # local (vLLM generation / HF forward-pass rating with hidden states)
    api: str | None = None  # OpenRouter id (behaviour only)
    vision: bool = True


MODELS = [
    M("smollm2-135m", "smollm", 0.135, 0.135, hf="HuggingFaceTB/SmolLM2-135M-Instruct", vision=False),
    M("smollm2-360m", "smollm", 0.36, 0.36, hf="HuggingFaceTB/SmolLM2-360M-Instruct", vision=False),
    M("smollm2-1.7b", "smollm", 1.7, 1.7, hf="HuggingFaceTB/SmolLM2-1.7B-Instruct", vision=False),
    M("qwen3.5-0.8b", "qwen", 0.8, 0.8, hf="Qwen/Qwen3.5-0.8B"),
    M("qwen3.5-2b", "qwen", 2, 2, hf="Qwen/Qwen3.5-2B"),
    M("qwen3.5-4b", "qwen", 4, 4, hf="Qwen/Qwen3.5-4B"),
    M("qwen3.5-9b", "qwen", 9, 9, hf="Qwen/Qwen3.5-9B", api="qwen/qwen3.5-9b"),
    M("qwen3.5-27b", "qwen", 27, 27, hf="Qwen/Qwen3.5-27B", api="qwen/qwen3.5-27b"),
    M("qwen3.5-35b-a3b", "qwen", 35, 3, hf="Qwen/Qwen3.5-35B-A3B", api="qwen/qwen3.5-35b-a3b"),
    M("qwen3.5-122b-a10b", "qwen", 122, 10, api="qwen/qwen3.5-122b-a10b"),
    M("qwen3.5-397b-a17b", "qwen", 397, 17, api="qwen/qwen3.5-397b-a17b"),
    M("gemma-4-e2b", "gemma", 5.1, 2, hf="google/gemma-4-E2B-it"),
    M("gemma-4-e4b", "gemma", 8, 4, hf="google/gemma-4-E4B-it"),
    M("gemma-4-26b-a4b", "gemma", 26, 4, hf="google/gemma-4-26b-a4b-it", api="google/gemma-4-26b-a4b-it"),
    M("gemma-4-31b", "gemma", 31, 31, hf="google/gemma-4-31B-it", api="google/gemma-4-31b-it"),
    # frontier (closed) generators / anchor judges; sizes unknown -> nan
    M("deepseek-v4.1-flash", "deepseek", float("nan"), float("nan"), api="deepseek/deepseek-v4.1-flash"),
    M("gemini-3.8-flash", "gemini", float("nan"), float("nan"), api="google/gemini-3.8-flash"),
    M("gpt-5.6-sol", "gpt", float("nan"), float("nan"), api="openai/gpt-5.6-sol"),
    M("claude-opus-5.5", "claude", float("nan"), float("nan"), api="anthropic/claude-opus-5.5"),
    M("claude-fable-5.1", "claude", float("nan"), float("nan"), api="anthropic/claude-fable-5.1"),
]
BY_NAME = {m.name: m for m in MODELS}
LOCAL_GENERATORS = [m.name for m in MODELS if m.hf]
API_GENERATORS = [m.name for m in MODELS if m.api and not m.hf]
LOCAL_RATERS = [m.name for m in MODELS if m.hf and m.vision]
