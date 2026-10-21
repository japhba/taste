#!/bin/bash
# Runs on the rented GPU box: model download, local haiku generation, forward-pass rating with hidden states.
set -eo pipefail
export PATH=$HOME/.local/bin:$PATH TASTE_ROOT=/workspace/root HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false UV_PROJECT_ENVIRONMENT=/workspace/venv
cd /workspace/taste && source /workspace/venv/bin/activate
LOCAL_GEN="smollm2-135m smollm2-360m smollm2-1.7b qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b gemma-4-e2b gemma-4-e4b gemma-4-26b-a4b gemma-4-31b"
RATERS="qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b gemma-4-e2b gemma-4-e4b gemma-4-26b-a4b gemma-4-31b"
python -m taste.download
python -m taste.rate qwen3.5-0.8b --sets images --check
python -m taste.rate gemma-4-e2b --sets images --check
python -m taste.gen_haiku local $LOCAL_GEN
python -m taste.items haiku
python -m taste.rate qwen3.5-0.8b --sets haiku --check
python -m taste.rate gemma-4-e2b --sets haiku --check
python -m taste.rate $RATERS --sets haiku images
echo ALL_DONE
