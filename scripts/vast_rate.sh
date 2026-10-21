#!/bin/bash
# Rating only (generation + item tables already on the box). One item per forward pass; hidden states saved.
set -eo pipefail
export PATH=$HOME/.local/bin:$PATH TASTE_ROOT=/workspace/root HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false UV_PROJECT_ENVIRONMENT=/workspace/venv
cd /workspace/taste && source /workspace/venv/bin/activate
python -m taste.rate gemma-4-e2b qwen3.5-0.8b qwen3.5-2b gemma-4-e4b qwen3.5-4b qwen3.5-9b gemma-4-26b-a4b qwen3.5-27b gemma-4-31b --sets haiku images
echo ALL_DONE
