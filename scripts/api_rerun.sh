#!/bin/bash
# wait for the running API chain (pid $1), then re-rate with logprob-capable providers only
while kill -0 $1 2>/dev/null; do sleep 20; done
cd ~/taste && source /var/tmp/jbauer/venvs/taste/bin/activate
for w in haiku images; do python -m taste.rate_api qwen3.5-397b-a17b $w; python -m taste.rate_api qwen3.5-122b-a10b $w; done
