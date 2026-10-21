#!/bin/bash
# After the DeepSeek job ($1) leaves the queue (requeues keep it queued): if its last run exited cleanly, rerun the analysis and
# export the slides, so the report includes DeepSeek without anyone watching. A failed run is left alone for debugging.
#   nohup scripts/after_ds.sh <jobid> >> logs/after_ds.log 2>&1 &
J=$1; cd ~/taste
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 120; done
L=logs/slurm-ds-hold-$J.out
echo "$(date '+%F %T') job $J left the queue; last exit: $(grep '=== rating run exited' $L | tail -1)"
tail -1 $L | grep -q DS_RATE_EXIT_OK || { echo "rating run did not finish cleanly: not running the analysis"; exit 1; }
set -eo pipefail
scripts/analyze.sh
PLAYWRIGHT_BROWSERS_PATH=/var/tmp/jbauer/ms-playwright uv run --no-project --with playwright --with python-dotenv python -m taste.export_slides
echo "$(date '+%F %T') AFTER_DS_DONE"
