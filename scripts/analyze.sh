#!/bin/bash
# Full analysis + report from the ratings / hidden states on ceph.
set -eo pipefail
cd ~/taste && source /var/tmp/jbauer/venvs/taste/bin/activate
rm -f /ceph/scratch/jbauer/taste/results.json
python -m taste.analysis_haiku
python -m taste.analysis_images
python -m taste.probes
python -m taste.pcs "$@"
python -m taste.taste_model
python -m taste.report
