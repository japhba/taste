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
python -m taste.taste_space            # judge-defined axes in each model's representation vs raw token embeddings (haiku)
python -m taste.taste_space_images     # the same on images, with pixels as the raw stimulus
python -m taste.lapis_taste            # people: one vote split into obvious / group / own / random (LAPIS)
python -m taste.taste_features         # unsupervised features, judges' weights, steering directions
python -m taste.taste_features --craft
python -m taste.steer_report           # steering results (the GPU runs: taste.steer, taste.steer_features)
python -m taste.taste_terms            # one panel per term of the model of taste
python -m taste.report
