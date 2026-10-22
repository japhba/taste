#!/bin/bash
# One DeepSeek tensor-parallel rank, launched by `srun --gpu-bind=none` (one task per GPU, one or more nodes): maps the SLURM task ids
# to the torch.distributed variables that taste.ds_rate reads. Extra arguments go to taste.ds_rate.
set -eo pipefail
cd ~/taste
export CUDA_HOME=/ceph/scratch/jbauer/envs/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH
source /var/tmp/jbauer/venvs/dsrun/bin/activate
set -a; source ~/.env; set +a
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
D=/ceph/scratch/jbauer/models/DeepSeek-V4.1-Flash-1017
t=$(date +%s); cat $D-TP8/model$RANK-mp8.safetensors > /dev/null   # warm the page cache: cold ceph reads can be ~0.1 GB/s
echo "rank $RANK on $(hostname) gpu $LOCAL_RANK: page cache warm in $(( $(date +%s) - t )) s"
exec python -m taste.ds_rate --ckpt $D-TP8 --hf $D "$@"
