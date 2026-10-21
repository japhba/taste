#!/bin/bash
# Grab 8 suitable GPUs for the DeepSeek run as soon as they exist: every 20 min, (a) test whether CUDA works on the 8xH200 node
# gpu-sr675-35 (broken since 2026-10-21) and submit there if so, (b) submit to the 8xH100 node if all 8 GPUs are idle.
# Waits for the weight job ($1) to finish first. Exits after submitting once.
WJOB=$1; cd ~/taste
while squeue -j $WJOB -h | grep -q .; do sleep 120; done
grep -q CONVERT_DONE logs/slurm-ds-weights-$WJOB.out || { echo "weights job did not finish cleanly"; exit 1; }
while true; do
  T=$(sbatch --parsable -p gpu_lowp -w gpu-sr675-35 --gres=gpu:h200:8 -c 8 --mem=32G -t 00:05:00 -o logs/tmp/h200test-%j.out --wrap "/var/tmp/jbauer/venvs/taste/bin/python -c 'import torch; print(\"CUDA_OK\" if torch.cuda.is_available() else \"CUDA_BROKEN\")'")
  while squeue -j $T -h | grep -q .; do sleep 20; done
  if grep -q CUDA_OK logs/tmp/h200test-$T.out 2>/dev/null; then J=$(sbatch --parsable -w gpu-sr675-35 --gres=gpu:h200:8 slurm/ds_rate.sbatch); echo "$(date +%T) H200 node works: submitted $J"; exit 0; fi
  used=$(squeue -w gpu-xd670-30 --all -h -t R -o "%b" | grep -oE "[0-9]+$" | paste -sd+ | bc); used=${used:-0}
  if [ "$used" -eq 0 ]; then J=$(sbatch --parsable -w gpu-xd670-30 --gres=gpu:h100:8 slurm/ds_rate.sbatch); echo "$(date +%T) H100 node idle: submitted $J"; exit 0; fi
  echo "$(date +%T) H200 CUDA broken, H100 node has $used GPUs in use; retry in 20 min"
  sleep 1200
done
