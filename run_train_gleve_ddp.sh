#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=5,6
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Visible GPU count=${VISIBLE_GPU_COUNT}"

if [ "${VISIBLE_GPU_COUNT}" -lt 1 ]; then
  echo "No visible CUDA devices. Stop."
  exit 1
fi

torchrun \
  --nproc_per_node="${VISIBLE_GPU_COUNT}" \
  train_gleve.py "$@"
