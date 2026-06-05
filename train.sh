#!/usr/bin/env bash
# Launch SFT training on 8× A100 80GB GPUs.
# Usage:
#   ./train.sh                          # fresh run
#   ./train.sh --resume checkpoints/checkpoint-50

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file accelerate_zero3.yaml \
    run_sft.py "$@"
