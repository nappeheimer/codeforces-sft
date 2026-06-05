#!/usr/bin/env bash
# Evaluate the fine-tuned Codeforces SFT model.
#
# Usage:
#   ./eval.sh                        # loss + 20 generations (default)
#   ./eval.sh --loss                 # loss only
#   ./eval.sh --generate --n 50      # 50 generations only
#   ./eval.sh --compare-base         # also run base model for side-by-side
#   ./eval.sh --adapter checkpoints/checkpoint-105   # specific checkpoint

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7   # GPUs 1-7; leave 0 free
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default: run both loss and generation
ARGS="${@:---loss --generate --n 20 --out eval_results.json}"

python eval.py $ARGS
