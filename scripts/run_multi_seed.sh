#!/bin/bash
# Repo-relative wrapper. Set DATA_ROOT and RESULTS_ROOT for your environment;
# see docs/reproducibility.md.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src:$REPO/src/inheritance:$REPO/src/common:$PYTHONPATH"
# Trains the independent and inherited BraTS models across seeds 42-51.
for SEED in 42 43 44 45 46 47 48 49 50 51; do
  python "$REPO/src/training/train_brats.py" --seed "$SEED" "$@"
done
