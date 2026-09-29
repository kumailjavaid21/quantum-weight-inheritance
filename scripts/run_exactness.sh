#!/bin/bash
# Repo-relative wrapper. Set DATA_ROOT and RESULTS_ROOT for your environment;
# see docs/reproducibility.md.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src:$REPO/src/inheritance:$REPO/src/common:$PYTHONPATH"
python "$REPO/src/evaluation/exactness.py" "$@"
