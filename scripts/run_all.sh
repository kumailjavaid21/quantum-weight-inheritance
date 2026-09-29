#!/bin/bash
# Runs every experiment in order. See docs/reproducibility.md for the
# environment and path variables each stage expects.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO/scripts/run_exactness.sh"
bash "$REPO/scripts/run_speedup.sh"
bash "$REPO/scripts/run_compute_cost.sh"
bash "$REPO/scripts/run_multi_seed.sh"
