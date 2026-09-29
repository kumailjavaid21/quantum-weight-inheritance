"""
Phase 2 Part B2: degradation pipeline. Given clean images, produces
degraded versions at multiple noise levels (sigma_gauss in {0.05, 0.10,
0.20}, each combined with Poisson shot noise, matching the noise model
qgan_utils_wigan.add_noise already established in Phase 0/1 - reused here
unchanged rather than inventing a second noise model).

Running this script as __main__ builds and saves the FIXED degraded
version of the canonical phase2_test_5000 set (prepared by
prepare_phase2_splits.py) at all three levels, for consistent evaluation
across every method in Part B3/B4 (WI-QGAN, DnCNN, Restormer, SwinIR all
denoise the SAME degraded images).

Output: ~/host_share/qgan_data/wigan_datasets/phase2/degraded_sigma{S}.npy
for S in {005, 010, 020}, plus a manifest.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from qgan_utils_wigan import add_noise  # noqa: E402

DATA_DIR = Path(os.path.expanduser("~/host_share/qgan_data/wigan_datasets/phase2"))
SIGMA_LEVELS = [0.05, 0.10, 0.20]
SEED = 44  # matches the seed convention used for TEST-split noise throughout Phase 1 (seed+2)


def degrade(clean, sigma, seed):
    """Gaussian(0, sigma) + Poisson shot noise, identical model to
    qgan_utils_wigan.add_noise (peak=255*poisson_scale, poisson_scale=1.0)."""
    return add_noise(clean, gaussian_sigma=sigma, poisson_scale=1.0, seed=seed)


def main():
    clean = np.load(DATA_DIR / "phase2_test_5000.npy").astype(np.float32)
    manifest = {"source": "phase2_test_5000.npy", "n_images": len(clean), "levels": {}}

    for sigma in SIGMA_LEVELS:
        degraded = degrade(clean, sigma, seed=SEED)
        tag = f"{sigma:.2f}".replace("0.", "0")
        out_path = DATA_DIR / f"degraded_sigma{tag}.npy"
        np.save(out_path, degraded)
        mse = float(np.mean((degraded - clean) ** 2))
        manifest["levels"][str(sigma)] = {"file": out_path.name, "mse_vs_clean": mse}
        print(f"[degradation] sigma={sigma}: saved {out_path.name}, shape={degraded.shape}, "
              f"mse_vs_clean={mse:.5f}", flush=True)

    with open(DATA_DIR / "degradation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[degradation] DONE, manifest written to {DATA_DIR / 'degradation_manifest.json'}")


if __name__ == "__main__":
    main()
