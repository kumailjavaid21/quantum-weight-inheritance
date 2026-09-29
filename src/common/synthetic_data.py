"""
Go/No-Go Step 2, Task 2.1: controlled synthetic restoration task.

32x32 clean images = sums of 2-5 random 2D Gaussian blobs (random
centers, widths, amplitudes), normalized to [0,1]. Noisy versions add
Gaussian noise at sigma=0.10. Train: 4,000 samples; test: 500 samples.
Deliberately simple/synthetic (not BraTS) to isolate the weight-
inheritance morphism's optimization effect from medical-imaging
complexity, per the experiment design.
"""
import json
import os
from pathlib import Path

import numpy as np

OUT_DIR = Path(os.path.expanduser("~/host_share/qgan_data/go_nogo_synthetic"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 32
N_TRAIN = 4000
N_TEST = 500
NOISE_SIGMA = 0.10
SEED = 42


def make_gaussian_sum_image(rng, size=IMG_SIZE):
    n_blobs = rng.integers(2, 6)  # 2-5 blobs
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size), dtype=np.float64)
    for _ in range(n_blobs):
        cx = rng.uniform(4, size - 4)
        cy = rng.uniform(4, size - 4)
        wx = rng.uniform(2.0, 6.0)
        wy = rng.uniform(2.0, 6.0)
        amp = rng.uniform(0.5, 1.0)
        img += amp * np.exp(-(((xx - cx) ** 2) / (2 * wx ** 2) + ((yy - cy) ** 2) / (2 * wy ** 2)))
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def add_gaussian_noise(images, sigma, seed):
    rng = np.random.default_rng(seed)
    noisy = images + rng.normal(0.0, sigma, size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def build_split(n_images, seed):
    rng = np.random.default_rng(seed)
    clean = np.stack([make_gaussian_sum_image(rng) for _ in range(n_images)])
    return clean


def main():
    train_clean = build_split(N_TRAIN, seed=SEED)
    test_clean = build_split(N_TEST, seed=SEED + 1)

    train_noisy = add_gaussian_noise(train_clean, NOISE_SIGMA, seed=SEED + 100)
    test_noisy = add_gaussian_noise(test_clean, NOISE_SIGMA, seed=SEED + 101)

    np.save(OUT_DIR / "train_clean.npy", train_clean)
    np.save(OUT_DIR / "train_noisy.npy", train_noisy)
    np.save(OUT_DIR / "test_clean.npy", test_clean)
    np.save(OUT_DIR / "test_noisy.npy", test_noisy)

    manifest = {
        "img_size": IMG_SIZE, "n_train": N_TRAIN, "n_test": N_TEST,
        "noise_sigma": NOISE_SIGMA, "seed": SEED,
        "description": "sums of 2-5 random 2D Gaussian blobs, normalized to [0,1], "
                        "Gaussian noise sigma=0.10 added and clipped to [0,1]",
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[synthetic_data] train_clean={train_clean.shape} test_clean={test_clean.shape}")
    print(f"[synthetic_data] train noisy PSNR proxy MSE={np.mean((train_clean-train_noisy)**2):.5f}")
    print(f"[synthetic_data] wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
