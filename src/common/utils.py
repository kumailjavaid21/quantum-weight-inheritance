"""Shared utilities for WI-QGAN Phase 1 - copied verbatim from
~/qgan_project/qgan_utils.py (Project A) so wigan_project/scripts/ is
self-contained and Project A's own files are never modified or imported
across project boundaries.
"""
import random

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise(images, gaussian_sigma=0.1, poisson_scale=1.0, seed=None):
    """Add Gaussian(0, sigma) + Poisson shot noise to a batch of [0,1] images."""
    r = np.random.default_rng(seed)
    images = images.astype(np.float32)
    peak = 255.0 * poisson_scale
    poisson_noisy = r.poisson(images * peak) / peak
    gaussian_noise = r.normal(0.0, gaussian_sigma, size=images.shape)
    noisy = poisson_noisy + gaussian_noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def batch_psnr_ssim(clean, other):
    """clean, other: numpy arrays (N, H, W) in [0,1]. Returns mean (psnr, ssim)."""
    psnrs, ssims = [], []
    for c, o in zip(clean, other):
        psnrs.append(sk_psnr(c, o, data_range=1.0))
        ssims.append(sk_ssim(c, o, data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))
