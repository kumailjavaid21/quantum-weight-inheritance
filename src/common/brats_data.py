"""
BraTS2021 128x128 data loading for WI-QGAN Phase 1, with the "deliberately
limited" stratified subsampling used to keep this de-risking experiment
within its compute budget:
  - train: 8,000 slices (spec'd budget), stratified on tumor-mask presence
    to preserve the full training set's ~49% tumor-positive ratio.
  - val: 1,500 slices (not spec'd; chosen so periodic in-training
    validation stays cheap - full 24,667-image val would dominate wall
    clock at 5-epoch eval cadence across 5 models). Logged here explicitly.
  - test: 2,000 slices (not spec'd; chosen so Task 4's robustness sweep -
    5 models x 4 noise levels x 10 mask repetitions - stays tractable;
    the full 24,279-image test set would mean ~4.9M image denoise+metric
    evaluations). Logged here explicitly.
All subsample indices are seeded (default 42) and IDENTICAL across every
run (A-E) and every script that calls these functions, so all five models
are compared on exactly the same data.
"""
import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.path.expanduser("~/host_share/qgan_data/wigan_datasets/brats2021_128"))


def stratified_indices(masks, n, seed):
    """masks: (N, H, W) array-like (uint8/bool). Returns n indices sampled
    without replacement, preserving the tumor-positive fraction of `masks`."""
    tumor_pos = masks.reshape(len(masks), -1).any(axis=1)
    pos_idx = np.where(tumor_pos)[0]
    neg_idx = np.where(~tumor_pos)[0]
    frac_pos = len(pos_idx) / len(masks)
    n_pos = min(int(round(n * frac_pos)), len(pos_idx))
    n_neg = min(n - n_pos, len(neg_idx))
    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(idx)
    return idx


def load_split_subsample(split, n, seed=42, data_dir=DATA_DIR):
    """split: 'train' | 'val' | 'test'. Loads the full masks array (needed to
    stratify), full images array is memory-mapped so only the selected rows
    are actually read into RAM. Returns (images (n,128,128) float32,
    masks (n,128,128) uint8, subject_ids (list[str] len n))."""
    import json
    masks = np.load(data_dir / f"brats2021_128_{split}_mask.npy")
    idx = stratified_indices(masks, n, seed)
    images_mm = np.load(data_dir / f"brats2021_128_{split}.npy", mmap_mode="r")
    images = np.array(images_mm[idx])
    masks_sub = masks[idx]
    with open(data_dir / f"brats2021_128_{split}_subject_ids.json") as f:
        all_ids = json.load(f)
    subj_ids = [all_ids[i] for i in idx]
    return images, masks_sub, subj_ids
