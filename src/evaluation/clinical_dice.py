"""
Phase 2 Part B4: downstream clinical-task evaluation harness.

Given ANY denoising model (a callable image_denoiser(noisy_tensor) ->
denoised_tensor, batched, (B,1,128,128) in [0,1]), this harness:
  1. Loads the fixed degraded phase2_test_5000 set at each sigma level
     (degradation.py's output).
  2. Denoises it with the given model.
  3. Passes the denoised images through the trained segmentation U-Net.
  4. Computes Dice and Hausdorff distance against the ground-truth masks.
  5. Compares against two reference points: (a) NO DENOISING (segment the
     raw degraded image directly) and (b) ORACLE (segment the true clean
     image - the upper bound achievable regardless of denoiser quality).

This is deliberately a MODEL-AGNOSTIC harness (registered via a small
adapter per model family - WI-QGAN's patch-based forward signature differs
from the classical baselines' plain image-in/image-out signature), not
hard-coded to one architecture, so the same script evaluates WI-QGAN,
DnCNN, Restormer-lite, and SwinIR-lite through one consistent protocol.

Usage:
    python eval_downstream.py --model none      # no-denoising reference
    python eval_downstream.py --model oracle     # oracle-clean reference
    python eval_downstream.py --model dncnn --checkpoint <path>
    python eval_downstream.py --model restormer --checkpoint <path>
    python eval_downstream.py --model swinir --checkpoint <path>
    python eval_downstream.py --model wiqgan --checkpoint <path> --n-qubits 6 --n-layers 5
    python eval_downstream.py --model wiqgan_growing_ring --checkpoint <path> \\
        --old-n-qubits 4 --old-n-layers 3 --new-n-qubits 6 --new-n-layers 5

Appends one row per sigma level to
~/host_share/qgan_results/wigan_phase2/downstream_results.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = SCRIPTS_DIR.parent / "models"
PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(PHASE2_DIR))

from unet_model import UNet2D, dice_coefficient  # noqa: E402
from dncnn_model import DnCNN  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402
from swinir_lite_model import SwinIRLite  # noqa: E402

DATA_DIR = Path(os.path.expanduser("~/host_share/qgan_data/wigan_datasets/phase2"))
RESULTS_DIR = Path(os.path.expanduser("~/host_share/qgan_results/wigan_phase2"))
UNET_CKPT = RESULTS_DIR / "segmentation_unet" / "checkpoints" / "unet_best.pt"
OUT_CSV = RESULTS_DIR / "downstream_results.csv"
SIGMA_LEVELS = [0.05, 0.10, 0.20]
EVAL_BATCH = 100


def hausdorff_distance(pred_mask, gt_mask):
    """Symmetric Hausdorff distance between two binary masks (H,W). Returns
    0 if both empty, and a large sentinel (image diagonal) if exactly one
    is empty (undefined distance to/from nothing)."""
    diag = np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)
    pred_empty = pred_mask.sum() == 0
    gt_empty = gt_mask.sum() == 0
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return diag
    dt_gt = distance_transform_edt(1 - gt_mask)
    dt_pred = distance_transform_edt(1 - pred_mask)
    d1 = dt_gt[pred_mask.astype(bool)].max()
    d2 = dt_pred[gt_mask.astype(bool)].max()
    return float(max(d1, d2))


def build_denoiser(args, device):
    """Returns a callable(noisy_np (N,128,128)) -> denoised_np (N,128,128),
    or None for the 'none'/'oracle' special cases handled separately."""
    if args.model == "none" or args.model == "oracle":
        return None
    if args.model == "dncnn":
        model = DnCNN().to(device)
    elif args.model == "restormer":
        model = RestormerLite().to(device)
    elif args.model == "swinir":
        model = SwinIRLite().to(device)
    elif args.model == "wiqgan":
        sys.path.insert(0, str(SCRIPTS_DIR))
        from qgan_models_wigan import WIQGANGenerator
        model = WIQGANGenerator(img_size=128, patch_size=8, n_qubits=args.n_qubits,
                                 n_layers=args.n_layers, noise_prob=0.0).to(device)
    elif args.model == "wiqgan_growing_ring":
        from inheritance_growing import GrowingRingQGANGenerator
        model = GrowingRingQGANGenerator(args.old_n_qubits, args.old_n_layers,
                                          args.new_n_qubits, args.new_n_layers,
                                          img_size=128, patch_size=8, noise_prob=0.0).to(device)
    else:
        raise ValueError(args.model)

    state = torch.load(os.path.expanduser(args.checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    is_wiqgan = args.model in ("wiqgan", "wiqgan_growing_ring")

    def denoiser(noisy_np):
        outs = []
        t = torch.from_numpy(noisy_np).unsqueeze(1)
        with torch.no_grad():
            for i in range(0, len(t), EVAL_BATCH):
                chunk = t[i:i + EVAL_BATCH].to(device)
                if is_wiqgan:
                    mask = model.sample_noise_mask(device, noise_prob=0.0)
                    out, _, _ = model(chunk, noise_mask=mask, patch_indices=None)
                else:
                    out = model(chunk)
                outs.append(out.cpu().numpy()[:, 0])
        return np.concatenate(outs, axis=0)

    return denoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["none", "oracle", "dncnn", "restormer", "swinir", "wiqgan", "wiqgan_growing_ring"])
    ap.add_argument("--model-label", default=None, help="override label written to CSV (default: --model)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--old-n-qubits", type=int, default=4)
    ap.add_argument("--old-n-layers", type=int, default=3)
    ap.add_argument("--new-n-qubits", type=int, default=6)
    ap.add_argument("--new-n-layers", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    label = args.model_label or args.model

    clean = np.load(DATA_DIR / "phase2_test_5000.npy").astype(np.float32)
    masks = np.load(DATA_DIR / "phase2_test_5000_mask.npy").astype(np.float32)

    unet = UNet2D().to(device)
    unet.load_state_dict(torch.load(UNET_CKPT, map_location=device, weights_only=True))
    unet.eval()

    def segment(images_np):
        preds = []
        t = torch.from_numpy(images_np).unsqueeze(1)
        with torch.no_grad():
            for i in range(0, len(t), EVAL_BATCH):
                chunk = t[i:i + EVAL_BATCH].to(device)
                p = unet(chunk).cpu().numpy()[:, 0]
                preds.append(p)
        return np.concatenate(preds, axis=0)

    denoiser = build_denoiser(args, device) if args.model not in ("none", "oracle") else None

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not OUT_CSV.exists()
    fields = ["model", "sigma", "mean_dice", "std_dice", "mean_hausdorff", "std_hausdorff", "n_images"]
    rows = []

    for sigma in SIGMA_LEVELS:
        tag = f"{sigma:.2f}".replace("0.", "0")
        degraded = np.load(DATA_DIR / f"degraded_sigma{tag}.npy").astype(np.float32)

        if args.model == "oracle":
            seg_input = clean
        elif args.model == "none":
            seg_input = degraded
        else:
            seg_input = denoiser(degraded)

        pred_probs = segment(seg_input)
        pred_binary = (pred_probs > 0.5).astype(np.float32)

        dices = dice_coefficient(pred_binary, masks)
        hausdorffs = np.array([hausdorff_distance(pred_binary[i], masks[i]) for i in range(len(masks))])

        rows.append({
            "model": label, "sigma": sigma,
            "mean_dice": round(float(dices.mean()), 4), "std_dice": round(float(dices.std()), 4),
            "mean_hausdorff": round(float(hausdorffs.mean()), 3), "std_hausdorff": round(float(hausdorffs.std()), 3),
            "n_images": len(masks),
        })
        print(f"[eval_downstream] {label} sigma={sigma}: Dice={dices.mean():.4f}+-{dices.std():.4f} "
              f"Hausdorff={hausdorffs.mean():.3f}+-{hausdorffs.std():.3f}", flush=True)

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[eval_downstream] appended {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
