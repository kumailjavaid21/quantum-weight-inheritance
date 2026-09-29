"""
WI-QGAN Phase 1 Task 4: robustness evaluation of the five trained models
(A, B, C, D, E) under Pauli inference-time noise, adapted from Project A's
paper_scripts/phase6_robustness_stats.py protocol:
  - eval noise levels p in {0.0, 0.05, 0.10, 0.20}
  - N_REPS = 10 independently-sampled Pauli masks per (model, noise level),
    except p=0.0 where the mask is deterministically all-zero (1 rep,
    reported honestly as std=0 rather than padded).
  - QNRS(p) = mean_delta_psnr(p) / mean_delta_psnr(0.0), per model, guarded
    against a near-zero/negative ideal-case denominator exactly as in
    Project A (QNRS left blank with a note in that case).

Evaluated on the SAME 2,000-slice stratified test subsample used throughout
Phase 1 (wigan_data.load_split_subsample("test", 2000, seed=42)) so results
are directly comparable across models and consistent with training/val
subsampling choices.

Output: ~/host_share/qgan_results/wigan_phase1/phase1_robustness.csv
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch

from qgan_models_wigan import WIQGANGenerator
from qgan_utils_wigan import add_noise, batch_psnr_ssim
from wigan_data import load_split_subsample

N_REPS = 10
EVAL_LEVELS = [0.0, 0.05, 0.10, 0.20]
QNRS_DENOM_MIN_ABS = 0.05
IMG_SIZE = 128
PATCH_SIZE = 8
EVAL_BATCH = 100

RESULTS = Path(os.path.expanduser("~/host_share/qgan_results/wigan_phase1"))
OUT_CSV = RESULTS / "phase1_robustness.csv"

MODELS = [
    # (model_name, n_qubits, n_layers, checkpoint_path)
    ("A_4q3L_independent", 4, 3, RESULTS / "A_4q3L" / "checkpoints" / "generator_best.pt"),
    ("B_6q5L_independent", 6, 5, RESULTS / "B_6q5L" / "checkpoints" / "generator_best.pt"),
    ("C_8q7L_independent", 8, 7, RESULTS / "C_8q7L" / "checkpoints" / "generator_best.pt"),
    ("D_inherit_4to6", 6, 5, RESULTS / "D_inherit_4to6" / "checkpoints" / "generator_best.pt"),
    ("E_inherit_6to8", 8, 7, RESULTS / "E_inherit_6to8" / "checkpoints" / "generator_best.pt"),
]


def denoise_in_batches(generator, noisy_t, device, noise_mask):
    """Runs generator.forward over noisy_t in chunks of EVAL_BATCH images
    (all 256 patches/image, no subsampling) to keep the single-call batch
    size bounded regardless of test-set size or qubit count."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(noisy_t), EVAL_BATCH):
            chunk = noisy_t[i:i + EVAL_BATCH].to(device)
            denoised, _, _ = generator(chunk, noise_mask=noise_mask, patch_indices=None)
            outs.append(denoised.cpu().numpy()[:, 0])
    return np.concatenate(outs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean, _, _ = load_split_subsample("test", 2000, seed=42)
    noisy = add_noise(clean, gaussian_sigma=0.1, seed=44)
    noisy_t = torch.from_numpy(noisy).unsqueeze(1)
    psnr_n, ssim_n = batch_psnr_ssim(clean, noisy)
    print(f"[eval_wigan_robustness] device={device} n_test={len(clean)} eval_batch={EVAL_BATCH}", flush=True)

    rows = []
    for model_name, n_qubits, n_layers, ckpt_path in MODELS:
        if not ckpt_path.exists():
            print(f"[eval_wigan_robustness] SKIP {model_name} - checkpoint missing: {ckpt_path}")
            continue
        generator = WIQGANGenerator(img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                                     n_qubits=n_qubits, n_layers=n_layers, noise_prob=0.0).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        generator.load_state_dict(state, strict=True)
        generator.eval()

        model_rows = {}
        for eval_noise in EVAL_LEVELS:
            n_reps = 1 if eval_noise == 0.0 else N_REPS
            psnrs, ssims = [], []
            for rep in range(n_reps):
                torch.manual_seed(10_000 + rep)
                mask = generator.sample_noise_mask(device, noise_prob=eval_noise)
                denoised = denoise_in_batches(generator, noisy_t, device, mask)
                pd, sd = batch_psnr_ssim(clean, denoised)
                psnrs.append(pd)
                ssims.append(sd)
            psnrs, ssims = np.array(psnrs), np.array(ssims)
            delta_psnrs = psnrs - psnr_n
            delta_ssims = ssims - ssim_n
            model_rows[eval_noise] = {
                "mean_psnr": float(psnrs.mean()), "std_psnr": float(psnrs.std()),
                "mean_ssim": float(ssims.mean()), "std_ssim": float(ssims.std()),
                "mean_delta_psnr": float(delta_psnrs.mean()), "std_delta_psnr": float(delta_psnrs.std()),
                "mean_delta_ssim": float(delta_ssims.mean()), "std_delta_ssim": float(delta_ssims.std()),
                "n_reps": n_reps,
            }
            print(f"[eval_wigan_robustness] {model_name} eval_noise={eval_noise}: "
                  f"PSNR={model_rows[eval_noise]['mean_psnr']:.3f}+-{model_rows[eval_noise]['std_psnr']:.4f} "
                  f"dPSNR={model_rows[eval_noise]['mean_delta_psnr']:+.4f}", flush=True)

        ideal_dpsnr = model_rows[0.0]["mean_delta_psnr"]
        for eval_noise in EVAL_LEVELS:
            r = model_rows[eval_noise]
            if abs(ideal_dpsnr) < QNRS_DENOM_MIN_ABS:
                qnrs, qnrs_note = "", f"ideal dPSNR too close to 0 ({ideal_dpsnr:+.4f})"
            else:
                qnrs, qnrs_note = round(r["mean_delta_psnr"] / ideal_dpsnr, 4), ""
            rows.append({
                "model": model_name, "n_qubits": n_qubits, "n_layers": n_layers,
                "eval_noise": eval_noise, "n_reps": r["n_reps"],
                "mean_psnr": round(r["mean_psnr"], 4), "std_psnr": round(r["std_psnr"], 4),
                "mean_ssim": round(r["mean_ssim"], 5), "std_ssim": round(r["std_ssim"], 5),
                "mean_delta_psnr": round(r["mean_delta_psnr"], 4), "std_delta_psnr": round(r["std_delta_psnr"], 4),
                "mean_delta_ssim": round(r["mean_delta_ssim"], 5), "std_delta_ssim": round(r["std_delta_ssim"], 5),
                "qnrs": qnrs, "qnrs_note": qnrs_note,
            })

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = ["model", "n_qubits", "n_layers", "eval_noise", "n_reps", "mean_psnr", "std_psnr",
              "mean_ssim", "std_ssim", "mean_delta_psnr", "std_delta_psnr", "mean_delta_ssim",
              "std_delta_ssim", "qnrs", "qnrs_note"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[eval_wigan_robustness] wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
