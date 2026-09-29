"""
WI-QGAN Phase 1 training script: trains WIQGANGenerator (qgan_models_wigan)
on BraTS2021 128x128 FLAIR slices, either from random initialization (Runs
A/B/C, independent baselines) or from an inherited checkpoint at a smaller
qubit/layer count (Runs D/E, growth runs), per the Phase 1 de-risking
experiment spec.

Adapted from Project A's train_qgan_pauli_noise.py:
  - img_size 64 -> 128, patches 8x8 -> 256/image (16x16 grid)
  - --patch-subsample K: at each gradient step, only K of the 256 patches
    (shared across the batch, freshly sampled every step) are pushed
    through the quantum encoder/decoder. The other patches pass through
    UNCHANGED (still noisy) into the folded image fed to the discriminator,
    so the GAN structure (adversarial + lambda_rec * L1) is preserved
    exactly as in Project A, but the L1 reconstruction term is computed
    ONLY over the K processed patches (comparing against the clean patches
    at the same K locations) - the untouched patches would otherwise
    dilute/bias that loss toward "no-op" since they trivially equal the
    noisy input, not the clean target.
  - --init-from: path to an existing generator checkpoint at a SMALLER
    (--init-from-n-qubits, --init-from-n-layers). If given, the new
    generator is built at --n-qubits/--n-layers and then
    qgan_models_wigan.inherit_weights() transplants the old weights in,
    per design_notes.md's mapping. Before any training step, the
    full-image (all-patches) L1 reconstruction loss on the validation
    subsample is measured and written to inheritance_log.json - this is
    the "loss immediately after inheritance, before any new training"
    quantity Phase 1 Task 3 asks for. Epoch 1's validation is always
    additionally run (regardless of --eval-every) so "loss after 1 epoch
    of adaptation" is always captured too.

Usage:
    # independent baseline (Run A)
    python train_wigan_phase1.py --n-qubits 4 --n-layers 3 \
        --out-dir wigan_phase1/A_4q3L --epochs 30

    # inheritance run (Run D, growing from Run A's checkpoint)
    python train_wigan_phase1.py --n-qubits 6 --n-layers 5 \
        --init-from ~/host_share/qgan_results/wigan_phase1/A_4q3L/checkpoints/generator_last.pt \
        --init-from-n-qubits 4 --init-from-n-layers 3 \
        --out-dir wigan_phase1/D_inherit_4to6 --epochs 30
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from qgan_models_wigan import WIQGANGenerator, PatchGANDiscriminator, inherit_weights
from qgan_utils_wigan import add_noise, batch_psnr_ssim, seed_everything
from wigan_data import load_split_subsample

IMG_SIZE = 128
PATCH_SIZE = 8


class DenoiseDataset(Dataset):
    def __init__(self, images, seed):
        clean = images.astype(np.float32)
        noisy = add_noise(clean, gaussian_sigma=0.1, seed=seed)
        self.clean = torch.from_numpy(clean).unsqueeze(1)
        self.noisy = torch.from_numpy(noisy).unsqueeze(1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def full_image_rec_loss(generator, loader, device):
    """Full-image (all 256 patches, no subsampling) mean L1 reconstruction
    loss - the metric used for the loss-jump analysis, chosen so it is
    comparable across the subsampled-training runs and unaffected by which
    patches happened to be sampled on a given step."""
    generator.eval()
    l1 = nn.L1Loss()
    losses = []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            mask = generator.sample_noise_mask(device, noise_prob=generator.noise_prob)
            denoised, _, _ = generator(noisy, noise_mask=mask, patch_indices=None)
            losses.append(l1(denoised, clean).item())
    generator.train()
    return float(np.mean(losses))


def evaluate(generator, loader, device):
    generator.eval()
    all_clean, all_noisy, all_denoised = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            mask = generator.sample_noise_mask(device, noise_prob=generator.noise_prob)
            denoised, _, _ = generator(noisy, noise_mask=mask, patch_indices=None)
            all_clean.append(clean.cpu().numpy()[:, 0])
            all_noisy.append(noisy.cpu().numpy()[:, 0])
            all_denoised.append(denoised.cpu().numpy()[:, 0])
    generator.train()
    clean = np.concatenate(all_clean)
    noisy = np.concatenate(all_noisy)
    denoised = np.concatenate(all_denoised)
    psnr_noisy, ssim_noisy = batch_psnr_ssim(clean, noisy)
    psnr_den, ssim_den = batch_psnr_ssim(clean, denoised)
    return {
        "psnr_noisy": psnr_noisy, "ssim_noisy": ssim_noisy,
        "psnr_denoised": psnr_den, "ssim_denoised": ssim_den,
        "delta_psnr": psnr_den - psnr_noisy, "delta_ssim": ssim_den - ssim_noisy,
    }, (noisy, denoised, clean)


def save_qualitative(noisy, denoised, clean, path, n=5):
    n = min(n, len(clean))
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.6))
    for i in range(n):
        axes[0, i].imshow(noisy[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(denoised[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(clean[i], cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            axes[r, i].axis("off")
    fig.suptitle("Noisy vs Denoised vs Clean (WI-QGAN Phase 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="relative to ~/host_share/qgan_results/")
    ap.add_argument("--n-qubits", type=int, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--patch-subsample", type=int, default=64,
                     help="patches processed per image per training step (of 256 total); 0 = all")
    ap.add_argument("--n-train-subsample", type=int, default=8000)
    ap.add_argument("--n-val-subsample", type=int, default=1500)
    ap.add_argument("--n-test-subsample", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-rec", type=float, default=70.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--noise-prob", type=float, default=0.05,
                     help="per-qubit-per-layer training-time Pauli noise probability")
    ap.add_argument("--init-from", default=None, help="checkpoint path to inherit weights from")
    ap.add_argument("--init-from-n-qubits", type=int, default=None)
    ap.add_argument("--init-from-n-layers", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    out_dir = Path(os.path.expanduser("~/host_share/qgan_results")) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    train_imgs, train_masks, _ = load_split_subsample("train", args.n_train_subsample, seed=args.seed)
    val_imgs, val_masks, _ = load_split_subsample("val", args.n_val_subsample, seed=args.seed)
    test_imgs, test_masks, _ = load_split_subsample("test", args.n_test_subsample, seed=args.seed)

    train_ds = DenoiseDataset(train_imgs, seed=args.seed)
    val_ds = DenoiseDataset(val_imgs, seed=args.seed + 1)
    test_ds = DenoiseDataset(test_imgs, seed=args.seed + 2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[train_wigan_phase1] out_dir={out_dir} n_qubits={args.n_qubits} n_layers={args.n_layers} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device} "
          f"patch_subsample={args.patch_subsample} noise_prob={args.noise_prob} "
          f"init_from={args.init_from}", flush=True)

    generator = WIQGANGenerator(img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                                 n_qubits=args.n_qubits, n_layers=args.n_layers,
                                 noise_prob=args.noise_prob).to(device)

    inheritance_log = None
    if args.init_from:
        assert args.init_from_n_qubits and args.init_from_n_layers, \
            "--init-from-n-qubits and --init-from-n-layers are required with --init-from"
        old_state = torch.load(os.path.expanduser(args.init_from), map_location=device, weights_only=True)
        new_state = inherit_weights(old_state, args.init_from_n_qubits, args.init_from_n_layers, generator)
        generator.load_state_dict(new_state, strict=True)
        loss_after_inherit = full_image_rec_loss(generator, val_loader, device)
        n_old_qparams = args.init_from_n_layers * args.init_from_n_qubits * 3
        n_new_qparams = args.n_layers * args.n_qubits * 3
        inheritance_log = {
            "init_from": args.init_from,
            "old_n_qubits": args.init_from_n_qubits, "old_n_layers": args.init_from_n_layers,
            "new_n_qubits": args.n_qubits, "new_n_layers": args.n_layers,
            "loss_immediately_after_inheritance": loss_after_inherit,
            "n_quantum_params_inherited": n_old_qparams,
            "n_quantum_params_new": n_new_qparams - n_old_qparams,
            "n_quantum_params_total_new_model": n_new_qparams,
        }
        print(f"[train_wigan_phase1] INHERITED {args.init_from_n_qubits}q/{args.init_from_n_layers}L -> "
              f"{args.n_qubits}q/{args.n_layers}L | full-image val L1 rec_loss immediately after "
              f"inheritance (before any training) = {loss_after_inherit:.6f}", flush=True)
        with open(out_dir / "inheritance_log.json", "w") as f:
            json.dump(inheritance_log, f, indent=2)

    discriminator = PatchGANDiscriminator(in_channels=2).to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    n_patches = generator.n_patches
    K = args.patch_subsample if args.patch_subsample and args.patch_subsample < n_patches else None

    log_path = out_dir / "train_log.csv"
    log_fields = ["epoch", "d_loss", "g_loss", "adv_loss", "rec_loss_subsampled", "epoch_time_s",
                  "val_full_rec_loss", "val_psnr_noisy", "val_ssim_noisy", "val_psnr_denoised",
                  "val_ssim_denoised", "val_delta_psnr", "val_delta_ssim"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    best_val_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        d_losses, g_losses, adv_losses, rec_losses = [], [], [], []

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            patch_indices = None
            if K is not None:
                patch_indices = torch.randperm(n_patches, device=device)[:K]
            mask = generator.sample_noise_mask(device)

            with torch.no_grad():
                denoised_full, _, _ = generator(noisy, noise_mask=mask, patch_indices=patch_indices)
            d_real = discriminator(noisy, clean)
            d_fake = discriminator(noisy, denoised_full)
            real_labels = torch.full_like(d_real, 0.9)
            fake_labels = torch.zeros_like(d_fake)
            d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            denoised_full, denoised_sel, _ = generator(noisy, noise_mask=mask, patch_indices=patch_indices)
            d_fake_for_g = discriminator(noisy, denoised_full)
            adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))

            clean_patches, _ = generator.extract_patches(clean)
            clean_sel = clean_patches if patch_indices is None else clean_patches[:, patch_indices, :]
            rec_loss = l1(denoised_sel, clean_sel)

            g_loss = adv_loss + args.lambda_rec * rec_loss
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            adv_losses.append(adv_loss.item())
            rec_losses.append(rec_loss.item())

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch, "d_loss": np.mean(d_losses), "g_loss": np.mean(g_losses),
            "adv_loss": np.mean(adv_losses), "rec_loss_subsampled": np.mean(rec_losses),
            "epoch_time_s": epoch_time,
            "val_full_rec_loss": "", "val_psnr_noisy": "", "val_ssim_noisy": "",
            "val_psnr_denoised": "", "val_ssim_denoised": "", "val_delta_psnr": "", "val_delta_ssim": "",
        }

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs) or (epoch == 1)
        if do_eval:
            val_full_rec = full_image_rec_loss(generator, val_loader, device)
            metrics, (v_noisy, v_denoised, v_clean) = evaluate(generator, val_loader, device)
            row.update({
                "val_full_rec_loss": val_full_rec,
                "val_psnr_noisy": metrics["psnr_noisy"], "val_ssim_noisy": metrics["ssim_noisy"],
                "val_psnr_denoised": metrics["psnr_denoised"], "val_ssim_denoised": metrics["ssim_denoised"],
                "val_delta_psnr": metrics["delta_psnr"], "val_delta_ssim": metrics["delta_ssim"],
            })
            print(f"[epoch {epoch}/{args.epochs}] d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                  f"rec_sub={row['rec_loss_subsampled']:.4f} val_full_rec={val_full_rec:.4f} | "
                  f"PSNR {metrics['psnr_noisy']:.2f}->{metrics['psnr_denoised']:.2f} "
                  f"(dPSNR={metrics['delta_psnr']:+.2f}) [{epoch_time:.1f}s]", flush=True)
            if metrics["psnr_denoised"] > best_val_psnr:
                best_val_psnr = metrics["psnr_denoised"]
                torch.save(generator.state_dict(), ckpt_dir / "generator_best.pt")
                torch.save(discriminator.state_dict(), ckpt_dir / "discriminator_best.pt")
                save_qualitative(v_noisy, v_denoised, v_clean, fig_dir / f"val_epoch{epoch:03d}.png")
        else:
            print(f"[epoch {epoch}/{args.epochs}] d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                  f"rec_sub={row['rec_loss_subsampled']:.4f} [{epoch_time:.1f}s]", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in log_fields])

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(generator.state_dict(), ckpt_dir / f"generator_epoch{epoch:03d}.pt")
        torch.save(generator.state_dict(), ckpt_dir / "generator_last.pt")
        torch.save(discriminator.state_dict(), ckpt_dir / "discriminator_last.pt")

    final_val_full_rec = full_image_rec_loss(generator, val_loader, device)
    test_metrics, (t_noisy, t_denoised, t_clean) = evaluate(generator, test_loader, device)
    save_qualitative(t_noisy, t_denoised, t_clean, fig_dir / "test_final.png", n=8)

    n_qparams = args.n_layers * args.n_qubits * 3
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "final_val_full_rec_loss": final_val_full_rec,
                   "n_qubits": args.n_qubits, "n_layers": args.n_layers,
                   "n_quantum_params": n_qparams,
                   "train_samples": len(train_ds), "val_samples": len(val_ds),
                   "test_samples": len(test_ds), "epochs": args.epochs,
                   "batch_size": args.batch_size, "noise_prob": args.noise_prob,
                   "init_from": args.init_from}, f, indent=2)

    print(f"[train_wigan_phase1] FINAL TEST: PSNR {test_metrics['psnr_noisy']:.2f} -> "
          f"{test_metrics['psnr_denoised']:.2f} (dPSNR={test_metrics['delta_psnr']:+.2f}), "
          f"SSIM {test_metrics['ssim_noisy']:.3f} -> {test_metrics['ssim_denoised']:.3f} "
          f"final_val_full_rec_loss={final_val_full_rec:.6f} n_quantum_params={n_qparams}", flush=True)
    print(f"[train_wigan_phase1] results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
