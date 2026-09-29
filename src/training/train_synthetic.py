"""
Go/No-Go Step 2: trains one of the 6 model variants on the synthetic
sums-of-Gaussians restoration task. Reuses (read-only import, not
modified) the EXISTING, already-validated WI-QGAN generator
(qgan_models_wigan.WIQGANGenerator) and growing-ring morphism
(models/inheritance_growing.GrowingRingQGANGenerator +
inherit_weights_growing_ring) from wigan_project - this experiment tests
whether THAT SAME validated morphism gives an optimization advantage on
a clean, controlled task, not a reimplementation that could subtly
differ from what Phase 1.5 actually proved exact.

Simplified per the experiment design: no discriminator, no adversarial
loss - pure L1 reconstruction loss only, trained for a FIXED NUMBER OF
STEPS (not epochs) so step-level metrics (loss at each of the first 10
steps, steps-to-reach-X-loss) are well-defined and comparable across
variants regardless of dataset/batch-size details.

Modes:
  scratch        - fresh WIQGANGenerator at (n_qubits, n_layers), random init.
  random_grow    - WIQGANGenerator at (n_qubits, n_layers); OLD qubits/layers'
                   weights copied from --init-from's checkpoint (matching
                   inherit_weights' old-portion mapping), NEW qubits/layers
                   and new encoder/decoder slices left at their random
                   from-scratch initialization (Model 2's "random new-gate
                   initialization" control).
  morphism_grow  - GrowingRingQGANGenerator, inherit_weights_growing_ring
                   applied (Model 3/4/5's exact-identity morphism).

Usage:
    python train_go_nogo.py --mode scratch --n-qubits 8 --n-layers 7 --seed 42 --out-dir model1
    python train_go_nogo.py --mode random_grow --n-qubits 8 --n-layers 7 \\
        --init-from .../base_4q3L_seed42/checkpoints/generator_last.pt \\
        --old-n-qubits 4 --old-n-layers 3 --seed 42 --out-dir model2
    python train_go_nogo.py --mode morphism_grow --n-qubits 8 --n-layers 7 \\
        --init-from .../base_4q3L_seed42/checkpoints/generator_last.pt \\
        --old-n-qubits 4 --old-n-layers 3 --seed 42 --out-dir model3
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WIGAN_SCRIPTS = Path(os.path.expanduser("~/qgan_project/wigan_project/scripts"))
WIGAN_MODELS = Path(os.path.expanduser("~/qgan_project/wigan_project/models"))
sys.path.insert(0, str(WIGAN_SCRIPTS))
sys.path.insert(0, str(WIGAN_MODELS))

from qgan_models_wigan import WIQGANGenerator, inherit_weights  # noqa: E402
from inheritance_growing import GrowingRingQGANGenerator, inherit_weights_growing_ring  # noqa: E402
from qgan_utils_wigan import seed_everything  # noqa: E402

DATA_DIR = Path(os.path.expanduser("~/host_share/qgan_data/go_nogo_synthetic"))
RESULTS_BASE = Path(os.path.expanduser("~/host_share/qgan_results/quantum_project/go_nogo"))

IMG_SIZE = 32
PATCH_SIZE = 8
TOTAL_STEPS = 2000
BATCH_SIZE = 32
LR = 2e-4
N_GRAD_PROBES = 8  # number of independent minibatches used to estimate grad norm/variance at expansion


def inherit_weights_random(old_state, old_n_qubits, old_n_layers, new_model):
    """Model 2's control: SAME old-portion copy as inherit_weights, but the
    new_model's OWN random initialization is left untouched everywhere else
    (no zero-init of new components) - i.e. only the overlap with the old
    model is overwritten; everything new stays at its random draw."""
    new_state = {k: v.clone() for k, v in new_model.state_dict().items()}

    new_qw = new_state["q_weights"].clone()
    new_qw[:old_n_layers, :old_n_qubits, :] = old_state["q_weights"]
    new_state["q_weights"] = new_qw

    new_ew = new_state["encoder.2.weight"].clone()
    new_eb = new_state["encoder.2.bias"].clone()
    new_ew[:old_n_qubits, :] = old_state["encoder.2.weight"]
    new_eb[:old_n_qubits] = old_state["encoder.2.bias"]
    new_state["encoder.2.weight"] = new_ew
    new_state["encoder.2.bias"] = new_eb

    new_dw = new_state["decoder.0.weight"].clone()
    new_dw[:, :old_n_qubits] = old_state["decoder.0.weight"]
    new_state["decoder.0.weight"] = new_dw
    new_state["decoder.0.bias"] = old_state["decoder.0.bias"].clone()

    for key in ["encoder.0.weight", "encoder.0.bias", "decoder.2.weight", "decoder.2.bias"]:
        new_state[key] = old_state[key].clone()

    return new_state


def load_data(device):
    train_clean = torch.from_numpy(np.load(DATA_DIR / "train_clean.npy").astype(np.float32)).unsqueeze(1)
    train_noisy = torch.from_numpy(np.load(DATA_DIR / "train_noisy.npy").astype(np.float32)).unsqueeze(1)
    test_clean = torch.from_numpy(np.load(DATA_DIR / "test_clean.npy").astype(np.float32)).unsqueeze(1)
    test_noisy = torch.from_numpy(np.load(DATA_DIR / "test_noisy.npy").astype(np.float32)).unsqueeze(1)
    return train_clean.to(device), train_noisy.to(device), test_clean.to(device), test_noisy.to(device)


def sample_batch(clean, noisy, batch_size, rng):
    idx = rng.choice(len(clean), size=batch_size, replace=False)
    idx_t = torch.from_numpy(idx).long()
    return noisy[idx_t], clean[idx_t]


def full_l1_loss(generator, noisy, clean, device):
    generator.eval()
    l1 = nn.L1Loss()
    with torch.no_grad():
        mask = generator.sample_noise_mask(device, noise_prob=0.0)
        denoised, _, _ = generator(noisy, noise_mask=mask, patch_indices=None)
        loss = l1(denoised, clean).item()
    generator.train()
    return loss


def grad_norm_and_variance(generator, clean, noisy, device, batch_size, n_probes, rng):
    """Gradient norm/variance AT the expansion point: n_probes independent
    minibatches evaluated at the SAME (post-inheritance, pre-update)
    parameter state."""
    l1 = nn.L1Loss()
    params = [p for p in generator.parameters() if p.requires_grad]
    norms = []
    for _ in range(n_probes):
        b_noisy, b_clean = sample_batch(clean, noisy, batch_size, rng)
        generator.zero_grad(set_to_none=True)
        mask = generator.sample_noise_mask(device, noise_prob=0.0)
        denoised, _, _ = generator(b_noisy, noise_mask=mask, patch_indices=None)
        loss = l1(denoised, b_clean)
        loss.backward()
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.detach().pow(2).sum().item()
        norms.append(total_norm ** 0.5)
    generator.zero_grad(set_to_none=True)
    return float(np.mean(norms)), float(np.var(norms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["scratch", "random_grow", "morphism_grow"])
    ap.add_argument("--n-qubits", type=int, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--old-n-qubits", type=int, default=None)
    ap.add_argument("--old-n-layers", type=int, default=None)
    ap.add_argument("--out-dir", required=True, help="relative to ~/host_share/qgan_results/quantum_project/go_nogo/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    out_dir = RESULTS_BASE / args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_clean, train_noisy, test_clean, test_noisy = load_data(device)
    print(f"[train_go_nogo] mode={args.mode} n_qubits={args.n_qubits} n_layers={args.n_layers} "
          f"seed={args.seed} out_dir={out_dir} train={len(train_clean)} test={len(test_clean)} "
          f"device={device}", flush=True)

    n_circuit_evals_per_step = None
    n_quantum_params_inherited = 0
    n_quantum_params_new = 0

    if args.mode == "scratch":
        generator = WIQGANGenerator(img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                                     n_qubits=args.n_qubits, n_layers=args.n_layers, noise_prob=0.0).to(device)
        n_quantum_params_new = args.n_layers * args.n_qubits * 3
    elif args.mode == "random_grow":
        assert args.init_from and args.old_n_qubits and args.old_n_layers
        old_state = torch.load(os.path.expanduser(args.init_from), map_location=device, weights_only=True)
        generator = WIQGANGenerator(img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                                     n_qubits=args.n_qubits, n_layers=args.n_layers, noise_prob=0.0).to(device)
        new_state = inherit_weights_random(old_state, args.old_n_qubits, args.old_n_layers, generator)
        generator.load_state_dict(new_state, strict=True)
        n_quantum_params_inherited = args.old_n_layers * args.old_n_qubits * 3
        n_quantum_params_new = args.n_layers * args.n_qubits * 3 - n_quantum_params_inherited
    else:  # morphism_grow
        assert args.init_from and args.old_n_qubits and args.old_n_layers
        old_state = torch.load(os.path.expanduser(args.init_from), map_location=device, weights_only=True)
        generator = GrowingRingQGANGenerator(args.old_n_qubits, args.old_n_layers, args.n_qubits, args.n_layers,
                                              img_size=IMG_SIZE, patch_size=PATCH_SIZE, noise_prob=0.0).to(device)
        new_state = inherit_weights_growing_ring(old_state, args.old_n_qubits, args.old_n_layers, generator)
        generator.load_state_dict(new_state, strict=True)
        n_quantum_params_inherited = args.old_n_layers * args.old_n_qubits * 3
        n_quantum_params_new = (args.n_layers * args.n_qubits * 3 - n_quantum_params_inherited
                                 + len(generator.edges))

    n_patches = generator.n_patches
    n_circuit_evals_per_step = BATCH_SIZE * n_patches  # individual patch-circuit evaluations per step

    # (a) initial loss at t=0, before any training step
    initial_loss = full_l1_loss(generator, test_noisy, test_clean, device)
    initial_train_loss = full_l1_loss(generator, train_noisy[:500], train_clean[:500], device)
    print(f"[train_go_nogo] initial (t=0) test_loss={initial_loss:.6f} train_loss(subset)={initial_train_loss:.6f}",
          flush=True)

    # gradient norm/variance AT the moment of expansion (before any update)
    grad_norm_mean, grad_norm_var = grad_norm_and_variance(
        generator, train_clean, train_noisy, device, BATCH_SIZE, N_GRAD_PROBES, rng)
    print(f"[train_go_nogo] grad_norm_at_expansion mean={grad_norm_mean:.6f} var={grad_norm_var:.8f}", flush=True)

    opt = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    log_path = out_dir / "step_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "step_time_s"])

    first10_losses = []
    t_start = time.time()
    best_test_loss = initial_loss

    for step in range(1, args.steps + 1):
        t0 = time.time()
        b_noisy, b_clean = sample_batch(train_clean, train_noisy, BATCH_SIZE, rng)
        mask = generator.sample_noise_mask(device, noise_prob=0.0)
        denoised, _, _ = generator(b_noisy, noise_mask=mask, patch_indices=None)
        loss = l1(denoised, b_clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_time = time.time() - t0

        loss_val = loss.item()
        if step <= 10:
            first10_losses.append(loss_val)

        if step % 20 == 0 or step == 1 or step == args.steps:
            test_loss = full_l1_loss(generator, test_noisy, test_clean, device)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(generator.state_dict(), ckpt_dir / "generator_best.pt")
            print(f"[step {step}/{args.steps}] train_loss={loss_val:.6f} test_loss={test_loss:.6f} "
                  f"[{step_time:.3f}s]", flush=True)
        else:
            test_loss = ""

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, loss_val, step_time])

    total_wall_time = time.time() - t_start
    torch.save(generator.state_dict(), ckpt_dir / "generator_last.pt")

    final_test_loss = full_l1_loss(generator, test_noisy, test_clean, device)
    total_circuit_evals = args.steps * n_circuit_evals_per_step

    summary = {
        "mode": args.mode, "n_qubits": args.n_qubits, "n_layers": args.n_layers,
        "old_n_qubits": args.old_n_qubits, "old_n_layers": args.old_n_layers,
        "seed": args.seed, "steps": args.steps,
        "initial_test_loss_t0": initial_loss,
        "first10_step_losses": first10_losses,
        "grad_norm_mean_at_expansion": grad_norm_mean,
        "grad_norm_variance_at_expansion": grad_norm_var,
        "final_test_loss": final_test_loss,
        "best_test_loss": best_test_loss,
        "total_wall_time_s": total_wall_time,
        "n_circuit_evals_per_step_individual_patches": n_circuit_evals_per_step,
        "total_circuit_evals_individual_patches": total_circuit_evals,
        "total_circuit_evals_qnode_calls": args.steps,
        "n_quantum_params_inherited": n_quantum_params_inherited,
        "n_quantum_params_new": n_quantum_params_new,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_go_nogo] FINAL: test_loss={final_test_loss:.6f} best={best_test_loss:.6f} "
          f"wall_time={total_wall_time:.1f}s", flush=True)
    print(f"[train_go_nogo] results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
