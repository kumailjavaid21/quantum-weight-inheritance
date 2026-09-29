"""
Phase Q1.3: rerun the chained 4->6->8 growth on the go/no-go synthetic task
using the CORRECTED multi-step inheritance (models/inheritance_chained.py),
replacing the Model 4 numbers that the single-step implementation's chaining
bug invalidated.

Reuses train_go_nogo.py's data loader, loss/gradient helpers and constants by
read-only import, so the task, batch size, learning rate, step budget and
metric definitions are byte-identical to the original Model 1/2/3/4 runs and
the numbers remain directly comparable. train_go_nogo.py itself is NOT
modified.

Usage:
    python train_chained_fixed.py --schedule 4:3,6:5 --init-from <4q3L ckpt> \\
        --out-dir model4fix_stepA --seed 42
    python train_chained_fixed.py --schedule 4:3,6:5,8:7 --init-from <stepA ckpt> \\
        --out-dir model4fix_stepB --seed 42
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

QUANTUM_SCRIPTS = Path(os.path.expanduser("~/qgan_project/quantum_project/scripts"))
WIGAN_MODELS = Path(os.path.expanduser("~/qgan_project/wigan_project/models"))
WIGAN_SCRIPTS = Path(os.path.expanduser("~/qgan_project/wigan_project/scripts"))
for p in (str(QUANTUM_SCRIPTS), str(WIGAN_MODELS), str(WIGAN_SCRIPTS)):
    sys.path.insert(0, p)

from train_go_nogo import (  # noqa: E402  (read-only reuse of the original protocol)
    IMG_SIZE, PATCH_SIZE, TOTAL_STEPS, BATCH_SIZE, LR, N_GRAD_PROBES,
    load_data, sample_batch, full_l1_loss, grad_norm_and_variance, RESULTS_BASE,
)
from inheritance_chained import (  # noqa: E402
    ChainedGrowingQGANGenerator, inherit_weights_chained, build_chained_edge_list,
)
from qgan_utils_wigan import seed_everything  # noqa: E402


def parse_schedule(text):
    stages = []
    for part in text.split(","):
        n, L = part.split(":")
        stages.append((int(n), int(L)))
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                    help="full growth history, e.g. '4:3,6:5,8:7' (base first)")
    ap.add_argument("--init-from", required=True, help="parent checkpoint")
    ap.add_argument("--out-dir", required=True,
                    help="relative to ~/host_share/qgan_results/quantum_project/go_nogo/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    schedule = parse_schedule(args.schedule)
    assert len(schedule) >= 2, "need at least a base stage and one growth step"
    parent_schedule = schedule[:-1]
    new_n, new_L = schedule[-1]

    seed_everything(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    out_dir = RESULTS_BASE / args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_clean, train_noisy, test_clean, test_noisy = load_data(device)
    print(f"[train_chained_fixed] schedule={schedule} parent={parent_schedule} "
          f"seed={args.seed} out_dir={out_dir} device={device}", flush=True)

    old_state = torch.load(os.path.expanduser(args.init_from), map_location=device, weights_only=True)
    parent_has_connect = "connect_params" in old_state
    print(f"[train_chained_fixed] parent checkpoint has connect_params: {parent_has_connect}", flush=True)
    if parent_has_connect:
        cp = old_state["connect_params"]
        print(f"[train_chained_fixed] parent connect_params: n={len(cp)} mean={cp.mean():.6f} "
              f"std={cp.std():.6f} max_abs={cp.abs().max():.6f}  <-- these are the trained angles "
              f"the buggy single-step path discarded", flush=True)

    generator = ChainedGrowingQGANGenerator(schedule, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                                             noise_prob=0.0).to(device)
    new_state = inherit_weights_chained(old_state, parent_schedule, generator)
    generator.load_state_dict(new_state, strict=True)

    n_carried_edges = len(build_chained_edge_list(parent_schedule)) if parent_has_connect else 0
    old_n, old_L = parent_schedule[-1]
    n_quantum_params_inherited = old_L * old_n * 3 + n_carried_edges
    n_quantum_params_new = (new_L * new_n * 3 - old_L * old_n * 3
                             + len(generator.edges) - n_carried_edges)
    print(f"[train_chained_fixed] inherited quantum params: {n_quantum_params_inherited} "
          f"(incl. {n_carried_edges} carried connecting angles); new: {n_quantum_params_new}", flush=True)

    initial_loss = full_l1_loss(generator, test_noisy, test_clean, device)
    initial_train_loss = full_l1_loss(generator, train_noisy[:500], train_clean[:500], device)
    print(f"[train_chained_fixed] initial (t=0) test_loss={initial_loss:.6f} "
          f"train_loss(subset)={initial_train_loss:.6f}", flush=True)

    grad_norm_mean, grad_norm_var = grad_norm_and_variance(
        generator, train_clean, train_noisy, device, BATCH_SIZE, N_GRAD_PROBES, rng)
    print(f"[train_chained_fixed] grad_norm_at_expansion mean={grad_norm_mean:.6f} "
          f"var={grad_norm_var:.8f}", flush=True)

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

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, loss_val, step_time])

    total_wall_time = time.time() - t_start
    torch.save(generator.state_dict(), ckpt_dir / "generator_last.pt")
    if not (ckpt_dir / "generator_best.pt").exists():
        torch.save(generator.state_dict(), ckpt_dir / "generator_best.pt")

    final_test_loss = full_l1_loss(generator, test_noisy, test_clean, device)
    final_psnr = -10.0 * np.log10(max(final_test_loss ** 2, 1e-12))

    summary = {
        "schedule": schedule,
        "parent_schedule": parent_schedule,
        "init_from": args.init_from,
        "parent_had_connect_params": bool(parent_has_connect),
        "n_carried_connect_edges": int(n_carried_edges),
        "n_total_connect_edges": int(len(generator.edges)),
        "n_quantum_params_inherited": int(n_quantum_params_inherited),
        "n_quantum_params_new": int(n_quantum_params_new),
        "initial_test_loss": initial_loss,
        "initial_train_loss_subset": initial_train_loss,
        "grad_norm_at_expansion_mean": grad_norm_mean,
        "grad_norm_at_expansion_var": grad_norm_var,
        "first10_train_losses": first10_losses,
        "final_test_loss": final_test_loss,
        "best_test_loss": best_test_loss,
        "final_test_psnr_proxy_db": final_psnr,
        "steps": args.steps,
        "seed": args.seed,
        "total_wall_time_s": total_wall_time,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_chained_fixed] DONE initial_test_loss={initial_loss:.6f} "
          f"final_test_loss={final_test_loss:.6f} best={best_test_loss:.6f} "
          f"wall={total_wall_time:.1f}s", flush=True)
    print(f"[train_chained_fixed] results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
