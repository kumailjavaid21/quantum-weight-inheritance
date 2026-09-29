"""
EXPERIMENT 4 -- compute-cost reality check.

Measures wall-clock time for one forward+backward pass and peak GPU memory at
each circuit size, on the controlled synthetic task's geometry
(32x32 images, 8x8 patches => 16 patch-circuits per image).

Run early, because its result determines whether Experiment 2 is tractable at
12 and 16 qubits within any sane budget: state-vector cost grows as 2^n, so an
8 -> 16 qubit move is a 256x increase in state size alone.

If a size cannot run at batch 32, the batch is halved (32 -> 16 -> 8 -> 4) and
the batch actually used is reported. The circuit size is never changed.
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

WIGAN = Path(os.path.expanduser("~/qgan_project/wigan_project"))
sys.path.insert(0, str(WIGAN / "scripts"))
sys.path.insert(0, str(WIGAN / "models"))
from qgan_models_wigan import WIQGANGenerator  # noqa: E402

OUT = Path(os.path.expanduser("~/host_share/qgan_results/quantum_paper/results"))
DATA = Path(os.path.expanduser("~/host_share/qgan_data/go_nogo_synthetic"))
IMG, PATCH = 32, 8
SCHEDULE = {4: 3, 6: 5, 8: 7, 12: 9, 16: 11}
WARMUP, TIMED = 2, 5


def bench(n, L, batch, device):
    gen = WIQGANGenerator(img_size=IMG, patch_size=PATCH, n_qubits=n,
                          n_layers=L, noise_prob=0.0).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=2e-4)
    l1 = nn.L1Loss()
    clean = torch.rand(batch, 1, IMG, IMG, device=device)
    noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for i in range(WARMUP + TIMED):
        if i == WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        mask = gen.sample_noise_mask(device, noise_prob=0.0)
        out, _, _ = gen(noisy, noise_mask=mask, patch_indices=None)
        loss = l1(out, clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / TIMED
    peak = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else float("nan")
    del gen, opt, clean, noisy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,6,8,12,16")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[exp4] device={device} ({gpu_name})\n", flush=True)

    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        L = SCHEDULE[n]
        batch = args.batch
        dt = peak = None
        note = ""
        while batch >= 4:
            try:
                dt, peak = bench(n, L, batch, device)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                note = f"OOM at batch {batch}; halved"
                print(f"  [note] n={n}: OOM at batch {batch}, retrying at {batch // 2}",
                      flush=True)
                batch //= 2
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                note = f"OOM at batch {batch}; halved"
                print(f"  [note] n={n}: OOM at batch {batch}, retrying at {batch // 2}",
                      flush=True)
                batch //= 2

        if dt is None:
            print(f"n={n}: INFEASIBLE even at batch 4", flush=True)
            rows.append({"n_qubits": n, "n_layers": L, "batch_size": 0,
                         "time_per_step_s": float("nan"), "peak_mem_mb": float("nan"),
                         "feasible": False, "note": "infeasible at batch>=4",
                         "gpu": gpu_name})
            continue

        est_2000 = dt * 2000 / 3600.0
        print(f"n={n}: time/step={dt:.3f}s, peak_mem={peak:.0f}MB, batch={batch}"
              + (f"   [{note}]" if note else ""), flush=True)
        print(f"        -> 2,000 steps would take {est_2000:.2f} h", flush=True)
        rows.append({"n_qubits": n, "n_layers": L, "batch_size": batch,
                     "time_per_step_s": round(dt, 4), "peak_mem_mb": round(peak, 1),
                     "est_2000_step_hours": round(est_2000, 3),
                     "feasible": True, "note": note, "gpu": gpu_name})

    with open(OUT / "compute_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[exp4] wrote {OUT / 'compute_cost.csv'}", flush=True)
    json.dump(rows, open(OUT / "compute_cost.json", "w"), indent=2)


if __name__ == "__main__":
    main()
