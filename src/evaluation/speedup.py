"""
EXPERIMENT 2 (v2) -- step-count speedup with a reachable target.

TARGET = 0.052 (fixed), chosen as ~6% above the observed noise floor (~0.0487)
so that every circuit size can reach it. v1 used 0.048654 (one specific 8q run's
best loss), which sat at the noise floor and produced "did not converge" for
three of four sizes despite losses within 2e-4 of it.

Task and protocol are otherwise identical to v1:
  32x32 sums-of-Gaussians, 4,000 train / 500 test, sigma=0.10, pure l1,
  no discriminator, batch 32, Adam(2e-4, betas 0.5/0.999).

  Arm A  from-scratch : fresh circuit, budget 2,000 steps
  Arm B  morphism     : chained inheritance from the previous size's converged
                        checkpoint, budget 200 steps

The morphism arm's t=0 loss is recorded explicitly, because inheritance is
exact and the child therefore starts at its parent's loss -- which may already
be at or below the target. A 0-step result is reported as 0, and the ratio is
then reported as undefined rather than given a fabricated value.

Does not modify v1's outputs; writes scaling_speedup_v2.csv.
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
from inheritance_chained import (  # noqa: E402
    ChainedGrowingQGANGenerator, inherit_weights_chained)
from qgan_utils_wigan import seed_everything  # noqa: E402

DATA = Path(os.path.expanduser("~/host_share/qgan_data/go_nogo_synthetic"))
OUT = Path(os.path.expanduser("~/host_share/qgan_results/quantum_paper/results"))
CKPT = OUT / "scaling_ckpts_v2"
IMG, PATCH = 32, 8
SCHEDULE = [(4, 3), (6, 5), (8, 7), (12, 9)]
LR = 2e-4
TARGET = 0.052


def load_data(device, n_test):
    f = lambda p: torch.from_numpy(np.load(DATA / p).astype(np.float32)).unsqueeze(1)
    return (f("train_clean.npy").to(device), f("train_noisy.npy").to(device),
            f("test_clean.npy")[:n_test].to(device), f("test_noisy.npy")[:n_test].to(device))


def test_loss(gen, noisy, clean, device, chunk):
    gen.eval()
    l1 = nn.L1Loss(reduction="sum")
    tot = cnt = 0
    with torch.no_grad():
        for i in range(0, len(noisy), chunk):
            nb, cb = noisy[i:i + chunk], clean[i:i + chunk]
            mask = gen.sample_noise_mask(device, noise_prob=0.0)
            out, _, _ = gen(nb, noise_mask=mask, patch_indices=None)
            tot += l1(out, cb).item()
            cnt += cb.numel()
    gen.train()
    return tot / cnt


def train(gen, tr_c, tr_n, te_c, te_n, budget, device, rng, chunk, tag,
          batch, dense, sparse):
    opt = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    t0 = time.time()
    l_init = test_loss(gen, te_n, te_c, device, chunk)
    best = l_init
    if l_init <= TARGET:
        print(f"      {tag}: t=0 loss={l_init:.6f} <= target -> 0 steps", flush=True)
        return 0, l_init, best, l_init, time.time() - t0
    print(f"      {tag}: t=0 loss={l_init:.6f}", flush=True)

    steps = None
    for step in range(1, budget + 1):
        idx = torch.from_numpy(rng.choice(len(tr_c), size=batch, replace=False)).long()
        mask = gen.sample_noise_mask(device, noise_prob=0.0)
        out, _, _ = gen(tr_n[idx], noise_mask=mask, patch_indices=None)
        loss = l1(out, tr_c[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step <= dense or step % sparse == 0 or step == budget:
            tl = test_loss(gen, te_n, te_c, device, chunk)
            best = min(best, tl)
            if steps is None and tl <= TARGET:
                steps = step
                print(f"      {tag}: reached {TARGET} at step {step} (loss={tl:.6f})",
                      flush=True)
                break
            if step % 200 == 0:
                print(f"      {tag}: step {step}/{budget} loss={tl:.6f}", flush=True)
    final = test_loss(gen, te_n, te_c, device, chunk)
    return steps, final, min(best, final), l_init, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-scratch-steps", type=int, default=2000)
    ap.add_argument("--morphism-steps", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--eval-chunk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-dense", type=int, default=20)
    ap.add_argument("--eval-sparse", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2v2] device={device}", flush=True)
    print(f"[exp2v2] TARGET = {TARGET} (fixed, ~6% above the ~0.0487 noise floor)",
          flush=True)
    print(f"[exp2v2] budgets: from-scratch={args.from_scratch_steps}, "
          f"morphism={args.morphism_steps}; batch={args.batch}; n_test={args.n_test}",
          flush=True)
    print(f"[exp2v2] eval grid: every step to {args.eval_dense}, then every "
          f"{args.eval_sparse}\n", flush=True)

    tr_c, tr_n, te_c, te_n = load_data(device, args.n_test)
    rows, prev = [], None

    for n, L in SCHEDULE:
        print(f"=== n={n}, L={L} ===", flush=True)

        seed_everything(args.seed)
        rng = np.random.default_rng(args.seed)
        gen = WIQGANGenerator(img_size=IMG, patch_size=PATCH, n_qubits=n,
                              n_layers=L, noise_prob=0.0).to(device)
        fs_steps, fs_final, fs_best, fs_init, fs_t = train(
            gen, tr_c, tr_n, te_c, te_n, args.from_scratch_steps, device, rng,
            args.eval_chunk, "from-scratch", args.batch, args.eval_dense, args.eval_sparse)
        fs_reached = fs_steps is not None
        torch.save(gen.state_dict(), CKPT / f"scratch_{n}q{L}L.pt")
        scratch_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        del gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mo_steps = mo_best = mo_init = mo_t = None
        mo_reached, mo_attempted = False, False
        if prev is not None:
            mo_attempted = True
            p_sched, p_state = prev
            child_sched = p_sched + [(n, L)]
            seed_everything(args.seed)
            rng = np.random.default_rng(args.seed)
            child = ChainedGrowingQGANGenerator(child_sched, img_size=IMG,
                                                patch_size=PATCH, noise_prob=0.0).to(device)
            child.load_state_dict(inherit_weights_chained(p_state, p_sched, child),
                                  strict=True)
            mo_steps, mo_final, mo_best, mo_init, mo_t = train(
                child, tr_c, tr_n, te_c, te_n, args.morphism_steps, device, rng,
                args.eval_chunk, "morphism", args.batch, args.eval_dense, args.eval_sparse)
            mo_reached = mo_steps is not None
            torch.save(child.state_dict(), CKPT / f"morph_{n}q{L}L.pt")
            prev = (child_sched, {k: v.detach().clone()
                                  for k, v in child.state_dict().items()})
            del child
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print("      morphism: n/a (no smaller circuit to grow from)", flush=True)
            prev = ([(n, L)], scratch_state)

        if fs_reached and mo_reached and mo_steps > 0:
            ratio, ratio_s = fs_steps / mo_steps, f"{fs_steps / mo_steps:.1f}x"
        elif fs_reached and mo_reached and mo_steps == 0:
            ratio, ratio_s = float("inf"), "undefined (morphism already <= target at t=0)"
        else:
            ratio, ratio_s = float("nan"), "n/a"

        fs_s = (f"{fs_steps} steps" if fs_reached
                else f"did not reach {TARGET} in {args.from_scratch_steps} steps")
        if not mo_attempted:
            mo_s = "n/a"
        elif mo_reached:
            mo_s = f"{mo_steps} steps"
        else:
            mo_s = f"did not reach {TARGET} in {args.morphism_steps} steps"
        mo_b = "n/a" if mo_best is None else f"{mo_best:.6f}"
        print(f"n={n}: from-scratch = {fs_s} (best loss {fs_best:.6f}) | "
              f"morphism = {mo_s} (best loss {mo_b}) | speedup = {ratio_s}\n", flush=True)

        rows.append({
            "n_qubits": n, "n_layers": L,
            "from_scratch_steps_to_0052": fs_steps if fs_reached else "",
            "morphism_steps_to_0052": "" if mo_steps is None else mo_steps,
            "speedup_ratio": ("" if ratio != ratio else
                              ("inf" if ratio == float("inf") else round(ratio, 2))),
            "from_scratch_best_loss": round(fs_best, 6),
            "morphism_best_loss": "" if mo_best is None else round(mo_best, 6),
            "from_scratch_reached": fs_reached,
            "morphism_reached": mo_reached,
            "morphism_attempted": mo_attempted,
            "from_scratch_initial_loss": round(fs_init, 6),
            "morphism_initial_loss": "" if mo_init is None else round(mo_init, 6),
            "from_scratch_wall_s": round(fs_t, 1),
            "morphism_wall_s": "" if mo_t is None else round(mo_t, 1),
            "target": TARGET,
            "from_scratch_budget": args.from_scratch_steps,
            "morphism_budget": args.morphism_steps,
        })
        with open(OUT / "scaling_speedup_v2.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    json.dump(rows, open(OUT / "scaling_speedup_v2.json", "w"), indent=2)

    bar = "=" * 64
    print(bar)
    print(f"SCALING SPEEDUP (target = {TARGET})")
    print(bar)
    for r in rows:
        fs = (f"{r['from_scratch_steps_to_0052']}" if r["from_scratch_reached"]
              else f"not reached in {r['from_scratch_budget']}")
        if not r["morphism_attempted"]:
            mo = "n/a"
        elif r["morphism_reached"]:
            mo = f"{r['morphism_steps_to_0052']}"
        else:
            mo = f"not reached in {r['morphism_budget']}"
        sp = r["speedup_ratio"]
        sp_s = ("n/a" if sp == "" else
                ("undefined (morphism 0 steps)" if sp == "inf" else f"{sp}x"))
        print(f"  n={r['n_qubits']:<3} from-scratch = {fs:<22} | morphism = {mo:<22} | "
              f"speedup = {sp_s}")
    print(bar)
    print(f"  saved: {OUT / 'scaling_speedup_v2.csv'}")
    print(bar)


if __name__ == "__main__":
    main()
