"""
EXPERIMENT 1 -- chained-growth exactness across the extended schedule
    (4,3) -> (6,5) -> (8,7) -> (12,9) -> (16,11)

Protocol (matches the existing Phase 1.5 / verify_chained_growth.py protocol):
  * before each growth step the parent's connecting angles and rotations are
    perturbed away from zero, with magnitude matched to the drift measured
    after 2,000 optimisation steps (connect_params: mean -0.0064, std 0.0174,
    max|.| 0.057). Without this the naive construction also passes and the
    test is vacuous.
  * every tensor is cast to float64 BEFORE the inheritance map runs, because
    the map allocates via zeros_like(child_param); a float32 child silently
    truncates a float64 parent and floors the measurable divergence at
    float32 epsilon.
  * divergence = max |Delta <Z>| over the shared wires, 200 random input
    vectors, new wires held at input angle 0.
  * the naive single-step construction (GrowingRingQGANGenerator +
    inherit_weights_growing_ring, which treats the parent as a plain circuit)
    is run on the identical inputs as a control.

NOTE ON THE SCHEDULE: the published schedule steps n by +2 and L by +2. The
extension to 12 and 16 qubits continues that pattern (L = 9 and 11). This is a
protocol choice, recorded here rather than left implicit.
"""
import csv
import json
import os
import sys
from pathlib import Path

import torch

WIGAN = Path(os.path.expanduser("~/qgan_project/wigan_project"))
sys.path.insert(0, str(WIGAN / "scripts"))
sys.path.insert(0, str(WIGAN / "models"))

from qgan_models_wigan import WIQGANGenerator  # noqa: E402
from inheritance_growing import (  # noqa: E402  (Phase 1.5 original = naive control)
    GrowingRingQGANGenerator, inherit_weights_growing_ring)
from inheritance_chained import (  # noqa: E402
    ChainedGrowingQGANGenerator, inherit_weights_chained)

OUT = Path(os.path.expanduser("~/host_share/qgan_results/quantum_paper/results"))
SCHEDULE = [(4, 3), (6, 5), (8, 7), (12, 9), (16, 11)]
N_PROBE = 200
PROBE_CHUNK = 25          # chunked so 16-qubit states stay within memory
TOL = 1e-14
# drift magnitudes measured after 2,000 real optimisation steps
DRIFT_CONNECT_STD = 0.0174
DRIFT_ROT_STD = 0.05


def to_double(state):
    return {k: (v.double() if v.is_floating_point() else v.clone()) for k, v in state.items()}


def probe_inputs(n_shared, n_total, n, seed=123):
    g = torch.Generator().manual_seed(seed)
    shared = (torch.rand(n, n_shared, generator=g, dtype=torch.float64) * 2 - 1) * torch.pi
    if n_total == n_shared:
        return shared, shared
    pad = torch.zeros(n, n_total - n_shared, dtype=torch.float64)
    return shared, torch.cat([shared, pad], dim=1)


def chunk_for(n_qubits):
    """State size grows as 2^n and PennyLane materialises a tensor per gate, so
    the probe batch must shrink as the register grows."""
    if n_qubits >= 16:
        return 2
    if n_qubits >= 12:
        return 8
    return PROBE_CHUNK


def run_qnode_chunked(model, inputs, n_qubits, n_layers, use_connect=True):
    mask = torch.zeros(n_layers, n_qubits, dtype=torch.long)
    step = chunk_for(n_qubits)
    outs = []
    # no_grad is essential: this is a pure forward check, and retaining the
    # autograd graph across ~L*(2n) gates at 2^16 amplitudes exhausts memory
    # (an earlier run was SIGKILLed at the 12->16 step for exactly this reason).
    with torch.no_grad():
        for i in range(0, len(inputs), step):
            chunk = inputs[i:i + step]
            if use_connect:
                o = model.qnode(chunk, model.q_weights, model.connect_params, mask)
            else:
                o = model.qnode(chunk, model.q_weights, mask)
            outs.append(torch.stack(o, dim=-1).clone())
            del o
    return torch.cat(outs, dim=0)


def perturb_(model, seed):
    """Simulate post-training drift so the exactness test is not vacuous."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.q_weights.data += torch.randn(
            model.q_weights.shape, generator=g, dtype=torch.float64) * DRIFT_ROT_STD
        model.connect_params.data += torch.randn(
            model.connect_params.shape, generator=g, dtype=torch.float64) * DRIFT_CONNECT_STD


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--n-probe", type=int, default=200,
                    help="probe vectors; reduce for the largest register if memory-bound")
    ap.add_argument("--max-step", type=int, default=len(SCHEDULE) - 1,
                    help="run only the first k growth steps")
    args = ap.parse_args()
    n_probe = args.n_probe

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    rows = []

    # --- stage 0: the base circuit, using Run A's real trained weights ---
    ckpt = Path(os.path.expanduser(
        "~/host_share/qgan_results/wigan_phase1/A_4q3L/checkpoints/generator_last.pt"))
    base_state = to_double(torch.load(ckpt, map_location="cpu", weights_only=True))
    n0, L0 = SCHEDULE[0]
    parent_model = WIQGANGenerator(img_size=128, patch_size=8, n_qubits=n0,
                                   n_layers=L0, noise_prob=0.0).double()
    parent_model.load_state_dict(base_state, strict=True)
    parent_is_plain = True
    parent_schedule = [SCHEDULE[0]]
    print(f"[exp1] base circuit {n0}q/{L0}L loaded from {ckpt.name}", flush=True)
    print(f"[exp1] schedule: {' -> '.join(f'{n}q/{L}L' for n, L in SCHEDULE)}", flush=True)
    print(f"[exp1] probes={n_probe} (chunk adapts with register size), "
          f"dtype=float64, tol={TOL:g}\n", flush=True)

    for step_i in range(1, min(args.max_step, len(SCHEDULE) - 1) + 1):
        n_p, L_p = SCHEDULE[step_i - 1]
        n_c, L_c = SCHEDULE[step_i]
        label = f"{n_p}->{n_c}"
        child_schedule = SCHEDULE[:step_i + 1]

        shared, padded = probe_inputs(n_p, n_c, n_probe, seed=321 + step_i)
        out_parent = run_qnode_chunked(parent_model, shared, n_p, L_p,
                                       use_connect=not parent_is_plain)
        parent_state = to_double({k: v.detach().clone()
                                  for k, v in parent_model.state_dict().items()})

        # ---- chained construction (carries every earlier stage's angles) ----
        child = ChainedGrowingQGANGenerator(child_schedule, img_size=128, patch_size=8,
                                            noise_prob=0.0).double()
        child.load_state_dict(inherit_weights_chained(parent_state, parent_schedule, child),
                              strict=True)
        out_child = run_qnode_chunked(child, padded, n_c, L_c)[:, :n_p]
        div_chained = (out_parent - out_child).abs().max().item()

        # ---- naive single-step reuse (treats the parent as a plain circuit) ----
        try:
            naive = GrowingRingQGANGenerator(n_p, L_p, n_c, L_c, img_size=128,
                                             patch_size=8, noise_prob=0.0).double()
            naive.load_state_dict(
                inherit_weights_growing_ring(parent_state, n_p, L_p, naive), strict=True)
            out_naive = run_qnode_chunked(naive, padded, n_c, L_c)[:, :n_p]
            div_naive = (out_parent - out_naive).abs().max().item()
            del naive
        except Exception as exc:
            div_naive = float("nan")
            print(f"  [note] naive construction raised at {label}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

        verdict = "PASS" if div_chained < TOL else "FAIL"
        print(f"Step {label}: chained max|dZ| = {div_chained:.3e}   [{verdict} at tol {TOL:g}]",
              flush=True)
        print(f"Step {label}: naive   max|dZ| = {div_naive:.3e}", flush=True)

        rows.append({"growth_step": label, "n_parent": n_p, "n_child": n_c,
                     "L_parent": L_p, "L_child": L_c,
                     "chained_max_divergence": div_chained,
                     "naive_max_divergence": div_naive,
                     "verdict": verdict})

        # the child becomes the next parent, perturbed to simulate fine-tuning
        perturb_(child, seed=777 + step_i)
        cp = child.connect_params.data
        print(f"           (child perturbed for next step: connect mean={cp.mean():+.6f} "
              f"std={cp.std():.6f} max|.|={cp.abs().max():.4f})\n", flush=True)
        del parent_model
        parent_model = child
        parent_is_plain = False
        parent_schedule = child_schedule

    with open(OUT / "scaling_exactness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[exp1] wrote {OUT / 'scaling_exactness.csv'}", flush=True)
    json.dump(rows, open(OUT / "scaling_exactness.json", "w"), indent=2)


if __name__ == "__main__":
    main()
