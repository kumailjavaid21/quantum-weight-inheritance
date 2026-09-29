"""
Phase Q1.2: verify chained qubit-count growth is exact at EVERY step.

Protocol:
  (a) Load the real trained 4q/3L checkpoint (Phase 1 Run A).
  (b) Grow 4 -> 6. Check the grown circuit reproduces the 4q circuit's output
      on wires 0-3 exactly.
  (c) Perturb the 6q model's q_weights AND connect_params away from zero --
      this is essential: it simulates the drift that 2,000 training steps
      actually produced (measured: connect_params mean -0.0064, max |.| 0.057)
      and is the ONLY regime in which the chaining bug is observable. With
      connect_params still at zero, the buggy implementation would pass too.
      Then grow 6 -> 8 and check the grown circuit reproduces the PERTURBED
      6q model's output on wires 0-5 exactly.
  (d) Run step (c)'s test against the ORIGINAL single-step implementation as
      a control, to confirm the bug is real and that the fix is what removes
      it rather than the test being vacuous.

Also checks that for a single growth step the new chained implementation is
identical to the original (so no Phase 1.5 result is invalidated).

DTYPE NOTE: every model and state_dict here is cast to float64 BEFORE the
inheritance mapping runs. This matters: `inherit_weights_*` allocates the new
tensors with `zeros_like(new_model_param)`, so a float32 child would silently
truncate a float64 parent's angles and floor the measured divergence at ~1e-8
(float32 epsilon) regardless of whether the topology is exact. Casting first
removes that harness artifact so the number being reported is the morphism's
own exactness, not the storage precision.
"""
import os
import sys
from pathlib import Path

import torch

WIGAN_ROOT = Path(os.path.expanduser("~/qgan_project/wigan_project"))
sys.path.insert(0, str(WIGAN_ROOT / "scripts"))
sys.path.insert(0, str(WIGAN_ROOT / "models"))

from qgan_models_wigan import WIQGANGenerator  # noqa: E402
from inheritance_growing import (  # noqa: E402  (read-only: the Phase 1.5 original, used as the control)
    GrowingRingQGANGenerator,
    inherit_weights_growing_ring,
    build_connect_edge_list,
)
from inheritance_chained import (  # noqa: E402
    ChainedGrowingQGANGenerator,
    inherit_weights_chained,
    build_chained_edge_list,
)

CKPT = Path(os.path.expanduser(
    "~/host_share/qgan_results/wigan_phase1/A_4q3L/checkpoints/generator_last.pt"))
OUT_DIR = Path(os.path.expanduser("~/host_share/qgan_results/quantum_paper/bug_fix"))
N_PROBE = 200
TOL = 1e-14


def probe_inputs(n_shared, n_total, n=N_PROBE, seed=123):
    """Random angles on the shared wires; new wires held at 0, matching
    phase1_diagnostic.py's and inheritance_growing.py's own convention."""
    g = torch.Generator().manual_seed(seed)
    shared = (torch.rand(n, n_shared, generator=g, dtype=torch.float64) * 2 - 1) * torch.pi
    if n_total == n_shared:
        return shared, shared
    padded = torch.cat([shared, torch.zeros(n, n_total - n_shared, dtype=torch.float64)], dim=1)
    return shared, padded


def to_double(state):
    return {k: (v.double() if v.is_floating_point() else v.clone()) for k, v in state.items()}


def run_qnode(model, inputs, n_qubits, n_layers, use_connect=True):
    mask = torch.zeros(n_layers, n_qubits, dtype=torch.long)
    if use_connect:
        out = model.qnode(inputs, model.q_weights, model.connect_params, mask)
    else:
        out = model.qnode(inputs, model.q_weights, mask)
    return torch.stack(out, dim=-1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    results = {}

    print("=" * 78)
    print("Q1.2 (a): load trained 4q/3L checkpoint")
    print("=" * 78)
    old_state = to_double(torch.load(CKPT, map_location="cpu", weights_only=True))
    base_schedule = [(4, 3)]
    small = WIQGANGenerator(img_size=128, patch_size=8, n_qubits=4, n_layers=3, noise_prob=0.0).double()
    small.load_state_dict(old_state, strict=True)
    print(f"loaded {CKPT}")
    print(f"q_weights shape {tuple(old_state['q_weights'].shape)}, "
          f"has connect_params: {'connect_params' in old_state}")

    print()
    print("=" * 78)
    print("Q1.2 (b): growth step 1, 4 -> 6 qubits (3 -> 5 layers)")
    print("=" * 78)
    sched_6 = [(4, 3), (6, 5)]
    grown6 = ChainedGrowingQGANGenerator(sched_6, img_size=128, patch_size=8, noise_prob=0.0).double()
    state6 = inherit_weights_chained(old_state, base_schedule, grown6)
    grown6.load_state_dict(state6, strict=True)

    shared4, padded6 = probe_inputs(4, 6)
    out_small = run_qnode(small, shared4, 4, 3, use_connect=False)
    out_grown6 = run_qnode(grown6, padded6, 6, 5)[:, :4]
    div_step1 = (out_small - out_grown6).abs().max().item()
    results["step1_4to6_max_divergence"] = div_step1
    print(f"max |divergence| on shared wires 0-3 : {div_step1:.3e}   "
          f"({'PASS' if div_step1 < TOL else 'FAIL'} at tol {TOL:.0e})")

    print()
    print("=" * 78)
    print("Q1.2 (c): growth step 2, 6 -> 8 qubits (5 -> 7 layers), CHAINED")
    print("=" * 78)
    # Simulate post-training drift: this is what makes the bug observable.
    g = torch.Generator().manual_seed(77)
    with torch.no_grad():
        grown6.q_weights.data += torch.randn(
            grown6.q_weights.shape, generator=g, dtype=torch.float64) * 0.05
        grown6.connect_params.data += torch.randn(
            grown6.connect_params.shape, generator=g, dtype=torch.float64) * 0.02
    cp = grown6.connect_params.data
    print(f"perturbed 6q model: connect_params mean={cp.mean():.6f} "
          f"std={cp.std():.6f} max|.|={cp.abs().max():.4f}  (n={len(cp)} edges)")
    print("  -> nonzero connecting angles, matching the drift measured after 2,000 real steps")

    state6_trained = to_double({k: v.detach().clone() for k, v in grown6.state_dict().items()})

    sched_8 = [(4, 3), (6, 5), (8, 7)]
    grown8 = ChainedGrowingQGANGenerator(sched_8, img_size=128, patch_size=8, noise_prob=0.0).double()
    state8 = inherit_weights_chained(state6_trained, sched_6, grown8)
    grown8.load_state_dict(state8, strict=True)

    shared6, padded8 = probe_inputs(6, 8, seed=321)
    out_6 = run_qnode(grown6, shared6, 6, 5)
    out_8 = run_qnode(grown8, padded8, 8, 7)[:, :6]
    div_step2 = (out_6 - out_8).abs().max().item()
    results["step2_6to8_chained_max_divergence"] = div_step2
    print(f"max |divergence| on shared wires 0-5 : {div_step2:.3e}   "
          f"({'PASS' if div_step2 < TOL else 'FAIL'} at tol {TOL:.0e})")

    print()
    print("=" * 78)
    print("Q1.2 (d): CONTROL - same step-2 test against the ORIGINAL single-step code")
    print("=" * 78)
    old_impl_ok = True
    try:
        grown8_old = GrowingRingQGANGenerator(6, 5, 8, 7, img_size=128, patch_size=8, noise_prob=0.0).double()
        state8_old = inherit_weights_growing_ring(state6_trained, 6, 5, grown8_old)
        grown8_old.load_state_dict(state8_old, strict=True)
        out_8_old = run_qnode(grown8_old, padded8, 8, 7)[:, :6]
        div_step2_old = (out_6 - out_8_old).abs().max().item()
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        old_impl_ok = False
        div_step2_old = float("nan")
        print(f"original implementation raised: {type(exc).__name__}: {exc}")
    results["step2_6to8_original_max_divergence"] = div_step2_old
    if old_impl_ok:
        print(f"max |divergence| on shared wires 0-5 : {div_step2_old:.3e}   "
              f"({'PASS' if div_step2_old < TOL else 'FAIL'} at tol {TOL:.0e})")
        print(f"  dropped connecting angles: {int((state8_old['connect_params'] == 0).sum())}"
              f"/{len(state8_old['connect_params'])} of the grown model's edges are zero")

    print()
    print("=" * 78)
    print("Regression check: one-step chained == one-step original")
    print("=" * 78)
    e_new = build_chained_edge_list(sched_6)
    e_old = build_connect_edge_list(4, 3, 6, 5)
    same_edges = e_new == e_old
    results["onestep_edge_lists_identical"] = bool(same_edges)
    print(f"edge lists identical: {same_edges}  ({len(e_new)} edges)")

    grown6_ref = GrowingRingQGANGenerator(4, 3, 6, 5, img_size=128, patch_size=8, noise_prob=0.0).double()
    ref_state = inherit_weights_growing_ring(old_state, 4, 3, grown6_ref)
    grown6_ref.load_state_dict(ref_state, strict=True)
    out_ref = run_qnode(grown6_ref, padded6, 6, 5)[:, :4]
    div_vs_ref = (out_ref - out_grown6).abs().max().item()
    results["onestep_chained_vs_original_max_divergence"] = div_vs_ref
    print(f"max |divergence| chained vs original (one step): {div_vs_ref:.3e}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, v in results.items():
        print(f"  {k}: {v}")

    import json
    with open(OUT_DIR / "verify_chained_growth.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, float) else v) for k, v in results.items()}, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'verify_chained_growth.json'}")

    ok = (div_step1 < TOL and div_step2 < TOL and same_edges and div_vs_ref < TOL)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
