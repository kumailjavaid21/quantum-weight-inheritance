"""
Corrected MULTI-STEP (chained) growing-ring weight inheritance.

WHY THIS IS A NEW FILE AND NOT AN EDIT: `models/inheritance_growing.py` is a
Phase 1.5 file whose results are already reported; the standing project rule
is not to modify prior-phase files. This module therefore supersedes it for
chained growth while leaving it byte-identical, so every Phase 1.5 and
go/no-go number remains exactly reproducible. For a SINGLE growth step this
module is mathematically identical to `inheritance_growing.py` (verified by
`verify_chained_growth.py`, which checks the one-step edge list and circuit
against the original).

THE BUG IN THE SINGLE-STEP IMPLEMENTATION, when applied twice (4->6 then 6->8):

  (1) `inherit_weights_growing_ring` unconditionally executes
      `new_state["connect_params"] = zeros_like(...)`, and never reads
      `old_state["connect_params"]`. Correct when the parent is a plain
      WIQGANGenerator (which has no such key), but when the parent is itself
      an already-grown model whose connecting-edge angles have drifted away
      from zero during training (measured after 2,000 steps: mean -0.0064,
      max |value| 0.057), those trained parameters are silently discarded.

  (2) The deeper, structural half of the bug: `make_growing_ring_circuit`
      rebuilds the inherited-layer entangling ring as
      `CNOT(i, (i+r) % old_n)` with `r = old_range_formula(l, old_n)`. On a
      second growth step old_n becomes 6, so the ring among the first six
      wires is recomputed mod 6 -- but the parent 4->6 model's actual circuit
      used a ring mod 4 on wires 0-3 plus a CRY chain 3->4->5->0. The
      resulting 8-qubit circuit therefore does not contain the parent's
      circuit as a sub-circuit AT ALL, so no amount of parameter copying can
      make the second step exact. Fixing (1) alone would not fix exactness.

THE FIX: carry the full growth history as a SCHEDULE
[(n_0, L_0), (n_1, L_1), ..., (n_S, L_S)] (n strictly increasing, L
non-decreasing) rather than a single (old_n, old_L) pair, and reconstruct the
entangling topology recursively. For each layer l, let s = the stage that
"owns" l (the smallest s with l < L_s). Then that layer emits:

  - if s == 0: the BASE model's unparametrized CNOT ring, mod n_0, over
    wires 0..n_0-1 (bit-for-bit the base circuit's own formula);
  - if s >= 1: stage s's zero-initializable CRY ring over its n_s wires
    (this is what stage s contributed when layer l was "new" to it);
  - then, for every LATER stage j > s, stage j's CRY connecting chain
    (n_{j-1}-1 -> n_{j-1} -> ... -> n_j-1 -> 0).

Because CRY(0) is the exact identity (not an approximation), every gate a
later stage adds is a no-op at inheritance time, so the parent's circuit is
reproduced exactly on the parent's wires -- at every step of the chain, not
just the first. Each stage's edges keep their own wire positions, which is
precisely what (2) destroyed.
"""
import os
import sys
from pathlib import Path

import pennylane as qml
import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_PAULI_GATES = {1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}


def old_range_formula(l, n_qubits):
    """The base model's own per-layer CNOT range (qgan_models_wigan's
    formula). Reproduced here verbatim so stage 0's ring is bit-identical."""
    return (l % (n_qubits - 1)) + 1 if n_qubits > 1 else 1


def validate_schedule(schedule):
    assert len(schedule) >= 1, "schedule must have at least the base stage"
    for (n, L) in schedule:
        assert n >= 1 and L >= 1
    for i in range(1, len(schedule)):
        assert schedule[i][0] > schedule[i - 1][0], "qubit counts must strictly increase"
        assert schedule[i][1] >= schedule[i - 1][1], "layer counts must be non-decreasing"
    return tuple((int(n), int(L)) for n, L in schedule)


def owning_stage(l, schedule):
    """Smallest stage index s with l < L_s: the stage for which layer l was
    a genuinely new layer (or the base, for s == 0)."""
    for s, (_, L) in enumerate(schedule):
        if l < L:
            return s
    raise ValueError(f"layer {l} beyond final layer count {schedule[-1][1]}")


def build_chained_edge_list(schedule):
    """Flat, deterministically ordered list of (layer, wire_a, wire_b) for
    every zero-initializable CRY edge in the chained topology. Index order
    matches the connect_params tensor. Stage 0's CNOT ring is NOT in this
    list (it is unparametrized)."""
    schedule = validate_schedule(schedule)
    S = len(schedule) - 1
    _, L_final = schedule[-1]
    edges = []
    for l in range(L_final):
        s = owning_stage(l, schedule)
        if s >= 1:
            n_s = schedule[s][0]
            for i in range(n_s):
                edges.append((l, i, (i + 1) % n_s))
        for j in range(s + 1, S + 1):
            n_prev = schedule[j - 1][0]
            n_j = schedule[j][0]
            chain = list(range(n_prev - 1, n_j)) + [0]
            for a, b in zip(chain[:-1], chain[1:]):
                edges.append((l, a, b))
    assert len(set(edges)) == len(edges), "edge list must not contain duplicates"
    return edges


def make_chained_circuit(schedule, edges):
    schedule = validate_schedule(schedule)
    n_final, L_final = schedule[-1]
    n_base = schedule[0][0]
    dev = qml.device("default.qubit", wires=n_final)

    edges_by_layer = {l: [] for l in range(L_final)}
    for idx, (l, a, b) in enumerate(edges):
        edges_by_layer[l].append((idx, a, b))
    stage_of_layer = {l: owning_stage(l, schedule) for l in range(L_final)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, connect_flat, noise_mask):
        qml.AngleEmbedding(inputs, wires=range(n_final), rotation="Y")
        for l in range(L_final):
            for q in range(n_final):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
            if stage_of_layer[l] == 0:
                r = old_range_formula(l, n_base)
                for i in range(n_base):
                    qml.CNOT(wires=[i, (i + r) % n_base])
            for idx, a, b in edges_by_layer[l]:
                qml.CRY(connect_flat[idx], wires=[a, b])
            for w in range(n_final):
                m = int(noise_mask[l, w])
                if m in _PAULI_GATES:
                    _PAULI_GATES[m](wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_final)]

    return circuit


class ChainedGrowingQGANGenerator(nn.Module):
    """WIQGANGenerator-compatible generator carrying an arbitrary-length
    growth history. `schedule[0]` is the base (pre-growth) model; each later
    entry is one growth step. With len(schedule) == 2 this is mathematically
    identical to GrowingRingQGANGenerator."""

    def __init__(self, schedule, img_size=128, patch_size=8, noise_prob=0.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.schedule = validate_schedule(schedule)
        new_n, new_L = self.schedule[-1]
        old_n, old_L = self.schedule[-2] if len(self.schedule) > 1 else self.schedule[-1]

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_dim = img_size // patch_size
        self.n_patches = self.n_patches_per_dim ** 2
        self.patch_dim = patch_size * patch_size
        self.old_n, self.old_L, self.new_n, self.new_L = old_n, old_L, new_n, new_L
        self.n_qubits, self.n_layers = new_n, new_L
        self.noise_prob = float(noise_prob)

        self.encoder = nn.Sequential(
            nn.Linear(self.patch_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, new_n),
            nn.Tanh(),
        )
        self.q_weights = nn.Parameter(torch.zeros(new_L, new_n, 3))
        self.decoder = nn.Sequential(
            nn.Linear(new_n, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.patch_dim),
        )

        self.edges = build_chained_edge_list(self.schedule)
        self.connect_params = nn.Parameter(torch.zeros(len(self.edges)))
        self.qnode = make_chained_circuit(self.schedule, self.edges)

    def sample_noise_mask(self, device, noise_prob=None):
        p = self.noise_prob if noise_prob is None else noise_prob
        if p <= 0:
            return torch.zeros(self.new_L, self.new_n, dtype=torch.long, device=device)
        trigger = torch.rand(self.new_L, self.new_n, device=device) < p
        pauli_choice = torch.randint(1, 4, (self.new_L, self.new_n), device=device)
        return torch.where(trigger, pauli_choice, torch.zeros_like(pauli_choice))

    def extract_patches(self, x):
        B = x.shape[0]
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, self.n_patches, self.patch_dim)
        return patches, B

    def fold_patches(self, patches, B):
        p = self.patch_size
        n = self.n_patches_per_dim
        x = patches.view(B, n, n, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, 1, self.img_size, self.img_size)
        return x

    def denoise_patches(self, patches, noise_mask=None):
        B, K, _ = patches.shape
        flat = patches.reshape(B * K, self.patch_dim)
        angles = self.encoder(flat) * torch.pi
        if noise_mask is None:
            noise_mask = self.sample_noise_mask(angles.device)
        q_out = self.qnode(angles, self.q_weights, self.connect_params, noise_mask)
        q_out = torch.stack(q_out, dim=-1).float()
        residual = self.decoder(q_out)
        residual = residual.view(B, K, self.patch_dim)
        denoised = torch.clamp(patches + residual, 0.0, 1.0)
        return denoised, noise_mask

    def forward(self, noisy, noise_mask=None, patch_indices=None):
        patches, B = self.extract_patches(noisy)
        if patch_indices is None:
            denoised_patches, mask_used = self.denoise_patches(patches, noise_mask)
            full = self.fold_patches(denoised_patches, B)
            return full, denoised_patches, mask_used
        sel = patches[:, patch_indices, :]
        denoised_sel, mask_used = self.denoise_patches(sel, noise_mask)
        full_patches = patches.clone()
        full_patches[:, patch_indices, :] = denoised_sel
        full = self.fold_patches(full_patches, B)
        return full, denoised_sel, mask_used


def inherit_weights_chained(old_state, old_schedule, new_model):
    """Map a parent's state_dict onto `new_model` (one further growth step).

    `old_schedule` is the parent's OWN full schedule: [(n,L)] for a plain
    WIQGANGenerator, or the parent's multi-stage schedule for an
    already-grown model. Unlike the single-step implementation, every trained
    connecting-edge angle the parent holds is carried over -- matched by the
    edge's (layer, wire_a, wire_b) identity, which is stable across growth
    because each stage keeps its own wire positions -- and ONLY genuinely new
    edges are zeroed.
    """
    old_schedule = validate_schedule(old_schedule)
    assert new_model.schedule[:len(old_schedule)] == old_schedule, (
        f"new_model's schedule {new_model.schedule} must extend the parent's {old_schedule}")

    new_state = {k: v.clone() for k, v in new_model.state_dict().items()}
    old_n, old_L = old_schedule[-1]

    new_qw = torch.zeros_like(new_state["q_weights"])
    new_qw[:old_L, :old_n, :] = old_state["q_weights"]
    new_state["q_weights"] = new_qw

    new_cp = torch.zeros_like(new_state["connect_params"])
    if "connect_params" in old_state:
        old_edges = build_chained_edge_list(old_schedule)
        old_cp = old_state["connect_params"]
        assert len(old_edges) == len(old_cp), (
            f"parent edge list ({len(old_edges)}) does not match its connect_params ({len(old_cp)}); "
            "the supplied old_schedule is probably not the parent's real schedule")
        old_map = {e: old_cp[i] for i, e in enumerate(old_edges)}
        n_carried = 0
        for i, e in enumerate(new_model.edges):
            if e in old_map:
                new_cp[i] = old_map[e]
                n_carried += 1
        assert n_carried == len(old_edges), (
            f"only {n_carried}/{len(old_edges)} parent edges found in the grown topology - "
            "the grown circuit does not contain the parent's circuit")
    new_state["connect_params"] = new_cp

    old_w, old_b = old_state["encoder.2.weight"], old_state["encoder.2.bias"]
    new_w = torch.zeros_like(new_state["encoder.2.weight"])
    new_b = torch.zeros_like(new_state["encoder.2.bias"])
    new_w[:old_n, :] = old_w
    new_b[:old_n] = old_b
    new_state["encoder.2.weight"] = new_w
    new_state["encoder.2.bias"] = new_b

    old_w = old_state["decoder.0.weight"]
    new_w = torch.zeros_like(new_state["decoder.0.weight"])
    new_w[:, :old_n] = old_w
    new_state["decoder.0.weight"] = new_w
    new_state["decoder.0.bias"] = old_state["decoder.0.bias"].clone()

    for key in ["encoder.0.weight", "encoder.0.bias", "decoder.2.weight", "decoder.2.bias"]:
        new_state[key] = old_state[key].clone()

    return new_state
