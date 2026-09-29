"""
Phase 1.5 Task 2: topology-preserving ("growing ring") weight inheritance.

Confirmed by phase1_diagnostic.md: the CURRENT Phase 1 inheritance
(qgan_models_wigan.inherit_weights, NOT modified here) recomputes the whole
CNOT ring from scratch using the GROWN qubit count's modulus, which silently
changes which wire pairs get CNOT'd among the OLD qubits too (the ring's
wraparound point moves), causing near-maximal divergence (~1.96, vs. a
threshold of 1e-3) in the old qubits' output even before any training. A
"naive" fix that only relocates new qubits to the end of the ring (still
using plain, unparametrized connecting CNOTs) does NOT solve this, because a
CNOT with an old (possibly superposed) qubit as control and a new (|0>)
qubit as target still dephases the old qubit's reduced state.

The actual fix implemented here: every entangling gate that touches at least
one NEW qubit or belongs to a genuinely NEW layer is a PARAMETRIZED
controlled-rotation (qml.CRY), zero-initialized. CRY(0) is the exact
identity gate (not just "close to" identity), so these connecting edges
contribute nothing at step 0, regardless of the old qubits' state - while
every edge purely among old qubits within an INHERITED layer is left as the
FS0exact, unmodified CNOT from the old circuit's own per-layer range formula
(computed with old_n, never new_n), so the old-qubit sub-circuit is
bit-for-bit identical to the smaller model's circuit. This is verified by
this file's own self-test (`python inheritance_growing.py`) to < 1e-6.

Topology, concretely, for growing old_n -> new_n qubits, old_L -> new_L layers:
  - Inherited layers (l < old_L): old qubits get the ORIGINAL rotation
    weights and the ORIGINAL CNOT ring (range computed mod old_n, exactly as
    qgan_models_wigan's small-circuit formula). New qubits get zero rotation
    (Rot(0,0,0) = identity) plus a chain of zero-init CRY gates:
    (old_n-1 -> old_n -> old_n+1 -> ... -> new_n-1 -> 0).
  - New layers (l >= old_L): zero rotation on all new_n qubits, plus a full
    zero-init CRY ring over all new_n qubits (range-1 ring).
All zero-init CRY parameters are ordinary trainable nn.Parameters (they are
NOT frozen) - "zero at step 0" only describes the inheritance-time
initialization, not a permanent constraint during training.
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
    return (l % (n_qubits - 1)) + 1 if n_qubits > 1 else 1


def build_connect_edge_list(old_n, old_L, new_n, new_L):
    """Flat list of (layer, wire_a, wire_b) for every zero-init CRY edge,
    in a fixed order matching the connect_params tensor's indexing."""
    edges = []
    for l in range(new_L):
        if l < old_L:
            chain = list(range(old_n - 1, new_n)) + [0]
            for a, b in zip(chain[:-1], chain[1:]):
                edges.append((l, a, b))
        else:
            for i in range(new_n):
                edges.append((l, i, (i + 1) % new_n))
    return edges


def make_growing_ring_circuit(old_n, old_L, new_n, new_L, edges):
    dev = qml.device("default.qubit", wires=new_n)
    edges_by_layer = {l: [] for l in range(new_L)}
    for idx, (l, a, b) in enumerate(edges):
        edges_by_layer[l].append((idx, a, b))

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, connect_flat, noise_mask):
        qml.AngleEmbedding(inputs, wires=range(new_n), rotation="Y")
        for l in range(new_L):
            for q in range(new_n):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
            if l < old_L:
                r = old_range_formula(l, old_n)
                for i in range(old_n):
                    qml.CNOT(wires=[i, (i + r) % old_n])
            for idx, a, b in edges_by_layer[l]:
                qml.CRY(connect_flat[idx], wires=[a, b])
            for w in range(new_n):
                m = int(noise_mask[l, w])
                if m in _PAULI_GATES:
                    _PAULI_GATES[m](wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(new_n)]

    return circuit


class GrowingRingQGANGenerator(nn.Module):
    """A WIQGANGenerator-compatible generator (same encoder/decoder/patch
    plumbing) whose quantum circuit uses the growing-ring topology above.
    Always represents a GROWN model relative to a specific (old_n, old_L);
    for a first-time (non-grown) model, use qgan_models_wigan.WIQGANGenerator
    directly instead (this class has no reason to exist without a growth
    step behind it)."""

    def __init__(self, old_n, old_L, new_n, new_L, img_size=128, patch_size=8, noise_prob=0.0):
        super().__init__()
        assert img_size % patch_size == 0
        assert new_n >= old_n and new_L >= old_L
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_dim = img_size // patch_size
        self.n_patches = self.n_patches_per_dim ** 2
        self.patch_dim = patch_size * patch_size
        self.old_n, self.old_L, self.new_n, self.new_L = old_n, old_L, new_n, new_L
        self.n_qubits, self.n_layers = new_n, new_L  # interface parity with WIQGANGenerator
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

        self.edges = build_connect_edge_list(old_n, old_L, new_n, new_L)
        self.connect_params = nn.Parameter(torch.zeros(len(self.edges)))
        self.qnode = make_growing_ring_circuit(old_n, old_L, new_n, new_L, self.edges)

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
        else:
            sel = patches[:, patch_indices, :]
            denoised_sel, mask_used = self.denoise_patches(sel, noise_mask)
            full_patches = patches.clone()
            full_patches[:, patch_indices, :] = denoised_sel
            full = self.fold_patches(full_patches, B)
            return full, denoised_sel, mask_used


def inherit_weights_growing_ring(old_state, old_n, old_L, new_model):
    """old_state: state_dict of a qgan_models_wigan.WIQGANGenerator at
    (old_n, old_L). new_model: a GrowingRingQGANGenerator already
    constructed at (old_n, old_L, new_n, new_L). Classical encoder/decoder
    mapping is identical to qgan_models_wigan.inherit_weights (that part was
    never implicated in the diagnostic); q_weights uses the same "copy old
    layers/qubits, zero the rest" rule; connect_params is explicitly zeroed
    (guarantees identity at step 0, verified in this file's self-test)."""
    new_state = {k: v.clone() for k, v in new_model.state_dict().items()}
    new_n, new_L = new_model.new_n, new_model.new_L

    new_qw = torch.zeros_like(new_state["q_weights"])
    new_qw[:old_L, :old_n, :] = old_state["q_weights"]
    new_state["q_weights"] = new_qw
    new_state["connect_params"] = torch.zeros_like(new_state["connect_params"])

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


def _self_test():
    """Verifies the growing-ring inheritance reproduces the smaller circuit's
    output on the shared qubits to < 1e-6, using Run A's REAL trained
    q_weights (not synthetic), matching phase1_diagnostic.py's convention
    (random shared input angles, new-qubit input angle fixed at 0)."""
    import numpy as np
    from qgan_models_wigan import WIQGANGenerator

    torch.manual_seed(123)
    old_n, old_L, new_n, new_L = 4, 3, 6, 5

    ckpt_path = Path(os.path.expanduser(
        "~/host_share/qgan_results/wigan_phase1/A_4q3L/checkpoints/generator_last.pt"))
    old_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    old_q_weights = old_state["q_weights"]

    small = WIQGANGenerator(img_size=128, patch_size=8, n_qubits=old_n, n_layers=old_L, noise_prob=0.0)

    grown = GrowingRingQGANGenerator(old_n, old_L, new_n, new_L, img_size=128, patch_size=8, noise_prob=0.0)
    new_state = inherit_weights_growing_ring(old_state, old_n, old_L, grown)
    grown.load_state_dict(new_state, strict=True)

    N = 200
    shared_inputs = (torch.rand(N, old_n) * 2 - 1) * torch.pi
    zeros_for_new = torch.zeros(N, new_n - old_n)
    grown_inputs = torch.cat([shared_inputs, zeros_for_new], dim=1)

    noise_mask_small = torch.zeros(old_L, old_n, dtype=torch.long)
    noise_mask_grown = torch.zeros(new_L, new_n, dtype=torch.long)

    small_out = torch.stack(small.qnode(shared_inputs, old_q_weights, noise_mask_small), dim=-1)
    grown_out_full = torch.stack(
        grown.qnode(grown_inputs, grown.q_weights, grown.connect_params, noise_mask_grown), dim=-1)
    grown_out = grown_out_full[:, :old_n]

    max_div = (small_out - grown_out).abs().max().item()
    print(f"[inheritance_growing self-test] max divergence on shared qubits: {max_div:.3e}")
    assert max_div < 1e-6, f"growing-ring inheritance FAILED self-test: max divergence {max_div:.3e} >= 1e-6"
    print("[inheritance_growing self-test] PASSED (< 1e-6)")


if __name__ == "__main__":
    _self_test()
