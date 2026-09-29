"""
WI-QGAN Phase 1 models: a generalized version of Project A's Pauli-noise-
injection patch generator (qgan_models_pauli_noise.QuantumPatchGeneratorPauliNoise),
parametrized so it can be grown across the 4q/3L -> 6q/5L -> 8q/7L stages,
plus the weight-inheritance mapping (design_notes.md, Phase 0) that transplants
a smaller trained circuit's weights into a larger one.

Architecture is otherwise identical to Project A's noise-aware generator
(same encoder/decoder MLP shapes, same StronglyEntanglingLayers-per-layer
circuit with the corrected GLOBAL-range CNOT topology, same stochastic
Pauli-mask noise injection on default.qubit). The only functional addition
is patch subsampling support (denoise_patches accepts any subset of patches,
not just all of them) needed to make training on 128x128 BraTS images with
256 patches/image tractable within Phase 1's limited compute budget.

Weight-inheritance mapping (see design_notes.md "Weight-inheritance mapping"
section for the full design rationale):
  - Quantum circuit weights (L, n, 3): the first old_L layers x first
    old_n qubits are copied verbatim; every other entry (new qubits within
    old layers, and every qubit within new layers) is zero-initialized,
    i.e. Rot(0,0,0) = identity rotation. Note this is only an IDENTITY
    ROTATION, not a full identity gate, because the StronglyEntanglingLayers
    ansatz's CNOT entangling ring is unparameterized and always fires -
    recomputing that ring for the grown qubit count is exactly the source
    of the "loss jump at inheritance" this experiment is designed to
    measure (flagged as a caveat in design_notes.md's proof sketch).
  - Classical encoder's last Linear (32 -> n_qubits): first old_n output
    rows copied; new output rows zero-initialized (weight AND bias), so a
    new qubit's angle input starts at exactly 0 regardless of the patch.
  - Classical decoder's first Linear (n_qubits -> 32): first old_n input
    columns copied; new input columns zero-initialized, so the decoder
    initially ignores whatever a new qubit outputs.
  - Everything else (encoder's first Linear, decoder's last Linear, all
    biases not tied to the qubit dimension) is qubit-count-independent and
    copied in full.
"""
import pennylane as qml
import torch
import torch.nn as nn

_PAULI_GATES = {1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}


def make_pauli_noise_quantum_layer(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, noise_mask):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for l in range(n_layers):
            # GLOBAL-index range, matching Project A's bugfix: local
            # per-layer indexing would silently collapse every layer's
            # range to 1, changing the entangling topology.
            layer_range = (l % (n_qubits - 1)) + 1 if n_qubits > 1 else 1
            qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_qubits), ranges=[layer_range])
            for w in range(n_qubits):
                m = int(noise_mask[l, w])
                if m in _PAULI_GATES:
                    _PAULI_GATES[m](wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class WIQGANGenerator(nn.Module):
    def __init__(self, img_size=128, patch_size=8, n_qubits=6, n_layers=5, noise_prob=0.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_dim = img_size // patch_size
        self.n_patches = self.n_patches_per_dim ** 2
        self.patch_dim = patch_size * patch_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_prob = float(noise_prob)

        self.encoder = nn.Sequential(
            nn.Linear(self.patch_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_qubits),
            nn.Tanh(),
        )
        self.q_weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.patch_dim),
        )

        self.qnode = make_pauli_noise_quantum_layer(n_qubits, n_layers)

    def sample_noise_mask(self, device, noise_prob=None):
        p = self.noise_prob if noise_prob is None else noise_prob
        if p <= 0:
            return torch.zeros(self.n_layers, self.n_qubits, dtype=torch.long, device=device)
        trigger = torch.rand(self.n_layers, self.n_qubits, device=device) < p
        pauli_choice = torch.randint(1, 4, (self.n_layers, self.n_qubits), device=device)
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
        """patches: (B, K, patch_dim) for any K (full n_patches, or a
        subsample). Returns (denoised_patches (B,K,patch_dim), mask_used)."""
        B, K, _ = patches.shape
        flat = patches.reshape(B * K, self.patch_dim)
        angles = self.encoder(flat) * torch.pi
        if noise_mask is None:
            noise_mask = self.sample_noise_mask(angles.device)
        q_out = self.qnode(angles, self.q_weights, noise_mask)
        q_out = torch.stack(q_out, dim=-1).float()
        residual = self.decoder(q_out)
        residual = residual.view(B, K, self.patch_dim)
        denoised = torch.clamp(patches + residual, 0.0, 1.0)
        return denoised, noise_mask

    def forward(self, noisy, noise_mask=None, patch_indices=None):
        """If patch_indices is None: full-image forward (used for eval/
        robustness testing), returns the folded full image.
        If patch_indices is given (a 1-D LongTensor of patch indices,
        shared across the batch): only those patches are pushed through the
        quantum encoder/decoder (training-time subsampling); every other
        patch is passed through UNCHANGED (still noisy). Returns
        (full_image, denoised_selected_patches, mask_used) so the caller can
        compute a reconstruction loss over exactly the processed patches
        without the untouched ones diluting it.
        """
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


class PatchGANDiscriminator(nn.Module):
    """Identical to Project A's qgan_models.PatchGANDiscriminator - fully
    convolutional, so it works unchanged at 128x128 input (just yields a
    larger patch-logit map than at 64x64)."""

    def __init__(self, in_channels=2):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, 32, norm=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256, stride=1),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, noisy, candidate):
        x = torch.cat([noisy, candidate], dim=1)
        return self.net(x)


def inherit_weights(old_state, old_n_qubits, old_n_layers, new_model):
    """Builds a state_dict for `new_model` (a WIQGANGenerator instance
    already constructed at the target n_qubits/n_layers) by inheriting
    weights from `old_state` (an older, smaller WIQGANGenerator's
    state_dict). See module docstring for the exact mapping. Returns the
    new state_dict; does not mutate `new_model` in place (caller must
    load_state_dict it).
    """
    new_state = {k: v.clone() for k, v in new_model.state_dict().items()}
    new_n_qubits, new_n_layers = new_model.n_qubits, new_model.n_layers
    assert new_n_qubits >= old_n_qubits and new_n_layers >= old_n_layers, \
        "inherit_weights only supports growing (qubits and layers must not shrink)"

    # 1. quantum circuit weights (L, n, 3)
    new_qw = torch.zeros_like(new_state["q_weights"])
    new_qw[:old_n_layers, :old_n_qubits, :] = old_state["q_weights"]
    new_state["q_weights"] = new_qw

    # 2. encoder last Linear (index 2): (32 -> n_qubits)
    old_w, old_b = old_state["encoder.2.weight"], old_state["encoder.2.bias"]
    new_w = torch.zeros_like(new_state["encoder.2.weight"])
    new_b = torch.zeros_like(new_state["encoder.2.bias"])
    new_w[:old_n_qubits, :] = old_w
    new_b[:old_n_qubits] = old_b
    new_state["encoder.2.weight"] = new_w
    new_state["encoder.2.bias"] = new_b

    # 3. decoder first Linear (index 0): (n_qubits -> 32)
    old_w = old_state["decoder.0.weight"]
    new_w = torch.zeros_like(new_state["decoder.0.weight"])
    new_w[:, :old_n_qubits] = old_w
    new_state["decoder.0.weight"] = new_w
    new_state["decoder.0.bias"] = old_state["decoder.0.bias"].clone()

    # 4. qubit-count-independent layers: copied in full
    for key in ["encoder.0.weight", "encoder.0.bias", "decoder.2.weight", "decoder.2.bias"]:
        new_state[key] = old_state[key].clone()

    return new_state
