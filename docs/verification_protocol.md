# Verification protocol

The exactness claim is a claim that two circuits compute the same
function on a shared register. A test of such a claim is only
meaningful if it is capable of failing. This document records the three
design decisions that make the harness in
`src/inheritance/verification.py` and `src/evaluation/exactness.py`
adversarial rather than confirmatory.

## 1. The parent is perturbed before each growth step

The inheritance map sets every connecting gate angle to zero, and a
controlled rotation at angle zero is exactly the identity. If the parent
circuit's connecting angles are also zero, then a construction that
*discards* them is indistinguishable from one that *carries* them, and
the test passes either way.

Before each growth step the parent's rotation and connecting parameters
are therefore perturbed away from zero, with magnitude matched to the
drift measured after 2,000 real optimisation steps:

| Parameter group | Measured after 2,000 steps |
|---|---|
| connecting angles | mean −0.0064, std 0.0174, max abs 0.057 |

The harness prints the perturbed parent's statistics before each step so
the perturbation is visible in the log rather than implied.

## 2. The superseded construction is run as a control

At every step the harness additionally runs the single-step construction
applied naively to a parent that is itself already a grown circuit. This
is the procedure the chained construction replaces.

It fails at every step after the first, with divergences of 1.786 to
1.939 against a maximum possible value of 2 for a difference of Pauli-Z
expectation values. At the first step the two agree exactly, because
there the parent genuinely is a plain circuit and there is nothing for
the naive path to drop.

A control that fails this hard is what establishes that the passing
result is not vacuous.

## 3. Precision is fixed before the inheritance map runs

Every tensor is cast to double precision *before* inheritance, not
after. The inheritance map allocates the child's tensors with
`zeros_like` on the child's parameters, so a single-precision child
silently truncates a double-precision parent. The measured divergence is
then floored at single-precision epsilon regardless of whether the
construction is correct.

An earlier revision of the harness reported 5.1e-09 for the chained
case for exactly this reason. The value was an artefact of storage
precision, not a property of the construction. Casting first removes the
floor and the true value appears at double-precision round-off.

## 4. Regression against the single-step implementation

For a single growth step the chained construction must reduce to the
original single-step one. The harness checks this directly: the two edge
lists are compared elementwise and the two circuits are evaluated on the
same inputs. They agree exactly (zero divergence, identical 21-edge
lists at the 4→6 step), which confirms the generalisation does not
perturb any previously reported single-step result.

## 5. Memory

The state vector grows as `2^n`, and the simulator materialises a tensor
per gate. The harness therefore evaluates probes in chunks whose size
decreases with register size, inside `torch.no_grad()`.

Chunking partitions the probe set and takes the maximum divergence
across chunks; it is mathematically identical to evaluating all probes
at once. The `no_grad` context is required because this is a pure
forward check — retaining an autograd graph across the full gate
sequence at 2^16 amplitudes exhausts system memory.

## Summary of what the harness reports

For each growth step the harness prints and records:

- `chained_max_divergence` — the construction under test
- `naive_max_divergence` — the control
- `verdict` — PASS if the chained value is below 1e-14

Raw output: `results/scaling_exactness.csv`.
