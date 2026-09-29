# Reproducibility notes

## Hardware

All experiments were run on a single NVIDIA Quadro RTX 8000 (48 GB,
reported as 46,080 MiB by `nvidia-smi`). Simulation is state-vector
based, so cost is dominated by the `2^n` amplitude count rather than by
the classical network.

## Environment

See `environment.yml` (conda) or `requirements.txt` (pip). The pinned
versions are PyTorch 2.4.1 and PennyLane 0.37.0. Circuits are
differentiated by backpropagation through the simulator
(`diff_method="backprop"` on `default.qubit`).

## Seeds

Every training run uses seed 42 unless stated otherwise. The multi-seed
study uses seeds 42–51 (ten seeds). The exactness harness uses fixed
probe seeds so its inputs are identical across runs and across
constructions being compared.

## Paths

The scripts under `scripts/` and the Python entry points reference the
directory layout of the machine the experiments were run on. Before
running them, set the data and output roots to match your environment:

- synthetic task data — `synthetic_data.py` generates it; the trainers
  expect it under a `go_nogo_synthetic` directory
- BraTS data — see below
- results are written under a `quantum_paper/results` directory

These are defined as module-level constants at the top of each script.

## Datasets

### Synthetic controlled task

32×32 sums-of-Gaussians, 4,000 train / 500 test, Gaussian noise
σ = 0.10, pure L1 objective, no discriminator. Produced by
`src/common/synthetic_data.py`. This task carries the exactness and
step-count measurements because it isolates growth behaviour from the
confounds of a full adversarial training loop.

### BraTS 2021

Not redistributed here. Obtain it from the official challenge
distribution. Preprocessing used: axial slices extracted, resampled to
128×128, scaled to [0, 1], with a combined Gaussian and Poisson
degradation applied at evaluation time. See
`src/common/brats_data.py` and `src/common/degradation.py`.

## Growth schedule

The published schedule steps the qubit count by +2 and the layer count
by +2 per stage:

| Stage | Qubits | Layers |
|---|---|---|
| base | 4 | 3 |
| 1 | 6 | 5 |
| 2 | 8 | 7 |
| 3 | 12 | 9 |
| 4 | 16 | 11 |

Stages 3 and 4 extend the pattern to larger registers. The layer counts
for those two stages are a protocol choice continuing the +2 increment,
recorded here rather than left implicit.

## Scope of the 16-qubit measurement

The 16-qubit circuit appears in the exactness measurement only.

For the compute-cost measurement it required batch 8: batch 32 and
batch 16 both exhausted GPU memory. Measured cost at batch 8 was
28.945 s per training step with a peak of 25,107 MB, which projects to
16.1 hours for a 2,000-step run.

It is excluded from the speedup experiment for two reasons. The reduced
batch size would not have been comparable with the other sizes, which
all ran at batch 32; and the projected runtime exceeded the compute
budget. The exactness measurement is unaffected by either constraint,
because it is a forward-pass check over probe vectors rather than a
training run, so the 12→16 row is reported.

## Measured compute cost

| Qubits | Layers | Batch | Time/step (s) | Peak memory (MB) |
|---|---|---|---|---|
| 4 | 3 | 32 | 0.048 | 20 |
| 6 | 5 | 32 | 0.101 | 37 |
| 8 | 7 | 32 | 0.217 | 154 |
| 12 | 9 | 32 | 7.040 | 3,986 |
| 16 | 11 | 8 | 28.945 | 25,107 |

Between 8 and 12 qubits, time per step increases by a factor of 32.5
and peak memory by a factor of 26.0. Raw values in
`results/compute_cost.csv`.

## Target definition in the speedup experiment

Steps-to-target depends on the target chosen, and two are reported
because the choice turns out to matter.

`results/scaling_speedup.csv` uses 0.048654, the best loss reached by
one specific from-scratch 8-qubit run. That value sits at the noise
floor of these models, so a fresh run has roughly even odds of reaching
it. Three of four sizes are recorded as not converging, with best losses
within 2.5e-05 to 2.25e-04 of the target. The one clean pair is at 12
qubits: 1,320 steps from scratch against 70 with inheritance.

`results/scaling_speedup_v2.csv` uses 0.052, chosen to be reachable by
every size. All four from-scratch runs reach it (360, 330, 300 and 280
steps). Every inheritance run reaches it in 0 steps, because an exact
morphism starts the child at its parent's loss, which is already below
0.052. The ratio is therefore undefined rather than large.

Both files are included because neither target yields a non-degenerate
ratio at every size, and the reason is structural: for an exact
morphism, steps-to-target is zero for any target above the parent's
loss and unreachable for any target below it.

## Evaluation grid

Test loss is computed every step up to step 20, then every 10 steps.
Steps-to-target is therefore exact below step 20 and quantised to the
nearest 10 above it. The grid is recorded in the output CSVs.
