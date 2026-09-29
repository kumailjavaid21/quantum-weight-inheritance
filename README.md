# Exact Qubit-Count Morphism for Scalable Quantum Machine Learning

Reference implementation and experiment data for the paper:

> **Exact Qubit-Count Morphism for Scalable Quantum Machine Learning
> via Progressive Weight Inheritance**

## Overview

We introduce a construction for growing a variational quantum circuit
from `n` to `n' > n` qubits while preserving the learned function
exactly on the original register. The construction is proven exact and
verified to machine precision across four chained growth steps
(4→6→8→12→16).

Measured maximum divergence in the Pauli-Z expectation values on the
shared register, over 200 random input vectors in double precision:

| Growth step | Chained construction | Naive single-step reuse |
|---|---|---|
| 4 → 6   | 2.776e-16 | 2.776e-16 |
| 6 → 8   | 4.441e-16 | 1.900e+00 |
| 8 → 12  | 2.331e-15 | 1.939e+00 |
| 12 → 16 | 3.442e-15 | 1.786e+00 |

The second column is a control: it applies the single-step construction
repeatedly, which is correct only for the first growth step. Raw values
are in `results/scaling_exactness.csv`.

## Repository contents

- `src/` — implementation of the growing-ring morphism, the ring
  ansatz, training scripts, and evaluation harnesses.
- `scripts/` — bash entry points reproducing each experiment.
- `results/` — raw CSV outputs of every experiment in the paper.
- `figures/` — figure PDFs as they appear in the paper.
- `docs/` — reproducibility notes and the adversarial verification
  protocol.

## Installation

```bash
conda env create -f environment.yml
conda activate qubit-morphism
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Reproducing results

Each experiment has a corresponding script:

```bash
bash scripts/run_exactness.sh      # -> results/scaling_exactness.csv
bash scripts/run_speedup.sh        # -> results/scaling_speedup_v2.csv
bash scripts/run_compute_cost.sh   # -> results/compute_cost.csv
bash scripts/run_multi_seed.sh     # -> results/table4_multiseed.csv
bash scripts/run_all.sh            # all of the above, in order
```

All experiments were run on a single NVIDIA Quadro RTX 8000 (48 GB).

The exactness harness evaluates the circuit in a pure forward pass under
`torch.no_grad()` with a register-size-adaptive probe chunk. Chunking
partitions the 200 probe vectors and does not change the computed
divergence; it is required because the state vector grows as `2^n`.

The 16-qubit circuit appears in the exactness measurement only. It is
excluded from the speedup and compute-cost experiments, where it
required a reduced batch size that would not have been comparable with
the other sizes. See `docs/reproducibility.md`.

## Data

The medical imaging experiment uses BraTS 2021, which is publicly
available and is not redistributed here. See `docs/reproducibility.md`
for preprocessing instructions.

## Citation

```bibtex
@article{jazeb2026qubitmorphism,
  title   = {Exact Qubit-Count Morphism for Scalable Quantum Machine
             Learning via Progressive Weight Inheritance},
  author  = {Jazeb, Muhammad Noman and Iqbal, Muhammad Waseem and
             Ali, Usman and Aslam, Sheraz and Ashraf, Nouman and
             Khan, Sajawal ur Rehman and Javaid, Kumail},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026},
  note    = {To appear}
}
```

## License

MIT (see LICENSE).

## Contact

Corresponding author: Nouman Ashraf (nouman.ashraf@tudublin.ie)
