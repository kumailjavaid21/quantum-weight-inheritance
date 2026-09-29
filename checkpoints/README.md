# Checkpoints

No model weights are committed to this repository. The `.gitignore`
excludes `*.pt`, `*.pth` and `*.ckpt` so that they cannot be added
accidentally.

## What exists

The experiments produce the following checkpoint sets on the machine
they were run on.

### Corrected chained growth, synthetic task

Produced by `src/training/train_chained.py`. Two stages of the
schedule, each with a final and a best-validation checkpoint:

- `model4fix_stepA` — 4→6 growth, 2,000 steps
- `model4fix_stepB` — 6→8 growth chained from step A, 2,000 steps

Step B's loss at inheritance equals step A's converged loss to six
decimal places, which is the exactness property visible in a live
training run rather than in the forward-pass probe.

### Scaling study, synthetic task

Produced by `src/evaluation/speedup.py`. One from-scratch and one
inherited checkpoint per register size in {4, 6, 8, 12}, for each of the
two target definitions described in `docs/reproducibility.md`.

### BraTS study

Produced by `src/training/train_brats.py`. Independent baselines at
4 and 6 qubits, and inherited 4→6 models across ten seeds (42–51).

## Sizes

Every individual checkpoint is well under 50 MB; the quantum circuits
have a few hundred parameters and the classical encoder and decoder are
two linear layers each. The exclusion is for repository hygiene rather
than a file-size limit.

## Availability

Checkpoints are available from the corresponding author on request, and
will be deposited in a public archive on publication. Every result in
`results/` can be regenerated from source using the scripts in
`scripts/`; nothing in the paper depends on a checkpoint that cannot be
reproduced.
