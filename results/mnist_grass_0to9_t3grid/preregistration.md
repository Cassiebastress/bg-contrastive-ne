# Tuned cPCA → t-SNE baseline (T3) on MNIST-on-grass 0–9: pre-registration

This document is committed before any run of this phase. The plan is `docs/plans/t3_tuning_plan.md`, including its
"Update after the single-stage check".

## Why

In the 0–9 phase, (r_c) beat T3 on every confirmation seed. But T3 ran at one setting (10 components, the label-free
α), while (r_c) had its m_B chosen with labels on the selection seeds. This phase gives T3 a tuning budget larger than
the method's, chosen by the **same label-based rule on the same selection seeds**, and repeats the comparisons against
the tuned T3 (T3\*).

## Data

The 0–9 datasets of the 0–9 phase (`data/processed/mnist_grass_0to9_seed{s}.npz`), unchanged. Every run checks its
dataset's SHA-256 against `data/MANIFEST.md`. Seeds 0–2 select, seeds 3–5 confirm.

## Grid

- **Components:** k ∈ {2, 5, 10, 20, 30, 50, 100}.
- **α:** the cPCA package grid `baselines/cpca.ALPHAS` ({0} ∪ 40 log-spaced values from 0.1 to 1000) plus α = 1, the
  value at which the population identity makes cPCA exact. 42 values, sorted; α = 0 is plain PCA-k.
- **7 × 42 = 294 settings**, against 6 m_B values for (r_c).
- **Pipeline per setting** (T3's, otherwise unchanged): `cpca.fit` on the raw target and background pixels,
  `cpca.project(f, α, k)`, then openTSNE with library defaults and random_state = the dataset seed
  (`scripts/mnist_grass_run.tsne`).
- **Code:** `scripts/t3grid_run.py`. One process per (seed, α) runs all 7 values of k.

## Selection (seeds 0–2)

- Every setting runs on every selection seed.
- **T3\*** is the setting with the highest mean digit kNN accuracy over seeds 0–2. Ties go to fewer components, then
  smaller α. This is the rule that chose m_B for (r_c).
- **PCA\*** (for a descriptive item below) is the best α = 0 setting by the same rule.
- Both are written to `selection.json`, which is committed before any confirmation-seed run.

## Confirmation (seeds 3–5)

The full grid runs again on seeds 3–5. T3\* and PCA\* are single points of it; the rest is descriptive.

## Metrics

Exactly as in the 0–9 phase, on the 2-D embedding of the 5000 target images:
- **digit kNN accuracy** (primary): k = 15, stratified 5-fold, 10 classes;
- recall vs clean digits and recall vs grass: kNN recall at k = 15;
- ARI: k-means with k = 10, n_init = 10, random_state = the dataset seed.

## Pre-registered comparisons

On confirmation seeds 3–5, on digit kNN accuracy, under "consistent across 3 paired seeds": the mean paired difference
is > 0 and every per-seed difference is > 0, in either direction; otherwise "not consistent". Secondary metrics are
reported alongside, without verdicts.

1. **(r_c) vs T3\***. (r_c) is the stored 0–9 confirmation run at its selected m_B = 5.
2. **(r_a) vs T3\***, where (r_a) is **single-stage (r_a)**, the form adopted in `REPORT_single_stage.md`: the stored
   runs `results/single_stage/runs/0to9_seed{3,4,5}_single.json`.

## Descriptive, no verdicts

- **Two-stage (r_a) vs T3\***, with the stored 0–9 (r_a) runs.
- **Oracle T3:** on each confirmation seed, the best grid setting for that seed, chosen with that seed's own labels
  (ties as above). It is an upper bound for T3, not a baseline, because the setting is scored on the data it was picked
  on. Shown against (r_c) per seed.
- **T3\* vs the label-free T3** of the 0–9 phase.
- **PCA\* vs T2** (PCA-50 → t-SNE).
- **Heatmaps** of mean digit kNN accuracy over (k, α), for the selection and the confirmation seeds separately.
- The 10 best settings on the selection seeds, with their confirmation-seed means.

## Reused runs

The (r_c), two-stage (r_a), T3 and T2 rows are the stored 0–9 runs (confirmation: job 6087, code `ba1f582`; selection
T3 and T2: job 6086). The single-stage (r_a) rows are the stored runs of job 6101 (code `d6f7be0`). They are reused,
not rerun, if:
- the code they depend on is unchanged since each run's commit (`git diff --quiet`);
- the stored `.npy` matches the SHA-256 in its JSON;
- the dataset hashes match the manifest.

## Reproducibility check

Each seed's label-free α (5.541, or 4.375 on seed 3) is a grid value. So the grid point (label-free α, k = 10) computes
exactly the stored T3 run, with the same `cpca.project` and `tsne` calls. **Its embedding SHA-256 must equal the stored
T3 run's on all 6 seeds.** This is the phase's determinism check.

## Compute

The Math Slurm cluster, CPU only, 1 thread per process, up to 24 processes per job. 126 processes per stage. Every
results JSON records the environment, including the Slurm job and the commit. No comparison mixes Mac and cluster
results.

## Stop and report

- The label-free grid point does not reproduce the stored T3 SHA-256 on some seed.
- A dataset hash mismatch.
- The reuse check fails: then (r_c) and (r_a) are rerun on seeds 3–5 before any comparison.
- A non-finite embedding.

## Outputs

`REPORT_mnist_grass_0to9_t3grid.md`, with a short pointer from Summary item 1 of `REPORT_mnist_grass_0to9.md` (the
0–9 report's results are otherwise unchanged), a NOTES section and a README phase row.
