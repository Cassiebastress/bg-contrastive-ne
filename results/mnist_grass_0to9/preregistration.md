# MNIST-on-grass 0–9 pre-registration

This document is committed before any cluster job of this phase: before the dataset build, the pre-checks, and any
embedding. No MNIST-on-grass 0–9 output of any kind exists before this commit.

## Data

- **Recipe:** `data/mnist_grass.py` with `digits = 0–9` (documented in `data/README_mnist_grass.md`):
  - balanced: 500 MNIST training images of each digit per dataset seed, each sampled without replacement, in one
    shuffled order;
  - target = 0.25 × digit + a random 28 × 28 grass patch from the target pool;
  - background = a grass patch from the background pool;
  - 5000 target and 5000 background images per seed;
  - 1274 grass images, split half/half into the two pools.

  Everything except the digit sample is as in the 0/1 runs.
- **Files:** `data/processed/mnist_grass_0to9_seed{s}.npz`, checksums in `data/MANIFEST.md`. Every run checks its
  dataset's SHA-256 against the manifest.
- **Seeds:** dataset seeds 0–2 for selection, 3–5 for confirmation. Each seed is an independent dataset, and also
  seeds every run on it.
- **Build checks** (`scripts/mnist_grass_0to9_build.py`):
  - shapes and ranges;
  - exact reconstruction;
  - disjoint pools;
  - bit-identical rebuild;
  - exactly 500 images per digit;
  - before the build, a regression: rebuilding the stored 0/1 datasets gives bit-identical arrays.

## Runs

Settings are those of the 0/1 runs (`results/mnist_grass/preregistration.md` and
`results/bsgraph/preregistration_phase2.md`). Nothing is re-tuned except the m_B grid.

| label | run | settings |
|---|---|---|
| (a) | two-stage Neg-t-SNE, background term off | exact kNN graph (k = 15) on raw 784-D pixels; PCA initialization; Stage 1 250 epochs, Stage 2 500 epochs |
| (c) | (a) + ξ_B background negatives | ξ_B on raw pixels (k_B = 15, top L = 30, Jaccard weights, tie cap 2L); m_B ∈ {1, 2, 5, 10, 20, 40} |
| (r_a) | two-stage Neg-t-SNE on the residual graph, background term off | residual graph **fixed at k_loc = 30, r = 2** (the label-free default from the 0/1 run; not re-selected); initialization: PCA of the raw target |
| (r_c) | (r_a) + ξ_B background negatives | ξ_B on raw pixels as in (c); m_B ∈ {1, 2, 5, 10, 20, 40} |
| T1 | openTSNE on the target images | library defaults, random_state = dataset seed |
| T2 | PCA-50 (fit on the target) → openTSNE | `PCA(50, svd_solver="full")`; openTSNE as T1 |
| T3 | cPCA (label-free α, top 10 components) → openTSNE | `baselines/cpca.py`; openTSNE as T1 |

Code: `scripts/mnist_grass_0to9_run.py`. It calls `scripts/mnist_grass_run.embed` for (a), (c), T1, T2 and T3, and
`scripts/bsgraph_run.embed` for (r_a) and (r_c), both unchanged.

## m_B selection (seeds 0–2)

(c) and (r_c) each run at every m_B in the grid; (a), (r_a), T1, T2 and T3 run once each. Each method gets its own
m_B: the one with the highest mean digit kNN accuracy over seeds 0–2, with ties going to the smaller m_B. Both choices
(`results/mnist_grass_0to9/selection.json`) are committed before any confirmation run.

## Metrics

All are computed on the 2-D embedding of the 5000 target images. Digit labels, clean digits and grass patches are used
for evaluation only.

| metric | role | definition |
|---|---|---|
| **digit kNN accuracy** | **primary** | k = 15, stratified 5-fold kNN classifier, 10 classes (chance 0.10) |
| recall vs clean digits | secondary | kNN recall at k = 15 between the embedding and the clean digit images (higher is better) |
| recall vs grass | secondary | kNN recall at k = 15 between the embedding and the target images' grass patches (lower is better) |
| ARI | secondary | k-means on the 2-D embedding with k = 10, n_init = 10, random_state = dataset seed; adjusted Rand index between the k-means labels and the digit labels |
| per-digit accuracy, confusion matrix | descriptive | out-of-fold predictions of the same kNN classifier and folds as the primary metric; per-digit tables for every run, confusion-matrix figures on seed 3 |
| flags | review only | collapse ((c) against (a), (r_c) against (r_a), same seed: final embedding std < 0.5 × the reference's) and fragmentation (more than 5% of points in kNN-graph components smaller than 50) |

Secondary metrics, ARI included, are reported for every run as mean [min, max] over seeds and per seed, **with no
verdicts**. Digit kNN accuracy is the only metric with pre-registered comparisons.

## Comparisons

All on confirmation seeds 3–5, on digit kNN accuracy, under "consistent across 3 paired seeds": the mean paired
difference is > 0 and every per-seed difference is > 0, in either direction; otherwise "not consistent". The secondary
metrics are reported alongside, without verdicts.

1. (r_c) vs T1
2. (r_c) vs T2
3. (r_c) vs T3
4. (r_a) vs T3
5. (r_c) vs (c)
6. (r_c) vs (r_a)

**Descriptive, no verdicts:**
- (a) vs T1, the algorithm confound;
- (c) vs T1, T2 and T3.

## Exclusion

A seed is excluded only if its dataset build fails a build check. Results never decide exclusion.

## Determinism

- T1 on seed 0 is run a second time (in the selection job), and (r_c) on seed 3 a second time (in the confirmation
  job). Both reruns must be bit-identical.
- Pre-checks are descriptive and decide nothing (`scripts/mnist_grass_0to9_prechecks.py`, all seeds, seeds 0–2 and
  3–5 reported separately):
  - graph digit accuracy, edge purity, grass recall and clean-digit recall, on raw pixels, PCA-50, cPCA-10 and the
    full residual grid (k_loc ∈ {15, 30, 100} × r ∈ {0, 1, 2, 5, 10});
  - the ξ_B checks.

## Comparison with the 0/1 runs (descriptive)

The report sets the 0–9 results beside the stored 0/1 cluster results. Chance is 0.10 here against 0.50 there, so
accuracies are not directly comparable. ARI for the 0/1 runs (k-means with k = 2, otherwise as above) is computed
**post hoc** from their saved embeddings (`scripts/mnist_grass_0to9_ari01.py`) and labeled as such. The 0/1 results
files and reports are not changed.

## Compute and reproducibility

- **Where:** the Math Slurm cluster, CPU only (`CNE_DEVICE=cpu`), no GPU allocated.
- **Threads:** one run per process, **1 thread** each (torch, BLAS and OpenMP), up to 24 processes per job, fixed for
  the whole phase.
- **No mixing:** no comparison mixes Mac-recorded and cluster results.
- **Records:** every results JSON records the environment, including the Slurm job, node, threads and commit.

## Stop and report

- The 0/1 regression fails, or any dataset build check fails, including the 500-per-digit balance.
- A determinism check fails.
- A dataset hash or environment mismatch with the recorded setup can't be resolved.
- A run is non-finite or its scale grows more than 10×.
