# MNIST-on-grass pre-registration

Committed before any (a), (c), T1, T2 or T3 run on MNIST-on-grass data. The only MNIST-on-grass outputs that exist before this commit are the dataset build checks (`results/mnist_grass/build.json`) and the descriptive pre-checks (`results/mnist_grass/prechecks.md`), which score no embedding.

## Data

- **Recipe:** `data/mnist_grass.py` (documented in `data/README_mnist_grass.md`; inputs and checksums in `data/MANIFEST.md`):
  - digits 0 and 1;
  - target = 0.25 × digit + a random 28 × 28 grass patch from the target pool;
  - background = a grass patch from the background pool;
  - 5000 target and 5000 background images per dataset seed;
  - 1274 grass images, split half/half into the two pools.
- **Seeds:** dataset seeds 0–2 for selection, 3–5 for confirmation. Each seed is an independent dataset, and also seeds every run on it.

## Runs

| label | run | settings |
|---|---|---|
| (c) | two-stage Neg-t-SNE + ξ_B negatives | exact kNN graph (k = 15) and ξ_B (k_B = 15, top L = 30, Jaccard weights, tie cap 2L) on raw 784-D pixels; PCA initialization; Stage 1 of 250 epochs, Stage 2 of 500 epochs; m_B ∈ {1, 2, 5, 10, 20, 40} |
| (a) | reference: the same pipeline with the background term off | as (c), without ξ_B |
| T1 | openTSNE on the target images | library defaults (perplexity 30, PCA initialization, default early exaggeration and learning rate), random_state = dataset seed |
| T2 | PCA (50 components, fit on the target) → openTSNE | `PCA(50, svd_solver="full")`; openTSNE as T1 |
| T3 | cPCA (label-free α, top 10 components) → openTSNE | `baselines/cpca.py`, the project's reimplementation of contrastive 1.2.0's representative-α procedure, validated in Phase 5 prep: the representative with the largest target-to-background variance ratio; openTSNE as T1. The 2-D cPCA projection at every α representative is saved (descriptive figure). |

No tuning beyond the m_B grid for (c).

## Metrics

All are computed on the 2-D embedding of the target images. Digit labels, clean digits and grass patches are used for evaluation only.

| metric | definition |
|---|---|
| **digit kNN accuracy (primary)** | k = 15, stratified 5-fold kNN classifier, as in earlier phases |
| signal recall | kNN recall at k = 15 between the embedding and the clean digit images (higher is better) |
| background retention | kNN recall at k = 15 between the embedding and the target images' grass patches (lower is better) |
| flags | collapse ((c) only: final embedding std < 0.5 × (a)'s on the same seed) and fragmentation (more than 5% of points in kNN-graph components smaller than 50), for review only |

## Selection (seeds 0–2)

(c) runs at every m_B in the grid. (a), T1, T2 and T3 run once each. The m_B for (c) is the one with the highest mean digit kNN accuracy over seeds 0–2; ties go to the smaller m_B. The selection is committed before any confirmation run.

## Confirmation (seeds 3–5)

(c) runs at the selected m_B, alongside (a), T1, T2 and T3. (c) on seed 3 is run a second time and must be bit-identical.

## Comparisons

All on confirmation seeds 3–5, on digit kNN accuracy, under "consistent across 3 paired seeds": the mean difference is > 0 and every per-seed difference is > 0, in either direction. The secondary metrics are reported alongside, without verdicts.

1. (c) vs T1
2. (c) vs T2
3. (c) vs T3
4. (c) vs (a)

## Exclusion

A confirmation seed is excluded only if its dataset build fails a build check (`scripts/mnist_grass_build.py`). Results never decide exclusion.

## Compute and reproducibility

- **Where:** the Math Slurm cluster (AMD EPYC 9474F), CPU only (`CNE_DEVICE=cpu`), no GPU allocated. The cluster was validated in Phase B: two runs bit-identical (`results/mnist_grass/validation/`).
- **Threads:** one run per process, **1 thread** each (`CNE_NUM_THREADS=1`; BLAS and OpenMP also 1), up to 24 processes per job, fixed for the whole phase.
- **No mixing:** no comparison mixes Mac-recorded and cluster results.
- **Records:** every results JSON records the environment, including the Slurm job, node, threads and commit.

## Stop and report

- A dataset build check fails.
- A determinism check fails: openTSNE run twice in the pre-checks, or the (c) seed-3 rerun.
- A run is non-finite or its scale grows more than 10×.
- openTSNE or the cPCA code can't be installed or doesn't run deterministically. This is reported, never replaced by another implementation.
