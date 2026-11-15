# Single-cell RNA-seq (Zheng et al. 2017, AML027): pre-registration

This document is committed before any pre-check that uses the labels and before any embedding of this phase.

## Why

Every real-data result so far is on MNIST-on-grass, where the background is *added* to the signal, so global cPCA is
exact in population. This phase tests BC-NE on the cPCA paper's single-cell data (its Figure 3).
- **Target:** bone marrow cells of acute myeloid leukemia patient 027, before and after a stem-cell transplant.
- **Background:** a healthy donor's bone marrow.
- **What they share:** cell-type variation (T cells, B cells, monocytes, erythroid cells, …).
- **What only the target has:** the pre vs post difference.
- **Additive or not:** the background was not built to be either. We cannot show it is non-additive; the report will
  say so.

- **The method is unchanged.** BC-NE is (r_a): the residual graph (k_loc = 30, r = 2, k = 15) with no background
  negatives, embedded by one 750-epoch Neg-t-SNE fit (`scripts/single_stage_run.fit("single")`). ξ_B is not built.
- **Platform:** every run is on the cluster. Nothing is compared with earlier phases.

## Data

`data/scrna.py`, committed before this document (`2727984`, `be255d1`); `data/MANIFEST_scrna.md` (`ce1cc59`) pins the
four tarballs' SHA-256 and the processed file's (`b6d30107…`). A rebuild was byte-identical.

| role | 10x sample | cells |
|---|---|---|
| target, label 0 (pre) | `aml027_pre_transplant` | 3933 |
| target, label 1 (post) | `aml027_post_transplant` | 3965 |
| background | `frozen_bmmc_healthy_donor1` | 1985 |
| larger background, descriptive only | `frozen_bmmc_healthy_donor2` (added to donor 1) | 2472 |

- **Preprocessing** (label-free, one transform for target and background):
  1. each cell's counts scaled to sum to 10,000, then log(1 + x);
  2. the 500 genes with the largest dispersion (variance / mean) over target + background (ties to the lower index);
  3. each gene standardized with the target + background mean and std.
  Donor 2 goes through the same transform, so the target is identical in both background variants.
- **Already known before this document:** cell counts, and the median log10 total counts per sample (pre 3.69, post
  3.48, donor 1 3.29). Nothing else about the labels has been looked at.

## Runs (each on seeds 0–9)

| run | pipeline |
|---|---|
| r_a | BC-NE, as above |
| a | plain Neg-t-SNE: the exact kNN graph (k = 15) of X_T, otherwise exactly as r_a |
| T1 | openTSNE on X_T, library defaults, random_state = seed (`scripts/mnist_grass_run.tsne`) |
| T2 | PCA-50 of X_T (sklearn, full SVD) → T1's openTSNE |
| U2 | PCA-50 of X_T → UMAP (umap-learn defaults, random_state = seed) |
| T3 | cPCA (`baselines/cpca.py`) at the label-free α (`select_alpha(f, seed)`), top 10 → T1's openTSNE |
| cPCA-2D | the 2-D cPCA projection at that α (same fit as T3) |
| PCA-2D | the 2-D PCA projection of X_T (`cpca.project(f, 0, 2)`) |

- **Seeds:** the data are fixed, so the seed changes only the algorithms' randomness (initialization, sampling,
  t-SNE, UMAP, k-means, and the spectral clustering inside the α rule).
- **No selection stage:** every setting above was fixed before any result, so all ten seeds count.
- **Code:** `scripts/scrna_run.py`.

## Metrics (on the 2-D embedding of the 7898 target cells)

- **Primary: label kNN accuracy** (pre vs post): k = 15, stratified 5-fold, `eval.metrics.knn_accuracy`. Chance is
  about 0.50.
- **Secondary, no verdicts:**
  - **ARI:** k-means with k = 2 (n_init = 10, random_state = seed) against the label.
  - **Background-structure retention:** the mean, over the embedding's 15-NN edges, of the Jaccard overlap of the
    two cells' 15 nearest donor-1 background cells in the 500-gene space (`eval.metrics.background_retention`).
    - **Reference values** (pre-checks): all cell pairs, and the 500-gene kNN graph itself.
    - **Not a score:** cell type is real biology, so this says what a map keeps, not whether it is better.
  - **Fragmentation:** the existing component check (`component_stats`, flag above 5% of cells in components under 50).
  - **Runtime.**

## Pre-registered rules

- **Comparisons:** r_a vs each of T1, T2, U2, T3, cPCA-2D, PCA-2D and a, on the primary metric.
  - "r_a better, consistent across 10 paired seeds" means the mean paired difference is > 0 and every per-seed
    difference is > 0.
  - "X better, consistent across 10 paired seeds" is the mirror image.
  - Anything else is "not consistent".
- **Ceiling rule:** if both T1 and r_a average at least 0.95 on the primary metric, the report states that this
  dataset cannot discriminate between the methods, whatever the verdicts.
- **Limitations, stated in the report whatever the results:**
  - **The seeds:** they measure only the algorithms' randomness.
  - **The biology:** there is one patient and one donor.
  - **Batch:** pre and post are separate 10x runs (and differ in sequencing depth), so the label is confounded with
    batch, as in the cPCA paper's figure.

## Pre-checks (descriptive, before any embedding, in the first job)

- **Label kNN accuracy in high dimension** (same classifier and folds): in the 500-gene space, PCA-50, cPCA-10 at the
  seed-0 label-free α, and the residuals (k_loc = 30, r = 2).
- **The label-free α** on each seed, and its representatives.
- **Background-retention reference values:** all pairs, the 500-gene kNN graph, the residual kNN graph.
- **Replication of the paper's figure:**
  - preprocessing: the paper's (500 genes by raw-count dispersion, then target and background each standardized on
    their own);
  - cPCA-2D at the paper's α values (0, 3.455, 28.94, 388.8, from its notebook) and at our label-free α;
  - result: label kNN accuracy of each projection, and a figure to compare with the paper's Figure 3a.

## Descriptive extras (no verdicts)

- **BC-NE sensitivity:** k_loc ∈ {10, 30, 60} × r ∈ {0, 2, 5}, seeds 0–9. The (30, 2) cell is r_a itself.
- **cPCA with α chosen using labels** (labeled as such; an upper bound, not a fair baseline): cPCA-2D and T3 at every
  α of `cpca.ALPHAS`, each seed, and the best α by the primary metric.
- **Larger background:** r_a, T3 and cPCA-2D with background = donors 1 + 2 (4457 cells), seeds 0–9.
  Background-structure retention is always measured against donor 1, so it stays comparable.

## Checks

- **Determinism:** every run type except the α grid is rerun at seed 0; each rerun must be bit-identical
  (embedding SHA-256). A failed rerun stops the merge.
- **Data:** every process checks the processed file's SHA-256 against `data/MANIFEST_scrna.md`.
- **Completeness:** the merge stops unless every expected run is present.
- **Run records:** every run records the environment, the Slurm job, 1 thread and a clean commit.
