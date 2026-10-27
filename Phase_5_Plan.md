# Plan: Phase 5 prep — global metrics and the cPCA baseline

## 0. Why this phase

Phase 4 showed that background negatives reveal signal *within* background islands, but the global layout stays organized by background. The next method (a target kNN graph in a background-suppressed metric) aims to change the global layout: same-signal points in different background clusters should become neighbors and pull islands together. Before building it we need:

- **A.** Global metrics that can register that change. Signal Spearman cannot: high-D X_T itself scores ≈ 0.01 on these datasets.
- **B.** The contrastive PCA (cPCA) baseline. Building a kNN graph after linearly suppressing the background is close to "cPCA, then embed". Any new method must beat it, and it shows where linear suppression already suffices.

No new method is built in this phase. Same determinism rules, datasets, seeds and Phase 4 protocol conventions apply. Ground-truth labels are used for evaluation only.

### Open decisions (defaults chosen; the user may override)
1. Neighborhood size for the new metrics: k = 15, matching Phase 4.
2. cPCA α selection: the unsupervised procedure from Abid et al. (2018), plus an oracle-best α reported separately and clearly labeled.
3. cPCA output: 2-D cPCA plotted directly, and the top 10 cPCA components fed into embedding.
4. Embedders for the baseline: our two-stage Neg-t-SNE pipeline (same as run (a)), plus umap-learn with default settings for the literal "cPCA → UMAP" baseline.

## 1. Part A — Global metrics (`eval/global_metrics.py`)

### A1. Define the metrics
All metrics use y_s (4 signal groups) and y_b (6 background clusters) for evaluation only.

- **G1, cross-cluster same-signal neighbor rate.** For each point, among its k nearest neighbors in the 2-D embedding, the fraction that share its signal group *and* belong to a different background cluster. Report the mean over points.
  - Reference values: chance (random neighbors, labels independent) = 0.25 × 5/6 ≈ 0.21. Phase 4 runs: expected ≈ 0, since islands are disjoint. A perfectly signal-organized layout that ignores background: ≈ 5/6 ≈ 0.83.
  - Higher is better.
- **G2, background kNN accuracy.** Already computed in Phase 4. In this phase it becomes meaningful again: it should drop toward chance (≈ 0.17) if islands merge. Report it under that interpretation.
- **G3, cell-level layout.** Split points into the 24 (signal group × background cluster) cells and take each cell's embedding centroid. For each cell, find its nearest other cell centroid and record whether that cell shares the signal group, the background cluster, or neither. Report the three fractions.
  - A background-organized layout gives "shares background" ≈ 1. A signal-organized layout gives "shares signal" ≈ 1.

### A2. Validate the metrics before using them
The metrics are only useful if they can move. Compute G1–G3 on:
1. **Existing Phase 4 embeddings** (a), best (b), (c), (d), on all seeds, main datasets. Expected: G1 ≈ 0, G2 ≈ 1, G3 "shares background" ≈ 1.
2. **A signal-only reference:** the same two-stage Neg-t-SNE run on S_T + ε (the clean signal with the dataset's noise), main datasets, selection seeds. Expected: G1 high, G2 near chance, G3 "shares signal" high. This is an upper reference, not a method.
3. **High-D X_T** (G1 and G2 computed on its 50-D kNN graph) as context.

Write `results/phase5prep/metric_validation.md` with these tables.

**Stop and report** if the signal-only reference does not clearly separate from the Phase 4 runs on G1 and G3. That would mean the metrics can't detect the intended change.

### A3. Freeze
After validation and before any Part B result, commit the metric definitions and a short note on how they will be used to judge the next method. A pre-registered direction: the next method should raise G1 and G3 "shares signal", and lower G2, without lowering signal kNN accuracy below (c)'s. Numeric thresholds are decided by the user after seeing A2.

## 2. Part B — cPCA baseline (`baselines/cpca.py`)

### B1. Implement cPCA
- Target and background are each centered by their own mean (standard cPCA).
- For a given α, take the top eigenvectors of C_T − α C_B, where C_T and C_B are the sample covariances.
- α selection, unsupervised (Abid et al.): an α grid of 0 plus about 40 log-spaced values between 0.1 and 1000. Compute the 2-D cPCA subspace for each α, cluster the subspaces by principal-angle similarity, and return the representative α of each cluster.
  - If the authors' reference package (`contrastive`, PyPI) installs and runs deterministically, use it and verify it against our implementation on one dataset. Otherwise use our implementation. Record which.
  - When the procedure returns several representative α, pick one by a rule fixed in advance and that uses no labels. Default: the representative with the largest ratio of target variance to background variance in its subspace. Report all representatives in an appendix.
- **Oracle-best α** (reported separately, labeled "oracle"): the α on the grid with the highest signal kNN accuracy. This is an upper bound on what cPCA can do, not a fair baseline.

### B2. Runs (main datasets; same selection and confirmation seeds as Phase 4, including hard's exclusion of seed 3)
- **(e1) cPCA-2D:** the 2-D cPCA projection itself, at the unsupervised α.
- **(e2) cPCA → Neg-t-SNE:** top 10 cPCA components at the unsupervised α, then the same two-stage Neg-t-SNE as run (a) (kNN graph built in the 10-D cPCA space).
- **(e3) cPCA → UMAP:** the same 10-D input to umap-learn defaults (n_neighbors=15, min_dist=0.1), with a fixed random_state.
- **(e2-oracle):** as (e2) at the oracle-best α.

Reuse the stored Phase 4 embeddings for (a) and (c) at m*_c; do not rerun them.

### B3. Metrics and outputs
- Phase 4 metrics (signal kNN accuracy, signal recall, background accuracy) plus G1–G3.
- `results/phase5prep/cpca_summary.md`: per level, a table with rows (a), (c), (e1), (e2), (e3), (e2-oracle), using mean [min, max] over the confirmation seeds and per-seed paired differences between (c) and (e2), using the Phase 4 "consistent across 3 paired seeds" convention.
- Figures: seed 0 embeddings per level, colored by y_s and by y_b, for (a), (c), (e1), (e2), (e3).
- Record the selected α, all representative α, and the oracle α per dataset.

### B4. What to report (interpretation, no new decision rules)
- Where cPCA → embed matches or beats (c) on signal accuracy and on the global metrics. Expected on easy: signal and background are orthogonal, so cPCA should nearly isolate the signal.
- Where linear suppression falls short. Expected on medium (overlap) and especially hard (nonlinear background).
- Whether the unsupervised α is close to the oracle α. If not, cPCA's weak point may be α selection rather than linearity.

This tells us where the background-suppressed graph needs to be nonlinear to add anything beyond cPCA.

## 3. Stop-and-report conditions
- The metric validation (A2) fails to separate the signal-only reference from Phase 4 runs.
- cPCA results are not deterministic, or the reference package disagrees with our implementation beyond numerical tolerance.
- The α-selection procedure returns no usable representative on some dataset.
- Any Phase 4 hash or reproduction check fails when reloading stored embeddings.

## 4. Out of scope
- The background-suppressed target kNN graph itself (next phase).
- The oracle same-cluster control (separate small task; independent of this plan).
- contrastiveVI, real datasets, the linear ξ_B variant.