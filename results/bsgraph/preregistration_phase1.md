# Background-suppressed graph, Phase 1 pre-registration (graph quality; no embeddings)

Committed before any residual-graph metric is computed on MNIST-on-grass. The data are the MNIST-on-grass datasets, seeds 0–5, exactly as built and checked in that phase (`data/MANIFEST.md`). Everything runs on the Math Slurm cluster: CPU only, 1 thread per process.

## Residual graph (`method/residual_graph.py`)

For each target image x_i (raw 784-D pixels):
1. Find its k_loc nearest background images (exact, Euclidean).
2. Take their mean μ_i and the top r principal directions V_i of the neighbors centered at μ_i (SVD).
3. The residual is r_i = (x_i − μ_i) − V_i V_iᵀ (x_i − μ_i). With r = 0, this is local mean subtraction only.
4. Build the kNN graph (k = 15, union-symmetrized as every earlier graph) on the residuals.

**Grid:** k_loc ∈ {15, 30, 100} × r ∈ {0, 1, 2, 5, 10}, which gives 15 variants.

## Reference graphs

- raw pixels X_T;
- PCA-50 of X_T;
- descriptive only: cPCA-10 at the label-free α (as in T3).

## Metrics

Per graph, per seed. Labels, clean digits and grass patches are used for evaluation only. Each metric is computed in the space the graph is built on.

| metric | definition |
|---|---|
| **graph digit accuracy (primary)** | digit kNN accuracy, k = 15, stratified 5-fold |
| edge purity | the fraction of the symmetrized graph's edges that join same-digit images |
| graph grass recall | kNN recall at k = 15 against the target images' grass patches (lower is better) |
| graph clean-digit recall | kNN recall at k = 15 against the clean digit images (descriptive) |

## Variant selection (seeds 0–2)

The variant with the highest mean graph digit accuracy. Ties within 0.002 go to the simpler variant: smaller r, then smaller k_loc.

## Gate (seeds 0–2)

Phase 2 runs only if the selected variant's graph digit accuracy is at least **0.02 above max(raw, PCA-50) on every selection seed**.

- **If the gate fails, the phase stops after Phase 1.** The report is written with all graph tables. Neither the grid nor the threshold is adjusted.
- **Either way,** the selected variant's graph metrics on confirmation seeds 3–5 are reported (descriptive), together with whether grass recall drops along with the accuracy gain.

## Stop and report

- A residual-construction test fails.
- A dataset hash differs from `data/MANIFEST.md`.
- A determinism check fails.
