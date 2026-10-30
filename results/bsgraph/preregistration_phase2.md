# Background-suppressed graph, Phase 2 pre-registration (embeddings)

Committed after the Phase 1 gate passed (`results/bsgraph/phase1.md`, commit `e594d76`), and before any Phase 2 embedding. The data and compute are the same as in Phase 1:
- the MNIST-on-grass datasets for seeds 0–5, with hashes checked against `data/MANIFEST.md`;
- the Math Slurm cluster, CPU only, 1 thread per process.

## Selected residual graph

**k_loc = 30, r = 2**, chosen in Phase 1. For each target image, take its 30 nearest background images, subtract their mean and their top 2 principal directions, then build the kNN graph (k = 15, union-symmetrized) on these residuals.

## New runs

Both use two-stage Neg-t-SNE on the residual graph, with the same initialization as (a) and (c): PCA of the raw X_T. Stage 1 runs 250 epochs and Stage 2 runs 500 epochs.

- **(r_a):** background term off.
- **(r_c):** ξ_B negatives, computed on raw pixels exactly as in (c): k_B = 15, L = 30, Jaccard weights, tie cap 2L. m_B ∈ {1, 2, 5, 10, 20, 40}.

**Selection** (seeds 0–2): the m_B for (r_c) with the highest mean digit kNN accuracy; ties go to the smaller m_B. It is committed before any confirmation run.

**Confirmation** (seeds 3–5): (r_a), and (r_c) at the selected m_B. (r_c) on seed 3 is run a second time and must be bit-identical.

## Reused results (confirmation seeds 3–5, from the MNIST-on-grass phase)

These are reused: (a), (c) at m_B = 5, T1, T2 and T3. Before reusing them, the script checks two things:
1. each dataset's SHA-256 against `data/MANIFEST.md`;
2. that the code (a) and (c) depend on is unchanged since commit `5556465`, the commit those results ran on (`git diff --quiet 5556465 HEAD`). The files covered are `common/embed.py`, `common/knn.py`, `common/experiment.py`, `common/repro.py`, `method/bg_term.py`, `method/xi_b.py`, `data/mnist_grass.py`, `eval/metrics.py` and `scripts/mnist_grass_run.py`.

If that code has changed, (c) on seed 3 is rerun and must be bit-identical before the stored results are reused. A dataset-hash mismatch stops the phase.

## Metrics

These are as in the MNIST-on-grass phase. All are computed on the 2-D embedding of the target images:
- **digit kNN accuracy** (primary; k = 15, stratified 5-fold);
- kNN recall at 15 against the clean digits;
- kNN recall at 15 against the target images' grass patches (lower is better);
- the flags (collapse vs (r_a) on the same seed, and fragmentation), for review only.

## Comparisons

All on confirmation seeds 3–5 and digit kNN accuracy, under "consistent across 3 paired seeds". The secondary metrics are reported alongside, without verdicts.

1. (r_c) vs (c): does the residual graph improve the method?
2. (r_c) vs T3 (cPCA → t-SNE).
3. (r_c) vs T1, and (r_c) vs T2.
4. (r_c) vs (r_a): does ξ_B still add anything on top of the residual graph?

Descriptive only: (r_a) vs (a).

## Stop and report

- A dataset hash mismatch.
- The reuse check requires a (c) rerun, and that rerun is not bit-identical.
- A determinism check fails.
- A run is non-finite or its scale grows by more than 10×.
