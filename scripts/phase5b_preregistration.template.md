# Phase 5b pre-registration (frozen before any (b), (c) or oracle-control run)

## Designs

- **Generator:** `data/synthetic_l.py`; {{rho}} (Phase 1 rule).
- **Valid designs, run in the main experiment:** {{designs}}.
- **Exclusions:** {{excluded}}.
- **Data:** n_T = n_B = 5000, D = 50, with exactly balanced clusters, identical in X_T and X_B.
- **Seeds:** selection seeds 0–2; confirmation seeds start at 3.

## Metrics

| metric | definition |
|---|---|
| WC (primary) | signal kNN accuracy (k = 15, stratified 5-fold CV, seed 0) inside each true background cluster, size-weighted over clusters |
| signal accuracy | kNN accuracy of y_s in the embedding |
| G1, G1 at k = 200, G2, G3 | as defined in Phase 5 prep (descriptive only) |

## Runs

- (a): plain two-stage Neg-t-SNE.
- (b): uniform extra negatives, m_B ∈ {1, 2, 5, 10, 20, 40}.
- (c): ξ_B negatives (built as in Phase 2), m_B ∈ {1, 2, 5, 10, 20, 40}.
- (o): oracle same-cluster negatives, m_B ∈ {1, 2, 5, 10, 20, 40, 80}.
- (e2) and (e3): cPCA → Neg-t-SNE / UMAP on the top 10 components at the label-free α.
- **cPCA variants:** cPCA → Neg-t-SNE for every α representative × k ∈ {3, 5, 10, 20}, plus, for each k, the α with the best high-D WC on the full grid.
- **Best cPCA:** the per-seed maximum WC over all cPCA variants, chosen with labels. It is a conservative upper bound.
- **Shared Stage 1:** (a), (b), (c) and (o) share one Stage 1 per seed.

## Selection (seeds 0–2)

For each of (b), (c) and (o), the m_B with the highest mean WC; ties go to the smaller m_B.

## Confirmation (fresh seeds from 3)

- **Runs:** (a), best (b), best (c), best (o), (e2), (e3) and all cPCA variants. Also, as descriptive runs: every labeled linear projection, embedded (L-cyclic: the 64 Q-block subsets with U; L-between: U, and U ∪ V).
- **Exclusion:** a seed is excluded if it fails V1 ((a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90), V2 (detectability ≥ 0.85) or V2b (high-D signal kNN accuracy ≥ 0.65). These criteria use the data and (a) only. The next seed is drawn until 3 valid seeds exist; more than 2 exclusions means stop.

## Pre-registered comparisons (confirmation seeds, on WC; "consistent across 3 paired seeds")

1. (c) vs label-free (e2).
2. (c) vs best cPCA: (c) must exceed the per-seed maximum over all cPCA variants on every seed.
3. (c) vs (o).
4. (c) vs (b).

**Framing for L-cyclic:** does (c) beat every "linear projection → standard Neg-t-SNE" pipeline, including the label-chosen ones?

**Descriptive** (no verdict): (c) vs the per-seed embedded linear ceiling on the confirmation seeds, under the same paired rule.

## Predictions

`results/phase5b/predictions.md` (P1–P5) is reported as confirmed or not confirmed, unrevised.

## Stop and report

- ξ_B fails the Phase 2 gates on any seed.
- Any run is non-finite or its scale grows by more than 10×.
- The rerun of selection seed 0 is not bit-identical.
- More than 2 confirmation seeds are excluded.
