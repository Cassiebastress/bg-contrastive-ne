# Phase 5b predictions

These are committed before any Design L calibration result, and before any Design L dataset is generated. Each prediction is reported in `REPORT_phase5b.md` as **confirmed** or **not confirmed** (or **not applicable** if its design is not run). None will be revised.

## Definitions used below

| Term | Meaning |
|---|---|
| WC | within-cluster signal accuracy: signal kNN accuracy (k = 15, stratified 5-fold CV, seed 0) computed inside each true background cluster, size-weighted over clusters |
| linear ceiling | the best WC over labeled linear projections, at the **embedding level**: each projection is embedded with (a)'s two-stage Neg-t-SNE, then its WC is scored. L-cyclic: the 64 subsets of the six Q blocks, each together with U. L-between: U alone, and U ∪ V. |
| cPCA variants | cPCA → Neg-t-SNE on the top k components, for every α representative × k ∈ {3, 5, 10, 20}, plus, for each k, the α on the full grid with the best high-D WC, embedded |
| best cPCA | the per-seed maximum WC over all cPCA variants, chosen with labels. It is a conservative upper bound, not a fair baseline. |
| beats | consistent across 3 paired confirmation seeds: the mean difference is > 0 and every per-seed difference is > 0. For "best cPCA", (c) must exceed the per-seed maximum on every seed. |

## Change to P1 before any result

The spec defined the linear ceiling as the high-D kNN WC of the projected data, with no embedding. That definition conflicts with V2. One of the 64 subsets is "all six blocks + U". Projecting onto it removes only the isotropic noise dimensions, so its high-D WC is at least the detectability, while V2 requires detectability ≥ 0.85 and V3 requires the ceiling ≤ 0.80. The project owner therefore moved the ceiling, and P1 with it, to the embedding level. This was decided during plan review, before any Design L dataset was generated. The high-D kNN ceilings are still reported, as descriptive.

## Predictions

- **P1 (L-cyclic):** the embedding-level linear ceiling is about 0.5–0.7, because at most 3 of the 6 clusters can have their signal revealed without their nuisance by any block subset. "About 0.5–0.7" is judged as a mean over selection seeds in [0.50, 0.70].
- **P2 (L-cyclic):** (c) beats (e2) on WC, and beats best cPCA, on the confirmation seeds.
- **P3 (L-between):** label-free (e2) fails V3 (WC ≤ 0.80): it selects α > 1, which suppresses U and the signal with it. The cPCA oracle ceiling is high: at least 0.90 on every selection seed (embedding level).
- **P4 (L-between):** (c) beats label-free (e2) on WC but does not beat best cPCA.
- **P5 (both designs):** (c) beats (o) and (b) on WC.
