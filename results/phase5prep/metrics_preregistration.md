# Pre-registration: global metrics and thresholds for the next method

Committed after the metric validation (`results/phase5prep/metric_validation.md`, commit `58b573e`) and **before any Part B (cPCA) result**. It fixes how the next method (a target kNN graph built in a background-suppressed metric) will be judged. Spec: `Phase_5_Plan.md` §1 A3. The thresholds were set by the project owner after seeing the validation.

## Metric definitions (`eval/global_metrics.py`)

All metrics use k = 15 exact nearest neighbors. Ground-truth labels y_s (4 signal groups) and y_b (6 background clusters) are used for evaluation only.

- **G1, cross-cluster same-signal neighbor rate.** The mean over points of the fraction of their 15 nearest neighbors in the embedding that share the point's signal group **and** belong to a different background cluster. Chance for random neighbors is 0.208 (the exact fraction of point pairs with the same y_s and a different y_b). A perfectly signal-organized layout scores about 0.83.
- **G2, background kNN accuracy.** Accuracy of a 15-nearest-neighbor classifier predicting y_b from the embedding (stratified 5-fold cross-validation, seed 0). Chance is about 0.17. It drops if background islands merge.
- **G3, cell-level layout.** Form the 24 cells (signal group × background cluster) and take each cell's centroid in the embedding. For each cell, find the nearest other centroid; an exact tie goes to the lower cell index. Classify that neighbor as sharing the signal group, sharing the background cluster, or neither. Report the three fractions over the 24 cells.
- **Signal kNN accuracy.** As in Phase 4 (REPORT.md, "Metrics").

## Validation summary (from `results/phase5prep/metric_validation.md`)

| embedding | G1 | G2 | G3 shares signal |
|---|---|---|---|
| Phase 4 runs (a), best (b), (c), (d), every level and seed | 0.000–0.001 | 0.997–1.000 | 0 |
| signal-only reference | 0.816–0.827 | 0.162–0.176 | 1 |

## Pre-registered direction

The next method should:
- raise G1;
- raise G3 "shares signal";
- lower G2;
- not lower signal kNN accuracy below (c)'s, within the tolerance below.

## Thresholds

These apply per level, on the confirmation seeds (easy 3–5, medium 3–5, hard 4–6, or the seeds an equivalent exclusion rule yields).

| # | criterion | threshold | how it is judged |
|---|---|---|---|
| 1 | G1 | ≥ 0.50 | on **every** confirmation seed |
| 2 | G2 | ≤ 0.50 | on **every** confirmation seed |
| 3 | G3 "shares signal" | ≥ 0.50 (at least 12 of 24 cells) | on **every** confirmation seed |
| 4 | signal kNN accuracy | confirmation mean ≥ (c)'s confirmation mean − 0.05: easy ≥ 0.844, medium ≥ 0.532, hard ≥ 0.818 | on the confirmation **mean**, not the strict per-seed rule |

The (c) confirmation means are 0.894 (easy), 0.582 (medium) and 0.868 (hard), from `results/phase4/headline.md`.

**Rationale.** Each threshold sits about halfway between the Phase 4 or chance values and the signal-only reference:
- **G1:** 0 (Phase 4) and 0.21 (chance) against about 0.82;
- **G2:** 1.0 (Phase 4) against about 0.17;
- **G3 "shares signal":** 0 against 1.

Criterion 4 allows a small loss of within-island signal accuracy in exchange for a changed global layout.

## Separate pre-registered comparison with cPCA

The next method is also compared with **(e2) cPCA → Neg-t-SNE** (Part B) using the Phase 4 rule: X improves over Y if the mean over the 3 paired confirmation seeds is higher **and** every per-seed difference is positive (for G2, lower is better). The comparison covers four metrics: signal kNN accuracy, G1, G2 and G3 "shares signal".

This comparison is **reported separately** from the thresholds above. Passing the thresholds and beating (e2) are two distinct results.

## Use on Part B

Criteria 1–4 are also applied **descriptively** to the Part B cPCA runs (e1), (e2), (e3) and (e2-oracle). That application does not change the thresholds.
