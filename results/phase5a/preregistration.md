# Phase 5a pre-registration (frozen before any (b), (c) or oracle-control run)

## Datasets

The rule for each design's ρ is the Phase 1 rule: the largest grid value whose plain-Neg-t-SNE run (a) is valid on
seed 0 (background accuracy ≥ 0.90, signal accuracy ≤ 0.50, signal accuracy in the signal itself ≥ 0.95), confirmed
on seeds 0–2.

| Design | Generator | Setting |
|---|---|---|
| J | `data/synthetic_j.py` | θ = 60°, Δ projected off span(B), MLP gain g = 2 (configuration t60_proj_g2), ρ_J = 0.812404 |
| K | `data/synthetic_k.py` | ρ_K = 0.2 |

In J, Ã z_s and Δ̃ are each scaled to unit total variance per dataset; the scale factors are recorded per seed.

n_T = n_B = 5000 and D = 50. Selection seeds are 0–2; confirmation seeds start at 3.

## Status at the freeze (from the calibration CSVs)

- **Design K is stopped.** It failed V3-K on every selection seed ((e2) WC 0.970–0.980 > 0.80), so it has no main runs. The K rules below are kept as written but are not applied.
- **Known caveat for J** (selection seeds 0–2): the chosen configuration meets V3 only through the G1/G2 thresholds. (e2) has G1 0.104–0.145 and G2 0.879–0.917, but signal accuracy 0.983–0.988, WC 0.983–0.988 and G3 shares signal 1.000–1.000. On seed 0, its embedding is organized by signal globally, with background sub-lobes inside each signal blob. The interaction term, averaged over the signal, depends on the background cluster: a background main effect outside span(B) that X_B never shows, so cPCA cannot remove it.

## Metrics

All metrics use k = 15 and the true labels. kNN accuracies use the existing classifier: stratified 5-fold CV, seed 0.

| Metric | Definition |
|---|---|
| signal accuracy | kNN accuracy of y_s in the 2-D embedding |
| background accuracy (G2) | kNN accuracy of y_b in the 2-D embedding |
| WC | signal kNN accuracy computed inside each true background cluster, size-weighted over clusters |
| G1 | mean fraction of a point's 15 embedding neighbors with the same y_s and a different y_b |
| G3 "shares signal" | fraction of the 24 cells (y_s × y_b) whose nearest other cell centroid shares y_s but not y_b |
| coherence (data) | mean over g and cluster pairs k ≠ k′ of cos(d_kg, d_k′g), d_kg = cell mean − cluster mean in X_T |
| detectability (data) | WC on high-D X_T |

## Runs

- (a): plain two-stage Neg-t-SNE.
- (b): uniform extra negatives.
- (c): ξ_B negatives, built as in Phase 2.
- (o): the oracle same-cluster control. m_B extra negatives are drawn uniformly from the other members of the head's
  true background cluster. It is an oracle: it uses labels.
- (b), (c) and (o) each run at m_B ∈ {1, 2, 5, 10, 20} and share one Stage 1 per seed.
- (e2): cPCA → Neg-t-SNE on the top 10 components. (e3): cPCA → UMAP on the same input. Both use the label-free α of
  Phase 5 prep.
- Design K only: (e2-k20) and (e2-k40), cPCA → Neg-t-SNE on the top 20 and top 40 components at the same α as (e2).
  V3-K stays defined with the 10-component (e2).

## Selection (seeds 0–2)

For each of (b), (c) and (o), the m_B with the highest mean over the selection seeds is chosen; ties go to the smaller
m_B. The selection metric is signal accuracy for J and WC for K.

## Confirmation (fresh seeds from 3)

- **Runs:** (a), best (b), best (c), best (o), (e2) and (e3); for K also (e2-k20) and (e2-k40).
- **Exclusion:** a seed is excluded if it fails V1 ((a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90), V2
  (detectability ≥ 0.85), or, for J, V4 (coherence ≥ 0.5). These criteria use the data and (a) only, never (e2).
  The next seed is then drawn, until 3 valid seeds exist. More than 2 exclusions in a design means stop and report.
- **Listing:** excluded seeds are listed with their validity numbers and enter no comparison.

## Comparisons (confirmation seeds)

- **Rule:** "consistent across 3 paired seeds" (Phase 4): the mean paired difference is > 0 and every per-seed
  difference is > 0, with the sign flipped where lower is better. Mean and per-seed values are reported, with no
  p-values.
- **Design J:** (c) vs (e2), and (c) vs (o), each on signal accuracy, G1, G2 (lower is better) and G3 "shares
  signal", reported separately. The frozen thresholds G1 ≥ 0.50, G2 ≤ 0.50 and G3 "shares signal" ≥ 0.50 are also
  checked per seed for every run.
- **Design K:** (c) vs (e2), (c) vs (e2-k20), (c) vs (e2-k40), and (c) vs (o), on WC (primary). Signal accuracy is
  secondary. (c) "beats cPCA" only if it is consistent across 3 paired seeds against each of (e2), (e2-k20) and
  (e2-k40). G1–G3 are reported but marked ill-posed for K: same-label groups in different clusters are unrelated.
- **Descriptive only** (no verdict): (c) vs (a), (c) vs best (b), and (c) vs (e3).

## Predictions

`results/phase5a/predictions.md` (P1–P3) is reported as confirmed, not confirmed or not applicable, unrevised.

## Stop and report

- ξ_B fails the Phase 2 gates (same-y_b precision ≥ 0.90, same-y_s excess ≤ 0.10) on any seed.
- Any run is non-finite or its scale grows by more than 10×.
- The rerun of selection seed 0 is not bit-identical.
- More than 2 confirmation seeds are excluded in a design.
