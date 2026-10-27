# Phase 5a report: benchmarks that break cPCA's additive assumption

Phase 5 prep (`REPORT_phase5prep.md`) found that cPCA → Neg-t-SNE nearly solves every Phase 4 dataset. The reason was structural: in each Phase 4 generator the signal was added to the background independently, so in the population the target covariance minus the background covariance equals the signal covariance. That identity is exactly what cPCA exploits.

This phase built two benchmark families designed to break the identity. It asked two questions:

1. **Design J.** Is there a regime where cPCA fails but the signal is still shared across background clusters, so that a global, signal-organized layout is meaningful?
2. **Design K.** When each background cluster has its own signal directions, do ξ_B negatives (run (c)) beat cPCA at revealing the signal within clusters?

No new embedding method was built.

## How to read this report

**Shared definitions.**
- `REPORT.md` (repository root) defines the project, the two-stage Neg-t-SNE, runs (a)–(c), ξ_B, the flags, and the "consistent across 3 paired seeds" rule. See its sections "How to read this report" and "Method and setup".
- `REPORT_phase5prep.md` defines the global metrics G1–G3, the cPCA runs (e2) and (e3), and the label-free α rule.

This report uses those definitions without repeating them, except where it restates them for clarity.

**The Phase 5a spec.** The project owner gave the plan for this phase in conversation; it is not a file in the repository. The section "Design" below states everything from it that the results depend on. The owner changed two things while reviewing the implementation plan:
- J's knobs became a fixed-strength mixing angle θ, with the projection and the MLP gain as grid factors.
- (e2-k20) and (e2-k40) were added for Design K, together with predictions P1–P3 and the data diagnostics.

**Files referred to:**

| short name | file | contents |
|---|---|---|
| **grid prediction** | `results/phase5a/grid_J.md` (top section) | the Design J prediction, committed before any J dataset was evaluated |
| **predictions** | `results/phase5a/predictions.md` | P1–P3, with their status at commit time |
| **pre-registration** | `results/phase5a/preregistration.md` | datasets, metrics, selection and comparison rules, committed before any (b), (c) or (o) run |
| results | `results/phase5a/` | CSV/JSON results, generated tables (`*.md`) and figures |
| scripts | `scripts/phase5a_*.py`, `common/benchmark.py`, `data/synthetic_j.py`, `data/synthetic_k.py`, `eval/benchmark_metrics.py` | the code that produced every result |

Tables marked **(generated)** are copied verbatim from `results/phase5a/*.md` by `scripts/phase5a_assemble.py`; none is typed by hand. A seven-character code such as `9ed1110` is a commit in this repository's git history.

**Run labels:**

| label | run |
|---|---|
| (a) | plain two-stage Neg-t-SNE on X_T |
| (b) | (a) plus m_B uniform extra negatives per edge |
| (c) | (a) plus m_B ξ_B negatives per edge (background-neighborhood negatives, built as in Phase 2) |
| (o) | (a) plus m_B **oracle same-cluster** negatives per edge: drawn uniformly from the other points of the head's *true* background cluster. It uses labels, so it is a control, not a method. It runs through the same sampling and loss code as (c). |
| (e2) | cPCA → Neg-t-SNE: the top 10 cPCA components at the label-free α, embedded like (a) |
| (e3) | cPCA → UMAP on the same 10-D input |
| (e2-k20), (e2-k40) | Design K only: (e2) on the top 20 or top 40 components, at the same α |

**New metrics** (k = 15; kNN accuracy uses the stratified 5-fold classifier of `REPORT.md`):

| metric | definition |
|---|---|
| WC | within-cluster signal accuracy: signal kNN accuracy computed inside each true background cluster separately, size-weighted over clusters. It measures local signal separation whatever the global layout. |
| detectability | WC computed on high-D X_T, before any embedding: whether the signal is locally present at all |
| coherence | for background cluster k and signal group g, d_kg = (mean of X_T over cell (k, g)) − (mean of X_T over cluster k). Coherence is the mean over g and cluster pairs k ≠ k′ of cos(d_kg, d_k′g). About 1 when "same signal" points the same way in every cluster; about 0 for unrelated cluster-specific signals. |

## Summary

1. **Design K stopped at calibration: cPCA already reveals the signal within clusters.**
   - (e2) reaches WC 0.970–0.980, against the validity bar of ≤ 0.80. (e2-k20) and (e2-k40) reach 0.978–0.984.
   - **Why:** K's per-cluster signal maps lie outside the background's span, and each has mean zero, so the target-minus-background covariance is still exactly the (now 18-dimensional) signal covariance. Breaking *independence* this way does not hurt cPCA.
   - **The diagnostics agree:** along K's signal directions, the background's global and within-cluster variances are equal (ratio 1.00), the case in which (c) is predicted to have no advantage.
   - **Consequence:** (c) was never run on K, so question 2 is unanswered.
2. **Design J has one valid configuration, but it is valid only through local metrics.**
   - Of 16 configurations, only θ = 60°, projected, g = 2 meets V1–V4 on every selection seed. At θ ≤ 30°, cPCA solves the dataset (V3 fails). At θ ≥ 60°, every other configuration fails V2 on at least one seed: the signal is no longer locally detectable in high-D.
   - In that configuration, (e2) still has signal accuracy 0.98 and G3 "shares signal" = 1. Its embedding is organized by signal globally, but each signal blob splits into background sub-lobes.
   - G1 and G2 look only at the 15 nearest neighbors, which fall inside one sub-lobe, so they read "background-organized" and V3 counts cPCA as failing.
   - **Mechanism** (interpretation): the interaction Δ = M(z_b, z_s) − M(z_b, 0), averaged over the signal, still depends on the background cluster. This background main effect lies outside span(B) and never appears in X_B (Δ is 0 at the reference signal), so cPCA cannot remove it. It accounts for about 9% of the signal variance here, and coherence, which subtracts cluster means, does not see it.
3. **On fresh seeds, cPCA passed the frozen global thresholds on 2 of 3 seeds.**
   - On confirmation seeds 3 and 4, the label-free α was 37 and 29, and (e2) met G1 ≥ 0.50, G2 ≤ 0.50 and G3 ≥ 0.50 (G1 0.63 and 0.61).
   - On seed 5 (α = 4.4), and on the selection seeds (α 2.7–3.5), G1 was 0.10–0.15.
   - So J's "cPCA fails" property depends on which α representative the label-free rule selects. The pattern is consistent with large α removing the sub-lobes, but it was not tested directly.
4. **(e2) beats (c) on every pre-registered metric, consistently across 3 paired seeds.**
   - Signal accuracy: 0.98 vs 0.86. G1: 0.45 vs 0.00. G2: 0.60 vs 1.00. G3: 1.00 vs 0.00.
   - (c) reveals the signal within background islands (signal accuracy 0.86, up from 0.30 for (a)), but its global layout stays organized by background, as in Phase 4.
5. **(c) beats the oracle same-cluster control by +0.39 signal accuracy, consistently across 3 paired seeds** (0.86 vs 0.47).
   - This is the control `REPORT.md` listed as Next steps #1. Repelling random points of the same true background cluster spreads each island but recovers little signal; ξ_B's fine-grained background neighborhoods account for most of (c)'s gain.
   - **Caveats:** shown here on one dataset family. (o)'s selection mean was still rising at the largest m_B (20).
6. **Predictions.**
   - **P1 not confirmed:** projected configurations failed V3 with coherence still high, through the hidden background main effect.
   - **P2 not confirmed** (descriptive): the one valid configuration is projected.
   - **P3 not applicable** for both designs: K has no main runs, and J's chosen configuration is projected.
7. **Overall answer to question 1: no.** Within this family, no J configuration was found where cPCA fails robustly while the signal stays shared.
   - **The only valid configuration** fails V3 through background sub-structure inside a signal-first layout, and only at some α.
   - **The configurations predicted to favor (c)** are the unprojected ones (global/local background-variance ratio 46–69). At θ = 30° cPCA solved them; at θ ≥ 60° they failed V2. None was run in the main runs.

## Design

### Design J: jointly nonlinear signal (`data/synthetic_j.py`)

    target:      X_T = B z_b + ρ (cos θ · Ã z_s + sin θ · Δ̃) + ε
    background:  X_B = B z_b′ + ε′
    Δ = M(z_b, z_s) − M(z_b, s0)      (projected off span(B) when "proj")

- **Shared with Phase 4 "easy":** the latents z_b and z_s, the maps B and A, the noise, and X_B are all those of "easy", bit for bit (tested). The reference signal s0 = 0 is the signal-simplex centroid, so X_B is the target generator evaluated at s0.
- **M** is a fixed random 2-layer tanh MLP: input [z_b / std_b ; z_s / std_s], 64 hidden units, first-layer weights scaled by the gain g. It is drawn from a separate seed stream.
- **Normalization:** Ã z_s and Δ̃ are each scaled to unit total variance per dataset, so θ mixes a background-independent and a background-dependent signal at fixed total strength.
- **Grid:** θ ∈ {0°, 30°, 60°, 90°} × projection P ∈ {projected, unprojected} × g ∈ {2, 6}, 16 configurations. At θ = 0° the dataset doesn't depend on P or g.
- **ρ_J = 0.812** (= 0.2 · √16.5), from the Phase 1 rule at θ = 0°.

### Design K: cluster-specific signal directions (`data/synthetic_k.py`)

    X_T = B z_b + ρ A_{y_b} z_s + ε,    X_B = B z_b′ + ε′

- Each background cluster k has its own random D × 3 map A_k with orthonormal columns, orthogonal to span(B).
- Everything else is "easy", bit for bit.
- **ρ_K = 0.2**, from the Phase 1 rule.

### Calibration and validity (selection seeds 0–2; runs (a), (e2), (e3) only)

| criterion | meaning | condition |
|---|---|---|
| V1 | background dominates | (a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90 |
| V2 | signal present | detectability ≥ 0.85 |
| V3 (J) | cPCA does not solve it | (e2) fails at least one of: G1 ≥ 0.50, G2 ≤ 0.50, G3 "shares signal" ≥ 0.50, signal accuracy > 0.80 |
| V3-K | cPCA does not reveal the within-cluster signal | (e2) WC ≤ 0.80 |
| V4 (J) | signal still shared | coherence ≥ 0.5 |

- **Validity:** a configuration is valid if it meets every criterion on every selection seed.
- **J choice:** the valid configuration with the largest mean coherence; ties go to the smaller θ, then projected, then the smaller g.
- **Stops:** the sanity configuration θ = 0° had to fail V3 (cPCA solves the additive case), and it did. K stops if V3-K fails.

### Main runs (Design J only; rules from the pre-registration)

- **Selection** (seeds 0–2): (a); (b), (c) and (o) at m_B ∈ {1, 2, 5, 10, 20}; (e2) and (e3). Within each group, m_B is chosen by the highest mean signal accuracy, with ties going to the smaller m_B.
- **Confirmation** (fresh seeds from 3): (a), best (b), best (c), best (o), (e2), (e3).
- **Exclusion:** a confirmation seed is excluded if it fails V1, V2 or V4 (judged from the data and (a), never from (e2)); at most 2 exclusions are allowed. No seed was excluded.
- **Pre-registered comparisons:** (c) vs (e2), and (c) vs (o), each on signal accuracy, G1, G2 and G3 "shares signal", under "consistent across 3 paired seeds".

## Results

### Pre-registered predictions (generated)

{{predictions}}

### Design J calibration grid (generated)

The top section of `results/phase5a/grid_J.md`, not repeated here, holds the prediction exactly as committed before the grid ran (`c93c891`).

{{grid_J}}

### Validity per seed (generated)

{{validity}}

### Data diagnostics (descriptive; generated)

{{diagnostics}}

The "background main effect" column was added after the grid results (descriptive). It quantifies the effect described in Summary item 2.

### Design K calibration runs (generated)

{{k_calibration}}

### Design J main runs (generated)

{{tables_J}}

### Figures

**Design J grid** (selection seeds 0–2): mean and range against θ. Dotted lines mark the V2, V3 and V4 bars.

![Design J grid](results/phase5a/grid_J.png)

**Design J main runs**, selection seed 0, colored by signal y_s (left) and background y_b (right):

![Design J embeddings](results/phase5a/embeddings_J.png)

**Design K calibration runs**, selection seed 0:

![Design K embeddings](results/phase5a/embeddings_K_calibration.png)

## Interpretation

- **Why K did not break cPCA.** cPCA needs only that the target-minus-background covariance points at the signal. In K, cluster-specific maps make the signal *depend* on the background, but each cluster's signal has mean zero and lies outside the background's span. So no cross-covariance with the background appears, and the identity survives with a higher-dimensional signal covariance. Ten components already reveal each cluster's signal (WC 0.97), and 20 or 40 do slightly better (0.98).
- **Why J's valid configuration is a weak test.** The non-additive term does more than make the signal background-dependent: it adds a background-dependent *mean shift* outside the background's span. The background data never shows this shift, so cPCA keeps it, and the embedding shows background sub-lobes inside signal blobs. The frozen G1 and G2 thresholds, which read local neighbor composition, score this as cPCA failing. G3, which reads the arrangement of cells, scores it as signal-organized (1.00 on every seed).
- **What (c) shows here.** (c) behaves as in Phase 4. It separates the signal within each background island (signal accuracy 0.86, WC 0.86) and leaves the global layout organized by background (G1 = 0, G2 = 1). The oracle same-cluster control isolates where that gain comes from: the same number of repulsions, drawn from the right background cluster but without ξ_B's neighborhood structure, recovers far less (0.47).
- **Coherence caveat.** Coherence measures alignment in ambient coordinates. It can call a signal incoherent even when the signal is consistent in the background's own frame (for example, with a curved background). V4 was not changed for this; it applies to every low-coherence result above. In this grid, every J configuration had coherence of at least 0.56 on every seed.

## Limitations

- **One valid J configuration, and its V3 status did not replicate.** cPCA failed the frozen thresholds on 1 of 3 confirmation seeds, and the outcome tracked the label-free α. V3 was judged with one α per dataset; a check across all α representatives was not part of the protocol.
- **G1 and G2 are local.** They count neighbors among the 15 nearest, so they cannot tell a background-organized layout from a signal-organized layout with background sub-structure. G3 can. In this phase it disagreed with G1/G2 for (e2) on every selection seed and on confirmation seed 5.
- **The regime predicted to favor (c) was not tested.** Unprojected J configurations have background variance along the signal that is 46–69 times larger globally than within clusters. At the calibrated ρ, cPCA solved them at θ = 30°, and they failed V2 at θ ≥ 60°, so none was run.
- **The oracle control's m_B grid may be too short.** (o)'s mean signal accuracy was still rising at m_B = 20 (0.45 on the selection seeds).
- **Synthetic data only**, with discrete background clusters, as in all earlier phases.

## Next steps

1. **Remove the hidden background main effect from J.** Center the interaction, using Δ − E[Δ | z_b] (estimated per background state), then rerun the grid. This tests whether cPCA's failure then requires low coherence, as the grid prediction expected.
2. **Test where the diagnostics predict an advantage for (c).** Use unprojected or partially projected configurations with a large global/local background-variance ratio and with detectability kept above the V2 bar (e.g. smaller θ or g, or a larger ρ checked against V1).
3. **Judge "cPCA fails" across all α representatives,** not only the label-free choice, so a benchmark counts as hard for cPCA only if no representative solves it.

## Reproducibility

Run with the cne-bg environment, in this order:

    python scripts/phase5a_calibrate.py --step rho
    python scripts/phase5a_calibrate.py --step grid
    python scripts/phase5a_calibrate.py --step k
    python scripts/phase5a_calibrate.py --step summary
    python scripts/phase5a_calibrate.py --step diagnostics
    python scripts/phase5a_calibrate.py --step freeze
    python scripts/phase5a_run.py --design J --stage selection
    python scripts/phase5a_run.py --design J --stage confirmation
    python scripts/phase5a_run.py --design J --recheck
    python scripts/phase5a_report.py
    python scripts/phase5a_assemble.py

**Order of commits:**
- J grid prediction: `c93c891`, before any J dataset.
- Predictions P1–P3: `4336ea0`.
- (e2-k20) and (e2-k40) added: `21585f0`, before K calibration.
- Calibration results: `843e189`.
- Freeze: `9ed1110`, before any (b), (c) or (o) run.
- J selection: `2aa4462`.

**Checks:**
- The rerun of J selection seed 0 is bit-identical in every row and embedding (`results/phase5a/recheck_J.json`).
- The Phase 4 easy recheck stayed bit-identical after this phase's code changes (`results/phase5a/phase4_recheck_after_5a_changes.json`).
- Every results JSON records the environment.
