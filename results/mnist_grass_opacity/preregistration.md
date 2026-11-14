# Digit opacity on MNIST-on-grass 0/1: pre-registration

This document is committed before any run of this phase.

## Why

At the standard recipe (target = 0.25 × digit + grass), both BC-NE and t-SNE separate 0 from 1: on the 0/1 confirmation
seeds, two-stage (r_a) reached 0.972 digit kNN accuracy and t-SNE (T1) 0.884. This phase makes the digits fainter to
find a range of opacities where **BC-NE still separates the digits and plain t-SNE does not**, and where that range
starts.

- **The method is unchanged.** BC-NE here is (r_a), the residual graph (k_loc = 30, r = 2) with no background
  negatives, in its adopted single-stage form (`REPORT_single_stage.md`): one 750-epoch Neg-t-SNE fit.
- **Two new baselines:** cPCA-2D and local cPCA. From this phase on, every experiment includes them.
- **Platform:** every run is on the cluster, including the w = 0.25 reference. Nothing is compared with earlier runs.

## Data

The six 0/1 datasets (`data/processed/mnist_grass_seed{0..5}.npz`), unchanged. Every run checks its dataset's SHA-256
against `data/MANIFEST.md` (`scripts/bsgraph_phase1.load`).

- **Each file stores the clean digit image (`digits`, in [0, 1]) and the target grass patch (`grass_T`) separately.**
  For opacity w, the target is X_T(w) = w · digits + grass_T, computed in float64 and cast to float32, as the
  stored targets are. The background X_B is unchanged.
- **So at every w the digits, grass patches and background are the same,** and only the opacity changes.
- **Reproduction check, in every run:** X_T(0.25) must equal the stored target bit for bit. Otherwise the run stops.
- **Opacity grid (11 values):** w ∈ {0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01}.
- **Seeds:** 0–2 select, 3–5 confirm.

## Runs

Each run is on one (seed, w).

| run | pipeline | role |
|---|---|---|
| r_a | BC-NE: residual graph (k_loc = 30, r = 2, k = 15) on X_T(w) and X_B, single-stage Neg-t-SNE, 750 epochs, PCA initialization of X_T(w) (exactly `scripts/single_stage_run.fit("single")`) | the method |
| T1 | openTSNE on X_T(w), library defaults, random_state = seed (`scripts/mnist_grass_run.tsne`) | plain t-SNE |
| a | plain Neg-t-SNE: the raw kNN graph (k = 15) of X_T(w), otherwise exactly as r_a | descriptive |
| T3 | cPCA (`baselines/cpca.py`) of X_T(w) against X_B at the label-free α, top 10 components → openTSNE as T1 | descriptive |
| cPCA-2D | the 2-D cPCA projection at the same label-free α, used as the embedding (same cPCA fit as T3) | descriptive |
| local cPCA | `baselines/local_cpca.py`, label-free version: k-means on X_B, K ∈ {2, …, 10} by silhouette; each target to its nearest centroid; per cluster, cPCA at its label-free α, top 10 directions; Z = [local coordinates \| s · one-hot]; then exactly as a, with Z in place of X_T(w) | descriptive |

- **Local cPCA when it can't run:** if a cluster has fewer than 50 target or 50 background points, the run is
  recorded as not applicable at that (seed, w), with no embedding.
- **Code:** `scripts/opacity_run.py`. T3 and cPCA-2D share one process.

## Metrics

Computed on the 2-D embedding of the 5000 target images.
- **Digit kNN accuracy** (primary): k = 15, stratified 5-fold, 2 classes. Chance is 0.5.
- **Secondary, no verdicts:**
  - ARI: k-means with k = 2, n_init = 10, random_state = seed;
  - recall vs clean digits and recall vs grass: kNN recall at k = 15.

## Selection (seeds 0–2)

All six runs at every w on seeds 0–2.
- **Window:** W = {w : mean r_a accuracy ≥ 0.80 and mean T1 accuracy ≤ 0.60}, means over seeds 0–2.
- **w\*** is the **largest w in W**, the most visible digit at which t-SNE is already near chance.
- **If W is empty,** there is no w\* and no confirmation verdicts. The report then gives the curves and, descriptively,
  the w with the largest mean r_a − T1 difference.
- `selection.json` (W, w\*, the means) is committed before any confirmation-seed run.

Choosing w\* uses labels. That is acceptable because w sets how hard the data are, not a setting of any method.

## Confirmation (seeds 3–5)

All six runs at every w on seeds 3–5. Pre-registered at w\* only:
- **C1:** r_a accuracy ≥ 0.80 on every confirmation seed.
- **C2:** T1 accuracy ≤ 0.60 on every confirmation seed.
- **C3:** r_a vs T1 under the usual rule. "Consistent across 3 paired seeds" means the mean paired difference is > 0
  and every per-seed difference is > 0. Otherwise the result is "not consistent".

## Descriptive, no verdicts

- **Curves:** every run's accuracy against w, on the selection seeds, the confirmation seeds, and all six.
- **Crossing points:** where mean r_a accuracy falls below 0.80, and where mean T1 accuracy falls to 0.60. Both are
  found by linear interpolation between grid points, on all six seeds.
- **Everything else:** a, T3, cPCA-2D and local cPCA at every w, and the secondary metrics.

## Visual judgment by the project owner

After the confirmation job, an inspection figure shows T1 and r_a at every w (seed 3), colored by digit and in plain
gray. Panel titles give only w.
- **The project owner records,** for each w, whether t-SNE separates the digits (yes / no / partly), and optionally
  the same for r_a.
- **The record** goes verbatim into `visual_judgment.md`, committed before the report.
- **In the report** it appears as a separate column, labeled as the project owner's visual judgment. It does not
  change C1–C3. Where it disagrees with the 0.60 rule, the report shows both and lists those w.

## Checks

- **Determinism:**
  - every run type is rerun at (seed 0, w = 0.25) in the selection job, and at (seed 3, w\*) in the confirmation
    job;
  - each rerun must be bit-identical (embedding SHA-256);
  - a failed rerun stops the merge.
- **Completeness:** the merge stops unless every expected run is present.
- **Run records:** every run records the environment, the Slurm job, 1 thread and a clean commit.
