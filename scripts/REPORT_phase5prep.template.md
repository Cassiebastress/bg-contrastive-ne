# Phase 5 prep report: global metrics and the cPCA baseline

This report covers the preparation for the next method in the background-contrastive Neg-t-SNE project: a target kNN graph built in a background-suppressed metric. Before building it, two things were needed:
- global metrics that can register whether an embedding is organized by signal rather than by background;
- the contrastive PCA (cPCA) baseline that the next method must beat.

No new method was built.

## How to read this report

**Shared definitions.** The project, data, runs (a)–(d), notation and Phase 4 metrics are defined in `REPORT.md` (repository root), in its sections "How to read this report" and "Method and setup". This report uses them without repeating them.

**Documents and files referred to:**

| short name | file | contents |
|---|---|---|
| **spec** | `Phase_5_Plan.md` (repository root) | the plan for this phase, written by the project owner; §1 is Part A, §2 is Part B |
| **REPORT.md** | `REPORT.md` (repository root) | the Phase 4 report |
| **pre-registration** | `results/phase5prep/metrics_preregistration.md` | global-metric definitions and thresholds, committed before any cPCA result |
| results | `results/phase5prep/` | CSV/JSON results, generated tables (`*.md`) and figures |
| scripts | `scripts/phase5prep_*.py` | the scripts that produced every result |

Tables marked **(generated)** are copied verbatim from `results/phase5prep/*.md` by `scripts/phase5prep_assemble.py`; none is typed by hand. A seven-character code such as `23459ff` is a commit in this repository's git history.

**New run labels:**

| label | run |
|---|---|
| (e1) | cPCA-2D: the 2-D cPCA projection itself, used as the embedding |
| (e2) | cPCA → Neg-t-SNE: the top 10 cPCA components, embedded with the same two-stage Neg-t-SNE as run (a) |
| (e3) | cPCA → UMAP: the same 10-D input, embedded with UMAP |
| (e2-oracle) | (e2) at the **oracle α chosen by 10-D cPCA signal accuracy**. It uses labels, so it is an upper bound, not a fair baseline |

## Summary

1. **The global metrics work.**
   - **Phase 4 runs:** every run scores G1 ≈ 0, G2 ≈ 1 and G3 "shares signal" = 0. That is a background-organized layout, even where (c) recovers the signal within islands.
   - **Signal-only reference:** a reference embedding of the clean signal scores G1 ≈ 0.82, G2 ≈ 0.17 (chance) and G3 "shares signal" = 1.
   - **Thresholds:** the project owner set them after seeing this, and they were committed before any cPCA result (`23459ff`).
2. **cPCA nearly solves all three synthetic levels.** cPCA → Neg-t-SNE (e2) and cPCA → UMAP (e3) pass all four pre-registered criteria on every level:
   - signal accuracy 0.977–0.989;
   - G1 0.80–0.82;
   - G2 ≈ 0.17;
   - G3 "shares signal" = 1;

   which is about the signal-only reference. (e2) improves on Phase 4's (c) in signal accuracy, G1, G2 and G3, consistently across 3 paired seeds at every level. The signal-accuracy gain is +0.10 on easy, +0.40 on medium and +0.11 on hard. Even the raw 2-D projection (e1) meets all four criteria on medium and hard.
3. **The spec's expectation did not hold.** The spec expected linear suppression to fall short on medium (overlap) and especially on hard (nonlinear background). It did not. **Interpretation (post-hoc):** in every level, a sizeable share of the signal lies in directions where the background has essentially no variance. That share is 100% on easy, 50% on medium and 17–35% on hard (table below), and cPCA only needs that part.
4. **α selection is not a weak point here.**
   - **10-D signal accuracy, unsupervised α vs oracle α:** within 0.003 on every easy and medium dataset. On hard it is within 0.005 on five datasets and 0.022 below on selection seed 1, where three of the four representatives clustered at small α (0, 0.66, 0.84) and the selected α = 74.4 was not the oracle's 11.3. (e2) still reaches 0.956 there.
   - **After embedding:** (e2) and (e2-oracle) differ by at most 0.003 in confirmation-mean signal accuracy at every level.
5. **Implication for the next phase (interpretation; a decision for the project owner).** On these datasets, cPCA → embed sets the bar at about the signal-only reference. A background-suppressed kNN graph can only show value beyond cPCA on data where the signal is **not** linearly separable from the background, for example with the signal placed entirely inside the background's high-variance directions, or entangled nonlinearly. Such a benchmark is needed alongside the next method.

## Part A: global metrics

**Definitions** (full text in the pre-registration). All use k = 15 exact neighbors and true labels:

| metric | definition | Phase 4 runs | signal-only reference | chance |
|---|---|---|---|---|
| **G1** | mean fraction of a point's 15 embedding neighbors with the **same signal group and a different background cluster** | 0 | ≈ 0.82 | 0.208 |
| **G2** | background kNN accuracy | 1.0 | ≈ 0.17 | about 0.17 |
| **G3** | over the 24 (signal × background) cells, the fraction whose nearest other cell centroid shares the signal group (or shares the background cluster, or neither) | 0 (shares signal) | 1 (shares signal) | |

**Validation** (Plan item A2).
- **Inputs:**
  - the stored Phase 4 embeddings of (a), best (b), (c) and (d) on every Phase 4 seed;
  - a **signal-only reference**: the two-stage Neg-t-SNE of run (a) applied to S_T + ε (the clean signal plus the dataset's own noise), on seeds 0–2;
  - the high-D X_T, for context.
- **Reload check:** all 72 stored Phase 4 embeddings reproduce their recorded signal and background accuracies exactly.
- **Criterion:** the metrics count as able to register the intended change if every signal-only seed exceeds every Phase 4 run by at least 0.3 on G1 and on G3 "shares signal". The gaps are +0.82 or more on G1 and +1.00 on G3, on every level.

(generated)

{{validation}}

**Frozen thresholds for the next method** (per level, on the confirmation seeds; pre-registration, `23459ff`):

| # | criterion | threshold | judged |
|---|---|---|---|
| 1 | G1 | ≥ 0.50 | on every seed |
| 2 | G2 | ≤ 0.50 | on every seed |
| 3 | G3 "shares signal" | ≥ 0.50 | on every seed |
| 4 | signal accuracy | ≥ (c)'s confirmation mean − 0.05 (0.844 / 0.532 / 0.818) | on the mean |

Separately pre-registered: the next method is compared with (e2) on signal accuracy and G1–G3 with the Phase 4 "consistent across 3 paired seeds" rule. Below, criteria 1–4 are applied **descriptively** to the cPCA runs; they don't change the thresholds.

## Part B: the cPCA baseline

### Method (`baselines/cpca.py`, `scripts/phase5prep_cpca.py`)

- **cPCA** (Abid et al. 2018):
  - center X_T and X_B by their own means;
  - compute covariances C_T and C_B (XᵀX / (n − 1));
  - take the top eigenvectors of C_T − α·C_B.
- **Automatic α selection** mirrors the reference package `contrastive` 1.2.0 (`CPCA(standardize=False)`):
  - α grid {0} ∪ 40 log-spaced values from 0.1 to 1000;
  - the affinity between two α is the product of the cosines of the principal angles between the 2-D column spaces of the projected target data;
  - spectral clustering into 4 groups (fixed `random_state`);
  - the group containing α = 0 is dropped, every other group contributes its exemplar, and α = 0 is added back. Usually this gives 4 representatives.
- **Selected α** (label-free, fixed before any dataset was run): the representative with the largest target-to-background variance ratio tr(VᵀC_T V) / tr(VᵀC_B V) in its 2-D subspace.
- **Oracle α:** the grid α whose 10-D cPCA representation has the highest signal kNN accuracy, chosen per dataset seed.
- **Runs:** (e1) at the selected α; (e2) and (e3) on the top 10 components at the selected α; (e2-oracle) at the oracle α. Every representative α also goes through (e1)–(e3) (appendix).
  - (e2) is the two-stage Neg-t-SNE of run (a) with its own exact k = 15 graph and PCA init in the 10-D space.
  - (e3) is `umap-learn` 0.5.12 with defaults (n_neighbors = 15, min_dist = 0.1) and `random_state` = dataset seed.
  - Phase 4's (a) and (c) are reused from their stored embeddings.
- **Seeds:** the Phase 4 seed sets: easy and medium 0–5, hard 0–2 and 4–6 (hard seed 3 stays excluded). The confirmation seeds are primary.
- **Pairing caveat:** (e2) builds its own kNN graph and Stage 1, so it is paired with (c) by dataset seed only, not by a shared starting embedding.
- **Checks:**
  - our cPCA matches the reference package on easy seed 0: maximum principal angle 8.7·10⁻¹¹ rad over all 41 α (tolerance 10⁻⁶), and identical representative α;
  - (e2) and (e3) reruns are bit-identical;
  - after installing `umap-learn` and `contrastive`, the Phase 4 easy seed 0 rerun is still bit-identical.

### Results (generated)

{{cpca}}

### Why linear suppression suffices on these datasets (post-hoc interpretation)

The table below is a data diagnostic, computed from the clean background Bg_T and signal S_T (generated by `scripts/phase5prep_summary.py`). It counts how many of the 50 directions carry background variance at or above the noise variance (0.01), and the fraction of signal variance in the remaining, effectively background-free directions.

{{outside}}

**Reading the table:**
- **easy:** the background spans 5 directions and the signal lies entirely outside them.
- **medium:** by construction, half of the signal is inside the background's span and half outside. cPCA uses the outside half.
- **hard:** the nonlinear background spreads over 34–38 directions, yet 17–35% of the signal still lies where the background has no variance, and more lies where it has little.

cPCA needs only a few directions in which target variance exceeds α times background variance. With 4 well-separated signal groups, these suffice. This is why the spec's expected failure on medium and hard did not appear.

### Figures (seed 0; colored by signal y_s and by background y_b)

![easy](results/phase5prep/cpca_embeddings_easy.png)

![medium](results/phase5prep/cpca_embeddings_medium.png)

![hard](results/phase5prep/cpca_embeddings_hard.png)

## Limitations

- **The datasets are synthetic** and, as shown above, linearly favorable to cPCA. The results say nothing about data where the signal is linearly entangled with the background.
- **The background is discrete clusters**; continuous backgrounds are untested (REPORT.md, "Failures, surprises, and limits").
- **G3 is coarse:** 24 cells give steps of 1/24, and every embedding so far sits at 0 or 1.
- **Signal recall is still small in absolute terms** (up to about 0.08). Still, (e2)'s recall is 4–7× that of Phase 4's (c) and 12–21× that of (a). It measures exact neighbors in the 3-D clean signal.
- **(e2) is not paired with (c) through a shared Stage 1** (see Method).

## Reproducibility

- **Determinism:** same rules as Phase 4 (deterministic algorithms, 4 threads, seeded generators, exact kNN graphs). `umap-learn` runs single-threaded with a fixed `random_state`. Reruns of (e2) and (e3) on easy seed 0 are bit-identical (`results/phase5prep/cpca_check.json`).
- **Environment:**
  - adds `umap-learn` 0.5.12 (with `numba` 0.68.0, `llvmlite` 0.50.0 and `pynndescent` 0.6.0) and `contrastive` 1.2.0. The latter is installed `--no-deps` from the PyPI source archive, sha256 `7d6c2ba9…f1f7b2`; it declares no license, so it is not vendored;
  - existing packages were pinned during installation and did not change (`env/`, commit `25f173a`);
  - the Phase 4 bit-identity check still passes (`results/phase5prep/phase4_recheck_after_install.json`).
- **Provenance:**
  - metric code `8067d25`;
  - validation `58b573e`;
  - thresholds frozen `23459ff`, before the cPCA code (`2ea1ae1`) and every cPCA result.
- **To regenerate:**

  ```
  python scripts/phase5prep_metrics.py
  python scripts/phase5prep_cpca.py
  python scripts/phase5prep_summary.py
  python scripts/phase5prep_assemble.py
  ```
