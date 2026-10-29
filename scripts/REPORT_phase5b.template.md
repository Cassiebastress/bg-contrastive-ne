# Phase 5b report: Design L, where linear suppression is structurally limited

In every benchmark up to Phase 5a, some linear subspace held the signal and left out the background variation that hides it, so cPCA → Neg-t-SNE could win. This phase built two benchmarks where that is impossible or hard:

- **L-cyclic (primary).** Cluster k's signal block is cluster k−1's nuisance block. Any linear projection that keeps cluster k's signal also keeps cluster k−1's nuisance.
- **L-between (secondary).** The signal lies in the between-cluster directions. cPCA can keep it only at α < 1, while its label-free α rule tends to pick α > 1.

It asked two questions, both on WC (within-cluster signal accuracy):

1. **L-cyclic:** does (c) (ξ_B negatives) beat every "linear projection → standard Neg-t-SNE" pipeline, including the ones chosen with labels?
2. **L-between:** is label-free cPCA's failure only an α-selection failure?

No new embedding method was built.

## How to read this report

**Shared definitions.**
- `REPORT.md` defines the project, the two-stage Neg-t-SNE, runs (a)–(c), ξ_B, the flags, and the "consistent across 3 paired seeds" rule.
- `REPORT_phase5prep.md` defines G1–G3, the cPCA runs (e2) and (e3), and the label-free α rule.
- `REPORT_phase5a.md` defines WC, detectability, coherence and the oracle same-cluster control (o).

**The Phase 5b spec** was given by the project owner in conversation. "Design" below states everything the results depend on. The owner made one change in plan review, before any Design L dataset existed: the linear and cPCA ceilings are judged at the **embedding level** (each projection embedded with (a)'s two-stage Neg-t-SNE), not by high-D kNN. The spec's high-D ceiling could not pass V3 whenever V2 held (`results/phase5b/predictions.md`, "Change to P1").

**Files referred to:**

| short name | file | contents |
|---|---|---|
| **predictions** | `results/phase5b/predictions.md` | P1–P5 and the definitions used to judge them, committed before any Design L result |
| **pre-registration** | `results/phase5b/preregistration.md` | runs, selection rule, comparisons, exclusion and stop rules, committed before any (b), (c) or (o) run |
| results | `results/phase5b/` | CSV/JSON results, generated tables (`*.md`) and figures |
| scripts | `scripts/phase5b_*.py`, `common/benchmark_l.py`, `data/synthetic_l.py` | the code that produced every result |

Tables marked **(generated)** are copied verbatim from `results/phase5b/*.md` by `scripts/phase5b_assemble.py`; none is typed by hand. A seven-character code such as `9c35a0b` is a commit in this repository's git history.

**Run labels:**

| label | run |
|---|---|
| (a) | plain two-stage Neg-t-SNE on X_T |
| (b) | (a) plus m_B uniform extra negatives per edge |
| (c) | (a) plus m_B ξ_B negatives per edge |
| (o) | (a) plus m_B oracle same-cluster negatives per edge (uses labels; a control) |
| (e2), (e3) | cPCA on the top 10 components at the label-free α, then Neg-t-SNE or UMAP |
| cPCA variants | cPCA → Neg-t-SNE for every α representative × k ∈ {3, 5, 10, 20}, plus, per k, the α with the best high-D WC on the full grid |
| best cPCA | the per-seed maximum WC over all cPCA variants, chosen with labels: a conservative upper bound, not a fair baseline |
| best linear | the per-seed maximum WC over the labeled linear projections, embedded (L-cyclic: the 64 subsets of the six Q blocks, each with U; L-between: U, and U ∪ V). Descriptive. |

## Summary

1. **L-cyclic: (c) beats every linear-projection pipeline, including the label-chosen ones, consistently across 3 paired seeds.**
   - WC on the confirmation seeds: (c) 0.914, best labeled linear projection 0.624, best cPCA variant 0.530, label-free (e2) 0.420, (a) 0.335 (means; tables below).
   - (c) beats best cPCA by +0.36 to +0.41 on every seed (pre-registered) and the best labeled linear projection by +0.29 (descriptive).
   - **Per cluster** (a cluster counts as revealed at per-cluster WC ≥ 0.80): (c) reveals all 6 clusters on every seed (per-cluster WC ≥ 0.85). The best linear projection is always an alternating subset (U+Q135 on every confirmation seed) and reveals exactly 3 clusters (WC 0.98–1.00) while leaving the other 3 near chance (0.23–0.29). Of the cPCA variants, none reveals more than one cluster, and the best-WC variant reveals none.
2. **L-between: label-free cPCA's failure is not only an α-selection failure; it also needs a small k.**
   - The label-free rule picked α > 1 on every seed (1.34–2.15), and every α > 1 gives WC about 0.25 at every k: it subtracts the between-cluster variance, which is where the signal lies.
   - But α < 1 alone is not enough. At the pipeline's fixed k = 10, the best α < 1 gives only 0.35: components 6–10 bring in the within-cluster nuisance V, and embedding U ∪ V gives the same 0.35.
   - Only α < 1 **and** k ≤ 5 reveals the signal (0.99, every cluster), which is the same as the labeled projection onto U. On 2 of 3 confirmation seeds, the best variant is plain PCA (α = 0, k = 5).
   - So on L-between, best cPCA beats (c) by 0.07–0.08 on every seed (means 0.989 vs 0.914), as predicted (P4).
3. **(c) beats the oracle same-cluster control and uniform negatives on both designs,** consistently across 3 paired seeds: by +0.21 / +0.22 over (o) at m_B = 80 and by +0.39 over (b) at m_B = 40. This repeats Phase 5a's finding for (o) on WC.
4. **All five predictions confirmed** (P1–P5; table below). P1: the embedding-level linear ceiling was 0.627 on the selection seeds, inside the predicted 0.5–0.7, and 0.624 on the confirmation seeds.
5. **Extension (added after the main results): a local, cluster-then-cPCA baseline beats (c) on both designs.**
   - Label-free k-means on X_B recovers the 6 background clusters exactly (K = 6, ARI 1.000, every seed). cPCA run within each cluster at its own label-free α reaches WC 0.984–0.992 at every k ∈ {3, 5, 10}, with label-free and true clusters alike.
   - On L-cyclic, that is +0.07 to +0.08 above (c) on every seed. Predictions P6–P9 were all confirmed.
   - So the answer to question 1 holds only for *global* linear pipelines. With discrete, well-separated background clusters, a piecewise-linear pipeline that needs no labels does better than (c). See "Extension: local cPCA" below.
6. **(c) reaches about the high-D detectability.** Its WC (0.91) is close to the high-D WC of X_T (0.92) on both designs. Interpretation (not tested): ξ_B removes most of the background's hold on the embedding but cannot do better than the high-D kNN graph it embeds, while a projection that drops the nuisance dimensions (U on L-between) can.

## Design

### Generator (`data/synthetic_l.py`)

    L-cyclic:   target  x = B_U c_k + σ_w Q_{k+1} w + ρ Q_k z_s + ε      background  x = B_U c_k + σ_w Q_{k+1} w + ε
    L-between:  target  x = B_U c_k + σ_w V w + ρ W z_s + ε              background  x = B_U c_k + σ_w V w + ε

- **Shared:** K = 6 background clusters with centers on a 5-D simplex (edge 30), mapped through B_U (D × 5). 4 signal groups on a 3-D simplex (edge 6, σ_s = 1). Noise std 0.1 in every dimension. Nuisance w ~ N(0, I_3) with σ_w = √(5/3), matching easy's within-cluster variance. n_T = n_B = 5000, D = 50.
- **Subspaces:** one orthonormal D × 23 draw per seed: B_U, then six 3-D blocks Q_0…Q_5 (indices mod 6). L-between uses V = Q_0 for the nuisance and W = B_U R (R a random 5 × 3 orthonormal matrix) for the signal, so the signal lies inside the between-cluster span.
- **Balance:** clusters are exactly balanced and identical in X_T and X_B, so C_T − C_B is the signal covariance in population: cPCA's target is the signal in both variants.
- **ρ = 0.2 for both variants,** from the Phase 1 rule.

### Validity (selection seeds 0–2)

| criterion | condition |
|---|---|
| V1 | (a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90 |
| V2 | detectability (high-D WC) ≥ 0.85 |
| V2b | high-D signal kNN accuracy ≥ 0.65 |
| V3 (L-cyclic) | label-free (e2) WC, the embedded linear ceiling and the embedded cPCA ceiling all ≤ 0.80 |
| V3 (L-between) | label-free (e2) WC ≤ 0.80 |

Both variants were valid on every selection seed. ξ_B passed the Phase 2 gates on every selection and confirmation seed (same-y_b precision 1.000, same-y_s excess +0.035 to +0.044).

### Main runs (rules from the pre-registration)

- **Selection** (seeds 0–2): (a); (b) and (c) at m_B ∈ {1, 2, 5, 10, 20, 40}; (o) at m_B ∈ {1, 2, 5, 10, 20, 40, 80}; (e2), (e3). Best m_B per group by mean WC; ties to the smaller m_B.
- **Confirmation** (seeds 3–5; none excluded): (a), best (b), best (c), best (o), (e2), (e3), every cPCA variant, and, descriptive, every labeled linear projection, embedded.
- **Pre-registered comparisons** on WC: (c) vs (e2), vs best cPCA, vs (o), vs (b).

## Results

### Pre-registered predictions (generated)

The predictions as committed are in `results/phase5b/predictions.md` (`04a1450`). P1 and P3 are judged on the selection seeds, the others on the confirmation seeds.

{{predictions}}

### Calibration (generated)

{{calibration}}

### L-cyclic main runs (generated)

{{tables_cyclic}}

### L-between main runs (generated)

{{tables_between}}

### Figures

Selection seed 0, colored by signal group y_s (left) and background cluster y_b (right). The last two rows are the best cPCA variant and the best labeled linear projection from calibration.

**L-cyclic.** (c) splits every background cluster into its 4 signal groups. The best linear projection (U+Q135) does so for clusters 1, 3 and 5 only.

![L-cyclic embeddings](results/phase5b/embeddings_cyclic.png)

**L-between.** Label-free cPCA (α > 1) removes the between-cluster directions, and with them both the clusters and the signal, leaving an unstructured disk. The labeled projection onto U and the best cPCA variant (α < 1, k = 5) separate the signal within every cluster.

![L-between embeddings](results/phase5b/embeddings_between.png)

## Extension: local cPCA (added after the main results)

**Why it was run.** Every competitor in the main runs applies one global projection, but the background clusters can be recovered without labels. This baseline was added at review, after the main results, with predictions P6–P9 committed before any of its results (`a791d9b`). It changes no verdict above.

**Method** (`baselines/local_cpca.py`, `scripts/phase5b_local_cpca.py`), on the main runs' confirmation seeds 3–5 (datasets verified by hash):
- **Label-free version:** k-means on X_B for K ∈ {2, …, 10}, keeping the K with the largest silhouette, and each target point assigned to the nearest centroid.
- **Oracle version:** the true background clusters of X_B and X_T.
- **Per cluster:** cPCA at that cluster's label-free α, top k ∈ {3, 5, 10} directions.
- **Embedding:** Z = [local coordinates | s · one-hot(cluster)] is embedded once with (a)'s two-stage Neg-t-SNE. s = 10 × the largest local radius, which ensures that no kNN edge crosses clusters (checked: 0 on every run).

**Results.**
- **Clustering:** label-free clustering found K = 6 with ARI 1.000 on every seed of both designs, so the label-free and oracle versions differ by at most 0.001 WC.
- **α:** every cluster's label-free α was above 1 (1.7 to 58.8). Within a cluster the background varies only in that cluster's nuisance block, so subtracting it removes the nuisance and keeps the signal.
- **Embedding layout (descriptive, from the figures):** the embeddings are 24 tight islands, one per cluster × signal group. The islands of one background cluster are not reliably placed together, because the one-hot offsets make all clusters equidistant. WC and G2 are local, so they do not register this.
- **Determinism:** both reruns of seed 3 were bit-identical (`results/phase5b/recheck_{cyclic,between}_local_cpca.json`).

### Local cPCA on L-cyclic (generated)

{{tables_cyclic_local}}

### Local cPCA on L-between (generated)

{{tables_between_local}}

### Figures (confirmation seed 3)

Each figure shows, top to bottom: (c), the best global cPCA variant, then local cPCA at k = 3 with label-free and with true clusters.

![L-cyclic local cPCA](results/phase5b/embeddings_cyclic_local_cpca.png)

![L-between local cPCA](results/phase5b/embeddings_between_local_cpca.png)

## Interpretation

- **Why L-cyclic defeats every linear pipeline.** Revealing cluster k's signal needs Q_k, and keeping Q_k also keeps cluster k−1's nuisance. A subset can avoid that conflict for at most every other cluster, which is why the best subsets are alternating and reveal exactly 3 of 6 clusters (P1). cPCA has no labels to find even those subsets: near α = 1 its top directions are the signal blocks, and in the data each block also carries another cluster's nuisance.
- **Why (c) is not limited the same way.** ξ_B works with neighborhoods, not directions. Its negatives are points that are close in the background data, so within each cluster the repulsion lands on whatever direction carries that cluster's nuisance, and a different direction can carry it in each cluster.
- **L-between is a selection problem in two parameters.** In population, cPCA at α < 1 with k ≤ 5 keeps exactly U, which holds the signal and excludes the nuisance V. The label-free rule fails on α (the representative with the largest variance ratio was an α > 1 on every seed), and the pipeline's fixed k = 10 fails on its own (it brings V back). A label-free rule that picks both well would win this design; (c) would not.
- **The oracle control again.** As in Phase 5a, repelling random members of the true background cluster spreads each island but recovers much less signal than ξ_B: (o) at its best m_B (80, the top of its grid) reaches 0.69–0.70, against 0.91 for (c).

## Limitations

- **(b) and (o) were selected at the top of their m_B grids** (40 and 80) on both designs, and their selection-seed mean WC was still rising: by +0.06 to +0.08 for (b) from m_B = 20 to 40, and by +0.01 to +0.02 for (o) from 40 to 80. Larger m_B was not tried, so their best possible WC is unknown; (b)'s trend is far below (c), and (o)'s gain is flattening.
- **The best cPCA variant is chosen with labels per seed,** so it is an upper bound on what cPCA can do. The label-free comparison is (c) vs (e2).
- **The linear projections on the confirmation seeds are descriptive.** They were added before the freeze (`2fb8b7a`), but no verdict depends on them.
- **L-cyclic is adversarial by construction.** Every cluster's nuisance block is exactly another cluster's signal block. The result shows that (c) *can* beat every global linear pipeline, not how often such structure occurs in real data.
- **No piecewise or local linear baseline was run in the main experiment.** Every competitor applies one global projection to all points. But the background clusters are recoverable without labels (ξ_B's same-y_b precision is 1.000 on every seed), so a cluster-then-cPCA pipeline, with a different projection per cluster, may solve L-cyclic. *Extension, added after the main results:* it does. Local cPCA beats (c) on both designs (see "Extension: local cPCA"). Design L's discrete, well-separated clusters make the clustering step exact, which is the most favorable case for it.
- **Synthetic data only,** with discrete, balanced background clusters, as in all earlier phases.
- **Dirty flag:** both selection summaries record `dirty: true` at `9c35a0b`. The only uncommitted file was the then-untracked `scripts/phase5b_report.py`, which the runs do not import; tracked code was identical to `9c35a0b`. The confirmation summaries and rechecks record a clean tree.

## Next steps

1. **A local-cPCA baseline.** *Done as the extension above:* it beats (c) on both designs. A follow-up suggestion (not part of the reviewed next steps): a design where (c) could still have an edge needs background structure that clustering cannot carve into linear pieces, such as a continuous or curved background, or nuisance that changes direction within a cluster.
2. **A non-adversarial Design L variant.** Draw each cluster's nuisance and signal subspaces at random inside a shared lower-dimensional space, so that overlaps occur by chance rather than by construction.
3. **Move compute to the Math Slurm cluster** (account YANG-2026). The GPU path is untested: the ξ_B sampling code has only run on CPU, and results will not be bit-identical to the Mac's.
4. **A label-free rule for (α, k).** L-between shows cPCA needs both chosen well. Test whether a rule that also picks k (for example, from the cPCA eigenvalue gap) recovers the signal without labels.
5. **Judge (c) against its ceiling.** (c)'s WC sits just below detectability on both designs; a design where the high-D kNN graph itself is poor would show whether (c) is limited by the graph.

## Reproducibility

Run with the cne-bg environment, in this order:

    python scripts/phase5b_calibrate.py --step rho
    python scripts/phase5b_calibrate.py --step validate --variant cyclic
    python scripts/phase5b_calibrate.py --step validate --variant between
    python scripts/phase5b_calibrate.py --step summary
    python scripts/phase5b_calibrate.py --step freeze
    python scripts/phase5b_run.py --design cyclic --stage selection
    python scripts/phase5b_run.py --design between --stage selection
    python scripts/phase5b_run.py --design cyclic --stage confirmation
    python scripts/phase5b_run.py --design cyclic --recheck
    python scripts/phase5b_run.py --design between --stage confirmation
    python scripts/phase5b_run.py --design between --recheck
    python scripts/phase5b_local_cpca.py --design cyclic
    python scripts/phase5b_local_cpca.py --design between
    python scripts/phase5b_local_cpca.py --design cyclic --recheck
    python scripts/phase5b_local_cpca.py --design between --recheck
    python scripts/phase5b_report.py
    python scripts/phase5b_assemble.py

**Order of commits:**
- Predictions P1–P5: `04a1450`, before any Design L dataset.
- Generator, metrics and tests: `527cfca`.
- Calibration results: `f736abd`.
- Freeze: `9c35a0b`, before any (b), (c) or (o) run.
- Selection: `65b919a`.
- Confirmation: `81d6cc1` (L-cyclic), `a7928b3` (L-between).
- Extension: predictions P6–P9 and the local-cPCA code `a791d9b`, before any local result; results `8fc5e5f`.

**Checks:**
- The reruns of selection seed 0 are bit-identical in every row and embedding, for both variants (`results/phase5b/recheck_{cyclic,between}.json`).
- The Phase 4 easy recheck stayed bit-identical after this phase's code changes (`results/phase5b/phase4_recheck_after_5b_changes.json`).
- Every results JSON records the environment.
