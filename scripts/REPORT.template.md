# Background-contrastive Neg-t-SNE: Phase 4 report

This covers the synthetic-data prototype (Initial_Project_Plan.md §5). Tables marked "generated" are copied verbatim from `results/phase4/*.md` by `scripts/phase4_assemble.py`; none is typed by hand.

## Summary

**Pre-registered verdicts** (success criterion of Plan §5, rule 4, judged on fresh confirmation seeds): easy **not met**, medium **met**, hard **not met**.

**What happened.**
- **Signal revealed within background clusters.** Background negatives drawn from ξ_B reveal the hidden signal inside each background cluster, at every level. This goes well beyond both controls: the same number of uniform extra negatives (b), and the move along the attraction–repulsion spectrum (d). The (c) improvement is consistent across 3 paired confirmation seeds at every level, against (a), against (b) at the same m_B, against the best (b) at any m_B, and against (d). It holds on both primary metrics.
- **Global layout still organized by background.** The 6 background clusters stay separate islands in every run. Background kNN accuracy stays at 1.000 on easy and hard.

**Why easy and hard "fail".** The only failing clause is "background kNN accuracy drops relative to (a)". By design, the method can't do this. Background negatives add only repulsion. No target kNN edges connect different background islands, which sit about 30 units apart in the data, so nothing pulls islands together. The method can reorganize points within an island; it can't merge islands. With background accuracy at 1.000 for every run, the paired difference is exactly 0, and the rule needs it to be strictly positive.

**Why medium "passes".** Under (c), medium's background accuracy is 0.997–0.999: about 7–13 of 5000 points sit nearer another island's points. These are boundary points of islands that spread until they touch (under (c), medium's 6 islands join into one kNN-graph component). **This is not background suppression.** Medium is also the level with the *smallest* signal gain (+0.18 over the best control, against +0.51 on easy and +0.44 on hard) and the highest ξ_B same-signal pairing (see below).

## Headline (generated; confirmation seeds; signal kNN accuracy, mean [min, max])

{{headline}}

## Setup

- **Data** (Plan §2): synthetic target X_T = ρ·A z_s + B(z_b) + ε, with background X_B = B(z_b′) + ε.
  - 6 background clusters and 4 signal groups; n_T = n_B = 5000; D = 50.
  - Levels: easy (orthogonal maps), medium (signal half inside the background span), hard (nonlinear MLP background).
  - Main ρ is 0.2 / 0.2 / 0.3. That is the largest ρ for which plain Neg-t-SNE still hides the signal, although the high-D kNN graph still carries much of it (high-D signal kNN accuracy 0.70–0.89).
- **ξ_B** (Plan §3): Jaccard overlap of each target point's 15 nearest background points, keeping the top 30 per row (ties kept up to 60). It is built from X_B only, with no labels.
- **Schedule** (Plan §4):
  - Stage 1 is plain Neg-t-SNE for 250 epochs from a PCA init.
  - Stage 2 is a fresh 500-epoch fit from the Stage 1 embedding. It is the only stage that differs between runs.
  - Every run on a dataset shares its Stage 1, kNN graph and seed.
- **Runs:**
  - (a) plain;
  - (b) `uniform_extra`: m_B extra negatives per edge, drawn uniformly from all target points;
  - (c) `xi_b`: m_B extra negatives per edge, drawn from ξ_B;
  - (d) the t-SNE-like spectrum reference, `s=0`. The literal `neg_spec=0` is degenerate in this package: the embedding never moves.
- **Protocol** (Plan §5, pre-registered before any Phase 4 result):
  - m_B ∈ {1, 2, 5, 10, 20} is chosen on selection seeds 0–2, then frozen in a commit.
  - The verdict uses fresh confirmation seeds. A confirmation seed whose dataset fails the validity check (judged from (a) and the oracle only) is excluded and the next seed drawn.
  - The best (b) is chosen **per metric**, which favors the control.
  - "Improves" means consistent across 3 paired seeds: a higher mean and a positive difference on every seed. No p-values.

## Results by level

Full per-run tables, with selection and confirmation seeds and every pre-registered comparison listed with per-seed differences, are in the appendix.

**easy.**
- (c) at m_B = 10 reaches signal accuracy 0.89–0.90. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.30–0.32 |
  | best (b), m_B = 20 | 0.37–0.41 |
  | (d) | 0.34–0.35 |

- Every comparison on both primary metrics is consistent across 3 paired seeds. Against the best (b), signal accuracy improves by +0.51 (per seed +0.52, +0.49, +0.51).
- Each background island splits into 4 clean signal sectors on all 6 seeds (`results/phase4/embeddings_allseeds_easy.png`).
- The effect grows with m_B up to 5 (selection seeds 0.59, 0.82, 0.89 for m_B = 1, 2, 5) and then plateaus.

**medium.**
- (c) at m_B = 20 reaches 0.57–0.59. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.36–0.39 |
  | best (b) | 0.39–0.42 |
  | (d) | 0.38–0.39 |

- Every comparison is consistent across 3 paired seeds, with a smaller effect: +0.18 over the best (b).
- Islands expand and become diffuse, and the signal groups are only partly sorted (`embeddings_allseeds_medium.png`).

**hard.**
- (c) at m_B = 10 reaches 0.84–0.90. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.34–0.35 |
  | best (b) | 0.40–0.46 |
  | (d) | 0.36–0.39 |

- Every comparison is consistent across 3 paired seeds: +0.44 over the best (b).
- The islands split into signal sectors as on easy (`embeddings_allseeds_hard.png`).
- **Confirmation seed 3 was excluded:** plain (a) already reached signal accuracy 0.5126 there, above the validity limit of 0.50. The confirmation seeds are 4–6. Runs that had already finished on seed 3 are listed only in the exclusion note (appendix).

**ξ_B same-signal pairing.** Pairing same-signal points works against the method, so this number is the first thing to check when a level underperforms. It is the excess of ξ_B pairs sharing a signal group over chance (0.25), from `results/phase2/xi_b_diagnostics.csv` and the confirmation summaries:

| level | selection seeds | confirmation seeds |
|---|---|---|
| easy | +0.015 to +0.018 | +0.015 to +0.020 |
| medium | +0.058 | +0.054 to +0.064 |
| hard | +0.022 to +0.029 | +0.022 to +0.037 |

Medium, the weakest level, has the highest excess. That fits its construction: half of the signal direction lies inside the background span, so a point's background neighbors depend slightly on its signal group. Every level stays well within the ξ_B gate (+0.10).

**Scale shift.** Extra repulsion spreads the embedding, so scale has to be ruled out as the explanation. Embedding std relative to (a), on the confirmation seeds:

| run | easy | medium | hard |
|---|---|---|---|
| (c) | 1.11× | 1.25× | 1.16× |
| best (b) | 1.44× | 1.46× | 1.47× |
| (d) | 1.79× | 1.82× | 1.87× |

(d) spreads the embedding the most and (b) the second most, yet both stay near (a) on signal accuracy. On easy and hard, (d) also has the lowest mean φ (most expanded on that measure too). The (c) gain is therefore not explained by a global scale change. On medium, (c) has the lowest mean φ (0.009, against 0.014 for (d)), but (d)'s larger std and much lower signal accuracy argue against scale there as well.

## Post-hoc: global metrics (clearly labeled; not a decision rule)

The pre-registered global metrics, signal kNN recall and signal distance Spearman against the clean signal S_T, for (a), best (b), (c) and (d) on the confirmation seeds (generated):

{{posthoc}}

**Reading.** These two metrics already show the picture, so the island separation-to-size measure was not added.
- **Recall shows the local picture.** Signal kNN recall under (c) is about 3–4× the level of every control on easy and hard (e.g. easy 0.0144 against 0.0036–0.0038), and about 1.5× on medium, consistently across seeds. Within islands, neighbors increasingly share the signal.
- **Spearman shows the global picture.** The signal distance Spearman correlation stays near 0 (≤ 0.02) for every run, (c) included. Global distances are set by the background islands, not by the signal.
- **The absolute values are small** because exact 15-nearest neighbors in the clean 3-D signal are a very fine target. Even the high-D X_T scores only about 0.009 recall.

## Failures, surprises, and limits

- **The background clause can't be met by design** (see Summary). The pre-registered verdicts stand. Medium's pass rests on about 10 boundary points.
- **Hard seed 3 failed validity.** No rule existed for this, so the run stopped (Plan §6). The exclusion rule was added before any further hard confirmation result: exclude, draw the next seed, and stop if more than 2 seeds are excluded per level. Hard ρ = 0.3 sits close to the validity boundary; (a) gives 0.34–0.37 on most seeds but 0.51 on seed 3.
- **Recall and Spearman are weak metrics here** (post-hoc section). Signal kNN accuracy carries the interpretation.
- **No collapse or fragmentation flags** on any run. Under (c), medium's islands touch and form a single kNN component, but no small fragments appear.
- **The evidence has a gap.** No run separates "ξ_B's fine-grained background similarity matters" from "repelling random points of the *same background cluster* is enough". A same-cluster uniform control, using the true background labels as a deliberate oracle, would answer it.
- **Operational notes.**
  - Laptop sleep inflated wall-clock runtimes until the timer was switched to one that pauses during sleep (Phase 3). Results were never affected.
  - Stage 1's std growth from the std-1 PCA init (≈10×) is normal. The 10× stop check applies to Stage 2 runs, all of which passed.

## Next steps

1. **Cross-background attraction (main next step).** The result shows what repulsion alone can and can't do. To also suppress the background, the method needs attraction across background clusters: for example, a target kNN graph built in a background-suppressed metric, so that same-signal points in different islands become neighbors and pull the islands together.
2. **A same-background-cluster uniform control** (oracle labels), to close the gap above.
3. **The secondary tasks in Plan §7:** cPCA→UMAP and other baselines, the linear ξ_B, and real data.

## Stress test (graph-level hiding), not used for any choice

{{stress}}

## Reproducibility

- **Determinism:** every fit runs with `torch.use_deterministic_algorithms(True)`, 4 threads, and exact kNN graphs. A full rerun of easy seed 0 is bit-identical (`results/phase4/recheck_easy.json`).
- **Hashes:** dataset hashes match Phase 1, and ξ_B hashes match Phase 2, for every selection seed.
- **Sanity check:** run (a) reproduces the Phase 1 accuracies exactly.
- **Provenance:**
  - The protocol was committed before any Phase 4 result (`dad1c09`).
  - Each `selection_{level}.json` was committed before its confirmation runs, and `stress_settings.json` before any stress fit (see git history).
  - Every results JSON records the environment (package versions, torch, thread count, CPU, git commit).

## Appendix A: full tables (generated)

{{tables}}

## Appendix B: excluded runs

{{excluded}}
