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

| level | confirmation seeds | (a) plain | best (b) uniform_extra | (d) s=0 | (c) xi_b | (c) background acc | pre-registered success |
|---|---|---|---|---|---|---|---|
| easy | 3, 4, 5 | 0.312 [0.304, 0.322] | 0.387 [0.374, 0.410] (m_B=20) | 0.346 [0.337, 0.352] | 0.894 [0.890, 0.899] (m_B=10) | 1.000 [1.000, 1.000] | no |
| medium | 3, 4, 5 | 0.368 [0.355, 0.388] | 0.405 [0.391, 0.416] (m_B=20) | 0.384 [0.382, 0.387] | 0.582 [0.573, 0.588] (m_B=20) | 0.998 [0.997, 0.999] | yes |
| hard | 4, 5, 6 | 0.347 [0.339, 0.353] | 0.429 [0.399, 0.464] (m_B=20) | 0.381 [0.364, 0.394] | 0.868 [0.843, 0.902] (m_B=10) | 1.000 [1.000, 1.000] | no |

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

Signal kNN recall and signal distance Spearman against S_T are pre-registered metrics; this view of them for (a), best (b), (c) and (d) was added after the verdicts. Recall chance level is 15/4999 = 0.0030. Values are mean [min, max] over the confirmation seeds; differences are (c) minus the other run, per seed.

### easy (confirmation seeds 3, 4, 5)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0036 [0.0033, 0.0037] | -0.0016 [-0.0037, 0.0012] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0038 [0.0036, 0.0040] | -0.0023 [-0.0050, 0.0008] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=10 | 0.0144 [0.0139, 0.0148] | 0.0156 [0.0108, 0.0189] | 1.000 [1.000, 1.000] |
| (d) s=0 reference | 0.0036 [0.0034, 0.0038] | -0.0025 [-0.0053, 0.0003] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0102, +0.0109, +0.0115 | yes | +0.0195, +0.0178, +0.0145 | yes |
| (b) uniform_extra, m_B=20 | +0.0101, +0.0107, +0.0112 | yes | +0.0198, +0.0181, +0.0158 | yes |
| (d) s=0 reference | +0.0104, +0.0110, +0.0111 | yes | +0.0197, +0.0186, +0.0161 | yes |

### medium (confirmation seeds 3, 4, 5)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0038 [0.0036, 0.0041] | 0.0013 [-0.0004, 0.0031] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0041 [0.0040, 0.0043] | 0.0015 [0.0000, 0.0034] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=20 | 0.0060 [0.0058, 0.0064] | 0.0075 [0.0069, 0.0079] | 0.998 [0.997, 0.999] |
| (d) s=0 reference | 0.0039 [0.0035, 0.0042] | 0.0020 [0.0013, 0.0033] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0028, +0.0020, +0.0018 | yes | +0.0073, +0.0048, +0.0066 | yes |
| (b) uniform_extra, m_B=20 | +0.0024, +0.0019, +0.0016 | yes | +0.0058, +0.0045, +0.0077 | yes |
| (d) s=0 reference | +0.0029, +0.0016, +0.0019 | yes | +0.0056, +0.0047, +0.0063 | yes |

### hard (confirmation seeds 4, 5, 6)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0036 [0.0032, 0.0039] | -0.0002 [-0.0027, 0.0032] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0047 [0.0041, 0.0054] | -0.0004 [-0.0032, 0.0029] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=10 | 0.0136 [0.0123, 0.0156] | 0.0146 [0.0112, 0.0176] | 1.000 [1.000, 1.000] |
| (d) s=0 reference | 0.0040 [0.0037, 0.0045] | -0.0004 [-0.0031, 0.0027] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0091, +0.0117, +0.0091 | yes | +0.0144, +0.0177, +0.0124 | yes |
| (b) uniform_extra, m_B=20 | +0.0083, +0.0102, +0.0082 | yes | +0.0147, +0.0182, +0.0121 | yes |
| (d) s=0 reference | +0.0087, +0.0111, +0.0089 | yes | +0.0149, +0.0181, +0.0121 | yes |

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

Pending: the stress test runs once, after the settings in `results/phase4/stress_settings.json` are committed.

## Reproducibility

- **Determinism:** every fit runs with `torch.use_deterministic_algorithms(True)`, 4 threads, and exact kNN graphs. A full rerun of easy seed 0 is bit-identical (`results/phase4/recheck_easy.json`).
- **Hashes:** dataset hashes match Phase 1, and ξ_B hashes match Phase 2, for every selection seed.
- **Sanity check:** run (a) reproduces the Phase 1 accuracies exactly.
- **Provenance:**
  - The protocol was committed before any Phase 4 result (`dad1c09`).
  - Each `selection_{level}.json` was committed before its confirmation runs, and `stress_settings.json` before any stress fit (see git history).
  - Every results JSON records the environment (package versions, torch, thread count, CPU, git commit).

## Appendix A: full tables (generated)

Values are mean [min, max] over seeds. Selection seeds 0-2 chose the settings; the verdict uses the fresh confirmation seeds (from 3 on; seeds whose dataset fails the validity check are excluded).

### easy

Selected on seeds 0-2: m*_c = 10; best (b) by signal accuracy m_B = 20, by signal recall m_B = 20.

#### Selection seeds 0-2

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.881 [0.875, 0.892] | 0.0091 [0.0086, 0.0095] | 1.000 [1.000, 1.000] | 0.0129 [0.0109, 0.0148] |  | none |
| Stage 1 only (reference) | 0.303 [0.295, 0.310] | 0.0031 [0.0030, 0.0032] | 1.000 [1.000, 1.000] | 0.0011 [0.0003, 0.0015] | 0.0776 [0.0761, 0.0787] | none |
| (a) plain | 0.303 [0.289, 0.314] | 0.0031 [0.0029, 0.0032] | 1.000 [1.000, 1.000] | 0.0008 [-0.0010, 0.0028] | 0.0765 [0.0749, 0.0775] | none |
| (b) uniform_extra, m_B=1 | 0.305 [0.294, 0.315] | 0.0030 [0.0030, 0.0031] | 1.000 [1.000, 1.000] | 0.0021 [0.0006, 0.0031] | 0.0692 [0.0677, 0.0701] | none |
| (b) uniform_extra, m_B=2 | 0.306 [0.296, 0.316] | 0.0031 [0.0030, 0.0033] | 1.000 [1.000, 1.000] | 0.0012 [0.0004, 0.0022] | 0.0634 [0.0620, 0.0642] | none |
| (b) uniform_extra, m_B=5 | 0.324 [0.317, 0.328] | 0.0034 [0.0033, 0.0036] | 1.000 [1.000, 1.000] | 0.0012 [0.0002, 0.0023] | 0.0510 [0.0498, 0.0517] | none |
| (b) uniform_extra, m_B=10 | 0.329 [0.318, 0.335] | 0.0036 [0.0034, 0.0039] | 1.000 [1.000, 1.000] | 0.0013 [0.0004, 0.0022] | 0.0387 [0.0378, 0.0392] | none |
| (b) uniform_extra, m_B=20 | 0.360 [0.356, 0.365] | 0.0039 [0.0038, 0.0041] | 1.000 [1.000, 1.000] | 0.0007 [-0.0010, 0.0028] | 0.0264 [0.0259, 0.0268] | none |
| (c) xi_b, m_B=1 | 0.591 [0.580, 0.602] | 0.0065 [0.0060, 0.0070] | 1.000 [1.000, 1.000] | 0.0071 [0.0065, 0.0075] | 0.0558 [0.0555, 0.0561] | none |
| (c) xi_b, m_B=2 | 0.818 [0.792, 0.838] | 0.0117 [0.0109, 0.0128] | 1.000 [1.000, 1.000] | 0.0162 [0.0152, 0.0177] | 0.0473 [0.0461, 0.0487] | none |
| (c) xi_b, m_B=5 | 0.886 [0.881, 0.894] | 0.0143 [0.0134, 0.0149] | 1.000 [1.000, 1.000] | 0.0179 [0.0155, 0.0209] | 0.0296 [0.0291, 0.0303] | none |
| (c) xi_b, m_B=10 | 0.888 [0.882, 0.895] | 0.0144 [0.0140, 0.0147] | 1.000 [1.000, 1.000] | 0.0195 [0.0166, 0.0248] | 0.0188 [0.0186, 0.0190] | none |
| (c) xi_b, m_B=20 | 0.879 [0.867, 0.886] | 0.0139 [0.0131, 0.0145] | 1.000 [1.000, 1.000] | 0.0186 [0.0168, 0.0210] | 0.0114 [0.0114, 0.0115] | none |
| (d) s=0 reference | 0.334 [0.324, 0.339] | 0.0036 [0.0034, 0.0037] | 1.000 [1.000, 1.000] | 0.0010 [-0.0006, 0.0027] | 0.0149 [0.0145, 0.0152] | none |

#### Confirmation seeds 3, 4, 5

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.887 [0.882, 0.892] | 0.0093 [0.0091, 0.0097] | 1.000 [1.000, 1.000] | 0.0105 [0.0038, 0.0161] |  | none |
| Stage 1 only (reference) | 0.313 [0.309, 0.319] | 0.0035 [0.0033, 0.0037] | 1.000 [1.000, 1.000] | -0.0017 [-0.0039, 0.0011] | 0.0777 [0.0774, 0.0783] | none |
| (a) plain | 0.312 [0.304, 0.322] | 0.0036 [0.0033, 0.0037] | 1.000 [1.000, 1.000] | -0.0016 [-0.0037, 0.0012] | 0.0765 [0.0763, 0.0768] | none |
| (d) s=0 reference | 0.346 [0.337, 0.352] | 0.0036 [0.0034, 0.0038] | 1.000 [1.000, 1.000] | -0.0025 [-0.0053, 0.0003] | 0.0150 [0.0149, 0.0153] | none |
| (c) xi_b, m_B=10 | 0.894 [0.890, 0.899] | 0.0144 [0.0139, 0.0148] | 1.000 [1.000, 1.000] | 0.0156 [0.0108, 0.0189] | 0.0188 [0.0187, 0.0189] | none |
| (b) uniform_extra, m_B=10 | 0.353 [0.349, 0.361] | 0.0037 [0.0035, 0.0041] | 1.000 [1.000, 1.000] | -0.0018 [-0.0050, 0.0017] | 0.0389 [0.0386, 0.0390] | none |
| (b) uniform_extra, m_B=20 | 0.387 [0.374, 0.410] | 0.0038 [0.0036, 0.0040] | 1.000 [1.000, 1.000] | -0.0023 [-0.0050, 0.0008] | 0.0265 [0.0264, 0.0266] | none |

#### Pre-registered comparisons (confirmation seeds 3, 4, 5)

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=10 vs (a) plain | acc_s | +0.5820 | +0.5886, +0.5892, +0.5682 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=10 | acc_s | +0.5406 | +0.5432, +0.5380, +0.5406 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | acc_s | +0.5071 | +0.5190, +0.4892, +0.5130 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | acc_s | +0.5479 | +0.5558, +0.5470, +0.5408 | yes |
| (c) xi_b, m_B=10 vs (a) plain | recall_s | +0.0109 | +0.0102, +0.0109, +0.0115 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=10 | recall_s | +0.0107 | +0.0102, +0.0106, +0.0113 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | recall_s | +0.0106 | +0.0101, +0.0107, +0.0112 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | recall_s | +0.0108 | +0.0104, +0.0110, +0.0111 | yes |
| (c) xi_b, m_B=10 vs (a) plain | acc_b (drop) | +0.0000 | +0.0000, +0.0000, +0.0000 | no |

Success criterion met: **no** ((c) runs flagged: no).

### medium

Selected on seeds 0-2: m*_c = 20; best (b) by signal accuracy m_B = 20, by signal recall m_B = 20.

#### Selection seeds 0-2

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.704 [0.700, 0.709] | 0.0068 [0.0063, 0.0071] | 1.000 [1.000, 1.000] | 0.0131 [0.0114, 0.0155] |  | none |
| Stage 1 only (reference) | 0.373 [0.362, 0.382] | 0.0037 [0.0036, 0.0039] | 1.000 [1.000, 1.000] | 0.0042 [0.0020, 0.0060] | 0.0722 [0.0709, 0.0730] | none |
| (a) plain | 0.376 [0.369, 0.387] | 0.0037 [0.0036, 0.0040] | 1.000 [1.000, 1.000] | 0.0038 [0.0014, 0.0056] | 0.0710 [0.0698, 0.0717] | none |
| (b) uniform_extra, m_B=1 | 0.379 [0.370, 0.388] | 0.0038 [0.0036, 0.0041] | 1.000 [1.000, 1.000] | 0.0040 [0.0021, 0.0053] | 0.0644 [0.0634, 0.0650] | none |
| (b) uniform_extra, m_B=2 | 0.380 [0.377, 0.385] | 0.0036 [0.0035, 0.0039] | 1.000 [1.000, 1.000] | 0.0040 [0.0024, 0.0055] | 0.0591 [0.0582, 0.0596] | none |
| (b) uniform_extra, m_B=5 | 0.385 [0.380, 0.388] | 0.0037 [0.0035, 0.0039] | 1.000 [1.000, 1.000] | 0.0045 [0.0020, 0.0066] | 0.0478 [0.0468, 0.0483] | none |
| (b) uniform_extra, m_B=10 | 0.388 [0.381, 0.393] | 0.0040 [0.0036, 0.0042] | 1.000 [1.000, 1.000] | 0.0045 [0.0018, 0.0069] | 0.0364 [0.0358, 0.0368] | none |
| (b) uniform_extra, m_B=20 | 0.401 [0.392, 0.411] | 0.0041 [0.0039, 0.0044] | 1.000 [1.000, 1.000] | 0.0046 [0.0014, 0.0068] | 0.0248 [0.0246, 0.0249] | none |
| (c) xi_b, m_B=1 | 0.416 [0.408, 0.424] | 0.0043 [0.0041, 0.0045] | 1.000 [1.000, 1.000] | 0.0049 [0.0027, 0.0062] | 0.0477 [0.0470, 0.0482] | none |
| (c) xi_b, m_B=2 | 0.462 [0.455, 0.471] | 0.0049 [0.0047, 0.0051] | 1.000 [1.000, 1.000] | 0.0057 [0.0030, 0.0073] | 0.0352 [0.0348, 0.0354] | none |
| (c) xi_b, m_B=5 | 0.519 [0.512, 0.528] | 0.0055 [0.0053, 0.0059] | 1.000 [1.000, 1.000] | 0.0073 [0.0058, 0.0090] | 0.0210 [0.0208, 0.0213] | none |
| (c) xi_b, m_B=10 | 0.553 [0.533, 0.575] | 0.0058 [0.0050, 0.0065] | 1.000 [1.000, 1.000] | 0.0080 [0.0057, 0.0099] | 0.0138 [0.0136, 0.0140] | none |
| (c) xi_b, m_B=20 | 0.561 [0.545, 0.572] | 0.0061 [0.0056, 0.0067] | 0.998 [0.998, 1.000] | 0.0104 [0.0092, 0.0115] | 0.0090 [0.0089, 0.0092] | none |
| (d) s=0 reference | 0.390 [0.379, 0.401] | 0.0039 [0.0034, 0.0041] | 1.000 [1.000, 1.000] | 0.0046 [0.0018, 0.0069] | 0.0140 [0.0137, 0.0142] | none |

#### Confirmation seeds 3, 4, 5

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.711 [0.700, 0.718] | 0.0069 [0.0066, 0.0074] | 1.000 [1.000, 1.000] | 0.0099 [0.0016, 0.0173] |  | none |
| Stage 1 only (reference) | 0.365 [0.352, 0.379] | 0.0038 [0.0038, 0.0038] | 1.000 [1.000, 1.000] | 0.0016 [0.0007, 0.0032] | 0.0720 [0.0717, 0.0726] | none |
| (a) plain | 0.368 [0.355, 0.388] | 0.0038 [0.0036, 0.0041] | 1.000 [1.000, 1.000] | 0.0013 [-0.0004, 0.0031] | 0.0711 [0.0707, 0.0714] | none |
| (d) s=0 reference | 0.384 [0.382, 0.387] | 0.0039 [0.0035, 0.0042] | 1.000 [1.000, 1.000] | 0.0020 [0.0013, 0.0033] | 0.0138 [0.0137, 0.0141] | none |
| (c) xi_b, m_B=20 | 0.582 [0.573, 0.588] | 0.0060 [0.0058, 0.0064] | 0.998 [0.997, 0.999] | 0.0075 [0.0069, 0.0079] | 0.0090 [0.0090, 0.0091] | none |
| (b) uniform_extra, m_B=20 | 0.405 [0.391, 0.416] | 0.0041 [0.0040, 0.0043] | 1.000 [1.000, 1.000] | 0.0015 [0.0000, 0.0034] | 0.0246 [0.0245, 0.0247] | none |

#### Pre-registered comparisons (confirmation seeds 3, 4, 5)

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=20 vs (a) plain | acc_s | +0.2141 | +0.2272, +0.1972, +0.2178 | yes |
| (c) xi_b, m_B=20 vs (b) uniform_extra, m_B=20 | acc_s | +0.1771 | +0.1810, +0.1686, +0.1816 | yes |
| (c) xi_b, m_B=20 vs (d) s=0 reference | acc_s | +0.1977 | +0.2064, +0.1976, +0.1890 | yes |
| (c) xi_b, m_B=20 vs (a) plain | recall_s | +0.0022 | +0.0028, +0.0020, +0.0018 | yes |
| (c) xi_b, m_B=20 vs (b) uniform_extra, m_B=20 | recall_s | +0.0019 | +0.0024, +0.0019, +0.0016 | yes |
| (c) xi_b, m_B=20 vs (d) s=0 reference | recall_s | +0.0021 | +0.0029, +0.0016, +0.0019 | yes |
| (c) xi_b, m_B=20 vs (a) plain | acc_b (drop) | +0.0019 | +0.0014, +0.0016, +0.0026 | yes |

Success criterion met: **yes** ((c) runs flagged: no).

### hard

Selected on seeds 0-2: m*_c = 10; best (b) by signal accuracy m_B = 20, by signal recall m_B = 20.

#### Selection seeds 0-2

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.867 [0.849, 0.882] | 0.0101 [0.0089, 0.0113] | 1.000 [1.000, 1.000] | 0.0099 [0.0055, 0.0134] |  | none |
| Stage 1 only (reference) | 0.354 [0.344, 0.372] | 0.0040 [0.0037, 0.0041] | 1.000 [1.000, 1.000] | 0.0024 [-0.0003, 0.0040] | 0.0550 [0.0542, 0.0557] | none |
| (a) plain | 0.353 [0.338, 0.371] | 0.0039 [0.0037, 0.0041] | 1.000 [1.000, 1.000] | 0.0021 [-0.0003, 0.0038] | 0.0539 [0.0530, 0.0545] | none |
| (b) uniform_extra, m_B=1 | 0.358 [0.345, 0.376] | 0.0040 [0.0038, 0.0042] | 1.000 [1.000, 1.000] | 0.0022 [-0.0003, 0.0034] | 0.0490 [0.0483, 0.0496] | none |
| (b) uniform_extra, m_B=2 | 0.361 [0.347, 0.385] | 0.0040 [0.0038, 0.0042] | 1.000 [1.000, 1.000] | 0.0024 [0.0002, 0.0037] | 0.0452 [0.0445, 0.0457] | none |
| (b) uniform_extra, m_B=5 | 0.368 [0.352, 0.396] | 0.0041 [0.0040, 0.0042] | 1.000 [1.000, 1.000] | 0.0022 [-0.0002, 0.0036] | 0.0372 [0.0367, 0.0375] | none |
| (b) uniform_extra, m_B=10 | 0.381 [0.363, 0.409] | 0.0045 [0.0042, 0.0046] | 1.000 [1.000, 1.000] | 0.0025 [-0.0003, 0.0044] | 0.0293 [0.0290, 0.0296] | none |
| (b) uniform_extra, m_B=20 | 0.415 [0.392, 0.441] | 0.0047 [0.0044, 0.0049] | 1.000 [1.000, 1.000] | 0.0023 [-0.0007, 0.0045] | 0.0212 [0.0210, 0.0214] | none |
| (c) xi_b, m_B=1 | 0.606 [0.565, 0.635] | 0.0075 [0.0066, 0.0081] | 1.000 [1.000, 1.000] | 0.0061 [0.0041, 0.0072] | 0.0392 [0.0383, 0.0401] | none |
| (c) xi_b, m_B=2 | 0.713 [0.657, 0.749] | 0.0100 [0.0089, 0.0105] | 1.000 [1.000, 1.000] | 0.0105 [0.0096, 0.0123] | 0.0330 [0.0315, 0.0343] | none |
| (c) xi_b, m_B=5 | 0.792 [0.745, 0.835] | 0.0111 [0.0103, 0.0120] | 1.000 [1.000, 1.000] | 0.0163 [0.0133, 0.0187] | 0.0230 [0.0223, 0.0233] | none |
| (c) xi_b, m_B=10 | 0.836 [0.810, 0.859] | 0.0128 [0.0121, 0.0133] | 1.000 [1.000, 1.000] | 0.0172 [0.0149, 0.0189] | 0.0157 [0.0154, 0.0159] | none |
| (c) xi_b, m_B=20 | 0.826 [0.799, 0.842] | 0.0121 [0.0117, 0.0126] | 1.000 [1.000, 1.000] | 0.0189 [0.0153, 0.0227] | 0.0104 [0.0104, 0.0104] | none |
| (d) s=0 reference | 0.372 [0.357, 0.400] | 0.0043 [0.0043, 0.0043] | 1.000 [1.000, 1.000] | 0.0025 [-0.0006, 0.0049] | 0.0110 [0.0108, 0.0112] | none |

#### Confirmation seeds 4, 5, 6

Excluded seed 3 (dataset not a valid test, judged from (a) and the oracle): (a) signal acc 0.5126 (limit 0.50), background acc 1.0000, oracle 0.9956. Its runs are in no table or comparison.

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.896 [0.887, 0.912] | 0.0103 [0.0097, 0.0113] | 1.000 [1.000, 1.000] | 0.0082 [0.0046, 0.0109] |  | none |
| Stage 1 only (reference) | 0.346 [0.341, 0.349] | 0.0038 [0.0036, 0.0039] | 1.000 [1.000, 1.000] | -0.0002 [-0.0030, 0.0028] | 0.0555 [0.0533, 0.0571] | none |
| (a) plain | 0.347 [0.339, 0.353] | 0.0036 [0.0032, 0.0039] | 1.000 [1.000, 1.000] | -0.0002 [-0.0027, 0.0032] | 0.0543 [0.0520, 0.0559] | none |
| (d) s=0 reference | 0.381 [0.364, 0.394] | 0.0040 [0.0037, 0.0045] | 1.000 [1.000, 1.000] | -0.0004 [-0.0031, 0.0027] | 0.0111 [0.0106, 0.0115] | none |
| (c) xi_b, m_B=10 | 0.868 [0.843, 0.902] | 0.0136 [0.0123, 0.0156] | 1.000 [1.000, 1.000] | 0.0146 [0.0112, 0.0176] | 0.0162 [0.0158, 0.0165] | none |
| (b) uniform_extra, m_B=10 | 0.392 [0.371, 0.410] | 0.0041 [0.0037, 0.0045] | 1.000 [1.000, 1.000] | -0.0005 [-0.0032, 0.0029] | 0.0296 [0.0286, 0.0304] | none |
| (b) uniform_extra, m_B=20 | 0.429 [0.399, 0.464] | 0.0047 [0.0041, 0.0054] | 1.000 [1.000, 1.000] | -0.0004 [-0.0032, 0.0029] | 0.0213 [0.0209, 0.0217] | none |

#### Pre-registered comparisons (confirmation seeds 4, 5, 6)

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=10 vs (a) plain | acc_s | +0.5211 | +0.5040, +0.5522, +0.5072 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=10 | acc_s | +0.4762 | +0.4724, +0.4918, +0.4644 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | acc_s | +0.4393 | +0.4436, +0.4386, +0.4358 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | acc_s | +0.4870 | +0.4788, +0.5082, +0.4740 | yes |
| (c) xi_b, m_B=10 vs (a) plain | recall_s | +0.0100 | +0.0091, +0.0117, +0.0091 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=10 | recall_s | +0.0094 | +0.0087, +0.0110, +0.0086 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | recall_s | +0.0089 | +0.0083, +0.0102, +0.0082 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | recall_s | +0.0095 | +0.0087, +0.0111, +0.0089 | yes |
| (c) xi_b, m_B=10 vs (a) plain | acc_b (drop) | +0.0000 | +0.0000, +0.0000, +0.0000 | no |

Success criterion met: **no** ((c) runs flagged: no).

## Appendix B: excluded runs

Hard confirmation seed 3: the dataset failed the validity check, judged from (a) and the oracle only: (a) signal kNN accuracy 0.5126 > 0.50, background accuracy 1.000, oracle 0.9956. Before the exclusion rule existed, the first confirmation run stopped there (Plan §6), after these runs had already finished. They are listed here only and appear in no table or comparison. Lines are verbatim from that run's log:

```
hard confirmation seed=3 a      acc_s 0.513 acc_b 1.000 recall_s 0.007 (11s)
hard confirmation seed=3 d      acc_s 0.562 acc_b 1.000 recall_s 0.008 (12s)
hard confirmation seed=3 c_m10  acc_s 0.877 acc_b 1.000 recall_s 0.015 (36s)
hard confirmation seed=3 b_m10  acc_s 0.572 acc_b 1.000 recall_s 0.007 (31s)
hard confirmation seed=3 b_m20  acc_s 0.601 acc_b 1.000 recall_s 0.008 (44s)
STOP (Plan §6): hard seed 3: fresh dataset is not a valid test {'emb_acc_b': 1.0, 'emb_acc_s': 0.5126000000000001, 'oracle_acc_s': 0.9956000000000002, 'valid': False}
```
