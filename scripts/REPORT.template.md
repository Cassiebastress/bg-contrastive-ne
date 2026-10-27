# Background-contrastive Neg-t-SNE: Phase 4 report

This project tests a change to Neg-t-SNE, a neighbor-embedding method, meant to reveal structure that is specific to one dataset (the **target**) but hidden by variation it shares with a second dataset (the **background**). This is the neighbor-embedding analogue of contrastive PCA. This report covers the final experiment on synthetic data with a known hidden signal.

## How to read this report

**Documents in this repository** that the report refers to:

| short name | file | contents |
|---|---|---|
| **Plan**, **Plan §n** | `Initial_Project_Plan.md` (repository root), section n | the project plan: goal (§0), Phases 0–4 (§1–§5), stop-and-report conditions (§6), later work (§7) |
| **NOTES** | `NOTES.md` (repository root) | working notes: package internals, decisions, and results of each phase |
| results of phase N | `results/phaseN/` | CSV/JSON results and figures; each JSON records the software environment |
| scripts | `scripts/` | the scripts that produced every result (e.g. `scripts/phase4_run.py`) |

Tables marked **(generated)** are copied verbatim from `results/phase4/*.md` by `scripts/phase4_assemble.py`; none is typed by hand. A seven-character code such as `dad1c09` is a commit in this repository's git history.

**Phases** (the project ran in five steps; each builds on the previous one):

| phase | what it did | where |
|---|---|---|
| Phase 0 | studied the `contrastive-ne` package (v0.4.0, vendored in `third_party/contrastive-ne/`), confirmed the training schedule, and fixed the determinism rules | `results/phase0/`, NOTES |
| Phase 1 | built the synthetic data generator and chose the signal strength ρ per difficulty level by a validity sweep; froze the datasets with hashes | `data/synthetic.py`, `results/phase1/` |
| Phase 2 | built ξ_B, the background-informed negative distribution, and checked it against ground truth | `method/xi_b.py`, `results/phase2/` |
| Phase 3 | added the background term to the trainer, with tests and a stability run | `method/bg_term.py`, `results/phase3/` |
| Phase 4 | this experiment | `scripts/phase4_*.py`, `results/phase4/` |

**Notation:**

| symbol | meaning |
|---|---|
| X_T, X_B | target and background data: 5000 points each, 50 features |
| y_s, y_b | ground-truth **signal** label (4 groups) and **background** label (6 clusters) of each target point. Used only for evaluation, never by the method |
| S_T | the clean signal part of X_T (ground truth, evaluation only) |
| ρ | signal strength (`signal_to_background`) |
| island | one background cluster as it appears in a 2-D embedding |
| m_B | number of extra (background) negatives per kNN edge |
| m*_c | the m_B selected for run (c) |
| (a), (b), (c), (d) | the four run types (see "Runs"). Labels like `b_m20` / `c_m10` mean run (b) / (c) with m_B = 20 / 10 |
| selection seeds | seeds 0–2, used to choose m_B |
| confirmation seeds | fresh seeds used only for the final verdict |
| `acc_s`, `acc_b`, `recall_s`, `spearman_s` | column names in the generated tables: signal kNN accuracy, background kNN accuracy, signal kNN recall, signal Spearman (see "Metrics"). "(drop)" marks a metric where lower is better, so the difference is reported as (a) minus (c) |
| same-y_b / same-y_s precision | the fraction of ξ_B pairs whose two points share a background / signal label (see "Checks with thresholds") |
| mean [min, max] | mean over seeds, with the smallest and largest seed value |

## Summary

**Pre-registered verdicts** (success criterion, rule 4 below, judged on the confirmation seeds): easy **not met**, medium **met**, hard **not met**.

**What happened.**
- **Signal revealed within background clusters.** Background negatives drawn from ξ_B reveal the hidden signal inside each background cluster, at every level. This goes well beyond both controls: the same number of uniform extra negatives (b), and the move along the attraction–repulsion spectrum (d). The (c) improvement is consistent across 3 paired confirmation seeds at every level, against (a), against (b) at the same m_B, against the best (b) at any m_B, and against (d). It holds on both primary metrics.
- **Global layout still organized by background.** The 6 background clusters stay separate islands in every run. Background kNN accuracy stays at 1.000 on easy and hard.

**Current claim.** ξ_B recovers background-cluster membership without labels (its pairs share a background cluster 100% of the time) and uses it to target repulsion at points of the same cluster.

**What the evidence does not yet show.**
- **(b) is a weak control.** Its uniform extra negatives mostly land in *other* islands (5 of 6), which are already far apart, so they add almost no force within an island (see "Why the islands split"). Beating (b) shows that the repulsion has to be targeted, not that ξ_B's targeting is the only one that works.
- **Two explanations remain open.** The experiment cannot yet tell "ξ_B's fine-grained background similarity matters" apart from "repelling random points of the same background cluster is enough". A same-cluster control that uses the true background labels as a deliberate oracle (Next steps #1) would separate them.

**Why easy and hard "fail".** The only failing clause is "background kNN accuracy drops relative to (a)". By design, the method can't do this. Background negatives add only repulsion. Attraction comes only from target kNN edges, and in every dataset used here **0** of its roughly 50,000 kNN edges join different background clusters (0 of about 925,000 across all 18 datasets) (measured; see "Background-cluster separation" below). So nothing pulls islands together.

The clusters are far apart: cluster centers sit 30 apart in the background latent space. In the data, the closest points of different clusters are 22–23 apart on easy and medium and 16–21 on hard, against a typical 15th-neighbor distance of about 2. The method can reorganize points within an island; it can't merge islands. With background accuracy at 1.000 for every run, the paired difference is exactly 0, and the rule needs it to be strictly positive.

**Why medium "passes".** Under (c), medium's background accuracy is 0.997–0.999: about 7–13 of 5000 points sit nearer another island's points. These are boundary points of islands that spread until they touch (under (c), medium's 6 islands join into one kNN-graph component). **This is not background suppression.** Medium is also the level with the *smallest* signal gain (+0.18 over the best control, against +0.51 on easy and +0.44 on hard).

## Headline (generated; confirmation seeds; signal kNN accuracy, mean [min, max])

{{headline}}

## Method and setup

### Data (Plan §2; generator `data/synthetic.py`; Phase 1)

- **Two latent variables per target point:**
  - **background:** 6 equal-weight Gaussian clusters in 5 dimensions, centers at the vertices of a regular simplex with edge 30, within-cluster std 1;
  - **signal:** 4 clusters in 3 dimensions, simplex edge 6, std 1.
- **Construction:**
  - target: X_T = ρ·A z_s + B(z_b) + ε;
  - background: X_B = B(z_b′) + ε, from the same background distribution but with no signal;
  - ε is isotropic noise with std 0.1, and D = 50.
- **Difficulty levels** (they describe how the background hides the signal from *plain* Neg-t-SNE):
  - **easy:** A and B are orthonormal maps into orthogonal subspaces;
  - **medium:** half of A's direction lies inside B's span (overlap α = 0.5);
  - **hard:** B is a random one-hidden-layer tanh network (width 64, gain 6), so the background is nonlinear. Its output is rescaled to the same total variance as the linear levels. A is a random orthonormal 50×3 map drawn independently of the network, so it overlaps the background's directions only as much as random directions do, not by construction.
- **Signal strength ρ.** Phase 1 chose ρ per level: the largest ρ for which plain Neg-t-SNE still hides the signal (validity criterion below). Main ρ = 0.2 (easy), 0.2 (medium), 0.3 (hard); stress-test ρ = 0.1 at every level.
- **Signal in the high-D graph.** At the main ρ the high-dimensional kNN graph of X_T still carries much of the signal. High-D signal kNN accuracy is 0.70–0.89 on selection seeds 0–2, and 0.70–0.91 over all seeds used in Phase 4. The signal is lost only in the 2-D embedding.
- **Seeds.** Each dataset is fully determined by its seed, and its hash is recorded in `results/phase1/datasets.csv`. In every run the method seed equals the dataset seed.
- **The background is discrete.** It consists of separated clusters. Continuous backgrounds (gradients, manifolds) are untested (see "Failures, surprises, and limits").

**Background-cluster separation** (generated by `scripts/phase4_data_geometry.py`, data only, no embeddings; every dataset used in Phase 4). The columns are:
- **latent center distance:** the minimum distance between background cluster centers in the 5-D latent;
- **data-space centroid distance:** the minimum (and mean) distance between the cluster centroids of X_T;
- **closest points:** the smallest distance between two target points from different clusters;
- **15th neighbor:** the median distance from a point to its 15th nearest neighbor, for scale;
- **kNN edges joining clusters:** undirected edges of the exact k = 15 target graph whose endpoints lie in different background clusters.

{{data_geometry}}

### ξ_B, the background-informed negative distribution (Plan §3; `method/xi_b.py`; Phase 2)

- **Background neighbors.** For each target point i, find its k_B = 15 nearest **background** points by exact Euclidean search from X_T into X_B.
- **Score target pairs** by the Jaccard overlap of their background-neighbor sets: s_B(i, j) = |N_B(i) ∩ N_B(j)| / |N_B(i) ∪ N_B(j)|.
- **Keep the top L = 30 partners** per point. All partners tied with the 30th are kept, up to 2L = 60; beyond that, a seeded random draw within the tied block. Probabilities are proportional to s_B. Target kNN edges are not excluded.
- **No labels used.** ξ_B uses X_B and X_T only.

### Training (Plan §4; `common/embed.py`, `method/bg_term.py`; Phases 0 and 3)

- **Package.** `contrastive-ne` v0.4.0, non-parametric Neg-t-SNE, with one local change: an 11-line hook that lets the trainer add a term to the batch loss (`third_party/contrastive-ne/UPSTREAM.md`).
- **Target kNN graph.** Exact 15-nearest-neighbor graph of X_T, symmetrized by union, binary.
- **Loss.** With φ = 1/(1 + d²) (Cauchy kernel on embedding distance d), the Neg-t-SNE per-edge loss for kNN edge (i, j) is

      L_ij = −log(φ_ij/(φ_ij + c)) − Σ_{u=1..m} log(1 − φ_iu/(φ_iu + c)) − γ Σ_{b=1..m_B} log(1 − φ_ib/(φ_ib + c))

  - The first two terms are the package's own: attraction, and m = 5 uniform negatives drawn from the current minibatch. c is the package's constant (`neg_spec` = 1.0 by default).
  - The third term is the new background term, with γ = 1 and m_B extra negatives per edge. It uses the same φ, c and clamps as the package's negatives, and joins the batch loss with the package's aggregation (a sum).
  - A unit test checks that one background negative and one uniform negative at the same φ contribute identical loss and gradient.
- **Optimizer.** SGD; learning rate 1.0, annealed linearly to 0 in each stage; minibatches of 32768 directed edges; gradients clipped to ±4.
- **Two-stage schedule:**
  - **Stage 1:** plain Neg-t-SNE for 250 epochs from a PCA initialization (first two principal components, scaled so the first has std 1).
  - **Stage 2:** a fresh 500-epoch fit starting exactly from the Stage 1 embedding.
  - Only Stage 2 differs between runs. Every run on a dataset shares its Stage 1, its kNN graph and its seed.
- **Runs (all are Stage 2 settings):**

  | run | name | what Stage 2 does |
  |---|---|---|
  | (a) | plain | no background term |
  | (b) | `uniform_extra` | m_B extra negatives per edge, drawn uniformly from all other target points. This control adds the same amount of repulsion as (c) with no background information |
  | (c) | `xi_b` | the method: m_B extra negatives per edge, drawn from ξ_B(i, ·) |
  | (d) | spectrum reference | plain Neg-t-SNE at the t-SNE-like end of the attraction–repulsion spectrum (`s=0`). The package sets Z̄ = 100·n at s = 0, and c = m·Z̄/n² = 500/n = 0.1 at n = 5000 (`third_party/contrastive-ne/src/cne/cne.py:333-339`). It tests whether simply shifting the repulsion balance recovers the signal. The literal `neg_spec=0` is degenerate in this package: the embedding never moves |

### Metrics (`eval/metrics.py`; ground truth used for evaluation only)

| metric | definition | better |
|---|---|---|
| signal kNN accuracy | accuracy of a 15-nearest-neighbor classifier predicting y_s from the embedding; stratified 5-fold cross-validation, shuffled with seed 0. Chance ≈ 0.25 | higher |
| background kNN accuracy | the same, predicting y_b. Chance ≈ 0.17 | diagnostic; pre-registered as should-drop (see Summary) |
| signal kNN recall | mean over points of the fraction of a point's 15 nearest neighbors in the embedding that are also among its 15 nearest neighbors in the clean signal S_T. Exact neighbors; chance = 15/4999 = 0.0030 | higher |
| signal Spearman | Spearman rank correlation between all pairwise embedding distances and all pairwise S_T distances (all 5000 points) | higher |
| mean φ | mean of φ = 1/(1 + d²) over a fixed sample of 100,000 random point pairs (per dataset and seed). Lower means a more expanded embedding | scale diagnostic |
| embedding std | standard deviation of the embedding coordinates, averaged over the two axes; also reported relative to (a) on the same seed | scale diagnostic |
| components | connected components of the 15-nearest-neighbor graph built in the 2-D embedding, and the fraction of points in components of fewer than 50 points | collapse/fragmentation diagnostic |

The reference rows "high-D X_T" (the classifier and recall applied to the 50-dimensional X_T itself) and "Stage 1 only" give context.

### Checks with thresholds

- **Dataset validity** (from Phase 1). A dataset is a valid test only if plain Neg-t-SNE, run (a), hides the signal but not the background:
  - embedding background kNN accuracy ≥ 0.90;
  - embedding signal kNN accuracy ≤ 0.50;
  - **oracle** signal kNN accuracy ≥ 0.95. The oracle is the same classifier applied directly to the clean signal S_T; it checks that the signal groups are separable at all.
- **ξ_B informativeness gates** (from Phase 2), on every dataset used:
  - ξ_B pairs share y_b at a rate ≥ 0.90;
  - ξ_B pairs share y_s at no more than chance + 0.10.

  Rates are exact expectations over ξ_B's probabilities, with each point weighted by its number of target kNN edges (as often as training draws it). The same-y_s **excess** over chance is always recorded. Pairing same-signal points works directly against the method.
- **Health:**
  - no non-finite loss or embedding at any epoch;
  - Stage 2 embedding std never grows above 10× its starting value. Stage 1's growth of about 10× from the std-1 PCA initialization is normal and not covered by this check.
- **Reproduction:**
  - dataset and ξ_B hashes match Phases 1–2;
  - run (a) reproduces the Phase 1 accuracies exactly;
  - a full rerun of easy seed 0 is bit-identical.

## Pre-registered protocol and rules (Plan §5; `common/experiment.py`; committed in `dad1c09` before any Phase 4 result)

- **Selection, then confirmation, per level.**
  1. **Selection:** on seeds 0–2, run (a), (b) and (c) at each m_B ∈ {1, 2, 5, 10, 20}, plus (d).
  2. **Freeze:** commit the chosen settings in `results/phase4/selection_{level}.json` before any confirmation run.
  3. **Confirmation:** on fresh seeds, run (a), (d), (c) at m*_c, (b) at m*_c, and (b) at each per-metric best m_B.
  4. **Checkpoint:** easy was run first, with a checkpoint before medium and hard.
- **Rule 1, metrics.**
  - **Primary:** signal kNN accuracy and signal kNN recall.
  - **Secondary:** background kNN accuracy (should drop), signal Spearman, and scale.
- **Rule 2, "improves".** Runs are paired by seed (same dataset, same Stage 1). "X improves over Y" means the mean over 3 seeds is higher **and** every per-seed difference is positive. This is reported as "consistent across 3 paired seeds", with the mean difference and the per-seed values. No p-values.
- **Rule 3, best settings (on the selection seeds).**
  - m*_c is the m_B with the highest mean signal kNN accuracy among the (c) runs.
  - The best (b) is chosen separately for each primary metric. This favors the control, because (c) is compared against the strongest (b) for each metric.
  - Ties go to the smaller m_B.
- **Rule 4, success (on the confirmation seeds).**
  - (c) at m*_c improves **both** primary metrics over (a), over (b) at m*_c, over the per-metric best (b), and over (d).
  - Background kNN accuracy drops relative to (a).
  - No (c) run is flagged.
- **Rule 5, flags (judged against (a) on the same seed).**
  - **collapse:** embedding std < 0.5× that of (a);
  - **fragmentation:** more than 5% of points are in components of fewer than 50 points.

  Flags mark runs for review; flagged runs are never dropped.
- **Rule 6, interpretation.** If the best (b) or (d) matches (c) on a primary metric, the effect is extra repulsion or a spectrum shift, not background information, and the report must say so.
- **Exclusion rule for confirmation seeds** (added after hard seed 3 failed, before any further hard confirmation result).
  - A confirmation seed whose dataset fails the validity check (judged from (a) and the oracle only, never from (b) or (c)) is excluded, and the next seed is drawn until 3 valid ones exist.
  - More than 2 exclusions in one level stops the experiment for that level.
  - Results already computed on an excluded seed appear only in Appendix B.
- **Stop-and-report conditions** (Plan §6, specialized to this experiment). The experiment stops and reports instead of adjusting anything if any of these happens:
  - a hash mismatch;
  - (a) failing to reproduce Phase 1;
  - a non-finite value, or Stage 2 std growth > 10×;
  - a fresh dataset failing the ξ_B gates;
  - a determinism failure;
  - more than 2 exclusions in one level;
  - uninterpretable results at the easy checkpoint.

## Results by level

Full per-run tables, with selection and confirmation seeds and every pre-registered comparison with per-seed differences, are in Appendix A.

**easy.**
- (c) at m_B = 10 reaches signal accuracy 0.89–0.90. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.30–0.32 |
  | best (b), m_B = 20 | 0.37–0.41 |
  | (d) | 0.34–0.35 |

- Every comparison on both primary metrics is consistent across 3 paired seeds. Against the best (b), signal accuracy improves by +0.51 (per seed +0.52, +0.49, +0.51).
- Each background island splits into 4 clean signal sectors on all 6 seeds (figure below).
- The effect grows with m_B up to 5 (selection seeds: 0.59, 0.82 and 0.89 for m_B = 1, 2 and 5) and then plateaus.

**medium.**
- (c) at m_B = 20 reaches 0.57–0.59. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.36–0.39 |
  | best (b) | 0.39–0.42 |
  | (d) | 0.38–0.39 |

- Every comparison is consistent across 3 paired seeds, with a smaller effect: +0.18 over the best (b).
- Islands expand and become diffuse, and the signal groups are only partly sorted.

**hard.**
- (c) at m_B = 10 reaches 0.84–0.90. The alternatives:

  | run | signal accuracy |
  |---|---|
  | (a) | 0.34–0.35 |
  | best (b) | 0.40–0.46 |
  | (d) | 0.36–0.39 |

- Every comparison is consistent across 3 paired seeds: +0.44 over the best (b).
- The islands split into signal sectors as on easy.
- **Confirmation seed 3 was excluded:** plain (a) already reached signal accuracy 0.5126 there, above the validity limit of 0.50. The confirmation seeds are therefore 4–6.

**Figures.** In each grid below:
- **rows** are seeds: 0–2 for selection, then the confirmation seeds;
- **columns** are (a), the best (b) and (c), each shown twice: colored by signal label y_s, then by background label y_b. Panel titles give the kNN accuracy.

The seed 0 embeddings of all 12 Stage 2 runs are in `results/phase4/embeddings_{easy,medium,hard}.png`.

easy (`results/phase4/embeddings_allseeds_easy.png`):

![easy: (a), best (b) and (c) on all seeds](results/phase4/embeddings_allseeds_easy.png)

medium (`results/phase4/embeddings_allseeds_medium.png`):

![medium: (a), best (b) and (c) on all seeds](results/phase4/embeddings_allseeds_medium.png)

hard (`results/phase4/embeddings_allseeds_hard.png`):

![hard: (a), best (b) and (c) on all seeds](results/phase4/embeddings_allseeds_hard.png)

**Metrics against m_B** (`results/phase4/metrics_vs_mB.png`). Rows are levels and columns are metrics.
- Lines and bands show (b) and (c) on the selection seeds (mean, min–max).
- The dashed lines and bands are (a) and (d).
- The open squares are the confirmation seeds at the selected m_B.

![metrics against m_B](results/phase4/metrics_vs_mB.png)

**Why the islands split (interpretation).** Inside a background island, two forces act on the points:
- **Attraction** along target kNN edges mostly joins points of the same signal group: 59% of edges on easy, against 25% by chance.
- **ξ_B repulsion** pushes each point away from partners chosen for background similarity, which within an island are nearly blind to signal (see the excess below).

Pushing everyone apart from random island-mates while still pulling same-signal neighbors together sorts each island into signal groups. In the terms of the method's theory, at the idealized optimum the embedding learns roughly "target similarity divided by background similarity". Within an island the background part is high and nearly equal for everyone, so dividing it out leaves the signal.

Uniform extra negatives (b) mostly land in *other* islands (5 of 6), which are already far apart, so they add almost no force within an island.

**ξ_B same-signal pairing.** This number is the first thing to check when a level underperforms. It is the excess of ξ_B pairs sharing a signal group over chance (0.25), from `results/phase2/xi_b_diagnostics.csv` (selection seeds) and `results/phase4/summary_{level}_confirmation.json` (confirmation seeds):

| level | selection seeds | confirmation seeds |
|---|---|---|
| easy | +0.015 to +0.018 | +0.015 to +0.020 |
| medium | +0.058 | +0.054 to +0.064 |
| hard | +0.022 to +0.029 | +0.022 to +0.037 |

Every level stays well within the gate (+0.10).

**Why medium is weaker than hard (post-hoc interpretation, added after the verdicts).** The level names describe how hard it is for *plain* Neg-t-SNE to show the signal, not how much signal the method has to work with.
- **The target kNN graph carries the signal; ξ_B only clears room.** Attraction along target kNN edges is what sorts each island into signal groups. On medium, half of the signal direction lies inside the background span, so within a background cluster that half looks like ordinary background spread. As a result, medium's high-D graph carries clearly less signal: fewer edges join same-signal points, and the high-D signal accuracy is lower. On easy and hard, (c) recovers about as much signal as the high-D graph holds. On medium it recovers less, and its islands become diffuse rather than splitting into sectors.
- **ξ_B works slightly against the method on medium.** The same entanglement makes ξ_B pair same-signal points a little more often than chance (the excess above), which opposes the method.
- **Hard also runs at a larger ρ** (0.3 vs 0.2). Its nonlinear background hides the signal from plain Neg-t-SNE for longer, so the validity criterion allowed a 1.5× stronger signal, along a random direction rather than one built to overlap the background.

Confirmation seeds (generated):

{{graph_signal}}

**Scale shift.** Extra repulsion spreads the embedding, so scale has to be ruled out as the explanation. Embedding std relative to (a), on the confirmation seeds:

| run | easy | medium | hard |
|---|---|---|---|
| (c) | 1.11× | 1.25× | 1.16× |
| best (b) | 1.44× | 1.46× | 1.47× |
| (d) | 1.79× | 1.82× | 1.87× |

(d) spreads the embedding the most and (b) the second most, yet both stay near (a) on signal accuracy. On easy and hard, (d) also has the lowest mean φ (most expanded on that measure too). The (c) gain is therefore not explained by a global scale change. On medium, (c) has the lowest mean φ (0.009, against 0.014 for (d)), but (d)'s larger std and much lower signal accuracy argue against scale there as well. Per-run scale against m_B is in `results/phase4/scale_vs_mB.png`.

## Post-hoc: global metrics (clearly labeled; not a decision rule)

The pre-registered global metrics, signal kNN recall and signal Spearman, for (a), best (b), (c) and (d) on the confirmation seeds (generated):

{{posthoc}}

**Reading.** Recall shows the local picture. Spearman turns out to be uninformative here (second point). No further measure, such as island separation relative to island size, was added: background kNN accuracy and the embedding figures already show that the islands stay separate.
- **Recall shows the local picture.** Signal kNN recall under (c) is about 3–4× the level of every control on easy and hard (e.g. easy 0.0144 against 0.0036–0.0038), and about 1.5× on medium, consistently across seeds.
- **Spearman cannot detect global signal organization here.** The signal Spearman correlation stays near 0 (≤ 0.02) for every run, (c) included. But the high-D X_T itself also scores only about 0.01 (0.008–0.013; Appendix A, "high-D X_T (reference)" rows). In these datasets, distances between background clusters dominate all pairwise distances, so this metric reflects the data, not the embedding. It cannot show whether an embedding organizes the signal globally.
- **The absolute values are small** because exact 15-nearest neighbors in the clean 3-D signal are a very fine target. Even the high-D X_T scores only about 0.009 recall.

## Failures, surprises, and limits

- **The background clause can't be met by design** (see Summary). The pre-registered verdicts stand. Medium's pass rests on about 10 boundary points.
- **Hard seed 3 failed validity.** No rule existed for this, so the run stopped (Plan §6). The exclusion rule above was added before any further hard confirmation result. Hard ρ = 0.3 sits close to the validity boundary; (a) gives 0.34–0.37 on most seeds but 0.51 on seed 3.
- **Recall and Spearman are weak metrics here** (post-hoc section). Signal kNN accuracy carries the interpretation.
- **No collapse or fragmentation flags** on any run. Under (c), medium's islands touch and form a single kNN component, but no small fragments appear.
- **The evidence has a gap.** No run separates "ξ_B's fine-grained background similarity matters" from "repelling random points of the *same background cluster* is enough". A same-cluster uniform control, using the true background labels as a deliberate oracle, would answer it (Next steps #1).
- **Only discrete backgrounds were tested.** The background here is separated clusters. With a continuous background (a gradient or manifold), there are no islands, ξ_B's neighborhoods overlap smoothly, and both the mechanism and the "islands can't merge" limit may behave differently. This is untested.
- **Operational note.** Laptop sleep inflated wall-clock runtimes until the timer was switched to one that pauses during sleep (Phase 3). Results were never affected.

## Next steps

1. **A same-background-cluster uniform control (first).** Draw the m_B extra negatives uniformly from points with the same *true* background label, a deliberate oracle. If it matches (c), the current result rests on recovering cluster membership; if (c) beats it, ξ_B's fine-grained background similarity adds something. This settles what the current claim can be.
2. **Cross-background attraction.** The result shows what repulsion alone can and can't do. To also suppress the background, the method needs attraction across background clusters: for example, a target kNN graph built in a background-suppressed metric, so that same-signal points in different islands become neighbors and pull the islands together.
3. **The later work in Plan §7:**
   - baselines: UMAP on X_T, contrastive PCA followed by UMAP, and contrastiveVI;
   - the linear ξ_B variant;
   - real datasets with known target-specific groups.

## Stress test (graph-level hiding), not used for any choice

**Design.**
- **Data:** the same generator at ρ = 0.1 for every level, seeds 0–2. At this ρ the signal is weak even in the high-dimensional kNN graph, not only in the embedding.
- **Settings:** fixed in advance from the main selection seeds and committed in `results/phase4/stress_settings.json` before any stress fit:
  - (a);
  - (d);
  - (c) at m*_c;
  - the best (b).
- **Run once.** Nothing was tuned or rerun on these data. ξ_B diagnostics are reported but not gated.

{{stress}}

Seed 0 embeddings of the stress runs. Rows alternate between coloring by y_s and by y_b, per level; columns are the runs.

![stress test embeddings, seed 0](results/phase4/stress/stress_embeddings.png)

**Reading.** At ρ = 0.1 the signal is hidden in the target kNN graph itself. Only 0.29–0.34 of kNN edges share a signal group (chance is 0.25), against 0.47–0.64 in the main datasets. Attraction therefore has almost nothing to sort islands by.
- **easy and hard:** (c) adds a small gain, consistent across seeds, of about +0.03 signal accuracy over the best (b), and stays far below the high-D X_T level.
- **medium:** no consistent gain.

This matches the graph-signal interpretation above: background negatives reveal signal that the target graph carries, but they cannot create it. That is a further reason to change the attraction graph itself (Next steps #2).

## Reproducibility

- **Environment:**
  - Python 3.12, torch 2.14.1 (CPU), numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1;
  - `contrastive-ne` v0.4.0 (upstream commit `d5f0b53`), on an Apple M3;
  - every results JSON records the full environment.
- **Determinism:**
  - every fit runs with `torch.use_deterministic_algorithms(True)`, 4 threads, all random generators seeded from the run's seed, and exact kNN graphs;
  - a full rerun of easy seed 0 is bit-identical (`results/phase4/recheck_easy.json`).
- **Hashes and sanity check:** dataset hashes match Phase 1, and ξ_B hashes match Phase 2, for every selection seed. Run (a) reproduces the Phase 1 accuracies exactly.
- **Provenance** (commits in this repository):
  - The protocol was committed before any Phase 4 result (`dad1c09`).
  - Each `results/phase4/selection_{level}.json` was committed before its confirmation runs.
  - `results/phase4/stress_settings.json` was committed before any stress fit (`18581cc`).
- **To regenerate:**

  ```
  python scripts/phase4_run.py --level L --stage selection    # then commit the selection file
  python scripts/phase4_run.py --level L --stage confirmation
  python scripts/phase4_stress.py
  python scripts/phase4_report.py
  python scripts/phase4_assemble.py
  ```

## Appendix A: full tables (generated)

{{tables}}

## Appendix B: excluded runs

"Plan §6" below refers to the stop-and-report conditions in `Initial_Project_Plan.md`.

{{excluded}}
