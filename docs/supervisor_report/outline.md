# Supervisor report: outline (checkpoint)

This is the outline for `main.tex`, for approval before drafting. For each section it gives the claims, their sources and their types, plus the planned tables and figures.

## Claim types

These tags are used here, in `claims.md`, and in the report itself.

| tag | type | how the report renders it |
|---|---|---|
| **[pub]** | published result (cited) | "(published)" |
| **[der]** | our derivation | "(our derivation)" |
| **[pre]** | pre-registered experimental verdict | "(pre-registered)" |
| **[desc]** | measured, no verdict attached | "(descriptive)" |
| **[interp]** | our explanation, not tested | "(interpretation)" |

## The report is standalone

**Revised after drafting.** The aim is that a reader needs no other file to follow the report; naming other files is not banned. The report now:
- defines the datasets and the protocol up front, in §2;
- is written mostly in prose;
- in Appendix A, maps its names to the detailed per-phase reports and results directories.

The bullets below record the original, stricter rule.

- **No references to earlier report files.** The report never names `REPORT*.md`, `NOTES.md` or the plan files, and never says "see the Phase X report". Everything it needs is defined inside it.
- **Source comments.** `% source:` comments in `main.tex` point to generated result files under `results/` (e.g. `results/bsgraph/tables_phase2.md`), never to report files. Where a number exists only in a report's prose, its report section is recorded in `claims.md`, not in the `.tex` file.
- **Appendix A** maps report names to internal phase and run labels and to result directories. It does not map to report files.
- **Below in this outline**, the source abbreviations refer to the earlier reports. They are for checking only and do not appear in the report.

| abbreviation | file |
|---|---|
| R4 | `REPORT.md` (Phase 4) |
| R5p | `REPORT_phase5prep.md` |
| R5a | `REPORT_phase5a.md` |
| R5b | `REPORT_phase5b.md` |
| RMG | `REPORT_mnist_grass.md` |
| RBS | `REPORT_bsgraph.md` |
| N | `NOTES.md` |
| P | `Initial_Project_Plan.md` |

## Names used in the report (no internal labels in the main text)

- **Methods:**

  | report name | what it is | internal label |
  |---|---|---|
  | `\methodname` (placeholder BC-NE) | the residual graph + two-stage Neg-t-SNE | (r_a) |
  | `\methodname` + background negatives | the same, plus background negatives | (r_c) |
  | *raw-graph background negatives* | the earlier method: raw kNN graph + background negatives | (c) |
  | *plain Neg-t-SNE* | — | (a) |

- **Controls:**

  | report name | internal label |
  |---|---|
  | *uniform extra negatives* | (b) |
  | *same-cluster oracle negatives* | (o) |
  | *t-SNE-end spectrum reference* | (d) |
  | *cPCA → Neg-t-SNE*, *cPCA → UMAP*, *cPCA → t-SNE* | (e2), (e3), T3 |
  | *t-SNE*, *PCA → t-SNE* | T1, T2 |
  | *local cPCA* | local cPCA |

- **Synthetic benchmarks:**

  | report name | internal label |
  |---|---|
  | *additive benchmarks* (orthogonal / overlapping / nonlinear background) | easy / medium / hard |
  | *weak-signal stress sets* | stress |
  | *interaction benchmark* | Design J |
  | *cluster-specific-signal benchmark* | Design K |
  | *cyclic benchmark* | L-cyclic |
  | *between-cluster benchmark* | L-between |

  Together, the cyclic and between-cluster benchmarks are called *the two subspace-conflict benchmarks*.
- **Metrics:**
  - *signal kNN accuracy*;
  - *within-cluster signal accuracy* (WC);
  - *detectability* (WC on the high-D data);
  - G1–G3 are named in words, and appear only where needed.

---

## 1. Summary (about 1 page)

- **The method in one paragraph:** build the kNN graph on local background residuals, then run two-stage Neg-t-SNE; background negatives are optional.
- **Key findings:**
  1. **The residual graph is what makes the method work.**
     - On MNIST-on-grass, `\methodname`+BN beats raw-graph background negatives on every confirmation seed (0.973 vs 0.931; +0.036 to +0.046). **[pre]** Source: `results/bsgraph/comparisons.json`.
     - Without background negatives, the residual graph alone lifts plain Neg-t-SNE by +0.10 to +0.13. **[desc]**
  2. **Background negatives are redundant once the graph is background-suppressed.** +0.0014, 0.0000, +0.0020: not consistent. **[pre]**
  3. **On MNIST-on-grass, the method ties cPCA → t-SNE and is more stable across seeds.**
     - The comparison is not consistent: +0.032, −0.006, +0.028. **[pre]**
     - The ranges are 0.972–0.974 vs 0.941–0.979. **[desc]**
  4. **Global cPCA wins on additive data,** by a population identity. **[der]**
     - Synthetic additive benchmarks: cPCA → Neg-t-SNE beats raw-graph background negatives at every level. **[desc]** It is a paired comparison with a consistency label, but no pre-registered verdict for this pair. Source: R5p.
     - MNIST-on-grass: cPCA → t-SNE beats raw-graph background negatives. **[pre]** Source: RMG.
  5. **Local cPCA beats raw-graph background negatives on both subspace-conflict benchmarks.** Local cPCA is a label-free cluster-then-cPCA pipeline. WC 0.984–0.992 vs 0.914. **[pre]** Source: R5b.
  6. **On the cyclic benchmark, raw-graph background negatives beat every global linear pipeline,** including the label-chosen ones: 0.914 vs best cPCA 0.530 **[pre]**, and vs best labeled projection 0.624 **[desc]**.
- **The main open question:** does the residual graph hold up where the background is not additive, the case where its advantage over global cPCA is supposed to lie?

## 2. Problem setup and notation

- X_T ∈ ℝ^{n_T×D} and X_B ∈ ℝ^{n_B×D}, in a shared feature space. The goal is a 2-D embedding of X_T that shows the structure absent from X_B.
- **Notation:**
  - p_ij: target kNN similarities;
  - q_ij = φ(‖y_i − y_j‖), with the Cauchy kernel φ(d) = 1/(1+d²);
  - ξ: the noise distribution over pairs;
  - m: uniform negatives per edge; m_B: background negatives per edge;
  - k = 15;
  - k_loc and r: residual-graph parameters;
  - C_T, C_B: covariances;
  - α: cPCA contrast.
- Transposes are written X^{T}.
- This section is definitions only.

## 3. Background: neighbor embeddings as contrastive learning

1. **t-SNE and UMAP as attraction over kNN pairs plus repulsion over noise pairs.** **[pub]** Damrich et al. 2023; van der Maaten & Hinton 2008; McInnes et al. 2018. The attraction–repulsion spectrum is from Böhm et al. 2022 **[pub]**.
2. **NCE vs negative sampling.**
   - NCE recovers p whatever the noise distribution (Gutmann & Hyvärinen 2010).
   - Negative sampling learns a rescaled p (Mikolov et al. 2013; Levy & Goldberg 2014).
   - In Damrich et al. 2023: Theorem 1 and Corollary 2. **[pub]** *Numbering and statements to verify against arXiv 2206.01816.*
3. **Our extension.** **[der]**
   - With classifier φ/(φ+c) and noise distribution ξ, the idealized optimum is q*_ij = c·p_ij / (m ξ_ij).
   - With background negatives: q*_ij = c·p_ij / (m ξ_U,ij + γ m_B ξ_B,ij).
   - **Assumptions:**
     - each q_ij can be set freely;
     - expectations replace sampling;
     - in-batch uniform sampling is treated as uniform over pairs;
     - clamping is ignored.
   - Package facts used (c constant, in-batch negatives): N §3–4.
4. **UMAP's effective loss differs from its stated loss.** **[pub]** Damrich & Hamprecht 2021.
5. **UMAP = negative sampling with a 1/d² kernel,** whose repulsion diverges at short range; Neg-t-SNE uses the bounded Cauchy kernel. **[pub]** Damrich et al. 2023, Lemma 3, *numbering to verify*.
6. **Initialization:** PCA with PC1 at std 1. **[pub]** Kobak & Linderman 2021; Damrich et al. 2023.

## 4. The method

1. **Residual graph (core).**
   - For target x_i: N_B(i) = its k_loc nearest background points, with mean μ_i and the top-r right singular vectors V_i of the centered neighbors.
   - The residual is r_i = (I − V_i V_i^{T})(x_i − μ_i).
   - The graph is the exact k = 15 kNN graph on the residuals, union-symmetrized and binary.
   - Defaults: k_loc = 30, r = 2.
   - Source: `method/residual_graph.py`, `results/bsgraph/preregistration_phase2.md`.
2. **Two-stage Neg-t-SNE.**
   - Stage 1: 250 epochs from the PCA init.
   - Stage 2: 500 epochs from the Stage 1 output, with a fresh LR annealing.
   - Optimizer: SGD, learning rate 1 annealed linearly, batch 2^15 edges, m = 5, c = 1, gradient clipping at ±4.
   - Source: N §4–5.
3. **Optional background negatives.**
   - ξ_B is built from the Jaccard overlap of background-neighbor sets: k_B = 15, L = 30, ties up to 2L.
   - The loss gains the term −γ Σ log(1 − φ_ib/(φ_ib + c)), with γ = 1, in Stage 2 only.
   - Read via §3.3 **[der]**, with an intuitive gloss **[interp]**.
4. **Algorithm box** with the full pipeline.
5. **Implementation note:**
   - built on `contrastive-ne` v0.4.0, plus an 11-line hook;
   - exact kNN graphs;
   - bit-identical reruns.

## 5. Design decisions and their evidence

One short subsection per decision: what was chosen, the alternatives, and the evidence.

1. **Neg-t-SNE rather than UMAP or t-SNE.**
   - **Why:** negative sampling leaves ξ out of the classifier, so changing ξ changes the optimum (§3.3) **[der]**. NCE and t-SNE normalize against the noise, so there is no such lever **[pub + der]**. The package confirms c is a constant scalar, not ξ (N §4, §9) **[desc]**.
   - **Alternative rejected, UMAP proper:** its 1/d² kernel's repulsion diverges at short range **[pub]**, which is exactly where background negatives deliberately act, on pairs that start close together **[interp]**.
   - **Trade-off:** without the background term, Neg-t-SNE scored below t-SNE on MNIST-on-grass: 0.858 vs 0.884, and 0.891 for PCA → t-SNE. **[desc]** Source: `results/mnist_grass/tables.md`.
2. **Putting the background into the noise distribution.**
   - **Why not NCE:** under NCE the noise cancels at the optimum. **[pub + der]**
   - **Evidence that it acts:** raw-graph background negatives beat the best uniform extra-negatives control at all three additive levels, consistently across seeds **[pre]**. Signal accuracy over the best uniform control:

     | level | gain |
     |---|---|
     | orthogonal | +0.51 |
     | overlapping | +0.18 |
     | nonlinear | +0.44 |

     They also beat the spectrum reference at every level **[pre]**.
   - **The uniform control is weak** **[desc]**: about 5 of 6 uniform negatives land in other background islands, which are already far apart, so they add little force within an island. Beating it shows that the repulsion has to be targeted, not that ξ_B's targeting is the only one that works. Cross-reference item 3, the stronger same-cluster control.
3. **Background similarity by shared background neighbors (Jaccard).**
   - **ξ_B pairs share the background, not the signal** **[desc]**:
     - synthetic data: same background cluster 100% of the time; same-signal excess over chance at most +0.064;
     - MNIST-on-grass: same-digit excess +0.055 to +0.074; ξ_B pairs' grass distance 3.8–4.0, against 7.9–8.1 for random pairs.
   - **Beats same-cluster oracle negatives** **[pre]**:
     - interaction benchmark: +0.39;
     - both subspace-conflict benchmarks (cyclic and between-cluster): +0.21 and +0.22.

     So fine-grained neighborhoods matter, not just cluster membership **[interp]**.
   - **Caveat** **[desc]**: on the subspace-conflict benchmarks, the oracle control was selected at the top of its m_B grid (80), and its selection mean was still rising slowly (+0.01 to +0.02 from 40 to 80). Its best possible value is unknown.
4. **The two-stage schedule.**
   - **Why:** the package's early exaggeration is a no-op for these settings and does not reset the learning rate (N §5) **[desc]**. Two separate fits give an exact, reproducible handoff **[desc]**.
   - **Not tested:** no single-stage ablation was run. The warm-up's original rationale was to let a global layout form before the background term, to avoid fragmentation. That is a design rationale, not a tested result **[interp]**.
5. **Moving the background into the graph.**
   - **Evidence the raw graph was a limit:**
     - on both subspace-conflict benchmarks (cyclic and between-cluster), raw-graph background negatives reach about the high-D detectability (0.91 vs 0.92) **[desc]**;
     - on the weak-signal stress sets, gains are small or absent when the graph carries little signal **[desc]**;
     - 0 of about 925,000 kNN edges join background clusters, so islands can't merge **[desc]**;
     - on MNIST-on-grass, the raw graph's grass recall is 0.44 **[desc]**.
   - **Counterexample** **[desc]**: on MNIST-on-grass, raw-graph background negatives reached 0.931, above the raw graph's own digit accuracy (0.901 on the same confirmation seeds; 0.903–0.905 on the selection seeds). So the "graph ceiling" is not strict **[interp]**.
   - **Main evidence:** the residual-graph result itself. `\methodname`+BN beats raw-graph background negatives **[pre]**.
6. **Local residual graph vs a global cPCA graph** (the alternative considered).
   - **The comparison** **[desc]**: on MNIST-on-grass, the global cPCA-10 graph was cleaner than the residual graph.

     | seeds | graph | graph digit accuracy | grass recall |
     |---|---|---|---|
     | selection | global cPCA-10 | 0.970 | 0.006 |
     | selection | residual | 0.962 | 0.056 |
     | confirmation | global cPCA-10 | 0.957 | — |
     | confirmation | residual | 0.962 | — |

     On the confirmation seeds the accuracy order reverses, but cPCA-10 keeps the lower grass recall (0.006).
   - **The case for the local graph rests on non-additive backgrounds, which are untested** **[interp]**; see §8.
7. **Defaults k_loc = 30 and r = 2.**
   - **Gate** **[pre]**: +0.025 to +0.029 over the better of raw and PCA-50 on every selection seed.
   - **Robustness** **[desc]**: all 15 variants reach mean graph digit accuracy 0.954–0.964 (per seed 0.951–0.965), above raw (0.904) and PCA-50 (0.935). A label-free default is supported on this one dataset **[interp]**.
   - Source: `results/bsgraph/phase1.md`.
8. **Current status of background negatives.**
   - **Redundant on MNIST-on-grass with the residual graph** **[pre]**. m_B = 1 was selected, and accuracy falls slowly with m_B **[desc]**.
   - **Untested** with the residual graph anywhere else.
   - **Same information, two uses** **[interp]**: both the residual graph and ξ_B use each target's nearest background points; ξ_B for repulsion, the residual graph for neighborhoods.

## 6. Theoretical insights: when linear suppression works

1. **Population identity.** If x_T = s + g + ε and x_B = g′ + ε′, with g independent of s, g′ ~ g and equal noise, then C_T − C_B = C_S, for any background map.
   - Short proof in the text. **[der; standard]**
   - For α > 1: C_T − αC_B = C_S − (α−1)(C_G + σ²I). **[der]**
2. **Breaking independence is not enough.** Zero-mean cluster-specific signal maps outside span(B) keep the identity **[der]**. The cluster-specific-signal benchmark stopped at calibration, with cPCA WC 0.970–0.980 **[desc]**.
3. **Finding vs cleaning.** **[der]** *Pending your answer (question 1 below).*
   - Projecting onto the signal's directions does not remove background variance lying in those directions.
   - Illustrations **[desc]**:
     - between-cluster benchmark: U ∪ V gives 0.35 at k = 10;
     - cyclic benchmark: block structure.
4. **Global vs local background variance.**
   - A local method can help where the background varies strongly globally along the signal but weakly within a neighborhood **[interp]**.
   - What decides whether cPCA fails is the fraction of signal variance outside the directions of high global background variance, not the global/local ratio. The ratio was dominated by a 2–10% tilt into span(B) **[desc]**.
5. **The cyclic construction.**
   - **The bound:** a block-aligned projection reveals at most alternating clusters, 3 of 6 **[der]**. For general projections this is an argument **[interp]**. *Pending question 2.*
   - **Observed:** the best of the 64 labeled block subsets reveals exactly 3 clusters (WC 0.624) **[desc]**.
   - **Per-cluster cPCA:** C_T^k − αC_B^k keeps Q_k for any α ≥ 1 **[der]**. Every per-cluster label-free α was > 1 (1.7–58.8) **[desc]**, and local cPCA's WC was 0.98–0.99 **[pre]**.
6. **The residual graph in these terms** **[interp; untested]**:
   - the local mean removes the cluster centroid;
   - the top-r directions remove the local nuisance;
   - the cyclic nuisance is 3-D, while the default is r = 2.

   *Pending question 3.*

## 7. Empirical evidence

### 7.1 Evaluation protocol

- **Synthetic data with known ground truth;** labels are used for evaluation only.
- **Pre-registration:** protocols and predictions were committed before the results they govern.
- **Seeds:** selection seeds 0–2, then fresh confirmation seeds.
- **"Consistent across 3 paired seeds":** the mean difference is > 0 and every per-seed difference is > 0.
- **Determinism checks:** bit-identical reruns.
- **Dataset validity checks** and the exclusion rule.
- **Controls and what each rules out:**

  | control | what it rules out |
  |---|---|
  | uniform extra negatives | more repulsion |
  | spectrum reference (Z̄ = 100n, c = 500/n) | a shift along the spectrum |
  | same-cluster oracle negatives | knowing the cluster suffices |
  | plain Neg-t-SNE | — |
  | cPCA pipelines | linear suppression |

- **Platforms:** synthetic experiments on a laptop, MNIST-on-grass on the Math Slurm cluster. No comparison mixes platforms.

### 7.2 Synthetic results by question

All synthetic results used the raw graph with background negatives; the residual graph has not been tested on them.

- **Q1. Beyond more repulsion, and beyond knowing the clusters?**
  - **Additive benchmarks, confirmation-seed signal accuracy:**

    | level | raw-graph BN | best uniform | spectrum ref | plain |
    |---|---|---|---|---|
    | orthogonal | 0.894 | 0.387 | 0.346 | 0.312 |
    | overlapping | 0.582 | 0.405 | 0.384 | 0.368 |
    | nonlinear | 0.868 | 0.429 | 0.381 | 0.347 |

    Every comparison is consistent **[pre]**.
  - **Success criterion:** met only on the overlapping level **[pre]**. The failing clause, "background accuracy drops", cannot be met by design: the method adds only repulsion, and no kNN edges join clusters **[desc + interp]**. The overlapping level's pass rests on about 10 boundary points **[desc]**.
  - **Scale is not the explanation:** the uniform control and the spectrum reference spread the embedding more **[desc]**.
  - **Same-cluster oracle:**
    - interaction benchmark: +0.386, and +0.363 at m_B = 40;
    - cyclic: +0.213; between-cluster: +0.223.

    All **[pre]**.
  - **Stress sets:** +0.03 on two levels, none on the overlapping level **[desc]**. Run once and not used for any choice.
- **Q2. When does global cPCA win?**
  - **Additive benchmarks:** cPCA → Neg-t-SNE reaches 0.980–0.989 and passes all frozen global criteria **[desc]**. It beats raw-graph BN by +0.10 / +0.40 / +0.11 **[desc]**.
  - **Global layout:** raw-graph BN keeps a background-organized layout **[desc]**.
  - **Interaction benchmark:** cPCA beats raw-graph BN (0.984 vs 0.858) **[pre]**. Its global-metric failure did not replicate on 2 of 3 seeds **[desc]**.
  - **Label-free vs oracle α:** within 0.003 after embedding **[desc]**.
- **Q3. When does the method beat every global linear projection, and when does a local pipeline beat it?**
  - **Cyclic benchmark:**
    - vs best cPCA: 0.914 vs 0.530 (+0.36 to +0.41) **[pre]**;
    - vs best labeled projection: 0.624 **[desc]**;
    - 6 of 6 clusters revealed vs 3 **[desc]**.
  - **Between-cluster benchmark:** best cPCA 0.989 beats raw-graph BN (−0.07 to −0.08) **[pre]**. Label-free cPCA fails on both α and k (0.255) **[desc]**.
  - **Local cPCA:** 0.984–0.992 on both; K = 6 found with ARI 1.0; it beats raw-graph BN by +0.07 to +0.08 **[pre]**. Discrete, separable clusters are its most favorable case **[interp]**.
  - **Detectability:** 0.91 vs 0.92 **[desc]**.

### 7.3 Real data: MNIST-on-grass

- **Recipe:** the construction from Abid et al. 2018.
  - Digits 0 and 1 at 25% intensity on 28×28 grass patches from 1274 ImageNet images.
  - 5000 target and 5000 background images per seed.
  - Selection seeds 0–2, confirmation seeds 3–5.
  - Five documented deviations from the authors' notebook.
- **Raw-graph background negatives:**
  - 0.931, beating t-SNE (0.884), PCA → t-SNE (0.891) and plain Neg-t-SNE (0.858) **[pre]**;
  - cPCA → t-SNE (0.955) beats it **[pre]**.
- **Residual graph:**
  - graph quality: §5.6–5.7;
  - embeddings: `\methodname`+BN 0.973, `\methodname` 0.972 **[pre / desc]**;
  - grass recall 0.003 vs 0.022 **[desc]**.
- **Background-negatives ablation:** not consistent, with one exact tie **[pre]**.
- **Checks:** a bit-identical rerun, and stored results reused under a code-and-hash check **[desc]**.

### Planned tables (compact; full versions in Appendix C)

| table | content | source |
|---|---|---|
| T1 | additive benchmarks, four runs × 3 levels | `results/phase4/headline.md` |
| T2 | additive benchmarks: raw-graph BN vs cPCA → Neg-t-SNE | `results/phase5prep/` generated tables |
| T3 | subspace-conflict benchmarks: WC of all runs, including local cPCA | `results/phase5b/tables_*.md` |
| T4 | MNIST-on-grass: seven runs | `results/bsgraph/tables_phase2.md` |
| T5 | MNIST-on-grass pre-registered comparisons | same |
| T6 | residual-graph grid summary vs raw, PCA-50 and cPCA-10 | `results/bsgraph/phase1.md` |

### Planned figures (copied, not regenerated)

| figure | file |
|---|---|
| F1 | `results/phase4/embeddings_easy.png` |
| F2 | `results/phase5b/embeddings_cyclic.png` |
| F3 | `results/phase5b/embeddings_cyclic_local_cpca.png` |
| F4 | `results/bsgraph/embeddings_seed3.png` |
| F5 (optional) | `results/mnist_grass/embeddings_seed3.png` |

## 8. What has and hasn't been tested

- **Summary table:** one row per experiment, with columns for data, background type, graph (raw or residual), competitors, and outcome.
- **Gaps:**
  1. non-additive real data. This is also where the local residual graph's case against a global cPCA graph must be made (§5.6);
  2. the residual graph on the subspace-conflict benchmarks, including whether r = 2 suffices against a 3-D nuisance;
  3. continuous or curved backgrounds;
  4. the role of background negatives;
  5. label-free defaults beyond MNIST;
  6. the t-SNE vs Neg-t-SNE confound;
  7. a single-stage schedule;
  8. nonlinear contrastive methods (contrastiveVI);
  9. scaling beyond n = 5000.

## 9. Proposed next steps

1. A real dataset with a non-additive background, with `\methodname` (with or without BN) as the main method; the dataset to be chosen with the supervisor.
2. Optional: the residual graph on the subspace-conflict benchmarks.
3. What background negatives are for.

## Appendices

- **A. Name map:** report name → internal phase and run label → result directory. No report files.
- **B. Reproducibility:**
  - the repository (public GitHub) and key commits;
  - environment: Python 3.12, torch 2.14.1 CPU, contrastive-ne v0.4.0 `d5f0b53`, openTSNE 1.0.4;
  - platforms;
  - determinism rules.
- **C. Full tables:** generated tables from `results/`, converted to booktabs.

## Citations (to verify at drafting; unverifiable ones become `\todo{}` and are listed in `claims.md`)

- Abid et al. 2018
- van der Maaten & Hinton 2008
- McInnes et al. 2018
- Damrich et al. 2023
- Damrich & Hamprecht 2021
- Böhm, Berens & Kobak 2022
- Kobak & Linderman 2021
- Gutmann & Hyvärinen 2010
- Mikolov et al. 2013
- Levy & Goldberg 2014
- Poličar et al. (openTSNE)
- Weinberger et al. 2023 (contrastiveVI)

## Open questions from the first checkpoint (still unanswered)

1. **Finding vs cleaning (§6.3)** has no earlier report text. I'd write it as our derivation, illustrated with existing results. Is that the meaning you intend?
2. **The cyclic "no single projection" claim (§6.5).** I can prove it only for block-aligned projections; for general projections it stays an argument. Is that OK?
3. **The r = 2 vs 3-D nuisance remark (§6.6 and §8 gap 2)** is my own observation. Include it, or drop it?
4. **Figures:** F1–F4 by default; F5 optional.
