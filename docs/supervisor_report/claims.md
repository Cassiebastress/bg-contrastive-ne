# Claims register for `main.tex`

This file lists every claim in the report with its type and source.

- **Types:**
  - `pub`: published, cited;
  - `der`: our derivation;
  - `pre`: pre-registered verdict;
  - `desc`: measured, no verdict;
  - `interp`: our explanation, not tested.
- **Sources:**
  - Generated result files (`results/...`) are primary.
  - Report sections (`REPORT*.md`) are given where a number exists only in report prose or where the report states the verdict. They are for checking only; the report itself does not cite them.
- **Unresolved `\todo{}`:** none.

## 1. Summary

| # | claim | type | source |
|---|---|---|---|
| S1 | BC-NE+BN beats raw-graph BN on MNIST-on-grass, 0.973 vs 0.931, +0.036 to +0.046 per seed | pre | `results/bsgraph/tables_phase2.md`, `comparisons.json` |
| S2 | the residual graph raises plain Neg-t-SNE by +0.10 to +0.13 | desc | `results/bsgraph/tables_phase2.md` (descriptive comparison: +0.133, +0.103, +0.104) |
| S3 | BC-NE+BN vs BC-NE: +0.0014, 0.0000, +0.0020, not consistent | pre | `results/bsgraph/comparisons.json`; REPORT_bsgraph.md Summary 5 |
| S4 | BC-NE+BN vs cPCA→t-SNE: +0.032, −0.006, +0.028, not consistent | pre | `results/bsgraph/tables_phase2.md` |
| S5 | ranges 0.972–0.974 vs 0.941–0.979 | desc | `results/bsgraph/tables_phase2.md` |
| S6 | additive ⇒ C_T − C_B = C_S | der | §6.1 proof; REPORT_phase5prep.md "Why cPCA succeeds" |
| S7 | cPCA→Neg-t-SNE beats raw-graph BN at every additive level | desc | `results/phase5prep/cpca_summary.md` ((c) vs (e2) tables). A paired comparison, but no verdict was pre-registered for this pair |
| S8 | cPCA→t-SNE beats raw-graph BN on MNIST-on-grass | pre | `results/mnist_grass/tables.md` |
| S9 | local cPCA 0.984–0.992 vs 0.914, beats raw-graph BN | pre (P6–P9) | `results/phase5b/tables_*_local_cpca.md`, `predictions_status.md` |
| S10 | cyclic: raw-graph BN 0.914 vs best cPCA 0.530 | pre | `results/phase5b/tables_cyclic.md` |
| S11 | cyclic: vs best labeled projection 0.624 | desc | `results/phase5b/tables_cyclic.md` |

## 3. Background

| # | claim | type | source |
|---|---|---|---|
| B1 | t-SNE and UMAP are attraction along kNN edges plus repulsion | pub | Damrich et al. 2023 §3; van der Maaten & Hinton 2008; McInnes et al. 2018 |
| B2 | attraction–repulsion spectrum | pub | Böhm et al. 2022 |
| B3 | NCE optimum q = p for any full-support ξ | pub | Gutmann & Hyvärinen 2010; Damrich et al. 2023 Theorem 1 (checked in arXiv v2 text) |
| B4 | negative sampling: classifier q/(q+1); with uniform ξ it learns (\|X\|/m) p | pub | Damrich et al. 2023 Corollary 2 and the text following it (checked); Mikolov et al. 2013; Levy & Goldberg 2014 |
| B5 | the package's classifier is q/(q+c) with constant c (default 1); c moves along the spectrum | pub + desc | Damrich et al. 2023; NOTES.md §4 and §6 (`cne.py:654`) |
| B6 | general-ξ optimum q* = c p/(mξ); with background negatives, eq. (3) | der | §3 derivation; Initial_Project_Plan.md §0 states the idea |
| B7 | the uniform-ξ case recovers Corollary 2 | der | substitution in eq. (2) |
| B8 | assumptions: free q, expectations, in-batch ≈ uniform, clamp [1e-4, 1] ignored | der | NOTES.md §3–4 (in-batch sampling; clamp values) |
| B9 | UMAP's effective loss differs from the stated one and has far less repulsion | pub | Damrich & Hamprecht 2021; Damrich et al. 2023 §3 ("about m/n of the repulsion") |
| B10 | UMAP = NEG with the 1/d² model, whose repulsion diverges at d → 0; Neg-t-SNE's is bounded | pub | Damrich et al. 2023 Lemma 3 and the text following it (checked) |
| B11 | PCA init with PC1 at std 1 | pub | Kobak & Linderman 2021; Damrich et al. 2023 (Supp. K.2, per NOTES.md §7) |

## 4. Method

| # | claim | type | source |
|---|---|---|---|
| M1 | residual definition, defaults k_loc = 30, r = 2 | (definition) | `method/residual_graph.py`; `results/bsgraph/preregistration_phase2.md` |
| M2 | invariant to sign flips of V's columns | der | `tests/test_residual_graph.py` (tested) |
| M3 | package defaults: m = 5 in-batch, c = 1, batch 2^15, SGD lr 1 linear annealing, clip ±4 | (fact) | NOTES.md defaults table |
| M4 | Stage 1 250 epochs / Stage 2 500 epochs, fresh annealing | (fact) | `common/embed.py`; REPORT.md "Training" |
| M5 | ξ_B construction (k_B = 15, L = 30, ties up to 2L, seeded draw) | (definition) | `method/xi_b.py`; REPORT.md "ξ_B" |
| M6 | a unit test checks equal loss and gradient for one background negative and one uniform negative | desc | `tests/test_bg_term.py`; NOTES.md Phase 3 |
| M7 | "target similarity divided by background similarity" | interp | REPORT.md "Why the islands split" |
| M8 | 11-line hook; exact graphs; package graph unseeded; own sampler generator; bit-identical when off | desc | NOTES.md §1, §8, Phase 3; `third_party/contrastive-ne/UPSTREAM.md` |

## 5. Design decisions

| # | claim | type | source |
|---|---|---|---|
| D1 | NEG leaves ξ out of the classifier, so ξ changes the optimum | der | eq. (2) |
| D2 | NCE optimum is p for any ξ; t-SNE has no noise distribution | pub | B3; van der Maaten & Hinton 2008 |
| D3 | the package uses a constant c, not ξ | desc | NOTES.md §4, §9 |
| D4 | UMAP's repulsion diverges at short range | pub | B10 |
| D5 | background negatives act on pairs that start close | interp | — |
| D6 | without the background term, Neg-t-SNE is below t-SNE on MNIST-on-grass: 0.858 vs 0.884 (0.891 PCA→t-SNE) | desc | `results/mnist_grass/tables.md` |
| D7 | raw-graph BN beats the best uniform control at all three additive levels: +0.51 / +0.18 / +0.44 | pre | `results/phase4/tables.md` (comparisons vs b_m20: means +0.5071, +0.1771, +0.4393) |
| D8 | it also beats the spectrum reference at every level | pre | `results/phase4/tables.md` |
| D9 | about 5 in 6 uniform negatives land in another cluster; they add little force within a cluster | desc | 6 equal clusters (by construction); REPORT.md "Why the islands split" |
| D10 | ξ_B pairs share the background cluster 100% of the time; same-signal excess ≤ +0.064 | desc | `results/phase2/xi_b_diagnostics.csv`; `results/phase4/summary_*_confirmation.json` (medium confirmation up to +0.064); phase5a/5b tables (≤ +0.044) |
| D11 | MNIST same-digit excess +0.055 to +0.074; grass distance 3.8–4.0 vs 7.9–8.1 | desc | `results/mnist_grass/prechecks.md` |
| D12 | ξ_B beats the oracle: +0.39 (interaction), +0.21 / +0.22 (cyclic / between) | pre | `results/phase5a/tables_J.md` (+0.386 mean); `results/phase5b/tables_cyclic.md` (+0.213), `tables_between.md` (+0.223) |
| D13 | fine-grained neighborhoods matter, not just the cluster | interp | REPORT_phase5a.md Summary 5 |
| D14 | the oracle was selected at the top of its grid (80), still rising by +0.01 to +0.02 from 40 to 80 | desc | REPORT_phase5b.md Limitations; `results/phase5b/selection_*.json` |
| D15 | early exaggeration does nothing at NEG defaults; no LR reset | desc | NOTES.md §5 and §10 (checked empirically) |
| D16 | no single-stage ablation; the warm-up rationale is a design choice | interp | Initial_Project_Plan.md §0/§4 (stated as a design choice; no ablation in any results directory) |
| D17 | raw-graph BN reaches about the detectability: 0.91 vs 0.915–0.924 | desc | `results/phase5b/tables_cyclic.md`, `tables_between.md` (detectability, confirmation seeds) |
| D18 | stress sets: small or absent gains | desc | `results/phase4/stress/tables_stress.md` |
| D19 | 0 of about 925,000 kNN edges join background clusters | desc | `results/phase4/data_geometry.md` (313574 + 312002 + 300033 = 925609) |
| D20 | MNIST raw graph: 44% of neighbors are grass neighbors | desc | `results/bsgraph/phase1.md` (raw grass recall 0.439) |
| D21 | counterexample: raw-graph BN 0.931 > raw graph 0.901 (confirmation); 0.903–0.905 (selection) | desc | `results/mnist_grass/tables.md`; `results/bsgraph/phase1.md`; `results/mnist_grass/prechecks.md` |
| D22 | so the graph ceiling is not strict | interp | — |
| D23 | global cPCA-10 graph vs residual: 0.970 / 0.006 vs 0.962 / 0.056 (selection); 0.957 / 0.006 vs 0.962 / 0.057 (confirmation) | desc | `results/bsgraph/phase1.md` |
| D24 | the local graph's case rests on non-additive backgrounds, untested | interp | — |
| D25 | gate passed: +0.025 / +0.027 / +0.029 (seeds 1 / 0 / 2) | pre | `results/bsgraph/phase1.md`, `results/bsgraph/preregistration_phase1.md` |
| D26 | all 15 variants 0.954–0.964 (per seed 0.951–0.965), above raw 0.904 and PCA-50 0.935; even r = 0 | desc | `results/bsgraph/phase1.md` |
| D27 | a label-free default is supported on this dataset | interp | — |
| D28 | ξ_B redundant with the residual graph; m_B = 1 selected; 0.973 → 0.968 at m_B = 40 | pre + desc | `results/bsgraph/tables_phase2.md` |
| D29 | the residual graph and ξ_B use the same information | interp | REPORT_bsgraph.md Interpretation |

## 6. Theory

| # | claim | type | source |
|---|---|---|---|
| T1 | population identity and proof | der (standard) | §6.1; REPORT_phase5prep.md |
| T2 | holds for any background map | der | proof (only Cov(g) enters) |
| T3 | for α > 1: C_T − αC_B = C_S − (α−1)(C_G + σ²I) | der | REPORT_phase5prep.md |
| T4 | zero-mean cluster-specific maps outside span(B) keep the identity | der | REPORT_phase5a.md Interpretation |
| T5 | cluster-specific benchmark stopped; cPCA WC 0.970–0.980 | desc | `results/phase5a/k_calibration.md` |
| T6 | Cov(P x_T) = P C_S P + P C_G P + σ²P; finding vs cleaning | der | new in this report (user-approved framing, outline Q1) |
| T7 | between-cluster: k ≤ 5 gives 0.989; k = 10 gives 0.35; U ∪ V gives 0.34–0.36 | desc | `results/phase5b/tables_between.md` (cPCA variants by α and k: 0.988 / 0.989 at k = 3 / 5, 0.352 at k = 10); `results/phase5b/runs_between_confirmation.csv` (lin_U+V: 0.354, 0.336, 0.356) |
| T8 | between-cluster offsets don't affect WC | der | definition of WC |
| T9 | a local method helps where background variance is high globally and low locally | interp | REPORT_phase5a.md data diagnostics rationale |
| T10 | what decides is the fraction of signal outside high-global-variance directions; ratio 46–69 from a 2–10% tilt; ≥ 87% outside | desc | `results/phase5a/data_diagnostics.md`; REPORT_phase5a.md "Reading the global/local ratio" (the corrected statement) |
| T11 | block-aligned projections reveal at most 3 of 6 clusters | der | §6.4; `results/phase5b/calibration.md` (subsets by clean reveals) |
| T12 | general projections: ρ²a_k ≫ σ_w²a_{k+1}; (σ_w/ρ)² ≈ 42 per step; not proven | interp | user-approved wording (outline Q2); σ_w² = 5/3, ρ = 0.2 from `data/synthetic_l.py` |
| T13 | best block subset reveals exactly 3 clusters (0.624); best cPCA 0.530 (0.551 max) | desc | `results/phase5b/tables_cyclic.md`, `calibration.md` |
| T14 | per-cluster cPCA keeps Q_k for any α ≥ 1 | der | §6.4 |
| T15 | per-cluster label-free α all > 1 (1.7–58.8) | desc | `results/phase5b/tables_*_local_cpca.md` ("Clusters and α"); REPORT_phase5b.md extension |
| T16 | the residual graph removes centroid and local nuisance; r = 2 vs a 3-D nuisance | interp | user-approved (outline Q3); 3-D blocks from `data/synthetic_l.py` |
| T17 | r = 5 nearly identical on MNIST (0.961 vs 0.962 at k_loc = 30) | desc | `results/bsgraph/phase1.md` |

## 7. Evidence

| # | claim | type | source |
|---|---|---|---|
| E1 | protocol: selection seeds 0–2, fresh confirmation seeds; "consistent" rule | (method) | `common/experiment.py` (`paired`); REPORT.md protocol |
| E2 | validity thresholds 0.50 / 0.90 / 0.95; one seed excluded | (method) + desc | REPORT.md "Checks with thresholds"; `results/phase4/excluded_runs_note.md` |
| E3 | no run flagged | desc | flag columns in every `runs_*.csv` |
| E4 | spectrum reference Z̄ = 100n, c = 500/n | (fact) | NOTES.md §6; REPORT.md runs table |
| E5 | platforms differ by ≤ 0.027 WC; no mixed comparisons | desc | `results/mnist_grass/tables.md` (cluster validation table) |
| E6 | benchmark constructions (n, D, simplex sizes, ρ) | (definition) | `data/synthetic*.py`; REPORT.md / REPORT_phase5a.md / REPORT_phase5b.md "Design" |
| E7 | interaction: 1 of 16 configurations valid | desc | `results/phase5a/grid_J.md` |
| E8 | Table 1 values | pre (comparisons) / desc (cPCA column) | `results/phase4/headline.md`; `results/phase5prep/cpca_summary.md` |
| E9 | success criterion met only on overlapping; the clause can't be met by design; about 10 boundary points | pre + desc | `results/phase4/verdicts.json`; REPORT.md Summary |
| E10 | spread vs plain: 1.11–1.25 / 1.44–1.47 / 1.79–1.87 | desc | `results/phase4/runs_*_confirmation.csv` (std_ratio_vs_a; recomputed: easy 1.11 / 1.44 / 1.79, medium 1.25 / 1.46 / 1.82, hard 1.16 / 1.47 / 1.87) |
| E11 | oracle: interaction +0.385 to +0.388 (0.858 vs 0.472); +0.36 at m_B = 40; cyclic +0.19 to +0.23; between +0.22 to +0.23 | pre | `results/phase5a/tables_J.md`, `extension_o.md`; `results/phase5b/tables_*.md` |
| E12 | stress: +0.03 consistent on orthogonal and nonlinear; none on overlapping | desc | `results/phase4/stress/tables_stress.md` |
| E13 | cPCA additive G1 0.805–0.820 vs 0.000–0.001; gains +0.10 / +0.40 / +0.11 | desc | `results/phase5prep/cpca_summary.md` |
| E14 | interaction: cPCA 0.984 vs 0.858; per seed +0.086 to +0.155 | pre | `results/phase5a/tables_J.md` |
| E15 | label-free vs oracle α: within 0.003 after embedding | desc | REPORT_phase5prep.md Summary 4; `results/phase5prep/cpca_summary.md` |
| E16 | cyclic: +0.36 to +0.41 vs best cPCA; +0.29 vs best projection; 6 vs 3 clusters, per-cluster ≥ 0.85 | pre / desc / desc | `results/phase5b/tables_cyclic.md`; REPORT_phase5b.md Summary 1 |
| E17 | between: best cPCA beats by 0.07–0.08; label-free 0.255 | pre / desc | `results/phase5b/tables_between.md` |
| E18 | local cPCA: K = 6, ARI 1.0; 0.984–0.992; beats by +0.07 to +0.08; predictions confirmed | desc / pre | `results/phase5b/tables_*_local_cpca.md`, `predictions_status.md` |
| E19 | discrete clusters favor local cPCA | interp | REPORT_phase5b.md Limitations |
| E20 | Table 2 values | pre / desc | `results/phase5b/tables_*.md`, `tables_*_local_cpca.md` |
| E21 | MNIST recipe (0/1 at 25%, 28×28, 1274 photos, disjoint pools, 5000 + 5000, 5 deviations) | (definition) | `data/README_mnist_grass.md`; `results/mnist_grass/build.json` |
| E22 | pre-checks: 0.90 / 0.93 / 1.00; grass recall 0.44 | desc | `results/mnist_grass/prechecks.md`; `results/bsgraph/phase1.md` |
| E23 | Tables 3–4 values and verdicts | pre / desc | `results/bsgraph/tables_phase2.md`; `results/mnist_grass/tables.md` |
| E24 | grass recall 0.022 → 0.003 | desc | `results/bsgraph/tables_phase2.md` |
| E25 | rerun bit-identical; reuse check passed | desc | `results/bsgraph/recheck.json`, `reuse.json` |

## 8–9. Gaps and next steps

All gap items are `desc` (what has not been run) or `interp`. Next steps follow the order in REPORT_bsgraph.md "Next steps", as revised by the user.

## Appendices

| claim | source |
|---|---|
| pre-registration commits | `git log --diff-filter=A` on each pre-registration file; `a791d9b` (local cPCA predictions) and `3dd613a` (MNIST selection) from REPORT_phase5b.md / REPORT_mnist_grass.md reproducibility sections |
| environment versions | REPORT.md Reproducibility; `env/`; results JSON `environment` blocks |
| Tables 6–8 | `results/bsgraph/phase1.md`; `results/bsgraph/tables_phase2.md` and `results/mnist_grass/tables.md` (appendices); `results/phase4/tables.md` |

## Citations

Each entry was checked against an authoritative record. None is unresolved.

| key | checked against | note |
|---|---|---|
| abid2018cpca | Crossref, DOI 10.1038/s41467-018-04608-8 | |
| vandermaaten2008tsne | jmlr.org/papers/v9/vandermaaten08a | |
| mcinnes2018umap | arXiv API, 1802.03426 | |
| damrich2023tsne | arXiv API, 2206.01816 (journal_ref ICLR 2023); theorem numbering from the v2 PDF text | |
| damrich2021umap | arXiv API, 2103.14608; NeurIPS proceedings page (vol. 34) | proceedings page numbers not found, so omitted |
| bohm2022spectrum | arXiv API, 2007.08902 (journal_ref JMLR 23(95):1–32) | |
| kobak2021init | Crossref, DOI 10.1038/s41587-020-00809-z | |
| gutmann2010nce | proceedings.mlr.press/v9/gutmann10a | |
| mikolov2013word2vec | arXiv API, 1310.4546; NeurIPS 26, pp. 3111–3119 | |
| levy2014neural | NeurIPS metadata JSON, pp. 2177–2185 | |
| policar2024opentsne | Crossref, DOI 10.18637/jss.v109.i03 | |
| weinberger2023contrastivevi | Crossref, DOI 10.1038/s41592-023-01955-3 | |
| deng2009imagenet | Crossref, DOI 10.1109/CVPR.2009.5206848 | |
| lecun1998mnist | Crossref, DOI 10.1109/5.726791 | |
