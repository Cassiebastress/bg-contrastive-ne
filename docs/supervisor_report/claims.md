# Claims register for `main.tex`

This file lists every claim in the report with its type and source.

## Conventions

- **Types:**
  - `pub`: published (cited);
  - `der`: our derivation;
  - `pre`: pre-registered verdict;
  - `desc`: measured, no verdict;
  - `interp`: our explanation, not tested.
- **Sources:**
  - Generated result files (`results/...`) are primary.
  - Report sections (`REPORT*.md`) are given where a number exists only in report prose, or where the report states the verdict.
- **Scope:** the report is self-contained. Appendix A maps its names to the repository's phases, detailed reports and results directories.
- **Unresolved `\todo{}`:** none.

**Cut in the shortening pass** (still documented in their phase reports, but no longer in this report):
- the subspace-conflict benchmarks (cyclic, between-cluster), local cPCA, and finding vs cleaning;
- the interaction and cluster-specific-signal benchmarks, and the same-cluster oracle;
- the weak-signal stress sets;
- the global/local variance discussion;
- the additive-benchmark figure, background-accuracy clause and embedding-spread comparison;
- the α-selection check;
- the UMAP effective-loss citation (Damrich & Hamprecht 2021, removed from `refs.bib`).

## §1 Summary

| # | claim | type | source |
|---|---|---|---|
| S1 | BC-NE+BN beats raw-graph BN on MNIST-on-grass, 0.973 vs 0.931 | pre | `results/bsgraph/tables_phase2.md`, `comparisons.json` |
| S2 | the residual graph raises plain Neg-t-SNE by 0.10–0.13 | desc | `results/bsgraph/tables_phase2.md` (descriptive comparison: +0.133, +0.103, +0.104) |
| S3 | BC-NE+BN vs BC-NE differ by ≤ 0.002 per seed, with one exact tie; not consistent | pre | `results/bsgraph/comparisons.json` (+0.0014, 0.0000, +0.0020) |
| S4 | BC-NE+BN vs cPCA→t-SNE not consistent; ranges 0.972–0.974 vs 0.941–0.979 | pre / desc | `results/bsgraph/tables_phase2.md` |
| S5 | additive ⇒ C_T − C_B = C_S | der | §5.6 proof |
| S6 | cPCA pipelines beat raw-graph BN on every additive benchmark run: synthetic (desc), MNIST (pre) | desc / pre | `results/phase5prep/cpca_summary.md`; `results/mnist_grass/tables.md` |

## §2 Problem, data and evaluation

| # | claim | type | source |
|---|---|---|---|
| P1 | synthetic template (n, D, noise 0.1, simplex 30 / 6, ρ rule, validity thresholds 0.50 / 0.90 / 0.95) | (definition) | `data/synthetic.py`; REPORT.md "Data" and "Checks with thresholds" |
| P2 | additive constructions; ρ = 0.2, 0.2, 0.3 | (definition) | `data/synthetic.py`; `results/phase1/validity.csv` |
| P3 | MNIST recipe: 0/1 at 25%, 28×28, 1274 photos, disjoint pools, 5000 + 5000, 5 deviations | (definition) | `data/README_mnist_grass.md`; `results/mnist_grass/build.json` |
| P4 | protocol: pre-registration, selection seeds 0–2, confirmation seeds, verdict rule, bit-identical reruns, no flags | (method) / desc | `common/experiment.py`; flag columns in every `runs_*.csv` |
| P5 | spectrum reference Z̄ = 100n, c = 500/n | (fact) | NOTES.md §6 |

## §3 Background

| # | claim | type | source |
|---|---|---|---|
| B1 | t-SNE and UMAP as attraction plus repulsion; the spectrum | pub | Damrich et al. 2023; Böhm et al. 2022; van der Maaten & Hinton 2008; McInnes et al. 2018 |
| B2 | NCE optimum q = p for any full-support ξ | pub | Gutmann & Hyvärinen 2010; Damrich et al. 2023 Thm 1 (checked in arXiv v2) |
| B3 | negative sampling with uniform ξ learns (\|X\|/m) p | pub | Damrich et al. 2023 Cor. 2 and the text after it (checked); Mikolov et al. 2013; Levy & Goldberg 2014 |
| B4 | the package classifier is q/(q+c), default c = 1; c moves along the spectrum | pub | Damrich et al. 2023; NOTES.md §4, §6 |
| B5 | general-ξ optimum, eq. (2); with background negatives, eq. (3); recovers Cor. 2 | der | §3 derivation |
| B6 | assumptions, including the clamp [1e-4, 1] | der | NOTES.md §3–4 |
| B7 | UMAP = NEG with the 1/d² model; its repulsion diverges, Neg-t-SNE's is bounded | pub | Damrich et al. 2023 Lemma 3 and the text after it (checked) |

## §4 Method

| # | claim | type | source |
|---|---|---|---|
| M1 | residual definition; defaults k_loc = 30, r = 2 | (definition) | `method/residual_graph.py`; `results/bsgraph/preregistration_phase2.md` |
| M2 | invariant to sign flips | der | `tests/test_residual_graph.py` |
| M3 | package defaults | (fact) | NOTES.md defaults table |
| M4 | PCA init with PC1 at std 1 | pub | Kobak & Linderman 2021; Damrich et al. 2023 |
| M5 | Stage 1 250 / Stage 2 500 epochs, fresh annealing | (fact) | `common/embed.py` |
| M6 | built-in early exaggeration does nothing at NEG defaults | desc | NOTES.md §5, §10 |
| M7 | warm-up rationale; no single-stage comparison | interp | Initial_Project_Plan.md §0/§4; no ablation in any results directory |
| M8 | ξ_B construction | (definition) | `method/xi_b.py` |
| M9 | unit test: equal loss and gradient for one background and one uniform negative | desc | `tests/test_bg_term.py` |
| M10 | the BN term lowers the similarity of background-overlapping pairs | der | eq. (3) |
| M11 | "target similarity divided by background similarity" | interp | REPORT.md "Why the islands split" |
| M12 | 11-line hook; exact graphs; own sampler generator; bit-identical when off | desc | NOTES.md §1, §8, Phase 3 |

## §5 Design decisions

| # | claim | type | source |
|---|---|---|---|
| D1 | NEG leaves ξ out of the classifier; NCE optimum p for any ξ; t-SNE has no noise distribution | der / pub | eq. (2); B2 |
| D2 | UMAP's repulsion diverges at short range | pub | B7 |
| D3 | background negatives act on pairs that start close | interp | — |
| D4 | Neg-t-SNE below t-SNE on MNIST without the term: 0.858 vs 0.884 / 0.891 | desc | `results/mnist_grass/tables.md` |
| D5 | under NCE the noise cancels at the optimum | pub | B2 |
| D6 | beats the best uniform control: +0.51 / +0.18 / +0.44; beats the spectrum reference | pre | `results/phase4/tables.md` |
| D7 | about 5 in 6 uniform negatives land in another cluster | desc | 6 equal clusters (construction) |
| D8 | ξ_B: same cluster 100%; same-signal excess ≤ +0.064 | desc | `results/phase2/xi_b_diagnostics.csv`; `results/phase4/summary_*_confirmation.json` |
| D9 | MNIST: same-digit excess +0.055 to +0.074; grass distance 3.8–4.0 vs 7.9–8.1 | desc | `results/mnist_grass/prechecks.md` |
| D10 | 0 of about 925,000 kNN edges join background clusters | desc | `results/phase4/data_geometry.md` |
| D11 | MNIST raw graph: 44% grass neighbors | desc | `results/bsgraph/phase1.md` (0.439) |
| D12 | counterexample: 0.931 > the raw graph's 0.901 | desc | `results/mnist_grass/tables.md`; `results/bsgraph/phase1.md` |
| D13 | the residual graph beats raw-graph BN | pre | `results/bsgraph/comparisons.json` |
| D14 | population identity, proof, any map, α > 1 | der | §5.6; REPORT_phase5prep.md |
| D15 | cPCA-10 graph vs residual graph (selection and confirmation numbers) | desc | `results/bsgraph/phase1.md` |
| D16 | the local graph's case rests on non-additive backgrounds | interp | — |
| D17 | gate +0.025 to +0.029 | pre | `results/bsgraph/phase1.md` |
| D18 | 15 variants 0.954–0.964 (per seed 0.951–0.965), above raw 0.904 and PCA-50 0.935; even r = 0 | desc | `results/bsgraph/phase1.md` |
| D19 | label-free default supported | interp | — |
| D20 | BN redundant with the residual graph; m_B = 1; 0.973 → 0.968 | pre / desc | `results/bsgraph/tables_phase2.md` |
| D21 | same information, two uses | interp | REPORT_bsgraph.md Interpretation |

## §6 Results

| # | claim | type | source |
|---|---|---|---|
| R1 | Table 1 values; BN beats every control | pre | `results/phase4/headline.md`, `tables.md` |
| R2 | cPCA→Neg-t-SNE 0.980–0.989; +0.10 / +0.40 / +0.11; neighbor fraction 0.805–0.820 vs 0.000–0.001 | desc | `results/phase5prep/cpca_summary.md` |
| R3 | MNIST pre-checks 0.90 / 0.93 / 1.00; grass recall 0.44 | desc | `results/mnist_grass/prechecks.md`; `results/bsgraph/phase1.md` |
| R4 | Tables 2–3 values and verdicts | pre / desc | `results/bsgraph/tables_phase2.md`; `results/mnist_grass/tables.md` |
| R5 | grass recall 0.022 → 0.003 | desc | `results/bsgraph/tables_phase2.md` |
| R6 | bit-identical rerun; reuse check | desc | `results/bsgraph/recheck.json`, `reuse.json` |

## §7–8 Gaps and next steps

All items are `desc` (not yet run) or `interp`.

## Appendices

| claim | type | source |
|---|---|---|
| pre-registration commits | — | `git log --diff-filter=A` on each pre-registration file; `3dd613a` from REPORT_mnist_grass.md |
| Mac vs cluster ≤ 0.027, no mixed comparisons | desc | `results/mnist_grass/tables.md` (cluster validation table) |
| environment versions | — | REPORT.md Reproducibility; results JSON `environment` blocks |
| Tables 5–7 | — | `results/bsgraph/phase1.md`; `results/bsgraph/tables_phase2.md` and `results/mnist_grass/tables.md` (appendices); `results/phase4/tables.md` |

## Citations

Each entry was checked against an authoritative record.

| key | checked against |
|---|---|
| abid2018cpca | Crossref, DOI 10.1038/s41467-018-04608-8 |
| vandermaaten2008tsne | jmlr.org/papers/v9/vandermaaten08a |
| mcinnes2018umap | arXiv API, 1802.03426 |
| damrich2023tsne | arXiv API, 2206.01816 (journal_ref ICLR 2023); theorem numbering from the v2 PDF |
| bohm2022spectrum | arXiv API, 2007.08902 (journal_ref JMLR 23(95):1–32) |
| kobak2021init | Crossref, DOI 10.1038/s41587-020-00809-z |
| gutmann2010nce | proceedings.mlr.press/v9/gutmann10a |
| mikolov2013word2vec | arXiv API, 1310.4546; NeurIPS 26, pp. 3111–3119 |
| levy2014neural | NeurIPS metadata JSON, pp. 2177–2185 |
| policar2024opentsne | Crossref, DOI 10.18637/jss.v109.i03 |
| weinberger2023contrastivevi | Crossref, DOI 10.1038/s41592-023-01955-3 |
| deng2009imagenet | Crossref, DOI 10.1109/CVPR.2009.5206848 |
| lecun1998mnist | Crossref, DOI 10.1109/5.726791 |
