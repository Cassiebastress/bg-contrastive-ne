# bg-contrastive-ne

**Background-contrastive neighbor embedding.** This is a t-SNE/UMAP-style 2-D visualization of one dataset (the *target*). It hides the variation that the target shares with a second, reference dataset (the *background*), so the structure specific to the target becomes visible.

This is a research codebase (a McGill COMP 400 project). It holds:
- the method;
- every experiment run so far, with pre-registered predictions, generated results and a report per experiment;
- a summary report for the supervisor.

- [The problem](#the-problem)
- [The method in brief](#the-method-in-brief)
- [What we have found so far](#what-we-have-found-so-far)
- [Where to read more](#where-to-read-more)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Using the method on your own data](#using-the-method-on-your-own-data)
- [Reproducing the experiments](#reproducing-the-experiments)
- [How the experiments are run](#how-the-experiments-are-run)
- [Glossary](#glossary)
- [Third-party code and references](#third-party-code-and-references)

---

## The problem

You often have two datasets in the same feature space:
- **target:** the data you care about. It contains the structure you want to see (the *signal*) mixed with nuisance variation.
- **background:** a dataset that contains only the nuisance variation.

Three examples:
- treated cells (target) vs untreated cells (background);
- tumor samples vs healthy tissue;
- the running example in this repo, MNIST digits pasted onto photos of grass (target) vs photos of grass alone (background).

A standard t-SNE or UMAP plot of the target is often dominated by the nuisance. In the MNIST-on-grass example, images group by how bright the grass is, not by which digit they show.

**Contrastive PCA (cPCA)** ([Abid et al., 2018](https://doi.org/10.1038/s41467-018-04608-8)) is the linear answer. It finds directions with high target variance and low background variance: the top eigenvectors of `C_target − α·C_background`. This project asks the neighbor-embedding version of the question: can we get a t-SNE-like picture of the target in which the background's structure is suppressed?

## The method in brief

Neighbor embeddings such as t-SNE and UMAP work in two steps. They build a k-nearest-neighbor (kNN) graph of the data, then lay out a 2-D embedding. Points joined by graph edges attract each other; other pairs repel. The method, called **BC-NE** in the reports, changes both steps.

1. **Background-suppressed (residual) kNN graph: the core.** For each target point:
   - find its `k_loc = 30` nearest background points;
   - subtract their mean and their top `r = 2` principal directions;
   - build the kNN graph on what remains.

   Neighbors in this graph are similar *after* removing the local background variation. This needs no clustering and no global projection. Code: `method/residual_graph.py`.
2. **Neg-t-SNE.** The graph is embedded with Neg-t-SNE ([Damrich et al., ICLR 2023](https://arxiv.org/abs/2206.01816)), a negative-sampling neighbor embedding that sits between t-SNE and UMAP. We use the authors' `contrastive-ne` package, vendored in `third_party/`.
   - **BC-NE:** one standard 750-epoch run with package defaults. This is the form adopted after `REPORT_single_stage.md`, which found it matches the earlier two-stage schedule closely.
   - **BC-NE+BN:** keeps the two-stage schedule, a 250-epoch warm-up and then a 500-epoch second stage in which background negatives act.
   - **Earlier reports:** they used the two-stage schedule for both.

   Code: `common/embed.py`.
3. **Optional background negatives.** In the second stage, extra repulsion is added between target points whose *background* neighborhoods overlap (Jaccard overlap of their nearest background points). Code: `method/xi_b.py`, `method/bg_term.py`.

   The theory: negative sampling, unlike NCE, does not put the noise distribution into its classifier. Drawing extra negatives from a background-similarity distribution therefore changes what the embedding learns, roughly "target similarity divided by background similarity". The derivation is in the supervisor report, §3.

   This step was the project's *original* method. The residual graph came later and turned out to matter more (see below).

## What we have found so far

These are the headline results. The supervisor report has the details and caveats.

| finding | evidence |
|---|---|
| **The residual graph is what makes the method work.** On MNIST-on-grass, BC-NE with background negatives reaches 0.973 digit kNN accuracy, against 0.931 for the original method (background negatives on the raw kNN graph). It wins on every confirmation seed. | pre-registered, `REPORT_bsgraph.md` |
| **Background negatives are redundant once the graph is background-suppressed.** With and without them, results differ by ≤ 0.002 per seed. | pre-registered, `REPORT_bsgraph.md` |
| **On MNIST-on-grass, BC-NE ties cPCA → t-SNE** (0.973 vs 0.955, not consistent across seeds) and is more stable across seeds. It clearly beats t-SNE (0.884) and PCA → t-SNE (0.891). | pre-registered, `REPORT_bsgraph.md`, `REPORT_mnist_grass.md` |
| **Global cPCA wins whenever the background is simply added to the signal.** If target = signal + independent background, the target-minus-background covariance is exactly the signal covariance, so cPCA is exact in the population. Every additive benchmark we ran, synthetic or real, behaves accordingly. | derivation; `REPORT_phase5prep.md`, `REPORT_mnist_grass.md` |
| **On synthetic benchmarks built so that no single global projection works,** the original method beats every global linear pipeline, including ones chosen with labels. A cluster-then-cPCA pipeline ("local cPCA") does better still. | pre-registered, `REPORT_phase5b.md` |

**Main open question.** The residual graph's intended advantage is suppressing a background that no global projection can remove. That has not yet been tested on real data where the background is *not* simply additive, and that is the next step.

## Where to read more

**Start with the supervisor report:** [`docs/supervisor_report/main.pdf`](docs/supervisor_report/main.pdf) (LaTeX source in the same folder). It is about 12 pages and self-contained. It covers the problem, the theory, the method, each design decision with its evidence, the results, and what remains untested.

**For a specific experiment**, the work ran in phases. Each phase has a report at the repository root and a results folder under `results/`. In chronological order:

| phase | question | report | answer in one line |
|---|---|---|---|
| 0 | How does `contrastive-ne` work internally? | `NOTES.md` §1–10 | Package internals, determinism rules, and the two-stage schedule. |
| 1–3 | Build the synthetic data, the background-negative distribution and the trainer hook. | `NOTES.md` (Phases 1–3) | Generators, diagnostics and tests. |
| 4 | Do background negatives reveal hidden signal on synthetic data? | `REPORT.md` | Yes within each background cluster, far beyond controls. But the global layout stays organized by the background. |
| 5 prep | How does cPCA do on the same data? | `REPORT_phase5prep.md` | It nearly solves it, because those benchmarks are additive. |
| 5a | Can we break cPCA with non-additive signals? | `REPORT_phase5a.md` | Mostly not. Background negatives beat a same-cluster oracle control. |
| 5b | Benchmarks where no global projection works ("Design L"). | `REPORT_phase5b.md` | Background negatives beat every global linear pipeline; local cPCA beats them. |
| MNIST-on-grass | Does it work on real images? | `REPORT_mnist_grass.md` | Beats t-SNE controls; cPCA → t-SNE is better. |
| bsgraph | Build the kNN graph on local background residuals. | `REPORT_bsgraph.md` | Big gain; ties cPCA → t-SNE; background negatives now redundant. |
| MNIST 0–9 | Repeat MNIST-on-grass with all ten digits. | `REPORT_mnist_grass_0to9.md` | The residual graph beats cPCA → t-SNE and the t-SNE controls; background negatives add a small, consistent gain. |
| single stage | Does BC-NE need the two-stage schedule? | `REPORT_single_stage.md` | No: one 750-epoch run matches it closely on 0/1 and 0–9, so it becomes BC-NE's form. |
| tuned T3 | Was cPCA → t-SNE under-tuned in the 0–9 phase? | `REPORT_mnist_grass_0to9_t3grid.md` | Yes: with α and the number of components tuned like m_B, it beats BC-NE and BC-NE+BN on every seed (0.689 vs 0.512 and 0.530). |

Other documents:
- `NOTES.md`: working notes, package internals with file/line references, and a short summary of each phase.
- `Initial_Project_Plan.md` and `Phase_5_Plan.md`: the original plans.
- `cluster/README.md`: how to run on the McGill Math Slurm cluster.
- `data/README_mnist_grass.md` and `data/MANIFEST.md`: the real-data recipe and its checksums.

## Repository layout

| path | contents |
|---|---|
| `method/` | the method: `residual_graph.py` (residual kNN graph), `xi_b.py` (background-negative distribution), `bg_term.py` (the extra loss term, hooked into the trainer) |
| `common/` | shared infrastructure: `embed.py` (PCA init, Neg-t-SNE fits, two-stage schedule), `knn.py` (exact kNN graphs), `repro.py` (determinism, environment logging), `experiment.py` (run labels, the paired-seed verdict rule, flags), `benchmark*.py` (benchmark runners) |
| `baselines/` | cPCA (`cpca.py`, which reproduces the reference `contrastive` package), cPCA → embedding (`cpca_embed.py`), local cPCA (`local_cpca.py`) |
| `eval/` | metrics: kNN accuracy and recall, global layout metrics, benchmark metrics, background-negative diagnostics |
| `data/` | dataset generators: `synthetic.py` (additive benchmarks), `synthetic_j.py`, `synthetic_k.py`, `synthetic_l.py` (later benchmark designs), `mnist_grass.py` (real data) |
| `scripts/` | one script per phase step (`phase4_run.py`, `bsgraph_run.py`, …), plus `*_report.py` and `*_assemble.py`, which generate each report's tables and figures from `results/` |
| `results/` | every result, one folder per phase: JSON/CSV with the full environment recorded, generated Markdown tables, figures, pre-registration files. Large arrays (`.npy`, `.npz`) are gitignored. |
| `tests/` | pytest suite (run before every experiment) |
| `cluster/` | Slurm job scripts and a parallel launcher for the McGill Math cluster |
| `third_party/contrastive-ne/` | vendored `contrastive-ne` v0.4.0, plus an 11-line hook so the trainer can add a loss term (see `UPSTREAM.md`) |
| `reference/` | the cPCA authors' MNIST-on-grass notebook, kept for reference and not run |
| `docs/supervisor_report/` | the summary report (LaTeX, PDF, claims register) |
| `env/` | pinned environments: `environment.yml` and `pip-freeze.txt` (Mac), `requirements-linux.txt` (cluster) |
| `REPORT*.md` | one report per phase, assembled from `scripts/REPORT*.template.md` and the generated tables |

## Setup

Everything runs on CPU. Python 3.12 and PyTorch 2.14.1 (CPU wheels).

**Linux (and the cluster).** `cluster/setup_env.sh` builds the environment and runs the tests. Read it for the exact steps:
- a conda env with Python 3.12;
- the CPU PyTorch wheels;
- `pip install -r env/requirements-linux.txt`;
- the `contrastive` package `--no-deps` from its checksummed source;
- the vendored `contrastive-ne` in editable mode.

**macOS.** The environment the synthetic experiments ran in is recorded in `env/environment.yml` and `env/pip-freeze.txt`. Install the vendored package last, so the hooked version is the one imported:

```sh
conda env create -f env/environment.yml
conda activate cne-bg
pip install -e third_party/contrastive-ne
```

**Check the install:**

```sh
python -m pytest            # from the repo root; pytest.ini puts the repo on the path
```

> If you use pyenv alongside conda, `conda run` can pick up the wrong interpreter. Call the env's Python by its full path (e.g. `~/miniconda3/envs/cne-bg/bin/python`).

## Using the method on your own data

```python
import numpy as np
from common.repro import setup_determinism
from common.embed import fit_neg, pca_init, two_stage
from method.residual_graph import residual_graph
from method.xi_b import build_xi_b

# X_T: (n_T, D) target, X_B: (n_B, D) background, same features and preprocessing
seed = 0
setup_determinism(seed)

# 1. Background-suppressed kNN graph (defaults k_loc=30, r=2, k=15)
G, _ = residual_graph(X_T, X_B, k_loc=30, r=2)

# 2. One 750-epoch Neg-t-SNE run from a PCA initialization of the raw target (BC-NE)
Y = fit_neg(pca_init(X_T), G, seed, n_epochs=750)["embd"]   # (n_T, 2) embedding

# Optional: background negatives (BC-NE+BN), two-stage: they act in Stage 2 after a 250-epoch warm-up
xi = build_xi_b(X_T, X_B, k_B=15, L=30, seed=seed)
_, stage2_bn = two_stage(pca_init(X_T), G, seed, bg_mode="xi_b", m_B=1, xi=xi)
```

**Runtime.** At n = 5000, one run takes a few minutes on one CPU core. The residual step does one small SVD per target point.

**Preprocessing.** Preprocess `X_T` and `X_B` identically (for real data, fit any scaling on both together).

## Reproducing the experiments

Each phase follows the same pattern. Each phase report ends with a **Reproducibility** section giving the exact commands.

```sh
python scripts/<phase>_run.py --stage selection ...      # choose settings on seeds 0–2
# commit results/<phase>/selection*.json before going on
python scripts/<phase>_run.py --stage confirmation ...   # fresh seeds 3–5, the verdicts
python scripts/<phase>_report.py                         # tables and figures from results/
python scripts/<phase>_assemble.py                       # REPORT_<phase>.md from its template
```

**Where things ran.** The synthetic experiments (Phases 0–5b) ran on a laptop. MNIST-on-grass and the residual-graph experiments ran on the McGill Math Slurm cluster, 24 single-threaded runs in parallel (see `cluster/README.md`). The newer scripts take `--make-jobs` to write a job list and `--merge` to combine per-seed outputs.

**MNIST-on-grass data.** The raw files are not committed:
- the 1274 ImageNet grass photos, from the cPCA authors' Google Drive folder;
- `mnist.npz`, from the Keras URL.

Their sources and SHA-256 checksums are in `data/MANIFEST.md`. Put them in `data/raw/`, then run `scripts/mnist_grass_manifest.py` and `scripts/mnist_grass_build.py`. Every run checks the dataset hash before using it.

## How the experiments are run

The project follows a few strict conventions. They make the results trustworthy, and they explain some of the repo's structure.

- **Pre-registration.** Each phase's runs, settings grid, comparisons and predictions are written to `results/<phase>/preregistration*.md` or `predictions.md`, and committed *before* the results they govern. Commit hashes are cited in each report.
- **Selection vs confirmation seeds.** Each seed generates a fresh dataset. Settings (e.g. how many background negatives) are chosen on seeds 0–2; verdicts use fresh seeds 3–5.
- **The verdict rule.** Runs are paired by seed. "X beats Y, consistent across 3 paired seeds" means the mean difference is positive **and** every per-seed difference is positive. Anything else is "not consistent". No p-values.
- **Generated tables only.** No results table is typed by hand. `scripts/*_report.py` writes them to `results/`, and `scripts/*_assemble.py` pastes them into the report.
- **Determinism.** Every run uses:
  - deterministic PyTorch algorithms;
  - a fixed thread count;
  - seeded generators;
  - exact kNN graphs (the package's approximate graph is unseeded, so it is never used).

  Reruns are checked to be bit-identical. Every results JSON records the software environment and git commit.
- **No mixed platforms.** Laptop and cluster results are each bit-reproducible, but they are not identical to each other. No comparison mixes them.
- **Stop and report.** If a check fails (a hash mismatch, a non-reproducible rerun, an invalid dataset), the run stops and the failure is reported. It is not tuned around.

## Glossary

| term | meaning |
|---|---|
| target / background | the dataset you want to visualize / the reference dataset that contains only nuisance variation |
| signal | target-specific structure (e.g. which digit); used only for evaluation, never by the method |
| Neg-t-SNE | negative-sampling neighbor embedding with a Cauchy kernel (Damrich et al., 2023) |
| background negatives, ξ_B | extra repulsion between target points whose nearest background points overlap |
| residual graph | the kNN graph built after removing each target point's local background mean and directions |
| BC-NE, BC-NE+BN | the method without and with background negatives (the reports' internal labels: `(r_a)`, `(r_c)`) |
| cPCA, α | contrastive PCA and its contrast parameter; the label-free α rule reproduces the `contrastive` package |
| local cPCA | k-means on the background, then cPCA inside each cluster |
| WC | within-cluster signal accuracy: kNN accuracy of the signal label, computed inside each background cluster |
| run labels in reports | `(a)` plain Neg-t-SNE · `(b)` uniform extra negatives (control) · `(c)` background negatives on the raw graph · `(d)` t-SNE-end spectrum reference · `(o)` same-cluster oracle negatives (control) · `(e2)`, `(e3)` cPCA → Neg-t-SNE / UMAP · `T1`–`T3` t-SNE, PCA → t-SNE, cPCA → t-SNE |
| easy / medium / hard | additive synthetic benchmarks: signal and background orthogonal / overlapping / background nonlinear |
| Designs J, K, L | later synthetic benchmarks: an interacting signal, cluster-specific signals, and signals placed where no global projection works (L-cyclic, L-between) |

## Third-party code and references

- `third_party/contrastive-ne/` is a vendored copy of [berenslab/contrastive-ne](https://github.com/berenslab/contrastive-ne) at v0.4.0 (`d5f0b53`), MIT-licensed (Damrich, Böhm, Hamprecht, Kobak, *From t-SNE to UMAP with contrastive learning*, ICLR 2023). It has one local change, an 11-line hook, documented in `third_party/contrastive-ne/UPSTREAM.md`.
- The `contrastive` package (cPCA reference, v1.2.0) is installed, not vendored, because it declares no license.
- Key papers:
  - Abid et al. 2018 (cPCA);
  - Damrich et al. 2023 (Neg-t-SNE and contrastive neighbor embeddings);
  - Böhm, Berens & Kobak 2022 (the attraction–repulsion spectrum).

  The full bibliography is in `docs/supervisor_report/refs.bib`.
