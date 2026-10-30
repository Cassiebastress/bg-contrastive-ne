# MNIST-on-grass report: background-contrastive Neg-t-SNE on real images

This phase tests run (c), two-stage Neg-t-SNE with ξ_B background negatives, on real images: the MNIST-on-grass dataset from the contrastive PCA (cPCA) paper. Digits 0 and 1 are overlaid at 25% intensity on patches of ImageNet grass photos; the background dataset is grass patches alone. (c) is compared with three controls set by the supervisor and one reference run, all on the same data:

| label | run |
|---|---|
| (c) | our method: two-stage Neg-t-SNE + ξ_B background negatives, as in Phases 4–5b |
| (a) | reference: the same pipeline with the background term off. It separates the effect of the background term from that of Neg-t-SNE vs t-SNE. |
| T1 | t-SNE (openTSNE, library defaults) on the target images |
| T2 | PCA (50 components, fit on the target) → t-SNE |
| T3 | cPCA (label-free α, top 10 components) → t-SNE. cPCA is `baselines/cpca.py`, the project's reimplementation of contrastive 1.2.0's representative-α procedure, validated in Phase 5 prep. |

There were no new synthetic benchmarks, no other baselines and no new method variants. Everything ran on the Math Slurm cluster, and no comparison mixes cluster and Mac results.

## How to read this report

- **Shared definitions:** `REPORT.md` defines the two-stage Neg-t-SNE, ξ_B, the flags and the "consistent across 3 paired seeds" rule. `REPORT_phase5prep.md` defines the label-free α rule.
- **The spec:** the plan for this phase was given by the project owner in conversation. `results/mnist_grass/preregistration.md` records everything the results depend on, and was committed before any embedding was run.
- **Generated content:** tables marked **(generated)** are copied verbatim from `results/mnist_grass/` by `scripts/mnist_grass_assemble.py`.
- **Metrics:** all are computed on the 2-D embedding of the 5000 target images.
  - **Digit kNN accuracy** (primary): k = 15, stratified 5-fold.
  - **Recall vs clean digits:** kNN recall at k = 15 between the embedding and the clean digit images. Higher is better.
  - **Recall vs grass:** kNN recall at k = 15 between the embedding and each target image's grass patch. Lower is better: it measures how much grass structure the embedding keeps.
- **Seeds:** dataset seeds 0–2 are for selection, 3–5 for confirmation. Each seed is an independent dataset and also seeds every run on it.

## Summary

1. **(c) beats plain t-SNE (T1) and PCA → t-SNE (T2) on every confirmation seed.** Digit kNN accuracy:

   | run | mean digit kNN accuracy | (c)'s per-seed lead |
   |---|---|---|
   | (c) | 0.931 | — |
   | T1 | 0.884 | +0.040 to +0.054 |
   | T2 | 0.891 | +0.038 to +0.043 |

   (c) also keeps much less grass structure: its recall vs grass is 0.022, against 0.135 for T1 and 0.131 for T2.
2. **The gain comes from the background term, not from Neg-t-SNE.**
   - (c) beats (a), the same pipeline without the background term, by +0.065 to +0.088 on every seed (0.931 vs 0.858).
   - Without the background term, Neg-t-SNE is actually *below* t-SNE here (0.858 vs 0.884 and 0.891; descriptive, no verdict was pre-registered for it).
   - So the background term accounts for all of (c)'s advantage over the t-SNE controls, and more.
3. **cPCA → t-SNE (T3) beats (c) on every confirmation seed,** by 0.013 to 0.042 (mean 0.023): 0.955 vs 0.931.
   - T3 also removes grass structure almost entirely (recall vs grass 0.003).
   - MNIST-on-grass is additive by construction: target = digit + independent grass, with the same grass distribution in both datasets. That is exactly the case where cPCA's target-minus-background covariance points at the signal (Phase 5 prep). The synthetic phases predicted this outcome.
4. **The pre-checks matched the design** (descriptive):
   - The digit signal is in the raw-pixel kNN graph: digit kNN accuracy 0.90 on raw X_T, 0.93 on PCA-50, against 1.00 on the clean digits.
   - ξ_B pairs share their grass: patch distance 3.8–4.0, close to each target's grass nearest neighbors (3.6–3.8) and half that of random pairs (7.9–8.1).
   - ξ_B pairs are close to random in digit: same-digit excess +0.055 to +0.074, below the old 0.10 gate.
5. **Reproducibility:**
   - The cluster was validated first (two runs bit-identical). Its difference from the Mac (at most 0.027 WC) is below every Phase 5b verdict margin.
   - openTSNE and the (c) seed-3 rerun were bit-identical.
   - No seed was excluded, and no run was flagged.

## Data (`data/mnist_grass.py`, `data/README_mnist_grass.md`)

The recipe reimplements the authors' notebook (`reference/cpca_mnist_notebook.py`, not run), with five documented deviations:
- sorted filenames;
- one RNG per dataset seed;
- a random sample from all 0/1 training digits;
- no `targetplus`;
- saved provenance.

Inputs and checksums are in `data/MANIFEST.md`.

### Dataset build (generated)

{{build}}

Every seed passed every build check: shapes and ranges, exact reconstruction, disjoint pools, and a bit-identical rebuild.

## Results

### Pre-checks, selection seeds 0–2 (descriptive; generated)

{{prechecks}}

### Main runs (generated)

{{tables}}

### Cluster validation (Phase B; generated)

L-cyclic confirmation seed 3 was rerun on the cluster with 4 threads, as on the Mac. The table compares it with the Mac's recorded values, next to the cluster's spread over optimization seeds 1000–1002 and the Mac's seed-to-seed spread.

{{validation}}

### Figures (confirmation seed 3)

Each run, colored by digit (left) and by the mean brightness of each image's grass patch (right):

![MNIST-on-grass embeddings](results/mnist_grass/embeddings_seed3.png)

- **(a), T1 and T2:** organized partly by grass brightness. The bright-grass end mixes the two digits.
- **(c):** splits the digits into two halves. Brightness still varies across the layout but no longer drives the neighborhoods (recall vs grass 0.024).
- **T3:** splits the digits along a single curve with no brightness structure.

The 2-D cPCA projection at each α representative, for comparison with the cPCA paper's figure:

![cPCA projections](results/mnist_grass/cpca_alphas_seed3.png)

The digits separate at the two intermediate representatives (α = 3.46 and 11.3, the label-free choice), but not at α = 0 (plain PCA) or at the largest α.

## Interpretation

- **On this dataset, the background term does what it was designed for** (interpretation). ξ_B's negatives are target points with similar grass. Repelling them pulls apart neighborhoods that formed because of grass, which lifts digit accuracy from 0.858 to 0.931 and cuts recall vs grass from 0.079 to 0.022.
- **Why cPCA still wins here.** The grass is added independently of the digit, and the target and background grass come from the same distribution. In population, the target-minus-background covariance is then the digit covariance, the identity behind every cPCA win in Phases 4–5. Phase 5b showed (c) winning only where that identity fails *globally* (L-cyclic). MNIST-on-grass is not such a case.
- **Recall vs clean digits is low for every run** (confirmation means 0.006–0.035, where chance is about 15/4999 ≈ 0.003). With only two digit classes, a 2-D layout cannot preserve the fine within-digit neighborhoods of 784-D clean digits. The metric is descriptive here and supports no conclusion.

## Limitations

- **One dataset with two digit classes.** It is additive by construction, so it favors cPCA.
- **The controls use t-SNE, our method uses Neg-t-SNE.** (a) isolates the background term, but (c) vs T1–T3 also changes the embedding algorithm. Without the background term, Neg-t-SNE was below t-SNE here.
- **m_B was selected on the same metric** used for the confirmation verdicts (on different seeds, as pre-registered). No other setting was tuned.
- **T3 is one label-free α per seed** (4.4–11.3 on the confirmation seeds). The comparison is with the label-free pipeline, not with the best α.

## Next steps

These are suggestions, not part of this phase's spec:
1. **A background-suppressed kNN graph,** marked out of scope in this phase's spec. Phase 5b suggested (c)'s limit is its raw-space graph, so a graph built after removing local background directions could close the gap to cPCA on additive data like this.
2. **A real dataset where the background is not additive,** for example one whose nuisance interacts with the signal. That is where the synthetic results predict (c) can beat cPCA.

## Reproducibility

On the cluster (see `cluster/README.md`), in this order:

    python scripts/mnist_grass_manifest.py                  # after unzipping the grass images and downloading mnist.npz
    sbatch cluster/mnist_grass_validate.sbatch              # Phase B
    sbatch cluster/mnist_grass_build.sbatch                 # build seeds 0-5, pre-checks, manifest
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_selection.txt
    python scripts/mnist_grass_run.py --merge selection
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_confirmation.txt
    python scripts/mnist_grass_run.py --merge confirmation
    python scripts/mnist_grass_report.py
    python scripts/mnist_grass_assemble.py

**Order of commits:**
- Data manifest: `dee01fa`.
- Builder and validation code: `d7505ec`.
- Phase B result: `1269dfa`.
- Build and pre-checks: `3da9bf7`.
- Pre-registration: `8f916ee`, before any embedding run.
- Selection: `3dd613a`, before any confirmation run.
- Confirmation job list: `5556465`.

**Environment:** every run used 1 thread on the cluster's AMD EPYC 9474F (CPU only, no GPU allocated). Every results JSON records the environment, the Slurm job and a clean commit.
