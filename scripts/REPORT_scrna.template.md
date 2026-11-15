# Single-cell RNA-seq: AML patient 027 against a healthy donor

This phase tests BC-NE on real data that was not built to be additive: the cPCA paper's single-cell example (its
Figure 3, from Zheng et al. 2017).
- **Target:** 7898 bone marrow cells from leukemia (AML) patient 027, 3933 before a stem-cell transplant and 3965
  after. The label is pre vs post.
- **Background:** 1985 bone marrow cells from a healthy donor.
- **What the two share:** cell-type variation (T cells, B cells, monocytes, erythroid cells, …). What only the target
  has: the pre vs post difference.

**Design:**
- **Claim type:** pre-registered (`results/scrna/preregistration.md`, `82ca18e`, committed before any label pre-check
  or embedding).
- **BC-NE** is (r_a) only: the residual graph (k_loc = 30, r = 2) and one 750-epoch Neg-t-SNE fit. No background
  negatives.
- **Baselines:** plain Neg-t-SNE, t-SNE, PCA-50 → t-SNE, PCA-50 → UMAP, cPCA → t-SNE, cPCA-2D and PCA-2D.
- **Seeds:** 0–9. The data are fixed, so the seeds vary only the algorithms' randomness, and every seed counts.
- **Preprocessing** (one transform for target and background): counts per cell normalized and log-transformed, the 500
  most dispersed genes, each gene standardized.
- **Platform:** the cluster only.

## Summary

1. **On the primary metric, BC-NE is the most accurate map, but by small margins.**
   - **Accuracy:** label kNN accuracy is 0.944 for BC-NE and 0.924–0.943 for every baseline except cPCA-2D,
     which is lower, at 0.844.
   - **Pre-registered verdicts:** BC-NE beats plain Neg-t-SNE, PCA-50 → t-SNE, PCA-50 → UMAP, cPCA → t-SNE, cPCA-2D
     and PCA-2D, consistently across 10 paired seeds. Against t-SNE it is not consistent: the mean difference is
     +0.001.
   - **The margins:** apart from cPCA-2D (+0.100), they are +0.004 to +0.020.
   - **Why the margins are small:** the pre-registered ceiling rule (both T1 and r_a at least 0.95) narrowly does not
     apply. But the high-dimensional data already reach 0.951, and every 2-D map comes close, so accuracy barely
     separates the methods here.
2. **The clear difference is in how the two samples are laid out** (ARI, descriptive).
   - **BC-NE:** splitting it into two k-means clusters recovers pre vs post with ARI 0.758 (0.755–0.773 over seeds).
   - **Every other map:** 0.525 at best (cPCA → t-SNE, which varies from 0.005 to 0.657 across seeds), and 0.36–0.39
     for t-SNE, PCA → t-SNE, UMAP and plain Neg-t-SNE.
   - **In the figure:** BC-NE puts the pre and post cells in two broad halves of the map. The other neighbor
     embeddings scatter both samples over many cell-type clusters, which a two-way split does not follow.
   - **The residual graph is what changes this:** plain Neg-t-SNE on the raw graph reaches only 0.355.
3. **BC-NE keeps less of the cell-type structure the background shares** (descriptive).
   - **The measure:** the overlap of embedding neighbors' nearest background cells.
   - **Values:** BC-NE 0.457, against 0.55–0.60 for t-SNE, PCA → t-SNE, UMAP and plain Neg-t-SNE, and 0.29–0.35 for
     the cPCA maps.
   - **For reference:** the 500-gene kNN graph scores 0.632, and random pairs 0.181.
   - **What it means:** BC-NE removes part of the background-shared organization. cPCA removes more but loses
     accuracy at its label-free α.
4. **cPCA's label-free α is a poor choice here.**
   - **The rule picks α = 307 on every seed:** cPCA → t-SNE then reaches 0.924 and cPCA-2D 0.844.
   - **Choosing α with the labels** (descriptive; an upper bound, as in the tuned-T3 phase): cPCA → t-SNE reaches
     0.955 at α = 5.54, a little above BC-NE (0.944), and cPCA-2D 0.933 at α = 1.7.
   - **As on MNIST:** a well-chosen α makes global cPCA competitive. The label-free rule does not find it.
5. **The paper's figure replicates.**
   - **The α values:** with the paper's own preprocessing, our cPCA finds exactly the paper's four representative α
     values (0, 3.46, 28.9, 389).
   - **Accuracy:** at α = 3.46, the value the paper shows, cPCA-2D separates pre from post with accuracy 0.977.
   - **But:** the label-free rule picks α = 389 there (0.823), and plain PCA-2D already reaches 0.905.
6. **The local background directions matter here, unlike on MNIST** (descriptive sensitivity).
   - **With mean subtraction only (r = 0),** ARI is 0.375–0.577. With r = 2 or 5 it is 0.734–0.761.
   - **Accuracy varies little:** 0.931–0.946.
   - **The default (k_loc = 30, r = 2)** is within 0.003 of the best cell on both.
7. **A larger background changes little for BC-NE.** Adding donor 2 (4457 background cells) gives 0.942 against
   0.944. It helps cPCA-2D (0.879 against 0.844).
8. **All checks pass.** Every seed-0 rerun was bit-identical, the main job reproduced the smoke test's embeddings
   exactly, and no map fragmented.

## What this does and does not show

- **Batch:** pre and post are separate 10x runs, and post cells have lower sequencing depth (median log10 total counts
  3.48 vs 3.69). In the maps, lower-depth regions largely coincide with the post cells (second embedding figure). So some of what any method
  separates may be technical, as in the paper's own figure.
- **Biology:** there is one patient and one donor. The seeds measure only the algorithms' randomness, so "consistent
  across 10 paired seeds" means stable on these cells, not general.
- **Additivity:** we cannot show the background is non-additive. This is real data with a structured background (many
  cell types), and the residual graph's local directions matter here (item 6), unlike on MNIST-on-grass.
- **The comparison that matters:** on accuracy, BC-NE is level with the best label-free baselines and below
  label-tuned cPCA → t-SNE. Its advantage is the global two-group layout (ARI), which no baseline matches. ARI was a
  secondary metric with no pre-registered verdict.

## Pre-checks (generated, before any embedding)

{{prechecks}}

## Results (generated)

{{tables}}

## Figures

**Seed 0, colored by sample** (the two linear projections are drawn on the central 99% of cells):

![Embeddings by sample](results/scrna/embeddings_seed0_label.png)

**Seed 0, colored by sequencing depth:**

![Embeddings by depth](results/scrna/embeddings_seed0_depth.png)

**Every seed: label kNN accuracy and ARI** (dots per seed, bar at the mean):

![Per seed](results/scrna/per_seed.png)

**cPCA against α** (mean over seeds; the label-free α is marked; BC-NE's mean for reference):

![Accuracy against alpha](results/scrna/alpha_curve.png)

**BC-NE sensitivity** (k_loc × r):

![Sensitivity](results/scrna/sensitivity.png)

**Replication of the paper's Figure 3a** (the paper's preprocessing; cPCA-2D at its α values):

![Replication](results/scrna/replication.png)

## Reproducibility

On the cluster (see `cluster/README.md`):

    python data/scrna.py                                   # download, verify, preprocess, write the manifest
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/scrna_precheck.txt
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/scrna_smoke.txt
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/scrna_main.txt
    python scripts/scrna_run.py --merge
    python scripts/scrna_report.py
    python scripts/scrna_assemble.py

- **Commits:**
  - data module and metric: `2727984`, `be255d1`;
  - manifest: `ce1cc59`;
  - pre-registration and run script: `82ca18e`;
  - results: `3ddc78c`;
  - figures redrawn with the linear projections zoomed: `d4f24d1`, `f74c28a`.
- **Jobs:**
  - 6282: pre-check, 35 s;
  - 6283: smoke test, 9 processes, 1 min 30 s;
  - 6284: main, 186 processes, 37 min.
- **Data:** `data/MANIFEST_scrna.md` pins the four 10x tarballs and the processed file. A rebuild was byte-identical.
- **Recorded in every run JSON:** the environment, the Slurm job, 1 thread and a clean commit. Embeddings (`.npy`)
  stay on the cluster.
