# MNIST-on-grass with digits 0–9

The earlier MNIST-on-grass phases used only digits 0 and 1. With two classes, accuracies sit close to the ceiling, and a map can score well without showing much structure. This phase repeats those experiments with **all ten digits, 500 of each per dataset**. Everything else is unchanged: the recipe (0.25 × digit + a grass patch), the sizes, the seeds, the controls, the metrics, the protocol and the cluster setup. The residual graph was **fixed** at k_loc = 30, r = 2, the label-free default from the 0/1 phase, and was not re-selected.

| label | run |
|---|---|
| (r_c) | residual graph + ξ_B background negatives (BC-NE+BN) |
| (r_a) | residual graph, background term off (BC-NE, the method's core) |
| (c) | raw-pixel graph + ξ_B background negatives (the original method) |
| (a) | raw-pixel graph, background term off (plain two-stage Neg-t-SNE) |
| T1 | t-SNE (openTSNE defaults) |
| T2 | PCA-50 → t-SNE |
| T3 | cPCA (label-free α, top 10 components) → t-SNE |

Everything ran on the Math Slurm cluster, CPU only, 1 thread per run. No comparison mixes cluster and Mac results.

## How to read this report

- **Definitions:** `REPORT_mnist_grass.md` defines the dataset recipe, (a), (c), T1–T3 and the metrics. `REPORT_bsgraph.md` defines the residual graph, (r_a) and (r_c).
- **Pre-registration:** `results/mnist_grass_0to9/preregistration.md` was committed (`a9329a6`) before any cluster job of this phase: before the dataset build, the pre-checks and every embedding.
- **Generated content:** tables marked **(generated)** are copied verbatim from `results/mnist_grass_0to9/` by `scripts/mnist_grass_0to9_assemble.py`.
- **Metrics:** all are computed on the 2-D embedding of the 5000 target images.
  - **Digit kNN accuracy** (primary): k = 15, stratified 5-fold, 10 classes. **Chance is 0.10** here, against 0.50 in the 0/1 runs.
  - **Recall vs clean digits** (higher is better) and **recall vs grass** (lower is better): kNN recall at k = 15.
  - **ARI** (secondary, pre-registered, no verdicts): k-means with k = 10 on the embedding, against the digit labels.
  - **Per-digit accuracy and confusion matrices** (descriptive): from the same classifier and folds.
- **Claim types:**
  - **pre-registered:** a comparison with a verdict under "consistent across 3 paired seeds";
  - **descriptive:** measured, no verdict;
  - **interpretation:** our explanation, not tested.

## Summary

1. **BC-NE+BN, (r_c), beats all three t-SNE controls on every confirmation seed** (pre-registered).

   | run | mean digit kNN accuracy | (r_c)'s per-seed lead |
   |---|---|---|
   | (r_c) | 0.530 | — |
   | T3 cPCA → t-SNE | 0.433 | +0.039, +0.137, +0.114 |
   | T2 PCA-50 → t-SNE | 0.255 | +0.268 to +0.287 |
   | T1 t-SNE | 0.253 | +0.268 to +0.293 |

   The verdict against T3 is new. With digits 0/1, (r_c) vs T3 was not consistent (0.973 vs 0.955). With ten digits, (r_c) is ahead on every seed.
2. **The method's core, (r_a), also beats cPCA → t-SNE on every seed** (pre-registered): +0.035, +0.106, +0.100 (0.514 vs 0.433).
3. **The residual graph is again what makes the difference** (pre-registered). (r_c) beats (c) by +0.289 to +0.316 (0.530 vs 0.226).
   - Both of the raw-graph runs fall below t-SNE here (descriptive): (c) by −0.020 to −0.038, and (a), plain Neg-t-SNE, sits at 0.115, near chance.
   - In the 0/1 runs, (c) beat T1 and T2. With ten digits, background negatives on the raw graph no longer rescue it.
4. **Background negatives now add a small, consistent gain** (pre-registered): (r_c) vs (r_a) is +0.004, +0.031 and +0.014 (mean +0.016). In the 0/1 runs this comparison was not consistent. The gain is small next to the residual graph's effect, and smaller than the seed-to-seed spread of T3.
5. **All the residual and cPCA maps remove the grass** (descriptive). Recall vs grass is 0.003–0.004 for (r_c), (r_a) and T3, against 0.14 for T1 and T2.
6. **The absolute level is modest** (descriptive).
   - At 0.53, (r_c) is far above chance (0.10). But the same classifier on the clean 784-D digits reaches 0.91, so much of the digit structure is still lost.
   - Digits 0, 1 and 6 are recovered best (0.69–0.87 for (r_c)); 5, 8 and 9 worst (0.29–0.35).
   - 4 and 9 are the largest confusion: on seed 3, 33% of the 9s are classified as 4.
7. **ARI ranks the runs the same way as the primary metric, but is low everywhere** (secondary, no verdicts).
   - (r_c) 0.244, (r_a) 0.225, T3 0.117; every other run ≤ 0.031.
   - With 0/1 (computed post hoc), (r_c) and (r_a) reach 0.87 and T3 0.80. k-means finds ten digit clusters only partly in any of these maps.
8. **Determinism:** the T1 seed-0 and (r_c) seed-3 reruns were bit-identical. No run was flagged. All build checks passed, including the 0/1 regression.

## Dataset build (generated)

{{build}}

## Pre-checks and graph quality (generated; descriptive)

{{prechecks}}

What the pre-checks show (descriptive):
- **Raw-pixel neighborhoods are mostly grass.** Mean graph digit accuracy is 0.36–0.38 on raw pixels, and about half of each image's graph neighbors (grass recall 0.48–0.50) are also among its 15 nearest images by grass patch.
- **The residual default gives the best graph:** mean graph digit accuracy 0.58–0.59, against 0.42 for PCA-50 and 0.46–0.49 for cPCA-10. Its grass recall is 0.05. cPCA-10 removes grass even more completely (0.005), but keeps less digit structure.
- **The default is not a lucky choice.** All 15 variants on the grid have means of 0.55–0.59. The lowest on both seed sets is k_loc = 100, r = 0: with many background neighbors, subtracting the mean alone isn't enough.
- **ξ_B pairs barely favor the same digit:** an excess of +0.008 to +0.010 over chance. They are close in grass (distance 3.8–3.9, against 7.9–8.0 for random pairs), as in the 0/1 data.

## Results (generated)

{{tables}}

### Figures (confirmation seed 3)

**By digit.** The colors follow Damrich et al. (2023), Fig. 1 (tab10 in digit order). Each scale bar gives a length in embedding units.

![MNIST-on-grass 0–9 embeddings colored by digit](results/mnist_grass_0to9/embeddings_seed3.png)

**The same embeddings, one panel per digit**: the digit in blue, every other image in gray. Some tab10 pairs (green/orange, gray/brown) are hard to tell apart for colorblind readers, and overlapping digits hide each other in a single scatter, so this view shows each digit on its own.

![MNIST-on-grass 0–9 embeddings, one panel per digit](results/mnist_grass_0to9/embeddings_by_digit_seed3.png)

- **(r_a) and (r_c):** 0, 1, 3, 6 and 7 each occupy their own region. 2, 5 and 8 spread across the middle, and 4 and 9 share one region.
- **T3:** each digit falls into several scattered patches.
- **(a), (c), T1 and T2:** every digit is spread over the whole map.

**By mean grass-patch brightness** (darker blue is brighter grass):

![MNIST-on-grass 0–9 embeddings by grass brightness](results/mnist_grass_0to9/embeddings_grass_seed3.png)

- **(a), (c), T1 and T2:** the layout runs from dark to bright grass, so grass brightness is the main axis of these maps.
- **(r_a), (r_c) and T3:** no brightness gradient.

Row-normalized kNN confusion matrices for (r_c), (r_a) and T3:

![MNIST-on-grass 0–9 confusion matrices](results/mnist_grass_0to9/confusion_seed3.png)

## Interpretation

- **With ten digits, the background dominates raw neighborhoods even more.** (interpretation) With 0/1, a 25%-intensity digit still determined most raw-pixel neighbors (raw graph digit accuracy 0.90). With ten digits of similar shapes, it mostly doesn't (0.37). Every run built on the raw graph inherits that, which is why (a) and (c) fall to or below t-SNE. The residual graph removes the grass before neighbors are chosen, so it keeps its advantage.
- **Why cPCA → t-SNE falls behind here, although the data are additive.** (interpretation, untested) In the population, cPCA's identity still holds. But T3 keeps only the top 10 cPCA components at one label-free α, and ten digit classes may need more directions than that. T3 also varies with α: seed 3, the only one with a different α (4.375), is its best seed (0.482 vs 0.408–0.411). This phase does not separate these causes.
- **Background negatives help a little once there is more to separate.** (interpretation) With the residual graph, ξ_B repels images with similar grass. With 0/1 there was nothing left for it to do. With ten digits the map is more crowded, and the extra repulsion gives a small, consistent gain.

## Limitations

- **Still an additive dataset.** Target = 0.25 × digit + grass, exactly as before. This is not the non-additive test the earlier reports call for.
- **Labels were used for selection.** m_B for (c) and (r_c) was chosen with labels on the selection seeds, as pre-registered.
  - For (r_c), m_B = 5 (0.5337) won over m_B = 2 (0.5336) by less than 0.0001; under the rule, only exact ties go to the smaller m_B.
  - The verdicts come from separate confirmation seeds.
- **Different embedding algorithms.** The comparisons with T1–T3 also change the algorithm (Neg-t-SNE vs t-SNE). (a) vs T1 shows that plain Neg-t-SNE is far below t-SNE here (−0.136 to −0.141, descriptive), so the residual graph's gain is larger than its margin over T1 suggests.
- **T3 is one label-free setting.** It uses a fixed 10 components, and cPCA with more components or another α was not tried.
- **ARI depends on k-means.** k-means looks for round clusters, and no map here has ten clean clusters, so the ARI values are low and secondary. The 0/1 ARI was computed post hoc.

## Next steps

These are suggestions, not part of this phase's spec:
1. **A real dataset with a non-additive background,** as proposed in the earlier reports. It is still the open test.
2. **Optional, descriptive: T3 with more cPCA components** (for example 20 or 50). This would show whether cPCA's gap here comes from keeping only 10 components.
3. **Revisit what ξ_B is for.** It is redundant with two classes and gives a small gain with ten. Does it help more as the number of classes, or the crowding of the map, grows?

## Reproducibility

On the cluster (see `cluster/README.md`), in this order:

    sbatch cluster/mnist_grass_0to9_build.sbatch
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_prechecks.txt
    python scripts/mnist_grass_0to9_prechecks.py --summarize
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_selection.txt
    python scripts/mnist_grass_0to9_run.py --merge selection
    python scripts/mnist_grass_0to9_run.py --make-jobs confirmation
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_confirmation.txt
    python scripts/mnist_grass_0to9_run.py --merge confirmation
    python scripts/mnist_grass_0to9_ari01.py
    python scripts/mnist_grass_0to9_report.py
    python scripts/mnist_grass_0to9_assemble.py

**Order of commits:**
- Builder, metrics and tests: `1aa2c55`.
- Pre-registration and run scripts: `a9329a6`.
- Datasets built: `a41619c`.
- Pre-checks: `065e966`.
- Selection: `9a3ec00`.
- Confirmation job list: `ba1f582`.
- Report scripts: `e16afa1`.
- Confirmation, ARI and tables: `f55ce92`.

**Jobs:** 6084 (build, 2 min), 6085 (pre-checks, 1 min 13 s), 6086 (selection, 11 min), 6087 (confirmation, 4 min 48 s). Every results JSON records the environment, the Slurm job, 1 thread and a clean commit.
