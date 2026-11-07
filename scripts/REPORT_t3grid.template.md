# Tuned cPCA → t-SNE (T3) on MNIST-on-grass 0–9

In the 0–9 phase (`REPORT_mnist_grass_0to9.md`), BC-NE+BN (r_c) beat cPCA → t-SNE (T3) on every confirmation seed,
0.530 vs 0.433 digit kNN accuracy. But the two were not tuned alike:
- **T3** ran at one setting: 10 components and the α picked by the cPCA package's label-free rule.
- **(r_c)** had m_B chosen with labels on the selection seeds.

This phase gives T3 a larger tuning budget than (r_c) had: 7 component counts × 42 values of α, 294 settings. T3\* is
chosen by the **same label-based rule on the same selection seeds** as m_B, then the comparisons are repeated against it.
- **Claim types:**
  - two pre-registered comparisons on digit kNN accuracy, (r_c) vs T3\* and single-stage (r_a) vs T3\*;
  - everything else is descriptive.
- **Rules committed before the results:**
  - the pre-registration (`results/mnist_grass_0to9_t3grid/preregistration.md`, `2ddd9ca`) before any grid run;
  - `selection.json` (`dbd9de7`) before any confirmation-seed run.
- **What (r_a) means here:** single-stage (r_a), the form adopted in `REPORT_single_stage.md`. Two-stage (r_a) is
  reported descriptively.
- **Not changed:** the 0–9 report's results stand as recorded, with a pointer to this report.

## Summary

1. **Tuned, T3 beats both forms of the method on every confirmation seed** (pre-registered).

   | run, seeds 3–5 | digit kNN accuracy | per-seed difference from T3\* |
   |---|---|---|
   | T3\*: k = 50, α = 1.70 | 0.689 | — |
   | (r_c) BC-NE+BN | 0.530 | −0.173, −0.142, −0.162 |
   | single-stage (r_a) BC-NE | 0.512 | −0.186, −0.175, −0.170 |

   **This reverses the 0–9 phase's verdict against T3.** That verdict compared against an under-tuned baseline.
2. **The gap is in digit structure, not in grass removal** (secondary, no verdicts).
   - **Grass:** recall vs grass is 0.004 for T3\*, 0.003 for (r_c) and 0.004 for (r_a). All three remove it.
   - **Clean digits:** recall vs clean digits is 0.165 for T3\* against 0.052 for (r_c).
   - **ARI:** 0.298 against 0.244.
3. **The choice of T3\* is not fragile** (descriptive).
   - **One ridge:** accuracy peaks on a single ridge at α ≈ 1.3–2.2 for every number of components, on both seed sets
     (heatmap below).
   - **Holds on new seeds:** T3\* drops only 0.012 from selection to confirmation (0.701 → 0.689), and ranks 2nd of 294
     settings on the confirmation seeds.
   - **Close to the oracle:** the best setting per confirmation seed, chosen with that seed's own labels, averages
     0.694, only 0.005 above T3\*.
4. **α is what held T3 back, not the number of components** (descriptive).
   - **The label-free α is off the ridge:** 5.54, or 4.38 on seed 3.
   - **α alone closes the gap:** at k = 10, the best α on the selection seeds gives 0.595 on the confirmation seeds.
     That is already above (r_c), against 0.433 for the label-free T3.
   - **Without α, k barely matters:** tuned PCA → t-SNE (α = 0, best k = 100) gives 0.262, about the same as T2 at
     0.255.
5. **All checks passed.**
   - **Reproduction:** the grid point (label-free α, k = 10) reproduced the stored T3 embedding bit for bit on all 6
     seeds.
   - **Data:** the dataset hashes matched.
   - **Reused runs:** they passed the reuse check, and the (r_c) and two-stage (r_a) reruns were bit-identical. See the
     note under Limitations about the first reuse check.

## Results (generated)

{{tables}}

## Figure

Mean digit kNN accuracy over the grid. Rows are the number of cPCA components, and columns are the α grid values,
equally spaced. The square marks T3\* and the circle the label-free T3 setting of seeds 0–2, 4 and 5. One-hue blue
ramp, shared by both panels.

![Mean digit kNN accuracy over (k, α)](results/mnist_grass_0to9_t3grid/heatmap_digit_acc.png)

## Interpretation

- **This is what the additive identity predicts.** (interpretation)
  - The data are target = 0.25 × digit + grass, with background = grass from a separate pool. In the population,
    C_T − 1·C_B is then exactly the digit covariance (scaled by 1/16), so cPCA is exact.
  - The README's headline finding, "global cPCA wins whenever the background is simply added to the signal", now holds
    on the 0–9 data too, once α is chosen well.
  - The 0–9 report's guess that T3 lost because 10 components are too few is mostly wrong. At a good α, 10 components
    already beat (r_c).
- **Why the best α is above 1, not at 1.** (interpretation, untested)
  - At α = 1, accuracy is only 0.39–0.41 for k ≥ 20 (table by number of components).
  - The grass is unscaled while the digit is scaled by 0.25, so grass variance is far larger than digit variance. The
    two grass pools are finite samples, and at α = 1 their covariances need not cancel. The leftover grass variance can
    still outrank the digit directions.
  - A somewhat larger α pushes the grass directions below them. Too large an α also starts to suppress digit
    directions that share variance with grass, which would explain why accuracy falls off again above about 2.
- **What the residual graph does differently.** (interpretation) BC-NE removes the grass locally, from each image's
  background neighbors, and with no tuned parameter. That matters when no single projection removes the background. On
  this dataset one projection does, and a well-chosen global projection keeps more of the digit structure than local
  residuals.

## Limitations

- **One dataset, additive by construction,** which is the case where cPCA is exact in the population. The result says
  nothing about non-additive backgrounds, where the residual graph is meant to help.
- **T3\* needs labels.**
  - It was chosen with labels on the selection seeds, under the same rule that chose m_B.
  - Single-stage (r_a) uses no labels at all. Its residual-graph settings, k_loc = 30 and r = 2, are fixed label-free
    defaults.
  - Without labels, the package's α rule lands off the ridge (0.433).
- **The tuning budgets are unequal by design:** 294 settings for T3\*, 6 for (r_c), none for (r_a). The residual
  graph's k_loc and r were not tuned here.
- **Different embedding algorithms.** T3 uses openTSNE and BC-NE uses Neg-t-SNE. The 0–9 phase showed that plain
  Neg-t-SNE is well below t-SNE on these data.
- **The first reuse check failed because of a bug in the check.**
  - **The bug:** its code list included `scripts/single_stage_run.py` for every stored run. That file was added after
    the (r_c), (r_a), T2 and T3 runs, and only the single-stage runs depend on it.
  - **No dependency changed:** no file those runs depend on had changed.
  - **The fix:** it applies that file to the single-stage rows only (`eb64dc8`).
  - **The pre-registered remedy, run anyway:** (r_c) and two-stage (r_a) were rerun on seeds 3–5 (job 6110). All 6
    reruns were bit-identical to the stored runs.
  - The comparisons were computed only after that.

## What changes

- **The 0–9 report's claim that (r_c) and (r_a) beat cPCA → t-SNE** holds only against the label-free T3. Against a
  label-tuned T3 it reverses. The 0–9 report now points here from its Summary item 1.
- **Additive data:** the evidence that a well-tuned global linear projection is enough there is now stronger. MNIST-on-grass,
  0/1 and 0–9, is additive.

## Next steps (not run)

1. **A real dataset with a non-additive background.** This is still the open test, and now the main one.
2. **A label-free α rule that finds the ridge,** for example one based on how much background variance is left in the
   projected target. It would show whether T3\*'s advantage survives without labels.
3. **Optional: tune the residual graph's k_loc and r** with the same label-based rule. That would match the tuning
   budgets in the other direction.

## Reproducibility

On the cluster (see `cluster/README.md`):

    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_t3grid_selection.txt
    python scripts/t3grid_run.py --merge selection          # selection.json; commit it before the next job
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_t3grid_confirmation.txt
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_0to9_t3grid_rerun.txt
    python scripts/t3grid_run.py --merge confirmation
    python scripts/t3grid_report.py
    python scripts/t3grid_assemble.py

- **Commits:**
  - pre-registration and scripts: `2ddd9ca`;
  - selection results and T3\*: `dbd9de7`;
  - the reuse-check fix and the rerun job: `eb64dc8`;
  - confirmation results: `ea4e712`.
- **Jobs:**
  - 6108: selection, 126 processes, 22 min;
  - 6109: confirmation, 126 processes, 22 min;
  - 6110: reruns, 6 processes.
- **Every grid run's JSON records** the environment, the Slurm job, 1 thread and a clean commit. The embeddings
  (`.npy`) are on the cluster and gitignored.
