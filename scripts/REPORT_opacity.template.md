# Digit opacity on MNIST-on-grass 0/1

At the standard recipe (target = 0.25 × digit + grass), both BC-NE and plain t-SNE separate 0 from 1. This phase makes
the digits fainter, X_T(w) = w · digit + grass for w from 0.25 down to 0.01, to find **a range of opacities where
BC-NE still separates the digits and t-SNE does not**.
- **Claim type:** pre-registered (`results/mnist_grass_opacity/preregistration.md`, `91fdb42`), with w\* chosen on
  seeds 0–2 and three checks at w\* on seeds 3–5. Everything else is descriptive.
- **BC-NE** is (r_a) in its adopted single-stage form: the residual graph (k_loc = 30, r = 2), one 750-epoch
  Neg-t-SNE run. **t-SNE** is openTSNE with library defaults (T1).
- **Baselines, new from this phase:** cPCA-2D and local cPCA, alongside cPCA → t-SNE (T3) and plain Neg-t-SNE (a).
- **Data:** the six stored 0/1 datasets. Only the opacity changes; the digits, grass patches and background are the
  same at every w. Every run checked that w = 0.25 reproduces the stored target bit for bit.
- **Platform:** every run is on the cluster. Nothing is compared with earlier phases' numbers.

## Summary

1. **By the pre-registered levels, the window is a single grid point, and the result is mixed.**
   - **Selection (seeds 0–2):** the only w with mean BC-NE ≥ 0.80 and mean t-SNE ≤ 0.60 is 0.10, so w\* = 0.10.
   - **Confirmation at w\* (seeds 3–5):**
     - C1 **holds**: BC-NE 0.827, 0.842, 0.841;
     - C2 **fails narrowly**: t-SNE 0.596, 0.611, 0.614, so two seeds are just above 0.60;
     - C3 **holds**: BC-NE beats t-SNE by +0.232, +0.230, +0.227.
   - **So at w = 0.10,** BC-NE separates the digits at the 0.80 level on every seed and leads t-SNE by 0.23, but
     t-SNE is not at or below 0.60 on every seed.
   - **The levels leave little room.** Over all six seeds, mean t-SNE falls to 0.60 at w ≈ 0.101 and mean BC-NE falls
     below 0.80 at w ≈ 0.089, by linear interpolation.
2. **By the project owner's visual judgment, the range is wider: w = 0.15, 0.10 and 0.08** (seed 3).
   - **Judgment:** t-SNE separates the digits at w = 0.25 and 0.20 only. BC-NE separates them down to 0.08. Below
     that, neither does: "there is no clear clustering by digit".
   - **Where it differs from the levels:**
     - t-SNE at w = 0.15: accuracy 0.704, above 0.60, but no visible clustering by digit;
     - BC-NE at w = 0.08: accuracy 0.749, below 0.80, but visibly separated.
   - It was recorded verbatim and committed (`14594f4`) before this report. It is one viewer on one seed and does not
     change C1–C3.
3. **BC-NE's lead over t-SNE is largest at w = 0.08–0.15** (all six seeds, mean): +0.196 at 0.15, +0.237 at 0.10,
   +0.210 at 0.08, against +0.086 at 0.25 and +0.136 at 0.20.
   - At w ≤ 0.05, every method is near chance (0.5); the best mean is BC-NE's 0.544 at 0.05.
4. **cPCA at its label-free α matches BC-NE at visible opacities and falls behind at faint ones** (all six seeds,
   mean):

   | w | BC-NE | cPCA → t-SNE (T3) | cPCA-2D | t-SNE |
   |---|---|---|---|---|
   | 0.25 | 0.972 | 0.961 | 0.953 | 0.886 |
   | 0.15 | 0.919 | 0.901 | 0.897 | 0.723 |
   | 0.10 | 0.835 | 0.748 | 0.751 | 0.598 |
   | 0.08 | 0.768 | 0.624 | 0.598 | 0.558 |

   - α was not tuned here. The tuned-T3 phase showed the label-free α can miss the best range, so a tuned cPCA might
     close this gap.
5. **Local cPCA sits between t-SNE and global cPCA from w = 0.20 to 0.06** (0.621 at w = 0.10).
   - It found two background clusters (K = 2) at every seed and w, and was applicable everywhere.
   - Its embedding keeps the two clusters as separate islands, each with both digits mixed in.
6. **Plain Neg-t-SNE (a) is below t-SNE at every w from 0.20 to 0.04** (below that, both are at chance), so BC-NE's advantage comes from the residual graph, not
   from Neg-t-SNE itself.
7. **Grass is removed at every w** by BC-NE (recall vs grass 0.004–0.011) and cPCA (0.003–0.006), against
   0.134–0.186 for t-SNE.
8. **ARI needs care here.** For t-SNE, plain Neg-t-SNE and local cPCA it is about 0 at every w, even where their kNN
   accuracy is 0.85–0.97: k-means' two-way split follows the layout's large-scale shape (t-SNE's stripes, local cPCA's
   two background islands), not the digits. BC-NE's ARI falls from 0.874 at w = 0.25 to 0.442 at 0.10, where its two
   digit groups touch.
9. **Checks pass.** Every rerun was bit-identical (seed 0 at w = 0.25, and seed 3 at w\*), and every run reproduced
   the stored w = 0.25 target.

## Results (generated)

{{tables}}

## Figures

**Accuracy against opacity** (all six seeds, mean with min–max band; chance at 0.5; the window shaded, w\* marked):

![Accuracy vs opacity](results/mnist_grass_opacity/accuracy_vs_opacity.png)

**How faint the digits are:** one 0 and one 1 (seed 3) at several w.

![Digits by opacity](results/mnist_grass_opacity/digits_by_opacity_seed3.png)

**t-SNE vs BC-NE at every w** (seed 3), each without labels and colored by digit. The project owner's visual judgment
was made from this figure.

![t-SNE vs BC-NE by opacity](results/mnist_grass_opacity/side_by_side_seed3.png)

**Every run at w\* = 0.10** (seed 3):

![Embeddings at w = 0.10](results/mnist_grass_opacity/embeddings_w010_seed3.png)

The per-run inspection figures are `results/mnist_grass_opacity/inspection_{T1,r_a}_seed3.png`.

## Limitations

- **The window rests on one grid point.** Between w = 0.10 and 0.15 there is no grid value, and the pre-registered
  levels (0.80 and 0.60) leave a range only about 0.012 wide in w. C2 missed by at most 0.014.
- **The levels are a choice.** 0.80 and 0.60 were picked before the runs as reasonable values, not derived. The
  visual judgment suggests 0.60 is strict for t-SNE: at 0.704 the project owner saw no clustering by digit.
- **The visual judgment is one viewer on one seed,** made with the colored panels in view, so it is not blind to the
  labels.
- **cPCA used its label-free α.** A tuned α was not tried at low opacity.
- **One dataset, two digits, additive background.** As in every MNIST-on-grass phase, target = digit + grass, the
  case where global cPCA is exact in the population.

## Reproducibility

On the cluster (see `cluster/README.md`):

    python scripts/opacity_run.py --make-jobs selection
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_opacity_selection.txt
    python scripts/opacity_run.py --merge selection       # writes selection.json; commit it before confirmation
    python scripts/opacity_run.py --make-jobs confirmation
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/mnist_grass_opacity_confirmation.txt
    python scripts/opacity_run.py --merge confirmation
    python scripts/opacity_report.py
    python scripts/opacity_assemble.py

- **Commits:**
  - pre-registration, run script and jobs: `91fdb42`;
  - selection results and `selection.json`: `0e957e2`, before any confirmation run;
  - confirmation results, tables and figures: `9b49103`;
  - side-by-side figure: `2658ddd`, `7854120`;
  - the project owner's visual judgment: `14594f4`, before this report.
- **Jobs:** 6264 (smoke test, 5 processes, 1 min 23 s), 6265 (selection, 170 processes, 11 min 53 s) and 6268
  (confirmation, 170 processes, 11 min 20 s).
- **Recorded in every run JSON:** the environment, the Slurm job, 1 thread and a clean commit. Embeddings (`.npy`)
  stay on the cluster.
