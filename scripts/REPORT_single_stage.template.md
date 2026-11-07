# Single-stage vs two-stage (r_a)

(r_a) is the method's core form: the residual graph (k_loc = 30, r = 2) embedded with Neg-t-SNE, with no background negatives. Until now it ran on the **two-stage schedule** inherited from the original method:
- a 250-epoch first stage;
- then a 500-epoch second stage from that embedding, with a fresh learning-rate schedule.

The second stage exists to switch on background negatives, which (r_a) doesn't use. So for (r_a), the only thing that happens at the handoff is the learning-rate reset.

This check asks whether **one standard Neg-t-SNE run of 750 epochs** ("r_a-single") gives essentially the same result. It covers MNIST-on-grass digits 0/1 and 0–9, seeds 0–5. Confirmation seeds 3–5 are primary.
- **Claim type:** descriptive, with no pre-registered verdict.
- **Rule committed in advance:** before any single-stage run, a definition of "matches closely" and an adoption rule were committed (`d6f7be0`). The plan is `docs/plans/single_stage_plan.md`.
- **Not affected:** (r_c), BC-NE with background negatives, keeps its two-stage schedule and is not part of this check. The existing 0/1 and 0–9 reports are unchanged.

## Summary

1. **r_a-single matches two-stage (r_a) closely on both datasets, on every seed.**
   - All 12 seeds, including the secondary seeds 0–2, are within tolerance on both digit kNN accuracy and grass recall.
   - Mean Δ over seeds 3–5 (single − two-stage):

     | dataset | Δ digit kNN accuracy | Δ grass recall |
     |---|---|---|
     | 0/1 | −0.0001 | +0.0001 |
     | 0–9 | −0.0016 | +0.0001 |

   - The closest call is 0–9 seed 3: Δ = −0.0086, against a tolerance of 0.0092.
   - **Under the committed adoption rule, single-stage (r_a) becomes the method's core form for future experiments.**
2. **Secondary metrics agree too** (descriptive):
   - recall vs clean digits is within 0.0011 on every seed;
   - mean ARI changes by +0.0015 (0/1) and +0.012 (0–9).
3. **On some seeds the two runs settle into a different global arrangement** (descriptive).
   - **Scale of training-seed noise:** two-stage runs that differ only in training seed agree at kNN overlap 0.71–0.75, with Procrustes disparity ≤ 0.015.
   - **Most seeds:** single vs two-stage agrees about as well as that (overlap 0.73–0.77).
   - **Exceptions:**
     - **0–9:** seeds 0, 4 and 5, with overlap 0.23–0.40 and Procrustes disparity up to 0.38;
     - **0/1:** seed 1, with overlap 0.64 and disparity 0.22.
   - **What the figure shows** (0–9 seed 5): the same digit groups form in both runs but sit in different places. Every quality metric is within tolerance there.
   - So the schedule can change *where* groups end up, but not *how well* the digits are separated or the grass removed.
4. **Runtime is about the same.** Both versions train for 750 epochs. The differences in the table come from runs in different jobs (63–119 s two-stage, 64–89 s single-stage) and aren't a speed effect.

## Criterion and package behavior (generated, committed before any single-stage run)

{{criterion}}

The learning rate per epoch, as each run recorded it:

![Learning rate per epoch](results/single_stage/learning_rate.png)

## Results (generated)

{{tables}}

## Figures

Seed 3 is the planned figure seed. The second figure for each dataset shows the confirmation seed where the two layouts agree least, chosen by lowest kNN overlap rather than by eye. Colors:
- **0/1:** blue and orange;
- **0–9:** tab10, as in the 0–9 report;
- **bottom rows:** grass brightness.

**Digits 0–9, seed 3** (layouts agree):

![0–9, seed 3](results/single_stage/embeddings_0to9_seed3.png)

**Digits 0–9, seed 5** (least agreement; the same groups in a different arrangement):

![0–9, seed 5](results/single_stage/embeddings_0to9_seed5.png)

**Digits 0/1, seeds 3 and 5:**

![0/1, seed 3](results/single_stage/embeddings_01_seed3.png)

![0/1, seed 5](results/single_stage/embeddings_01_seed5.png)

## Limitations

- **Thin noise reference.** The tolerances come from three training seeds on one dataset seed. For 0/1 and for grass recall on both datasets, the 0.005 floor sets the tolerance, not the measured noise.
- **Close call on 0–9 seed 3.** Δ = −0.0086 against a tolerance of 0.0092. On the other two confirmation seeds the differences are −0.0024 and +0.0062. The mean difference over seeds 3–5 is −0.0016, and it is −0.0043 over seeds 0–2. So there is no sign of a systematic gap, but it can't be ruled out at this resolution.
- **Two datasets, both additive.** The check covers only MNIST-on-grass, where target = 0.25 × digit + grass.

## What changes

- **(r_a), BC-NE:** from now on it means the residual graph followed by one 750-epoch Neg-t-SNE run with package defaults (`fit_neg(pca_init(X_T), G, seed, n_epochs=750)`).
- **(r_c), BC-NE+BN:** keeps its two-stage schedule and stays a reference run in every future experiment.
- **Earlier reports:** their results stay as recorded, with two-stage (r_a).
- **The tuned-T3 experiment** (`docs/plans/t3_tuning_plan.md`): its comparison 2 becomes r_a-single vs T3\*. It reuses this check's 0–9 single-stage runs on seeds 3–5. Two-stage (r_a) vs T3\* is reported there descriptively.

## Future work (not run)

- **Is the warm-up needed for (r_c)?** Test one run with background negatives active from the first epoch against two-stage (r_c), using the same noise-based tolerance approach. This would test the warm-up's original rationale for the first time: letting a layout form before the background term acts.

## Reproducibility

On the cluster (see `cluster/README.md`):

    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/single_stage_1.txt
    python scripts/single_stage_run.py --check          # writes criterion.md; commit it before job 2
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/single_stage_2.txt
    python scripts/single_stage_run.py --merge
    python scripts/single_stage_report.py
    python scripts/single_stage_assemble.py

- **Commits:**
  - scripts and plan: `7200fc4`;
  - job 6100 results and the criterion: `d6f7be0`;
  - report additions after the results (training-seed layout agreement, least-agreeing-seed figure): `aafc8de` and `1072e8e`.
- **Jobs:** 6100 (reuse check and noise reference, 6 runs, 1 min 14 s) and 6101 (12 single-stage runs and 2 reruns, 1 min 44 s).
- **Recorded in every run JSON:** the environment, the Slurm job, 1 thread and a clean commit.
