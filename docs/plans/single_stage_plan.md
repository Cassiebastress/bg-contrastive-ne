# A. Single-stage vs two-stage (r_a), MNIST-on-grass 0/1 and 0–9 (descriptive)

Your spec is kept as written. What I found in the code, and the decisions that follow from it, are marked **[code]**.

## A0. Goal and scope

- **Unchanged:** (r_c) stays a reference run in all future experiments and keeps its two-stage schedule. It is not part
  of this check.
- **Two-stage (r_a)** is Stage 1 for 250 epochs, then Stage 2 for 500 epochs from Stage 1's embedding, with a fresh
  learning-rate schedule and no background term.
- **r_a-single** is one Neg-t-SNE run on the same residual graph (k_loc = 30, r = 2): package defaults (neg_spec = 1.0,
  early exaggeration on), 750 epochs, one learning-rate schedule.
- **Datasets:** 0/1 and 0–9, seeds 0–5. Seeds 3–5 are primary; seeds 0–2 go in secondary tables.
- **Claim type:** descriptive only.
- **Where it runs:** on the cluster, CPU, 1 thread, in the environment of the stored runs.
- **The existing 0/1 and 0–9 reports are not changed.**

## A1. Before running

1. **Package behavior, already checked in the code.** Both answers will also go into the result file.
   - **(a) Early exaggeration is a no-op at neg_spec = 1.0** **[code]**. In
     `third_party/contrastive-ne/src/cne/cne.py:462`, the exaggerated parameter is used only when
     `self.early_exaggeration and self.s is not None`. With neg_spec = 1.0 and `s` unset, the objective is unchanged.
     So stop condition A5.3 doesn't fire. The run will also assert that `params["s"] is None`.
   - **(b) The learning rate** **[code]** is `new_lr` (`cne.py:702`), linear annealing from `learning_rate` toward
     `lr_min_factor` × that rate over the fit's own `n_epochs`.
     - Two-stage anneals to its minimum at epoch 250, then restarts.
     - Single-stage anneals once, over 750 epochs.
     - `fit_neg` already records the per-epoch rates (`lrs`). Both schedules will be saved.
2. **Check that the stored two-stage runs can be reused.**
   - **Which runs:**
     - **0/1:** `results/bsgraph/runs/*_r_a.json`, cluster job 6036 (confirmation), code `1511f6c`, plus the selection
       job.
     - **0–9:** `results/mnist_grass_0to9/runs/*_r_a.json`, jobs 6086 and 6087.
   - **What is checked:**
     - the dataset checksums match the manifest;
     - `git diff --quiet` on the (r_a) code path since each run's commit;
     - the stored `.npy` exists on the cluster, and its SHA-256 matches the JSON. The `.npy` files are gitignored, and
       they are needed for kNN overlap and Procrustes.
   - **If any check fails:** rerun two-stage (r_a) on seed 3 of that dataset and require bit-identity first.
   - **Change made during implementation:** the seed-3 rerun now runs for both datasets regardless, in job 1, and must
     be bit-identical. The reason: `data/mnist_grass.py` changed after the 0/1 runs (`sample_digits`, `1aa2c55`), so
     the 0/1 `git diff` check fails by construction. The 0–9 rerun costs one extra run and confirms that the new
     script's two-stage path is exactly `scripts/bsgraph_run.embed("r_a")`. The stored `.npy` SHA-256 and digit
     accuracy must also be reproduced.
3. **Optimization-noise reference** (done before any single-stage run). On seed 3 of each dataset, rerun two-stage (r_a)
   with training seeds 1003 and 2003, on the same dataset and graph.
   - **[code]** The new script passes the training seed to `fit_neg(..., seed=train_seed)` for both stages. The dataset
     seed still drives `load`, so shared code is untouched.
     - `pca_init` (full SVD) is deterministic.
     - The residual graph uses exact kNN.
     - So only the optimizer's randomness changes.
   - **noise_acc / noise_grass:** the range over the stored run plus the 2 new ones.
4. **Commit `results/single_stage/criterion.md` before any single-stage run.** It holds:
   - the "matches closely" definition, with the numeric tolerances;
   - the adoption rule: adopt single-stage (r_a) only if it matches closely on both datasets, otherwise keep two-stage.
     (r_c) stays two-stage either way.

## A2. Runs

- **r_a-single:** for each dataset × seed 0–5, `fit_neg(pca_init(X_T), G, seed, batch_hook=BackgroundTerm("none"),
  n_epochs=750)`.
  - The same initialization and training seed as Stage 1 of the stored run.
  - The `BackgroundTerm("none")` hook only logs, exactly as in Stage 1.
- **Determinism:** rerun seed 3 of each dataset and require bit-identity.

## A3. Metrics and "matches closely"

Per dataset and seed, for both versions:
- digit kNN accuracy (k = 15, stratified 5-fold);
- recall vs grass and recall vs clean digits;
- ARI: k-means with k = 2 (0/1) or 10 (0–9), n_init = 10, random_state = dataset seed. This is `eval.metrics.kmeans_ari`.
- kNN overlap at k = 15 between the two embeddings;
- Procrustes disparity, from `scipy.spatial.procrustes`;
- wall-clock time.

**Matches closely** is judged per dataset, on every seed in 3–5. It holds when
|Δ digit acc| ≤ max(0.005, noise_acc) and |Δ grass recall| ≤ max(0.005, noise_grass).

## A4. Outputs

- `results/single_stage/noise_reference.md`
- `results/single_stage/tables.md`: 0/1 and 0–9 shown separately, seeds 3–5 as the primary table and seeds 0–2 as a
  secondary one.
- Seed-3 figures for each dataset, side by side:
  - **0–9:** colored by digit, in the paper style.
  - **0/1:** a validated two-color pair.
  - **Both:** a version colored by grass brightness, using the one-hue blue ramp.
- `REPORT_single_stage.md`, 1–2 pages:
  - the tolerances;
  - whether "matches closely" held, per dataset and seed;
  - the timing;
  - the adoption outcome;
  - Future work: is the warm-up needed for (r_c)?

## A5. Stop and report

1. The stored two-stage runs can't be verified or reproduced.
2. A determinism check fails.
3. Early exaggeration turns out to change the objective. **[code]** The code says it doesn't; the per-run assert
   enforces this.

**Out of scope:** any change to (r_c), k_loc, r, m_B or Z̄, to other datasets, or to the existing reports.

## A steps (each committed and pushed)

1. Save this plan to `docs/plans/single_stage_plan.md`. Then:
   - write `scripts/single_stage_run.py`, with modes `--noise`, `--single`, `--merge`, `--make-jobs`;
   - write the report and assemble scripts;
   - lint, and smoke-test locally.
2. Cluster job 1: the reuse check (A1.2) and the noise reference (A1.3), 4 runs. Then:
   - merge;
   - fill the tolerances into `criterion.md`;
   - **commit it before job 2.**
3. Cluster job 2: 12 r_a-single runs plus 2 determinism reruns. Then merge, generate the tables and figures, and write
   the report.
4. Record the adoption outcome, and update NOTES and the README phase row.

