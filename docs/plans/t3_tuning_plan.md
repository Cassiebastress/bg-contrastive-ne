# Plan (saved for later): a tuned cPCA → t-SNE baseline (T3) on MNIST-on-grass 0–9

## Context

In the 0–9 phase, (r_c) beat T3 (cPCA → t-SNE) on every confirmation seed (0.530 vs 0.433). But T3 ran at one
setting: 10 components and the label-free α. Meanwhile (r_c) had its m_B tuned with labels on the selection seeds. You
don't want to conclude the method is best because the baseline was under-tuned.

This phase gives T3 a tuning budget at least as large as the method's, chosen by the **same label-based rule on the same
selection seeds**, and reruns the comparison against it. It also measures how good T3 could possibly be on the
confirmation seeds themselves, as a descriptive upper bound. Scope: the 0–9 data only (your choice).

## Design

- **Grid:** number of cPCA components k ∈ {2, 5, 10, 20, 30, 50, 100} × α ∈ the cPCA package's grid
  ({0} ∪ 40 log-spaced values from 0.1 to 1000, as in `baselines/cpca.ALPHAS`) ∪ {1}.
  - That is 7 × 42 = 294 settings, against 6 m_B values for (r_c).
  - α = 1 is added because it's the value at which the population identity makes cPCA exact.
  - α = 0 is plain PCA-k → t-SNE, so the grid also gives a tuned version of T2 for free.
  - The pipeline is otherwise T3's:
    - `cpca.fit` on raw pixels;
    - `cpca.project(f, α, k)`;
    - openTSNE with library defaults and random_state = the dataset seed, via `scripts/mnist_grass_run.tsne`.
- **Selection (seeds 0–2):** every grid setting on every seed. T3\* is the setting with the highest mean digit kNN
  accuracy; ties go to fewer components, then smaller α. This is the same rule that picked m_B. The choice is committed
  before any confirmation-seed run.
- **Confirmation (seeds 3–5):** the full grid again on seeds 3–5. T3\* is one point of it; the rest is descriptive.
- **Pre-registered comparisons** (seeds 3–5, digit kNN accuracy, "consistent across 3 paired seeds"; secondary metrics
  alongside, no verdicts):
  1. (r_c) vs T3\*
  2. (r_a) vs T3\*
- **Descriptive, no verdicts:**
  - **Oracle T3:** on each confirmation seed, the best grid setting *for that seed*, chosen with its own labels. This is
    optimistic for T3, since the setting is picked on the very data it's scored on; it is an upper bound, not a
    baseline. Shown against (r_c) per seed.
  - T3\* vs the label-free T3 from the 0–9 phase.
  - The best α = 0 setting (tuned PCA-k → t-SNE) vs T2.
  - Heatmaps of mean digit kNN accuracy over (k, α), separately for selection and confirmation seeds.
- **Metrics:**
  - digit kNN accuracy (primary);
  - recall vs clean digits;
  - recall vs grass;
  - ARI (k-means, k = 10);

  all exactly as in the 0–9 phase.
- **Reused rows:** the (r_c), (r_a) and label-free T3 rows are the stored 0–9 confirmation runs (job 6087, code
  `ba1f582`). They're reused, not rerun, if:
  - the code they depend on is unchanged since `ba1f582`, checked with `git diff --quiet`;
  - every new run's dataset hash matches the manifest.

## Reproducibility check (built in)

The label-free α of each seed (5.541, or 4.375 on seed 3) is on the grid. So the grid point (label-free α, k = 10) is
exactly the stored T3 run, computed by the same `cpca.project` and `tsne` calls. Its embedding SHA-256 must equal the
stored T3 run's on all 6 seeds, or the phase stops. This replaces a separate rerun.

## Files

**New:**
- `results/mnist_grass_0to9_t3grid/preregistration.md`: the grid, the selection rule, comparisons 1–2, the descriptive
  items, the reuse and SHA checks, and the stop conditions. Committed before any grid run.
- `scripts/t3grid_run.py`, laid out like `scripts/mnist_grass_0to9_run.py`. It reuses that script's `load` (with its
  dataset-hash check), `eval.metrics` and `scripts.mnist_grass_run.tsne`.
  - `--stage S --seed s --alpha-index i`: one process per (seed, α) runs all 7 values of k, about 4–5 minutes, and
    writes `runs/{stage}_seed{s}_a{i}.json` with per-k metrics and SHA-256s (embeddings `.npy`, gitignored).
  - `--make-jobs STAGE`: 126 commands.
  - `--merge selection`: `runs_selection.csv`, `selection.json` (T3\*) and the SHA check on seeds 0–2.
  - `--merge confirmation`: `runs_confirmation.csv`, the SHA check on seeds 3–5, and `reuse.json`.
- `scripts/t3grid_report.py`, which writes `tables.md`, `comparisons.json` and `heatmap_digit_acc.png`. The tables:
  - T3\* and its settings;
  - comparisons 1–2;
  - the oracle per seed against (r_c);
  - T3\* vs label-free T3;
  - tuned PCA vs T2;
  - the top 10 settings on selection seeds.

  The heatmap uses the one-hue blue ramp already validated, and marks T3\* and the label-free setting.
- `scripts/REPORT_t3grid.template.md` and `scripts/t3grid_assemble.py`, which produce `REPORT_mnist_grass_0to9_t3grid.md`.

**Updated after results:**
- `REPORT_mnist_grass_0to9.md`: a short note under Summary item 1 pointing to the tuned-baseline result, whichever way
  it goes.
- `NOTES.md`: a section.
- `README.md`: a phase row.

## Steps (each committed and pushed; all compute on the cluster, CPU only, 1 thread per process)

1. Run script, report script and pre-registration. Lint and a local smoke test of the merge logic, with openTSNE
   stubbed, since it isn't installed on the Mac. Commit.
2. Selection job: 126 processes, about 25–30 min. Merge, check that T3\* is committed and the SHA check passes, commit.
3. Confirmation job: 126 processes, about 25–30 min. Merge, run the reuse check, run the report on the cluster, commit.
4. Write and assemble the report, add the pointer in the 0–9 report, NOTES and README, commit.

**Stop and report:**
- the label-free grid point doesn't reproduce the stored T3 SHA;
- a dataset hash mismatch;
- the reuse check fails: then rerun (r_c) and (r_a) on seeds 3–5 before comparing;
- a non-finite embedding.

## Verification

- Both SHA checks pass on all 6 seeds.
- Every JSON records a clean commit, the Slurm job and 1 thread.
- The pre-registration commit comes before the selection results, and `selection.json` comes before the confirmation
  results.
- Every number in the report comes from generated files.
