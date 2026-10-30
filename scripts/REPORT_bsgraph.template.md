# Background-suppressed kNN graph report (MNIST-on-grass)

In run (c), the ξ_B background negatives change only the repulsion. The kNN graph, which supplies the attraction, is still built on raw pixels, where grass shapes many neighborhoods. This phase built the graph on **local background residuals**: each target image minus its 30 nearest background images' mean and their top 2 principal directions. It tested whether that graph is cleaner (Phase 1, no embeddings, with a pre-registered gate) and then whether embeddings built on it are better (Phase 2).

The data are the MNIST-on-grass datasets, seeds 0–5. Everything ran on the Math Slurm cluster, CPU only, 1 thread per run. There were no new datasets or baselines, and ξ_B and the initialization are unchanged (both on raw pixels).

## How to read this report

- **Definitions:** see `REPORT_mnist_grass.md` for the dataset, the runs (a), (c), T1, T2 and T3, and the metrics (digit kNN accuracy is primary; recall vs clean digits and recall vs grass patches are secondary).
- **New runs:**
  - **(r_a):** two-stage Neg-t-SNE on the residual graph, background term off.
  - **(r_c):** the same plus ξ_B negatives.

  Both use the same initialization (PCA of raw X_T) and the same raw-pixel ξ_B as (a) and (c), so only the graph differs.
- **Pre-registrations:**
  - `results/bsgraph/preregistration_phase1.md`, committed before any residual-graph metric (`3f0fc27`);
  - `results/bsgraph/preregistration_phase2.md`, committed after the gate and before any embedding (`42bd31c`).
- **Generated content:** tables marked **(generated)** are copied verbatim from `results/bsgraph/` by `scripts/bsgraph_assemble.py`.
- **Reused results:** (a), (c) at m_B = 5, T1, T2 and T3 are the stored MNIST-on-grass confirmation runs. They were reused because the code they depend on is unchanged since the commit they ran at (`5556465`, checked with `git diff`), and every run here rechecks the dataset hashes.

## Summary

1. **The Phase 1 gate passed.**
   - The selected residual graph is k_loc = 30, r = 2. It has graph digit accuracy 0.961–0.963 on the selection seeds, against 0.932–0.938 for the better of the raw and PCA-50 graphs. That is +0.025 to +0.029 on every seed, past the +0.02 bar.
   - The graph's neighborhoods also hold almost no grass: grass recall is 0.057, against 0.44 for the raw graph and 0.34 for PCA-50.
   - The confirmation seeds show the same (0.956–0.967, descriptive).
2. **The residual graph improves the method a lot: (r_c) beats (c) on every confirmation seed,** by +0.036 to +0.046 (0.973 vs 0.931). Recall vs grass falls from 0.022 to 0.003, and recall vs clean digits rises from 0.013 to 0.056.
3. **(r_c) vs cPCA → t-SNE (T3) is not consistent.**
   - (r_c) is ahead on seeds 3 and 5 (+0.032, +0.028) and behind on seed 4 (−0.006), mean +0.018.
   - (r_c) is very stable (0.972–0.974); T3 varies with its label-free α (0.941–0.979).
   - Both remove grass structure to the same degree (recall vs grass 0.003).
   - The MNIST-on-grass phase found T3 consistently ahead of (c). The residual graph closes that gap.
4. **(r_c) beats T1 and T2 on every seed,** by +0.08 to +0.10.
5. **Once the graph is background-suppressed, ξ_B adds nothing measurable.**
   - (r_c) vs (r_a): +0.0014, 0.0000 and +0.0020, so not consistent (seed 4 is an exact tie).
   - The smallest m_B on the grid (1) was selected, and accuracy fell slowly as m_B grew (0.973 at m_B = 1, 0.968 at 40).
   - The residual graph alone accounts for the gain: (r_a) beats (a) by +0.10 to +0.13 (descriptive).
6. **Determinism and reuse:** the (r_c) seed-3 rerun was bit-identical, and the reuse check passed. No run was flagged.

## Results

### Phase 1: graph quality (generated)

{{phase1}}

### Phase 2: embeddings (generated)

{{phase2}}

### Figure (confirmation seed 3)

Colored by digit (left) and by mean grass-patch brightness (right):

![Background-suppressed graph embeddings](results/bsgraph/embeddings_seed3.png)

- **(r_a) and (r_c):** the two digits form two separate islands, with no grass-brightness gradient.
- **(c), on the raw graph:** splits the digits within one blob that still runs from dark to bright grass.
- **T3:** lays the digits along one curve, also with no brightness structure.

## Interpretation

- **The limit was the graph, as the Phase 5b and MNIST-on-grass reports suggested** (interpretation).
  - On raw pixels, about 44% of each target's graph neighbors are also among its 15 nearest images by grass patch. The attraction therefore pulls same-grass images together, and ξ_B's repulsion can only partly undo that.
  - Removing each image's local background mean and top 2 local background directions *before* the graph is built removes that pull at its source.
  - Then no repulsion is needed: (r_a) ≈ (r_c).
- **This is local, not global, background suppression.** The residual uses each image's own 30 nearest background images, with no global linear assumption. On this additive dataset it matches cPCA → t-SNE (not consistently better or worse). Whether it keeps working where cPCA's global identity fails, as in Phase 5b's L-cyclic, is untested.
- **Why ξ_B becomes redundant here** (interpretation). ξ_B pairs are targets with similar grass. On the residual graph, same-grass targets are no longer neighbors, so there is little left for the repulsion to separate.

## Limitations

- **One dataset, and an additive one.** MNIST-on-grass favors any method that subtracts the background linearly. The synthetic designs where cPCA fails were not tested with the residual graph.
- **Labels were used for selection.** The graph variant and m_B were chosen with labels, on the selection seeds (graph digit accuracy, then embedding digit accuracy), as pre-registered. The verdicts come from separate confirmation seeds.
- **The grid's top variant was not chosen.** The best mean on the grid (k100_r5, 0.964) was within the tie margin of the selected k30_r2 (0.962), and the simpler variant was chosen by rule.
- **Different embedding algorithms.** The comparisons with T1–T3 also change the algorithm (Neg-t-SNE vs t-SNE). (r_a) vs (a) isolates the graph's effect.
- **Stored results.** The (a), (c) and T1–T3 rows come from the MNIST-on-grass phase. They are reused under the pre-registered code-and-hash check, not rerun.

## Next steps

These are suggestions, not part of this phase's spec:
1. **Test the residual graph on the synthetic designs where cPCA fails** (Phase 5b's L-cyclic and L-between), against (c) and local cPCA. That is the test of local vs global suppression.
2. **A non-adversarial real dataset with a non-additive background,** the step deferred from the MNIST-on-grass report.
3. **Decide what ξ_B is for.** Here it is redundant once the graph is background-suppressed. Test whether it helps when the residual graph is imperfect, for example with a mis-sized k_loc or r.

## Reproducibility

On the cluster (see `cluster/README.md`), in this order:

    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/bsgraph_phase1.txt
    python scripts/bsgraph_phase1.py --merge
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/bsgraph_selection.txt
    python scripts/bsgraph_run.py --merge selection
    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/bsgraph_confirmation.txt
    python scripts/bsgraph_run.py --merge confirmation
    python scripts/bsgraph_report.py
    python scripts/bsgraph_assemble.py

**Order of commits:**
- Residual module and tests: `476a4e6`.
- Phase 1 pre-registration: `3f0fc27`.
- Phase 1 code: `bc6520f`.
- Phase 1 results (gate passed): `e594d76`.
- Phase 2 pre-registration: `42bd31c`.
- Phase 2 code: `9405a42`.
- Selection: `b474581`.
- Confirmation job list: `1511f6c`.

**Jobs:** 6032 (Phase 1, 1 min 12 s), 6034 (selection, 9 min), 6036 (confirmation, 1 min 38 s). Every results JSON records the environment, the Slurm job and a clean commit.
