# Plan: Background-contrastive Neg-t-SNE prototype (contrastive-ne)

## 0. Context and goal

We have two datasets in the same feature space:
- **Target** X_T (n_T × D): contains background variation plus target-specific "signal".
- **Background** X_B (n_B × D): contains background variation only.

Goal: a 2D embedding of X_T that reveals the signal structure while suppressing structure shared with the background. This is the neighbor-embedding analogue of contrastive PCA (cPCA).

Method: start from Neg-t-SNE (negative sampling with the Cauchy kernel) in the `contrastive-ne` package (github.com/berenslab/contrastive-ne; Damrich et al., ICLR 2023). Standard Neg-t-SNE pulls together target kNN edges and pushes apart m uniformly sampled in-batch negatives. We add **m_B extra negatives per edge**, drawn from a background-informed distribution ξ_B over pairs of *target* points that look similar for background reasons. The theory: because NEG's classifier does not include the noise distribution, the learned similarity becomes ~ p_target / (m·ξ), so pairs the background explains get extra repulsion.

Per-edge loss for edge (i, j), with Cauchy kernel φ and the package's existing constant c (do NOT change c and do NOT put ξ_B into the classifier):

    L_ij = -log(φ_ij / (φ_ij + c))
           - Σ_{u=1..m}   log(1 - φ_iu / (φ_iu + c))        # existing uniform in-batch negatives
           - γ Σ_{b=1..m_B} log(1 - φ_ib / (φ_ib + c))      # NEW: background negatives, b ~ ξ_B(i, ·)

The background term is off for the early epochs and switched on afterwards.

Scope of this plan: **non-parametric mode only**, synthetic data only, CPU is fine. Baselines (cPCA→UMAP, contrastiveVI, real data) are explicitly out of scope; see Section 7.

### Open decisions (defaults chosen; the user may override)
1. Synthetic data design (Section 2): defaults given.
2. ξ_B definition: Jaccard overlap of background-kNN sets (Section 3). An alternative is listed but not implemented initially.
3. Preprocessing: identical scaling for X_T and X_B; no separate per-dataset centering for the ξ_B kNN step.
4. Sweep values: m_B ∈ {0, 1, 2, 5, 10, 20}, γ = 1.

## 1. Phase 0 — Understand the package (no modifications yet)

1. Create a fresh environment. Install contrastive-ne from GitHub at a pinned commit (record the hash). Do not edit the installed package; fork or vendor it into `third_party/contrastive-ne` so changes are tracked in git.
2. Read the code and write `NOTES.md` answering, with file/line references:
   - Where the kNN graph is built (k, symmetrization, weights).
   - How minibatches of directed edges are formed.
   - Where and how negatives are sampled (in-batch? from which candidates? with replacement? is the head excluded?).
   - The exact Neg-t-SNE loss code, and confirmation that the classifier fraction uses a constant (e.g. Z̄·m/C(n,2)) rather than the true noise probability.
   - How the learning rate is annealed, and how the early-exaggeration phase (fixed Z̄ = |X|/m for the first epochs, then annealing reset) is implemented.
   - Which argument selects Neg-t-SNE (expected: `loss_mode="neg"`) and how the spectrum parameter `s` / Z̄ is set.
3. Sanity run: plain Neg-t-SNE on a 10k MNIST subset with package defaults. Save the plot. Confirm it looks UMAP-like (compact, separated digit clusters).
4. **Stop and report** if any answer in step 2 contradicts the assumptions above, especially if the classifier uses the true ξ.

## 2. Phase 1 — Synthetic data generator (`data/synthetic.py`)

Function `make_dataset(level, n_T=5000, n_B=5000, D=50, seed)` returns X_T, X_B, and ground truth used **only for evaluation**: signal labels y_s, background labels y_b for the target points, and the clean signal component S_T.

Construction:
- Background latent: K_b = 6 Gaussian clusters in a d_b = 5 dimensional latent space. Map to D dims with map B. Large scale (σ_b, cluster separation scaled so background dominates distances).
- Signal latent: K_s = 4 Gaussian clusters in a d_s = 3 dimensional latent space. Map to D dims with map A. Smaller scale than the background.
- Target point: x = A(z_s) + B(z_b) + ε. Background point: x = B(z_b') + ε, using the same B and the same background latent distribution. ε is isotropic Gaussian noise.

Difficulty levels:
- `easy`: A and B are linear with orthogonal column spaces.
- `medium`: A and B are linear with overlapping (random, non-orthogonal) column spaces.
- `hard`: B is a fixed random 2-layer MLP with tanh (nonlinear background); A is linear.

Add a `signal_to_background` scale parameter.

**Validity check** (required before using a dataset): run plain Neg-t-SNE on X_T and compute kNN classification accuracy (k = 15, 5-fold CV) in the embedding for y_b and y_s. The dataset is a valid test only if background accuracy is high and signal accuracy is clearly lower, i.e. plain Neg-t-SNE organizes the embedding by background. If not, increase background dominance and recheck. Report the chosen scales.

## 3. Phase 2 — Build ξ_B (`xi_b.py`)

Default definition: **background-neighbor overlap**.
1. Apply identical preprocessing to X_T and X_B (same feature scaling, no separate centering).
2. Build a kNN index on X_B (exact or approximate; e.g. pynndescent or sklearn). For each target point i, find its k_B = 15 nearest background points, N_B(i).
3. Build an inverted index from each background point to the target points that have it in N_B. Candidate pairs (i, j) are target points sharing at least one background neighbor. This avoids computing all n_T² pairs.
4. For candidates, compute the Jaccard similarity s_B(i, j) = |N_B(i) ∩ N_B(j)| / |N_B(i) ∪ N_B(j)|. Exclude j = i.
5. For each i, keep the top L = 30 candidates with s_B > 0. Store as CSR arrays (indptr, indices, probs), where probs are s_B normalized per row. This is a per-head sampling distribution ξ_B(i, ·).
6. Allow target kNN edges (i, j) to appear in the list. A positive edge that the background explains is supposed to receive extra repulsion.

Diagnostics (save to `results/xi_b_diagnostics.json` and plots):
- Distribution of list sizes. Number of target points with an empty list (these get no background negatives; report the count).
- On synthetic data: for sampled ξ_B pairs, the fraction sharing a background label vs. sharing a signal label. Compare with uniform random pairs. ξ_B is informative only if same-background-label precision is much higher than chance and same-signal-label precision is near chance.
- Fraction of target kNN edges that appear in ξ_B lists.

Alternative (do not implement now; note only): linear ξ_B from a kernel on target-pair differences projected onto the top principal components of X_B.

## 4. Phase 3 — Modify the trainer

Implement in the forked package (or as a subclass), keeping the original code path intact and bit-for-bit reproducible when the new feature is disabled.

New arguments:
- `bg_mode`: `"none"` (default) | `"xi_b"` | `"uniform_extra"` (control: m_B extra negatives drawn uniformly from all target points instead of from ξ_B).
- `bg_neighbors`: CSR (indptr, indices, probs) from Phase 2.
- `m_B` (int), `gamma` (float, default 1.0).
- `bg_start_epoch` (int; default = the epoch where the early-exaggeration phase ends, so the switch-on coincides with the learning-rate reset. Confirm against NOTES.md).

Behavior:
- For each directed edge in a batch with head i, after the existing negative sampling, sample m_B tails from row i of `bg_neighbors` (with replacement, by `probs`). Skip if the row is empty. Background tails index the full embedding table, not just the batch; this is fine in non-parametric mode.
- Compute the repulsive term with the same φ and constant c as existing negatives. Multiply by γ and add to the loss.
- The term is active only when epoch ≥ bg_start_epoch.
- Log per epoch: loss components (attraction, uniform repulsion, background repulsion) and Σφ over a fixed random sample of pairs (embedding-scale diagnostic).

Tests (`tests/`):
- With `bg_mode="none"` or m_B = 0, embeddings match the unmodified package exactly for a fixed seed.
- The sampler's empirical frequencies match `probs` for a few rows (chi-square or tolerance check).
- The background term's gradient only pushes sampled pairs apart (sign check on a tiny example).
- Before bg_start_epoch, results match `bg_mode="none"`.

## 5. Phase 4 — Experiment

Datasets: `easy` first. Run `medium` and `hard` only after `easy` gives interpretable results.

Runs (3 seeds each, package defaults otherwise, same epochs/batch size/k for all):
- (a) plain Neg-t-SNE (`bg_mode="none"`)
- (b) control: `bg_mode="uniform_extra"`, m_B ∈ {1, 2, 5, 10, 20}
- (c) method: `bg_mode="xi_b"`, m_B ∈ {1, 2, 5, 10, 20}

Metrics (ground truth used for evaluation only; never passed to the method):
- kNN classification accuracy (k = 15, 5-fold CV) in the embedding for signal labels y_s (higher is better) and background labels y_b (lower is better).
- kNN recall (k = 15): overlap between embedding neighbors and neighbors in the clean signal S_T.
- Spearman correlation between embedding distances and S_T distances (5000 random points, all pairs).
- Embedding scale: Σφ on the fixed pair sample, to detect a global shift along the attraction–repulsion spectrum.

Outputs:
- `results/metrics.csv` (one row per run).
- Figures: an embedding grid (rows = runs, columns = colored by y_s / by y_b); metric-vs-m_B curves with seed error bars for (b) vs (c).
- `REPORT.md`: a short summary of what happened, including failures.

Success criterion: for some m_B, (c) improves signal kNN accuracy and signal kNN recall over both (a) and (b) at the same m_B, while background kNN accuracy drops, without the embedding fragmenting or collapsing. If (b) helps as much as (c), the effect is extra repulsion, not background information. Say so plainly in the report.

## 6. Stop-and-ask conditions

Stop and report to the user (do not tune around the problem silently) if:
- Plain Neg-t-SNE already recovers the signal on a dataset (the dataset is not a valid test).
- The ξ_B diagnostics show it does not preferentially pair same-background points.
- Training becomes unstable (NaNs, exploding scale) when the background term switches on.
- The package internals contradict Section 0's assumptions.

## 7. Secondary tasks (later, not part of this plan)

- Baselines: UMAP on raw X_T; cPCA (Abid et al. 2018) components → UMAP; plain Neg-t-SNE with Z̄ tuned to match the method's Σφ (scale-matched control); contrastiveVI if feasible.
- The linear ξ_B alternative.
- cPCA's MNIST-digits-on-grass dataset, then real datasets with known target-specific groups.
- Parametric mode (embedding background points with the same map).