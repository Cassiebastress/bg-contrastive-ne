# NOTES: contrastive-ne internals (Phase 0)

The package is vendored at `third_party/contrastive-ne/`, from upstream v0.4.0 `d5f0b530f35be4d6247300a601fb5a860bc0a1c8`. All references below are to that unmodified copy. Paths are abbreviated as follows:
- `cne.py` = `third_party/contrastive-ne/src/cne/cne.py`
- `_cne.py` = `third_party/contrastive-ne/src/cne/_cne.py`

Call path for non-parametric Neg-t-SNE:
1. `CNE.fit` (`_cne.py:380`): build the graph, then the model.
2. `ContrastiveEmbedding.fit` (`cne.py:389`): the epoch loop.
3. `train` (`cne.py:8`): the batch loop.
4. `ContrastiveLoss.forward` (`cne.py:553`).

Defaults that matter for `CNE(loss_mode="neg")`:

| Setting | Default | Source |
|---|---|---|
| k | 15 | `_cne.py:314` |
| batch size | 2**15 | `cne.py:94` |
| m (negatives per edge) | 5 | `cne.py:95` |
| n_epochs | 200 | `cne.py:96` |
| optimizer | SGD, lr 1.0, momentum 0 | `_cne.py:519-521`, `cne.py:100` |
| LR schedule | linear annealing to 0 | `_cne.py:319`, `cne.py:214-215` |
| `neg_spec` | 1.0 | `cne.py:250-255` |
| eps | 1 | `cne.py:108` |
| clamping | [1e-4, 1] | `cne.py:277,286` |
| loss aggregation | sum | `cne.py:125` |
| gradient clipping | elementwise, at 4 | `cne.py:57-58` |

## 1. kNN graph

- **Where it's built:** `_cne.py:400-455`. On CPU with `graph=None`, annoy is used (`_cne.py:402-412`). The pykeops path (`_cne.py:431-451`) needs CUDA.
- **annoy:** a Euclidean index with 50 trees, built **without a seed** (`_cne.py:418-420`). Query k+1 neighbors per point and drop the first, assumed to be the point itself (`_cne.py:425-426`).
- **Symmetrization:** union. `adj[i, neighs] = 1; adj[neighs, i] = 1` (`_cne.py:427-428`). An edge (i,j) exists if j ∈ kNN(i) or i ∈ kNN(j).
- **Weights:** binary (1) and in practice **ignored**. The loader turns the matrix into COO `row`/`col` and discards `data` (`_cne.py:100-102`). Every *stored* entry becomes an edge, including any explicit zeros in a user-supplied graph.
- **User-supplied graph:** `graph=<scipy sparse>` is taken as-is via `graph.tocsr()` (`_cne.py:455`). Because annoy is unseeded, we never use the package's graph construction. Instead we build an **exact** kNN graph with the same structure (`common/knn.py`: k = 15, union-symmetrized, binary), once per dataset, and pass it to every fit (see §7 and Plan, Reproducibility rules).

## 2. Minibatches of directed edges

- **Loader:** `FastTensorDataLoader(neighbor_mat, shuffle=True, batch_size=cne.batch_size, seed=...)` (`_cne.py:537-542`). Each stored entry (i,j) of the symmetric matrix is one directed edge, so both (i,j) and (j,i) appear.
- **Epoch:** one pass over all directed edges in a fresh `torch.randperm` order (`_cne.py:128-134`). There are `ceil(nnz / batch_size)` batches; the last one is smaller (`drop_last=False`, `_cne.py:121-126`).
- **Batch layout:** in `train` (`cne.py:36-45`), `item` holds the head indices and `neigh` the tail indices. `images = cat([heads, tails])` has length 2b. In non-parametric mode `features = model(images)` is an `nn.Embedding` lookup, so rows `0..b-1` are heads and rows `b..2b-1` are their positive tails.
- **Duplicates:** a data point can appear in several batch slots, as the head or tail of several edges.

## 3. Negative sampling

`make_neighbor_indices` (`cne.py:739-774`), called from `ContrastiveLoss.forward` (`cne.py:575-581`):
- Row r of `neigh_inds` is `[b + r, n_1, ..., n_m]`. Column 0 is the positive tail (`cne.py:770-772`); `neigh_mask[:, 0] = False` marks it as attractive (`cne.py:585-586`).
- Negatives are `randint(0, 2b-1) + (r + 1 - 2b)` (`cne.py:753-754`). These are negative indices into the length-2b `features` tensor and resolve to slots `(r+1 ... r+2b-1) mod 2b`. So:
  - **in-batch:** candidates are the 2b slots of the batch (heads and tails of all edges in it), not the full dataset;
  - **uniform over slots**, not over data points. A data point is drawn in proportion to how many slots it occupies, roughly its degree;
  - **with replacement** (independent `randint` draws);
  - **the head's own slot r is excluded.** The head's *own positive tail* (slot b+r) is **not** excluded, and neither are other slots holding the same data point as the head. Such a slot gives d=0, so φ = 1/eps = 1.
- **m:** `min(negative_samples, 2b-2)` (`cne.py:573`). If m ≥ 2b−2, the full batch is used (`cne.py:756-767`).
- **Resampling frequency:** the docstring says "once every epoch" when `force_resample=None` (`cne.py:31`, `cne.py:165`). In fact `train` overwrites its local `force_resample` with `idx == 0` (True) on the first batch, and it is then not None for every later batch (`cne.py:48`). **Negatives are therefore resampled every batch.**
- **RNG:** the global torch RNG (`torch.randint`).

## 4. The Neg-t-SNE loss and the classifier constant

`ContrastiveLoss.forward`, `loss_mode == "neg"`, Euclidean metric (`cne.py:651-660`):

```
sq_dists = ||y_head - y_other||^2                       (cne.py:597, gather branch)
estimator = 1 / (1 + spec_param * (sq_dists + eps))     (cne.py:654)
loss = -[col 0] log(clamp(estimator)) - [cols 1..m] log(clamp(1 - estimator))   (cne.py:658-660)
loss = loss.sum()                                       (cne.py:684-685)
```

- With eps = 1, the Cauchy kernel is φ = 1/(1 + d²), and `estimator = φ / (φ + c)` with **c = `spec_param`**. That is exactly the plan's `φ/(φ+c)`.
- **The classifier uses a constant.** `spec_param` is a scalar tensor, the same for every pair (`cne.py:405`, `cne.py:459`). There is no per-pair noise probability, so the true noise distribution ξ is **not** in the classifier. ✓
- Clamping happens before the log: [1e-4, 1.0] on `estimator` and on `1 - estimator` (`cne.py:658-660`).
- With m = 5, 2-D output and b = 2**15, the gather branch (`cne.py:590-607`) is used, not the all-pairs branch (`cne.py:609-630`).
- **Relation to Z̄:** `neg_spec = m · Z̄ / n²` (`cne.py:328-330`, comment: "assume uniform noise distribution over n**2 many edges"). Plan §1 wrote Z̄·m/C(n,2); the code uses n², a factor of about 2. This doesn't matter, because we never change c.
- The default `neg_spec = 1.0` with m = 5 corresponds to Z̄ = n²/5 = `Z_umap` (`cne.py:323`), i.e. the s = 1, UMAP-like regime.
- **Gradient clipping:** elementwise to ±4 on all model gradients (`cne.py:57-58`), after `loss.backward()` on the summed loss.

## 5. Learning-rate annealing and early exaggeration

- **LR:** `CNE` sets `learning_rate = 1.0` for non-parametric runs (`_cne.py:519-521`) and passes `anneal_lr=True` (`_cne.py:319,526`), which maps to `"linear"` (`cne.py:214-215`). Each epoch, `new_lr` (`cne.py:692-736`) gives `lr = 1.0 · max(lr_min_factor=0, 1 − epoch/n_epochs)`, set on param group 0 (`cne.py:466-479`). It starts at 1.0, ends at 1/n_epochs, and has no warm-up (`warmup_epochs=0`).
- **(a) Early exaggeration is on by default but does nothing for neg defaults.**
  - `early_exaggeration=True` (`cne.py:129`). `spec_param_early` is computed with `s_early = max(s, 1)` if `s` is set, else 1 (`cne.py:441-443`).
  - It is applied only `if self.early_exaggeration and self.s is not None and epoch < n_epochs // 3` (`cne.py:453`).
  - With `loss_mode="neg"` defaults, `s` stays `None` (only `neg_spec = 1.0` is set, `cne.py:250-255`), so the early branch never runs.
  - Even with `s` set, `s ≥ 1` gives the same value. Early exaggeration only has an effect for `s < 1`, where the first third of training runs at s = 1.
- **(b) No LR reset at the phase boundary.** The LR schedule spans all `n_epochs` with no reset at `n_epochs // 3`. The only reset code (Adam only) is commented out (`cne.py:461-463`).
- **Mismatch with Plan §1:** "fixed Z̄ = |X|/m for the first epochs, then annealing reset" does not match the code. The early phase uses `Z_umap = n²/5` and nothing resets. This is why we use the explicit two-stage schedule (Plan §4).

## 6. Selecting Neg-t-SNE and setting the spectrum

- `loss_mode="neg"` (`cne.py:151`). `"neg_sample"` is a deprecated alias (`cne.py:240-241`).
- **Priority** when several are given: `s` > `Z_bar` > `neg_spec` > `noise_in_estimator`, with a warning (`cne.py:257-261`). Resolution happens in `process_spec_param` (`cne.py:309-352`), called at the start of `ContrastiveEmbedding.fit` (`cne.py:397`), with `n = len(init)` from `CNE.fit` (`_cne.py:544`):
  - `s` → `Z̄ = Z_tsne · (Z_umap / Z_tsne)^s`, where `Z_tsne = 100 n` and `Z_umap = n²/5` (`cne.py:321-326`).
  - `Z̄` → `neg_spec = m · Z̄ / n²` (`cne.py:328-330`).
  - `neg_spec` → `spec_param`, the constant c in the loss (`cne.py:332-333`).
- **Default:** no spectrum argument means `neg_spec = 1.0` (`cne.py:250-255`).

## 7. Initialization (schedule fact (c)) and RNG inventory

**(c) A custom init array is accepted, used unchanged, and keeps its dtype.**
- `CNE.fit(X=None, init=<ndarray>, graph=<sparse>)` is allowed: `X` may be None if `graph` is sparse (`_cne.py:393`).
- The init is only shape-checked (`_cne.py:479-485`), then becomes the weights via `torch.nn.Embedding.from_pretrained(torch.tensor(init))` and `requires_grad_(True)` (`_cne.py:515-516`).
- **There is no rescaling.** Only the string inits are rescaled: `"pca"` to std 1e-4 (`_cne.py:490-494`); `"le"` and `"diffmaps"` to std 1e-4 plus jitter (`_cne.py:495-507`, `_cne.py:301-303`).
- Nothing in `ContrastiveEmbedding.fit` touches the weights before the first epoch (`cne.py:396-447`).
- **Caveat:** `torch.tensor(init)` keeps the dtype. A float64 init would run the whole optimization in float64. We always pass float32, which matches the package's own path.

**Init choice:**
- The v0.4.0 default is `init=None` with X given, which becomes `"pca"` (sklearn `PCA(2)`) **rescaled so PC1 has std 1e-4** (`_cne.py:471-473`, `_cne.py:490-494`).
- The docstring at `_cne.py:384` says "PCA rescaled so that first PC has standard deviation 1"; the code contradicts it.
- **We use:** the first 2 PCs rescaled so PC1 has std 1 (paper Supp. K.2), using the package's own `rescale(x, target_std=1.0)` (`_cne.py:202-222`), cast to float32 and passed explicitly.

**RNG inventory:**

| Source | Seeded? | Where |
|---|---|---|
| annoy index build (kNN graph) | **No** | `_cne.py:418-420` |
| global torch seed at loader creation | `torch.manual_seed(seed)` | `_cne.py:119` |
| global torch seed again when the loss is created (inside `ContrastiveEmbedding.fit`, after the loader) | `torch.manual_seed(seed)` | `cne.py:549`, `cne.py:400` |
| per-epoch edge shuffle | global torch RNG | `_cne.py:130` |
| per-batch negatives | global torch RNG | `cne.py:753` |
| `"pca"` init | sklearn `PCA` with no `random_state` (bypassed: we pass an explicit init) | `_cne.py:492` |
| `"le"` / `"diffmaps"` init and jitter | seeded with `seed` | `_cne.py:497-507` |

Consequences:
- A fit is deterministic given (init, graph, seed): the global torch RNG is reseeded inside `fit`, and then only the shuffle and negative sampling draw from it.
- Anything else that draws from the global torch RNG during training changes the run.

## 8. Phase 3 hook points

- **Loss assembly:** `ContrastiveLoss.forward(features, ...)` (`cne.py:553`) only sees the 2b batch feature rows, not data indices. The head data indices are `item` in `train` (`cne.py:36-40`). The background term needs both the head data index and access to arbitrary rows of the embedding table, so the natural hook is `train` (`cne.py:45-49`). There, compute `y_bg = model(bg_tail_idx)`, the Cauchy kernel and `estimator` with the same `criterion.spec_param`, `eps` and clamps, and add `γ · Σ −log(1 − estimator)` to `loss` before `backward()`.
- **Embedding table:** in non-parametric mode `model` is the `nn.Embedding` holding all n rows (`_cne.py:515`). Indexing rows outside the batch is fine, and gradients flow to them. Gradient clipping (`cne.py:57-58`) is applied to the summed gradient of all terms together.
- **Uniform negatives index the batch, not the full table** (§3). The `uniform_extra` control should copy that sampling for comparability, or draw from all n points as Plan §4 says; this decision is for Phase 3.
- **Loss components:** the loss is summed into a scalar (`cne.py:684-687`). Logging attraction and uniform repulsion separately requires splitting column 0 from columns 1..m in `forward`, before aggregation (`cne.py:658-660`).
- **Epoch counter:** the loop variable at `cne.py:447`, passed to `train` as `epoch` (`cne.py:482-489`). With the two-stage schedule the background term is on for all of Stage 2, so no epoch gating is needed.
- **RNG for bit-identity:** the background sampler must use its **own `torch.Generator`**, not the global torch RNG. Otherwise `m_B = 0` or `bg_mode="none"` would still shift the shuffle and negative-sampling streams and break the "matches the unmodified package" test.

## 9. Assumption check (Plan §0 / §1)

| Plan assumption | What the code does | Match? |
|---|---|---|
| Neg-t-SNE = NEG with the Cauchy kernel | `estimator = 1/(1 + c(d² + 1)) = φ/(φ + c)` (`cne.py:654`) | ✓ |
| Classifier uses a constant c, not the true noise probability | `c = spec_param`, a scalar for all pairs (`cne.py:405,459,654`) | ✓ |
| Example form of the constant: Z̄·m/C(n,2) | `neg_spec = m·Z̄/n²` (`cne.py:330`) | ≈ (n² instead of C(n,2), irrelevant since c is unchanged) |
| m uniformly sampled in-batch negatives | Uniform over the 2b−1 batch slots other than the head slot, with replacement, resampled every batch. The head's own positive tail and duplicate slots of the head point are not excluded (`cne.py:739-774`) | ✓ with caveats |
| Edges are directed kNN edges | Union-symmetrized k = 15 graph; each undirected edge appears in both directions; weights ignored (`_cne.py:423-430`, `_cne.py:100-102`) | ✓ |
| Early exaggeration = fixed Z̄ = \|X\|/m for the first epochs, then an annealing reset | A no-op for neg defaults (`s=None`). When active it uses s = 1 (Z̄ = n²/5) for the first n_epochs//3. No LR reset | ✗, resolved by the two-stage schedule |
| `loss_mode="neg"` selects Neg-t-SNE | Yes (`cne.py:151,250`) | ✓ |
| Background tails can index the full embedding table (non-parametric) | `model` is an `nn.Embedding` over all n points (`_cne.py:515`) | ✓ |

**Stop condition check:** the classifier does **not** use the true ξ. The package internals do not contradict the core assumption of the method (Plan §0). The early-exaggeration mismatch was already handled by switching to the two-stage schedule.

## 10. Empirical confirmation (`scripts/phase0_mnist_sanity.py`, `results/phase0/summary.json`)

**Setup:**
- Determinism: `common.repro.setup_determinism(0)` (deterministic torch algorithms, 4 threads, torch/numpy/random seeded).
- Data: 10k random MNIST points (seed 0) drawn from all 70k, raw pixels.
- kNN graph: exact k = 15 on a 50-dim PCA, union-symmetrized, built once and reused.
- Init: PCA rescaled so PC1 has std 1, float32.
- Hardware and software: Apple M3, torch 2.14.1. The full environment is in `summary.json` under `environment`.

**Graph:** 10,000 nodes and 210,584 directed edges. Degree is min 15, mean 21.1, max 54. The graph is symmetric, binary, and has no self-loops. Building it twice gives identical graphs.

**Results:**

| Run | Epochs | kNN acc (k=15, 5-fold) | Runtime |
|---|---|---|---|
| Plain (package defaults) | 200 | 0.917 | 7.5 s |
| Stage 1 | 250 | 0.918 | 9.4 s |
| Stage 2 (from Stage 1) | +500 | 0.918 | 19.8 s |

**Qualitative comparison** (`results/phase0/comparison.png`):
- The structure is the same as the package README's Neg-t-SNE figure (full 70k): compact, well-separated digit clusters.
- 4/7/9 form one block and 3/5/8 sit next to each other; 1 is a long thin arc; 0, 2 and 6 are isolated clusters.
- At 10k points the clusters are sparser, as expected. Stage 2 is stable: no fragmentation and no change in scale.

**Schedule facts (all confirmed empirically):**
- **(a) Early exaggeration is a no-op for neg defaults.** Plain runs with EE on and off are bit-identical.
- **(b) No LR reset.** The recorded per-epoch learning rates fall monotonically in every fit (`results/phase0/lr_schedule.png`). Stage 2 restarts at 1.0 and falls to 0.002.
- **(c) A custom init is used unchanged.** The Stage 2 start weights (read through the epoch −1 callback) are exactly equal to the Stage 1 output, float32. The plain run starts exactly at the PCA init.
- **Approach used:** Stage 2 is a new `CNE.fit(init=stage1_embd, graph=G)`. The in-loop fallback is not needed.

**Reproducibility:**
- **Within a run:** the Stage 2 rerun is bit-identical.
- **Across two separate invocations of the script:** the kNN graph, PCA init, and plain, Stage 1 and Stage 2 embeddings are all bit-identical at 4 threads. The 1-thread fallback was not needed.
- **Deterministic mode:** no op raised an error.

**History, before the project-wide determinism rule:**
- With torch defaults (4 threads, non-deterministic algorithms), same-seed reruns diverged: max |diff| was 2.3e-5 after 1 epoch and 7.4 after 20 epochs.
- Both `torch.use_deterministic_algorithms(True)` (8.3 s vs 8.4 s per 200 epochs) and `torch.set_num_threads(1)` (14.3 s) removed the divergence.
- The likely source is multithreaded scatter-add in the backward pass (the `features[neigh_inds]` gather and/or the embedding backward), whose float summation order varies between runs.
- With the package's annoy graph (now replaced by the exact graph), the plain run's kNN accuracy was 0.926.

# Phase 1: synthetic generator (`data/synthetic.py`)

## Decisions and why

- **Simplex centers.** Each latent's cluster centers are the K = d + 1 vertices of a regular simplex, randomly rotated per seed. Edge length is 30 for the background (K_b = 6, d_b = 5) and 6 for the signal (K_s = 4, d_s = 3). All pairs are equidistant, so one number per latent sets the separation.
  - We first tried i.i.d. Gaussian centers with sep_s = 3. They often nearly collided (minimum signal center distance 0.48–2.8 across seeds 0–3), and the oracle y_s kNN accuracy on S_T was only 0.78–0.92 on 4 of 5 seeds.
  - With simplex centers (edge 6) the oracle accuracy is 0.995–0.997.
- **medium overlap.** The overlap is controlled: A = √α·B R + √(1−α)·Q₂, with α = ‖BᵀA‖²_F / d_s = 0.5. A keeps orthonormal columns, so the signal geometry, and therefore ρ, is comparable to easy. Independent random 3- and 5-dim subspaces in 50-D would be nearly orthogonal.
- **hard nonlinearity is measured within clusters.** Global affine-fit R² of the background on z_b is always ≥ 0.95 here: 6 centers in 5-D are fit exactly by an affine map, so between-cluster variance is always "linear".
  - Within-cluster R² by gain: 0.99 at gain 1.5, 0.96 at gain 3, and ≈ 0.87–0.89 at **gain 6 (chosen)**.
  - At gain 6, about 71% of pre-activations satisfy |pre| > 2, yet within-cluster background variance *grows*: 12–15 vs 5 for the linear levels. Total variance is matched to the linear levels.
  - Consequence: at a given ρ, hard hides the signal more strongly than easy or medium.
- **Randomness.** Independent `SeedSequence` streams cover centers, maps, target samples and background samples. ρ changes no random draw (S_T scales exactly), and n_B does not affect X_T.
- **Graph and init.** The validity check builds the exact kNN graph and the PCA init on X_T directly, because D = 50 makes the plan's PCA-50 step a pure rotation.

## Validity sweep (`scripts/phase1_validity.py`, `results/phase1/`)

The baseline embedding is Stage 1 (250 epochs) + Stage 2 (500 epochs), with no background term and n_T = n_B = 5000. kNN accuracy uses k = 15 and 5-fold CV. Chance is 0.167 for y_b and 0.25 for y_s.

Seed 0, embedding y_s accuracy by ρ (embedding y_b accuracy is 1.000 everywhere; oracle y_s accuracy is 0.996 everywhere):

| level | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| easy | **0.250** | **0.289** | 0.961 | 0.990 | 0.993 | 0.995 | 0.996 |
| medium | **0.264** | **0.371** | 0.584 | 0.841 | 0.947 | 0.978 | 0.987 |
| hard | **0.259** | **0.296** | **0.371** | 0.512 | 0.837 | 0.986 | 0.991 |

Bold marks valid configurations: y_b ≥ 0.90, y_s ≤ 0.50, oracle ≥ 0.95. The valid ρ values form one contiguous range at every level.

**Chosen ρ** (the largest valid value, confirmed valid on seeds 0, 1 and 2):

| level | ρ | emb y_s (seeds 0/1/2) | high-D X_T y_s (seeds 0/1/2) |
|---|---|---|---|
| easy | 0.2 | 0.289 / 0.306 / 0.314 | 0.892 / 0.876 / 0.875 |
| medium | 0.2 | 0.371 / 0.369 / 0.387 | 0.709 / 0.703 / 0.700 |
| hard | 0.3 | 0.371 / 0.338 / 0.349 | 0.882 / 0.870 / 0.849 |

**Signal is hidden by the embedding, not by the graph.**
- At the chosen ρ, the high-D kNN y_s accuracy on X_T is 0.70–0.89. The target kNN graph therefore still carries much of the signal: most neighbors share a signal label.
- Plain Neg-t-SNE loses it. Each background cluster becomes a compact, well-separated blob with the signal colors mixed (`embeddings_chosen.png`). Stage 1 alone gives the same accuracies as Stage 2.
- Implication for Phase 4: extra repulsion alone may already expand the blobs and expose within-cluster signal structure. The `uniform_extra` control (b) is therefore essential.
- A stricter criterion that also requires the high-D y_s accuracy to be low would force ρ ≈ 0.1. At that point the high-D y_s accuracy is still 0.51 (easy), 0.40 (medium) and 0.39 (hard).

**Other observations:**
- easy has a sharp transition between ρ = 0.2 (0.29) and 0.3 (0.96). medium and hard degrade more gradually; at a given ρ, overlap with the background directions and hard's larger within-cluster background variance hide the signal longer.
- `raw_pca.png` shows X_B covering X_T's background clusters at every level.
- **Determinism:** two full runs of the sweep give a byte-identical `validity.csv` and bit-identical chosen embeddings.

## Frozen datasets and Phase 4 additions

- **Approved scales:** `data.synthetic.MAIN_RHO` = {easy 0.2, medium 0.2, hard 0.3}.
- **Stress test:** `STRESS_RHO` = 0.1 for every level, the "graph-level hiding" set. Its high-D X_T y_s kNN accuracy is 0.50–0.51 (easy), 0.40 (medium) and 0.38–0.39 (hard), against 0.70–0.89 for the main sets.
- **Hashes:** `scripts/phase1_datasets.py` records SHA-256 hashes of X_T and X_B for every main and stress dataset (seeds 0–2) in `results/phase1/datasets.csv`. Two runs give an identical file.
- **Stress-test rule:** the stress sets are run only once, at the end of Phase 4, with the best main-run settings and no tuning.
- **`neg_spec=0` is degenerate:** `estimator = 1/(1 + 0·(d²+1)) ≡ 1` (`cne.py:654`). The attractive log term is log(1) = 0, the repulsive term log(clamp(0)) = log(1e-4) is constant, and the clamp passes no gradient. The embedding therefore never moves from its init (checked: a 20-epoch fit returns the init bit-for-bit).
- **The (d) spectrum reference uses `s=0` instead:** Z̄ = 100·n, so `neg_spec = 500/n` (0.1 at n = 5000). A Stage 2 run with it trains normally.

# Phase 2: ξ_B (`method/xi_b.py`)

## Decisions and why

- **Construction:**
  - exact background kNN (k_B = 15) from X_T to X_B on raw features;
  - candidates via the integer sparse product M Mᵀ (the inverted index);
  - Jaccard from integer overlap counts, so it is exact and independent of order.
- **Tie rule.** With k_B = 15, Jaccard takes only 15 values, so in 90–94% of rows the top-L cut falls inside a tied block. The tie rule therefore decides list membership. The rule:
  - keep the whole tied block up to 2L = 60 entries;
  - beyond the cap, a random draw within the block, keyed by `SeedSequence(seed, spawn_key=(row,))`.

  Breaking ties by index would drop pairs by position. Breaking them by target distance would bias ξ_B toward same-signal pairs.
- **Label agreement** is an exact expectation, Σ_i w_i Σ_j ξ_B(i, j)[same label], with heads weighted by target-graph degree (the way training draws heads). No sampling is involved; the Phase 3 sampler gets its own frequency test.
- **Stress datasets:** not touched. Their ξ_B is built only at the end of Phase 4.

## Diagnostics (`scripts/phase2_xi_b.py`, `results/phase2/`)

All informativeness gates pass on every level and seed: same-y_b ≥ 0.90, and same-y_s ≤ uniform + 0.10. Two runs give byte-identical CSV, JSON and `.npz` files.

| level | same-y_b | same-y_s excess over 0.25 (seeds 0/1/2) | target kNN edges same-y_s | ξ_B mass on target kNN edges (same-y_s / different-y_s) |
|---|---|---|---|---|
| easy | 1.000 | +0.015 / +0.018 / +0.017 | 0.58–0.59 | 0.49 (0.22 / 0.27) |
| medium | 1.000 | +0.058 / +0.058 / +0.058 | 0.47 | 0.58 (0.23 / 0.34) |
| hard | 1.000 | +0.028 / +0.029 / +0.022 | 0.58–0.63 | 0.51–0.54 (0.24–0.25 / 0.26–0.30) |

- **ξ_B follows background geometry within a cluster, not just the cluster label.**
  - The mean ‖Bg_i − Bg_j‖ of ξ_B pairs (1.34–1.82) matches that of target kNN edges (1.35–1.84). Both are far below random same-cluster pairs (2.99–5.07).
  - The Spearman correlation of s_B with background distance is −0.43 to −0.58.
- **About half of ξ_B mass falls on target kNN edges**, as intended (Plan §3 step 6), and 72–83% of target edges appear in ξ_B. That mass leans toward edges whose endpoints differ in signal, i.e. edges the background explains.
- **medium leaks a little signal (+0.058).** This is expected: half of A lies inside span(B), so a target point's background neighbors depend slightly on z_s. Pairing same-signal points works against the method. If medium underperforms in Phase 4, check this first (Plan §5).
- **Lists:**
  - no empty rows;
  - list-length median 35–38;
  - the 2L cap triggers in 0.1–0.9% of rows (`list_sizes.png`).
- **Sensitivity** (`sensitivity_seed0.csv`, informational only; defaults unchanged):
  - same-y_b is 1.000 for every k_B ∈ {5, 15, 30} and L ∈ {15, 30, 60};
  - the same-y_s excess shrinks slightly as k_B grows (medium: +0.067 at k_B = 5, +0.057 at 15, +0.052–0.060 at 30).
- **Provenance flag fix:** `environment_info()["project_git"]["dirty"]` used to count a run's own untracked outputs in `results/`, so every Phase 0 and Phase 1 results file says `dirty: true` even though the code matched the recorded HEAD. Since `92fbcbc`, `results/` is excluded; the Phase 2 diagnostics record `dirty: false`.

# Phase 3: background term in the trainer

## Decisions and why

- **Hook in the vendored package (11 lines).** We use this instead of monkeypatching a copy of `train()`, so the training loop isn't duplicated. It's documented in `third_party/contrastive-ne/UPSTREAM.md`, and the changed lines are marked `# local change (bg-contrastive-ne)`.
- **The term joins the batch loss with the package's own aggregation** (`"sum"`, so L_ij matches the plan formula exactly). A unit test checks that one background negative and one uniform in-batch negative at the same φ contribute identical loss and gradient, under `"sum"` and under `"mean"`.
- **`uniform_extra` samples uniformly from all other target points**, not in-batch. ξ_B tails are drawn from the full table too, so (b) and (c) differ only in the distribution.
- **The sampler has its own generator,** seeded from `SeedSequence([seed, 1])`. A generator seeded with the bare run seed would replay the global torch stream, which drives the edge shuffle and the package's negatives.
- **Logging runs in every mode and both stages, using no gradients and no random draws.** It records loss components, the background share of the loss, mean φ over a fixed pair sample, the embedding std, and finiteness. It changes nothing: the bit-identity tests and the regression check pass with it on.

## Verification

- **Regression check:** with the patched package and the logging helpers, the Phase 1 chosen embeddings (3 levels, seed 0) are bit-identical to those saved before the patch.
- **Tests:** 14 new, 44 in total, all passing.

## Smoke run (`scripts/phase3_smoke.py`, `results/phase3/`; easy, seed 0; stability only)

| Stage 2 run | runtime | std growth (max / start) | mean φ end (start 0.078) | background share (Stage 2 range) |
|---|---|---|---|---|
| none (a) | 12 s | 1.30 | 0.077 | 0 |
| uniform_extra m_B=5 | 18 s | 1.52 | 0.052 | 0.05–0.16 |
| uniform_extra m_B=20 | 38 s | 1.89 | 0.027 | 0.11–0.25 |
| xi_b m_B=5 | 24 s | 1.40 | 0.030 | 0.19–0.36 |
| xi_b m_B=20 | 51 s | 1.55 | 0.012 | 0.18–0.67 |
| s=0 reference (d) | 13 s | 2.34 | 0.015 | 0 |

- **All §6 checks pass:**
  - exact Stage 2 start;
  - the background share is exactly 0 throughout Stage 1 and > 0 in every Stage 2 epoch with m_B > 0;
  - no non-finite batch or embedding;
  - a bit-identical rerun.
- **There is a transient at the stage boundary in every run, including (a).** The fresh learning rate of 1.0 expands the embedding in the first Stage 2 epochs: (a)'s mean φ drops from 0.078 to about 0.04, then recovers to 0.077 by the end. For `xi_b` at m_B = 20, the background share peaks at 0.67 in the first Stage 2 epoch, because ξ_B pairs start close together. It falls to about 0.18 within a few epochs. Gradient clipping (±4 per element) bounds each step, and nothing diverges.
- **Scale differs between runs** (relevant for Phase 4):
  - A lower mean φ means a more expanded embedding.
  - `xi_b` at m_B = 5 lowers mean φ almost as much (0.030) as `uniform_extra` at m_B = 20 (0.027), with a quarter of the extra negatives. `xi_b` at m_B = 20 lowers it most (0.012).
  - By embedding std, the `s=0` reference (d) expands most (2.34×).
  - The extra repulsion from ξ_B is concentrated on background-similar pairs, about half of them target kNN edges (Phase 2). That is why `xi_b`'s attraction stays high: at m_B = 20 it ends near 288k, against 85k for (a).
  - Phase 4 compares against the best (b) over all m_B and against (d), and logs Σφ for every run, so scale shifts are visible next to the signal metrics.
- **Phase 4 runtime estimate:**
  - Stage 2 grows from about 12 s (m_B = 0) to about 51 s (m_B = 20).
  - The 12 Stage 2 runs per (level, seed) take about 5 min, so about 45 min for 3 levels × 3 seeds, plus the stress test at the end.
- **Runtimes exclude sleep.** Runtimes were timed with `time.time()`, which counts laptop sleep: one rerun recorded about 7500 s for a 51 s fit. Since `150285d` they use `time.perf_counter()` (`mach_absolute_time` on macOS, which pauses during sleep). The table above comes from that commit. All three smoke runs gave identical results apart from the environment block and runtimes.

# Phase 4: experiment (`REPORT.md`, `results/phase4/`)

- **Protocol:**
  - pre-registered in plan §5 and `common/experiment.py`, committed before any result (`dad1c09`);
  - selection on seeds 0–2, with `selection_{level}.json` committed before confirmation;
  - the verdict uses fresh confirmation seeds.
- **Verdicts:**

  | level | verdict | (c) signal accuracy | best control |
  |---|---|---|---|
  | easy | not met | 0.89 | 0.39 |
  | medium | met | 0.58 | 0.41 |
  | hard | not met | 0.87 | 0.43 |

  The only failing clause is "background accuracy drops". Background negatives add only repulsion, and no kNN edges join islands, so islands can't merge. Medium's pass rests on about 10 boundary points (background accuracy 0.997–0.999), not on background suppression.
- **Mechanism:** inside each background island, kNN attraction mostly joins same-signal points, while ξ_B repulsion is nearly signal-blind. The island therefore sorts into signal sectors (4 per island on easy and hard). Uniform extra negatives mostly land in other, already distant islands, so they do almost nothing within an island.
- **Hard seed 3 failed validity:** (a) reached 0.5126 > 0.50. The user set an exclusion rule: draw the next seed, stop if more than 2 are excluded per level, and list excluded runs only in `results/phase4/excluded_runs_note.md`. Hard confirmation used seeds 4–6.
- **Checks:**
  - (a) reproduces the Phase 1 accuracies exactly;
  - all hashes match;
  - a rerun of easy seed 0 is bit-identical;
  - no collapse or fragmentation flags.
- **Recall and Spearman are near zero in absolute terms** (fine-grained target; global layout set by islands). Signal accuracy carries the interpretation.
- **Main next step:** cross-background attraction, for example a kNN graph in a background-suppressed metric.

# Phase 5 prep: global metrics and the cPCA baseline (`REPORT_phase5prep.md`, `results/phase5prep/`)

- **Spec:** `Phase_5_Plan.md`.
- **Global metrics G1–G3** (`eval/global_metrics.py`) were validated before use. Phase 4 runs score G1 ≈ 0, G2 ≈ 1 and G3 "shares signal" = 0; the signal-only reference scores ≈ 0.82, ≈ 0.17 and 1. The thresholds, set by the project owner, were frozen in `23459ff` before any cPCA result.
- **Reference package `contrastive` 1.2.0.** It is a single module with no declared license, so it was installed `--no-deps`, not vendored. Its automatic α selection:
  - **affinity:** the product of the principal-angle cosines between the 2-D column spaces of the *projected target data* (QR of X_T·V), not of the feature-space directions;
  - **clustering:** spectral clustering into 4 groups with numpy's global generator; the cluster containing α = 0 is dropped, each other cluster contributes its exemplar (largest summed affinity), and α = 0 is added back;
  - **defaults to override:** it standardizes unless `standardize=False`, and uses `eig`, not `eigh`.

  Our `baselines/cpca.py` mirrors this, using `eigh`, sign fixing and an explicit `random_state`. On easy seed 0 it matches the package to within 8.7·10⁻¹¹ rad, with identical representatives.
- **Result:** cPCA → Neg-t-SNE / UMAP reaches about the signal-only reference on all three levels (signal accuracy about 0.98, G1 about 0.81) and passes all four frozen criteria. It beats (c) on every metric, consistently across the paired seeds.
- **Why it succeeds: a population identity.** The target is S + G + ε and the background is G′ + ε′, with G independent of S, distributed like G′, and the noise equal in both. So C_T − C_B = C_S at α = 1, for linear and nonlinear background maps alike. cPCA is exact in the population; only finite-sample error remains.
- **Why α > 1 is also fine (secondary diagnostic):** the share of signal variance lying in directions where the background has no variance is 100% (easy), 50% (medium) and 17–35% (hard).
- **Consequence:** benchmarks for the next method must break additivity or independence, so that the signal's effect depends on the background state. Examples are x = F(z_b, z_s) with background F(z_b′, s₀), or background-cluster-specific signal maps; the latter makes G1 and G3 ill-posed.
- **Installing umap-learn / contrastive** changed no existing package, and the Phase 4 bit-identity check still passes.

# Phase 5a: benchmarks that break cPCA's additive assumption (`REPORT_phase5a.md`, `results/phase5a/`)

- **Spec:** given by the project owner in conversation and summarized in REPORT_phase5a.md "Design".
  - J's knobs were changed in plan review to θ × projection × MLP gain (16 configurations).
  - Added later: predictions P1–P3, the data diagnostics, and (e2-k20)/(e2-k40) for K.
- **Generators** (`data/synthetic_j.py`, `data/synthetic_k.py`): both reuse easy's latents, maps, noise and X_B bit for bit. New draws come from a fifth seed stream. `data/synthetic.py` now also records `eps_T` in `truth`.
- **Oracle same-cluster control:** `bg_mode="oracle_cluster"` runs through the ξ_B sampling path, over `same_cluster_xi(y_b)`. The Phase 4 easy recheck stayed bit-identical after the change.
- **ρ:** ρ_J = 0.812 (= 0.2·√16.5, since J's signal has unit total variance) and ρ_K = 0.2, both from the Phase 1 rule.
- **K stopped at calibration:** (e2) WC 0.97–0.98 > 0.80, and the top 20/40 components give 0.98. The cluster-specific maps keep C_T − C_B equal to the signal covariance (zero-mean, outside span(B)).
- **J: one valid configuration, t60_proj_g2.**
  - It meets V3 only through G1/G2: (e2) is signal-organized globally (G3 = 1) with background sub-lobes.
  - **Cause:** the interaction carries a background-cluster mean shift outside span(B), which X_B never shows (about 9% of the signal variance). Coherence can't see it.
  - **Did not replicate:** on confirmation seeds 3 and 4 (label-free α 37 and 29), (e2) passed the frozen G1–G3 thresholds.
- **J main runs:**
  - (e2) beats (c) on signal accuracy, G1, G2 and G3, consistent across 3 seeds.
  - (c) beats the oracle same-cluster control on signal accuracy by +0.39 (0.86 vs 0.47), consistent across 3 seeds. This is REPORT.md Next steps #1: ξ_B's neighborhood structure, not same-cluster repulsion alone, carries (c)'s gain.
- **Predictions:** P1 not confirmed; P2 not confirmed (descriptive); P3 not applicable for J and K.
- **Extension:** (o) at m_B 40/80 peaks at 40 (0.48 on the selection seeds). At 40, (c) still beats (o) by +0.36 signal accuracy on every confirmation seed.
- **Global/local ratio:** dominated by a 2–10% tilt of the signal into span(B); cPCA keeps the other 90%+. What decides whether cPCA fails is the fraction of the signal outside high-global-variance directions (at least 87% everywhere here).
- **Next:**
  1. Design L: the signal entirely inside span(B), along between-cluster directions, with tight clusters.
  2. Judge V3 across all α representatives.
  3. Pre-register a non-local global metric (e.g. G1 at k = 200) alongside G3.
  4. Center J's interaction (lower priority).

# Phase 5b: Design L, where linear suppression is structurally limited (`REPORT_phase5b.md`, `results/phase5b/`)

- **Spec:** given by the project owner in conversation and summarized in REPORT_phase5b.md "Design". In plan review, the linear and cPCA ceilings moved to the embedding level (the spec's high-D ceiling could not pass V3 when V2 held); P1 moved with them, before any Design L result.
- **Generator** (`data/synthetic_l.py`): six balanced clusters, identical in X_T and X_B, so C_T − C_B is the signal covariance. L-cyclic: cluster k's signal block Q_k is cluster k−1's nuisance block. L-between: the signal lies in W ⊂ U (between-cluster span), the nuisance in V. ρ = 0.2 for both.
- **Ceilings** (`common/benchmark_l.py`): every labeled projection and cPCA variant is embedded with (a)'s two-stage Neg-t-SNE and scored on WC; high-D versions are descriptive.
- **Selection** (WC): L-cyclic b 40, c 20, o 80; L-between b 40, c 10, o 80. (b) and (o) at the top of their grids.
- **L-cyclic:** (c) 0.914 beats best cPCA (0.530) and the best labeled linear projection (0.624, always an alternating subset that reveals 3 of 6 clusters), consistent across 3 seeds. (c) reveals all 6 clusters.
- **L-between:** best cPCA (0.989) beats (c) (0.914) on every seed. Label-free cPCA fails (0.255) on α (it picks α > 1 on every seed) **and** on k: at k = 10 even the best α < 1 gives 0.35, because components 6–10 bring in the nuisance V. Only α < 1 with k ≤ 5 (often plain PCA) works.
- **(c) vs (o) and (b):** (c) wins on both designs, by about +0.22 and +0.39.
- **Predictions:** P1–P5 all confirmed.
- **Compute:** run on the laptop; the selection chain drained the battery (runs now use `caffeinate -s` and start only on AC). Selection summaries are dirty only through an untracked, unimported report script.
- **Next:**
  1. Move compute to the Math Slurm cluster (account YANG-2026); the GPU path is untested.
  2. A label-free rule for both α and k for cPCA.
  3. A design where the high-D kNN graph itself is poor, to see whether (c) is limited by the graph (its WC sits just under detectability here).
