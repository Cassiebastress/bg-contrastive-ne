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
