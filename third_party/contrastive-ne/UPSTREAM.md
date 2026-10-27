# Vendored: contrastive-ne

- Upstream: https://github.com/berenslab/contrastive-ne
- Commit: `d5f0b530f35be4d6247300a601fb5a860bc0a1c8` (v0.4.0, "bumps version to 0.4.0", 2025-11-07)
- License: MIT (see `LICENSE` in this directory), © the contrastive-ne authors
- Copied without upstream `.git`. The commit that adds this directory is the unmodified snapshot;
  all later changes to the package are diffs against it.

## Local changes

1. **`batch_hook` in the trainer** (`src/cne/cne.py`; Phase 3 of the project plan).
   - **What it adds:** `ContrastiveEmbedding(..., batch_hook=None)` stores the hook, and `fit` passes it to `train(..., batch_hook=None)`. There, right after `loss = criterion(...)`, the hook is called as `batch_hook(item, neigh, features, model, criterion, epoch)`, and a non-`None` return value is added to the batch loss before `backward()`. `CNE(**kwargs)` already forwards unknown keyword arguments to `ContrastiveEmbedding`, so `_cne.py` is unchanged.
   - **Why:** the project's background-negative term needs the head data indices (`item`) and the full embedding table (`model`), which the loss object does not see.
   - **When off:** with `batch_hook=None` the code path is the original one. The project's Phase 1 embeddings are reproduced bit-for-bit (Plan §4 tests and the Phase 3 regression check).
   - **Marked in the code** with `# local change (bg-contrastive-ne)`.
