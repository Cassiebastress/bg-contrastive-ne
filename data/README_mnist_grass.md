# MNIST-on-grass dataset (`data/mnist_grass.py`)

The real-data example of the contrastive PCA paper: MNIST digits 0 and 1 at 25% intensity on 28 × 28 patches of ImageNet grass photos (target), and grass patches alone (background). The reference recipe is the authors' Colab notebook, kept unrun in `reference/cpca_mnist_notebook.py`. Inputs and their checksums are listed in `data/MANIFEST.md`.

## Recipe

- **Grass:** every `.JPEG`/`.JPG`/`.jpg` file in `data/raw/grass/` (1274 images), converted to grayscale (PIL mode "L"), LANCZOS-resized to 100 × 100, and scaled to [0, 1].
- **Digits:** 5000 MNIST training images of digits 0 and 1, scaled to [0, 1].
- **Pools:** the grass images are split half/half into a target pool and a background pool.
- **Patches:** each patch comes from a random image of its pool, cropped 28 × 28 at a random origin in [0, 70) on each axis.
- **Target row** = 0.25 × digit + patch. **Background row** = patch.
- **n** = 5000 target and 5000 background rows per dataset seed.

## Deviations from the notebook

1. **Sorted filenames.** The notebook reads files in `os.listdir` order, which depends on the filesystem, so its exact dataset cannot be reproduced. With sorted names, a build depends only on the seed.
2. **One RNG per dataset seed.** All randomness comes from `numpy.random.default_rng(seed)`, in a fixed order: the digit sample, the pool permutation, the target patches (source images, then crop origins), the background patches. The notebook uses the global `np.random.seed(0)` and interleaves its draws.
3. **Random digit sample.** The 5000 digits are sampled without replacement from *all* training images of digits 0 and 1. The notebook takes the first 5000 in file order. Each seed therefore gives an independent dataset.
4. **No `targetplus`.** The notebook also builds an "augmented" target (1.5 × digit + patch), which isn't needed here. It also has a bug: it picks a source image from the target pool, then crops from a random image across *all* images.
5. **Provenance is saved.** For each seed, `data/processed/mnist_grass_seed{s}.npz` (gitignored) holds:
   - X_T and X_B;
   - the digit labels and MNIST indices;
   - the clean digits and the 0.25 scale;
   - every patch's source image and crop origin;
   - the clean target grass patches;
   - both pools;
   - a hash of the sorted filename list.

   The notebook's "resize and crop" function only resizes; we do the same.

## Checks

These run in `tests/test_mnist_grass.py` on fake inputs, and on the real data in `scripts/mnist_grass_build.py`:
- shapes and value ranges;
- exact reconstruction (X_T = 0.25 × digits + saved patches; X_B = saved patches);
- disjoint pools;
- bit-identical rebuild;
- digit class counts.
