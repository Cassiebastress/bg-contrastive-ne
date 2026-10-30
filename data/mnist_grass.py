"""MNIST-on-grass (the cPCA paper's real-data example), rebuilt deterministically from a dataset seed.

Recipe (reference/cpca_mnist_notebook.py; deviations documented in data/README_mnist_grass.md):
- grass: every .JPEG/.JPG/.jpg in the grass folder, in sorted filename order; grayscale (PIL mode "L"), LANCZOS
  resize to 100 x 100, scaled to [0, 1];
- one numpy.random.default_rng(seed), drawn from in this fixed order:
    1. n digits sampled without replacement from all MNIST training images of digits 0 and 1;
    2. a permutation of the grass images: the first half is the target pool, the second the background pool;
    3. target patches: source image (uniform over the target pool), then crop origins (row, col), each in [0, 70);
    4. background patches: source image (uniform over the background pool), then crop origins;
- target row i = 0.25 * digit_i / 255 + its 28 x 28 grass patch; background row j = its patch.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

GRASS_SUFFIXES = (".JPEG", ".JPG", ".jpg")
GRASS_SIZE = 100
PATCH = 28
MAX_ORIGIN = 70  # crop origin in [0, 70) on each axis, as in the notebook
DIGIT_SCALE = 0.25
DIGITS = (0, 1)
N = 5000


def grass_files(grass_dir):
    """Grass image paths in sorted filename order (the notebook's os.listdir order is filesystem-dependent)."""
    return sorted((p for p in Path(grass_dir).iterdir() if p.name.endswith(GRASS_SUFFIXES)), key=lambda p: p.name)


def load_grass(files):
    """(n_images, 100, 100) float64 in [0, 1]: grayscale, LANCZOS-resized (resized only, not cropped)."""
    out = np.empty((len(files), GRASS_SIZE, GRASS_SIZE))
    for i, p in enumerate(files):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert(mode="L").resize((GRASS_SIZE, GRASS_SIZE), Image.Resampling.LANCZOS),
                                dtype=np.float64) / 255
    return out


def load_mnist(path):
    with np.load(path) as f:
        return f["x_train"], f["y_train"]


def crop(grass, src, loc):
    """Flattened 28 x 28 patches: grass[src[i]] cropped at origin loc[i] = (row, col)."""
    return np.stack([grass[s, r:r + PATCH, c:c + PATCH].ravel() for s, (r, c) in zip(src, loc)])


def build(seed, grass_dir, mnist_path, n=N):
    """The dataset for one seed, with its provenance (see the module docstring for the draw order)."""
    files = grass_files(grass_dir)
    grass = load_grass(files)
    x_train, y_train = load_mnist(mnist_path)
    rng = np.random.default_rng(seed)
    mnist_index = rng.choice(np.flatnonzero(np.isin(y_train, DIGITS)), size=n, replace=False)
    perm = rng.permutation(len(files))
    target_pool, background_pool = perm[: len(files) // 2], perm[len(files) // 2:]
    t_src = rng.choice(target_pool, size=n)
    t_loc = rng.integers(MAX_ORIGIN, size=(n, 2))
    b_src = rng.choice(background_pool, size=n)
    b_loc = rng.integers(MAX_ORIGIN, size=(n, 2))
    digits = x_train[mnist_index].reshape(n, -1).astype(np.float64) / 255
    grass_T = crop(grass, t_src, t_loc)
    return {"X_T": DIGIT_SCALE * digits + grass_T, "X_B": crop(grass, b_src, b_loc), "y": y_train[mnist_index],
            "mnist_index": mnist_index, "digits": digits, "digit_scale": DIGIT_SCALE, "grass_T": grass_T,
            "target_src": t_src, "target_loc": t_loc, "background_src": b_src, "background_loc": b_loc,
            "target_pool": np.sort(target_pool), "background_pool": np.sort(background_pool), "seed": seed,
            "filenames_sha256": hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest()}


def check(d, grass):
    """The build checks (raises AssertionError): shapes, ranges, exact reconstruction, disjoint pools."""
    n = len(d["y"])
    assert d["X_T"].shape == d["X_B"].shape == d["digits"].shape == d["grass_T"].shape == (n, PATCH * PATCH)
    assert d["digits"].min() >= 0 and d["digits"].max() <= 1 and d["X_B"].min() >= 0 and d["X_B"].max() <= 1
    assert d["X_T"].min() >= 0 and d["X_T"].max() <= 1 + d["digit_scale"]
    assert set(np.unique(d["y"])) <= set(DIGITS) and len(np.unique(d["mnist_index"])) == n
    assert np.array_equal(d["X_T"], d["digit_scale"] * d["digits"] + crop(grass, d["target_src"], d["target_loc"]))
    assert np.array_equal(d["X_B"], crop(grass, d["background_src"], d["background_loc"]))
    assert not set(d["target_pool"]) & set(d["background_pool"])
    assert set(d["target_src"]) <= set(d["target_pool"]) and set(d["background_src"]) <= set(d["background_pool"])


def save(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
