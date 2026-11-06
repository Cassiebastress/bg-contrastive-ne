"""Tests for the MNIST-on-grass builder (data/mnist_grass.py), on fake grass JPEGs and a fake MNIST file."""
import hashlib

import numpy as np
import pytest
from PIL import Image

from data import mnist_grass as mg

N_IMAGES, N = 12, 40
DIGITS_0TO9 = tuple(range(10))
# SHA-256 of the RNG-derived arrays of the default (0/1) build of seed 3 below, recorded before the digits parameter
# was added (they don't depend on how the fake JPEGs decode): the 0/1 draws must never change
DRAWN = ["mnist_index", "y", "digits", "target_pool", "background_pool", "target_src", "target_loc", "background_src",
         "background_loc"]
GOLDEN_0_1 = "5bfe07da7689251257619df7f7f1f2fe474a08d1330cdf9d9f65aa730a378280"


def make_inputs(root, order):
    """Fake grass JPEGs (created in the given order, of varying sizes) and a fake mnist.npz."""
    grass = root / "grass"
    grass.mkdir()
    rng = np.random.default_rng(7)
    images = {i: rng.integers(0, 256, size=(120 + 7 * i, 150, 3), dtype=np.uint8) for i in range(N_IMAGES)}
    for i in order:
        Image.fromarray(images[i]).save(grass / f"n12102133_{1000 + i}.JPEG")
    (grass / "notes.txt").write_text("not an image")
    x = rng.integers(0, 256, size=(200, 28, 28), dtype=np.uint8)
    y = np.tile(np.arange(10), 20)
    np.savez(root / "mnist.npz", x_train=x, y_train=y, x_test=x[:5], y_test=y[:5])
    return grass, root / "mnist.npz"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return make_inputs(tmp_path_factory.mktemp("in"), range(N_IMAGES))


@pytest.fixture(scope="module")
def built(inputs):
    return mg.build(3, *inputs, n=N)


@pytest.fixture(scope="module")
def balanced(inputs):
    return mg.build(3, *inputs, n=N, digits=DIGITS_0TO9)


def digest(d, keys):
    h = hashlib.sha256()
    for k in sorted(keys):
        a = np.asarray(d[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_shapes_ranges_reconstruction_and_disjoint_pools(inputs, built):
    grass = mg.load_grass(mg.grass_files(inputs[0]))
    mg.check(built, grass)  # raises on any failure
    assert grass.shape == (N_IMAGES, 100, 100)
    assert built["X_T"].shape == built["X_B"].shape == (N, 784)
    assert len(built["target_pool"]) == len(built["background_pool"]) == N_IMAGES // 2


def test_only_images_with_the_grass_suffixes_are_used(inputs):
    assert [p.suffix for p in mg.grass_files(inputs[0])] == [".JPEG"] * N_IMAGES


def test_digits_are_zeros_and_ones_sampled_without_replacement(built):
    assert set(np.unique(built["y"])) == {0, 1}
    assert len(np.unique(built["mnist_index"])) == N
    counts = np.bincount(built["y"], minlength=2)
    assert counts.sum() == N and counts.min() > 0


def test_rebuild_is_bit_identical(inputs, built):
    again = mg.build(3, *inputs, n=N)
    assert all(np.array_equal(built[k], again[k]) for k in built)


def test_other_seed_differs(inputs, built):
    assert not np.array_equal(built["X_T"], mg.build(4, *inputs, n=N)["X_T"])


def test_file_creation_order_does_not_matter(tmp_path, built):
    shuffled = make_inputs(tmp_path, np.random.default_rng(1).permutation(N_IMAGES))
    again = mg.build(3, *shuffled, n=N)
    assert all(np.array_equal(built[k], again[k]) for k in built)


def test_save_load_roundtrip(tmp_path, built):
    mg.save(built, tmp_path / "d.npz")
    back = mg.load(tmp_path / "d.npz")
    assert all(np.array_equal(built[k], back[k]) for k in built)


def test_check_catches_a_broken_reconstruction(inputs, built):
    grass = mg.load_grass(mg.grass_files(inputs[0]))
    broken = dict(built, X_T=built["X_T"] + 1e-12)
    with pytest.raises(AssertionError):
        mg.check(broken, grass)


def test_default_digits_draws_are_unchanged(built):
    assert digest(built, DRAWN) == GOLDEN_0_1
    assert "digit_classes" not in built


def test_explicit_default_digits_equal_the_default(inputs, built):
    again = mg.build(3, *inputs, n=N, digits=(0, 1))
    assert set(again) == set(built) and all(np.array_equal(built[k], again[k]) for k in built)


def test_balanced_build_has_equal_digit_counts_and_passes_check(inputs, balanced):
    mg.check(balanced, mg.load_grass(mg.grass_files(inputs[0])))
    assert np.array_equal(np.bincount(balanced["y"], minlength=10), np.full(10, N // 10))
    assert np.array_equal(balanced["digit_classes"], np.arange(10))
    assert len(np.unique(balanced["mnist_index"])) == N


def test_balanced_rebuild_is_bit_identical_and_other_seed_differs(inputs, balanced):
    again = mg.build(3, *inputs, n=N, digits=DIGITS_0TO9)
    assert set(again) == set(balanced) and all(np.array_equal(balanced[k], again[k]) for k in balanced)
    assert not np.array_equal(balanced["mnist_index"], mg.build(4, *inputs, n=N, digits=DIGITS_0TO9)["mnist_index"])


def test_balanced_order_is_shuffled(balanced):
    assert not np.all(np.diff(balanced["y"]) >= 0)


def test_n_not_divisible_by_the_digits_raises(inputs):
    with pytest.raises(ValueError):
        mg.build(3, *inputs, n=N + 1, digits=DIGITS_0TO9)


def test_check_catches_unbalanced_digits(inputs, balanced):
    grass = mg.load_grass(mg.grass_files(inputs[0]))
    y = balanced["y"].copy()
    y[0] = (y[0] + 1) % 10
    with pytest.raises(AssertionError):
        mg.check(dict(balanced, y=y), grass)


def test_processed_paths():
    assert mg.processed_path(2, "/r").name == "mnist_grass_seed2.npz"
    assert mg.processed_path(2, "/r", name="mnist_grass_0to9").name == "mnist_grass_0to9_seed2.npz"
