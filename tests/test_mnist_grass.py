"""Tests for the MNIST-on-grass builder (data/mnist_grass.py), on fake grass JPEGs and a fake MNIST file."""
import numpy as np
import pytest
from PIL import Image

from data import mnist_grass as mg

N_IMAGES, N = 12, 40


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
