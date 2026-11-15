"""Tests for the single-cell data module (data/scrna.py), on fake 10x tarballs, and the background-retention metric."""
import io
import tarfile

import numpy as np
import scipy.io
import scipy.sparse as sp

from data import scrna
from eval.metrics import background_retention, mean_pair_background_jaccard
from method.xi_b import background_neighbor_overlaps

N_GENES_FAKE = 40


def fake_tarball(path, counts, genes, barcodes):
    """A 10x v1 filtered-matrix tarball: genes x cells matrix.mtx, genes.tsv, barcodes.tsv."""
    def add(tar, name, data):
        info = tarfile.TarInfo(scrna.MEMBER.format(name))
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf = io.BytesIO()
    scipy.io.mmwrite(buf, sp.coo_matrix(counts.T))
    with tarfile.open(path, "w:gz") as tar:
        add(tar, "matrix.mtx", buf.getvalue())
        add(tar, "genes.tsv", "".join(f"ENSG{i}\tG{i}\n" for i in range(len(genes))).encode())
        add(tar, "barcodes.tsv", "".join(f"{b}\n" for b in barcodes).encode())


def make_raw(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    sizes = {"aml027_pre_transplant": 30, "aml027_post_transplant": 25, "frozen_bmmc_healthy_donor1": 20,
             "frozen_bmmc_healthy_donor2": 15}
    monkeypatch.setattr(scrna, "N_CELLS", sizes)
    monkeypatch.setattr(scrna, "N_GENES", 10)
    d = scrna.raw_dir(tmp_path)
    d.mkdir(parents=True)
    counts = {}
    for s, n in sizes.items():
        c = rng.poisson(rng.uniform(0.1, 5, size=N_GENES_FAKE), size=(n, N_GENES_FAKE))
        c[:, 7] = 0  # a gene never expressed: never selected
        c[:, 0] += 1  # no empty cells
        counts[s] = c
        fake_tarball(d / f"{s}.tar.gz", c, range(N_GENES_FAKE), [f"{s}-{i}" for i in range(n)])
    return counts


def test_build(tmp_path, monkeypatch):
    counts = make_raw(tmp_path, monkeypatch)
    out = scrna.build(tmp_path)
    assert out["X_T"].shape == (55, 10) and out["X_B"].shape == (20, 10) and out["X_B2"].shape == (15, 10)
    assert list(np.bincount(out["y"])) == [30, 25]
    assert 7 not in out["genes"] and 7 not in out["genes_paper"]
    pooled = np.vstack([out["X_T"], out["X_B"]])
    assert np.allclose(pooled.mean(axis=0), 0) and np.allclose(pooled.std(axis=0), 1)
    # the dense reference computation of the same preprocessing
    raw = {s: c.astype(float) for s, c in counts.items()}
    log = {s: np.log1p(1e4 * c / c.sum(axis=1, keepdims=True)) for s, c in raw.items()}
    T = np.vstack([log["aml027_pre_transplant"], log["aml027_post_transplant"]])
    P = np.vstack([T, log["frozen_bmmc_healthy_donor1"]])
    disp = np.where(P.mean(0) > 0, P.var(0) / np.where(P.mean(0) > 0, P.mean(0), 1), -np.inf)
    genes = np.sort(np.argsort(-disp, kind="stable")[:10])
    assert np.array_equal(genes, out["genes"])
    mu, sd = P[:, genes].mean(0), P[:, genes].std(0)
    assert np.allclose(out["X_T"], (T[:, genes] - mu) / sd)
    assert np.allclose(out["X_B2"], (log["frozen_bmmc_healthy_donor2"][:, genes] - mu) / sd)
    assert np.allclose(out["X_T_paper"].mean(0), 0) and np.allclose(out["X_B_paper"].mean(0), 0)


def test_background_retention():
    # 1-D: background pairs at 0 and at 100; targets two near each. With k_B = 2 each target's background
    # neighbors are its side's pair, so same-side targets have Jaccard 1 and cross-side targets 0.
    X_B = np.array([[0.0], [0.1], [100.0], [100.1]])
    X_T = np.array([[0.02], [0.05], [100.02], [100.05]])
    ov = background_neighbor_overlaps(X_T, X_B, 2)
    assert background_retention(X_T, ov, k_B=2, k=1) == 1.0  # each target's nearest target is on its side
    crossed = np.array([[0.0], [10.0], [0.01], [10.01]])  # embedding pairs each target with one from the other side
    assert background_retention(crossed, ov, k_B=2, k=1) == 0.0
    assert np.isclose(mean_pair_background_jaccard(ov, 2), 4 / 12)  # 4 of the 12 ordered pairs are same-side
