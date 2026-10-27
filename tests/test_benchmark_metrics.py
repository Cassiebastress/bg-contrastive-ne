"""Tests for the Phase 5a metrics (eval/benchmark_metrics.py)."""
import numpy as np

from data.synthetic import make_dataset
from data.synthetic_k import make_k
from eval.benchmark_metrics import coherence, within_cluster_accuracy

rng = np.random.default_rng(0)


def _cells(n_per=60, n_b=3, n_s=4):
    y_b = np.repeat(np.arange(n_b), n_s * n_per)
    y_s = np.tile(np.repeat(np.arange(n_s), n_per), n_b)
    return y_s, y_b


def test_coherence_shared_vs_cluster_specific():
    y_s, y_b = _cells()
    shared = 5.0 * np.eye(8)[y_s] + 50.0 * np.eye(8)[4 + y_b % 4] + 0.1 * rng.standard_normal((len(y_s), 8))
    assert coherence(shared, y_s, y_b) > 0.99
    # each cluster maps the groups to its own orthogonal directions: offsets orthogonal across clusters
    specific = np.zeros((len(y_s), 3 * 4))
    specific[np.arange(len(y_s)), 4 * y_b + y_s] = 5.0
    assert abs(coherence(specific, y_s, y_b)) < 1e-12  # disjoint supports: every cosine is exactly 0


def test_within_cluster_accuracy_ignores_global_layout():
    y_s, y_b = _cells()
    # within each cluster the groups separate, but group positions are shuffled between clusters and clusters are far
    perm = np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2]])
    X = np.column_stack([10.0 * perm[y_b, y_s], 1000.0 * y_b]) + 0.5 * rng.standard_normal((len(y_s), 2))
    assert within_cluster_accuracy(X, y_s, y_b) > 0.99
    noise = rng.standard_normal((len(y_s), 2))
    assert within_cluster_accuracy(noise, y_s, y_b) < 0.45


def test_coherence_on_generators():
    easy = make_dataset("easy", n_T=3000, n_B=10, seed=0, signal_to_background=0.2)
    k = make_k(seed=0, rho=0.2, n_T=3000, n_B=10)
    assert coherence(easy.X_T, easy.truth["y_s"], easy.truth["y_b"]) > 0.8
    assert abs(coherence(k.X_T, k.truth["y_s"], k.truth["y_b"])) < 0.2
