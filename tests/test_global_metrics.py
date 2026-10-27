"""Tests for the global layout metrics (eval/global_metrics.py) on hand-built layouts."""
import numpy as np

from eval.global_metrics import cell_layout, cross_cluster_same_signal_rate, g1_chance, global_scores

rng = np.random.default_rng(0)
N_PER_CELL = 40
S, B = 4, 6


def _labels():
    y_s = np.repeat(np.tile(np.arange(S), B), N_PER_CELL)
    y_b = np.repeat(np.repeat(np.arange(B), S), N_PER_CELL)
    return y_s, y_b


def _layout(organize_by):
    """'background': islands per background cluster (100 apart), each split by signal (5 apart), like Phase 4 (c).
    'signal': one blob per signal group (100 apart), background ignored (mixed) inside each blob."""
    y_s, y_b = _labels()
    if organize_by == "background":
        pos = 100.0 * y_b + 5.0 * y_s
    else:
        pos = 100.0 * y_s
    X = np.stack([pos, np.zeros(len(pos))], axis=1) + rng.normal(0, 0.3, (len(pos), 2))
    return X, y_s, y_b


def test_background_islands():
    X, y_s, y_b = _layout("background")
    assert cross_cluster_same_signal_rate(X, y_s, y_b) == 0.0
    g3 = cell_layout(X, y_s, y_b)
    assert g3["G3_shares_background"] == 1.0 and g3["n_cells"] == S * B


def test_signal_organized():
    X, y_s, y_b = _layout("signal")
    g1 = cross_cluster_same_signal_rate(X, y_s, y_b)
    assert g1 > 0.75  # neighbors share the signal group and mostly come from other background clusters
    assert cell_layout(X, y_s, y_b)["G3_shares_signal"] == 1.0
    scores = global_scores(X, y_s, y_b)
    assert scores["G2"] < 0.5  # background no longer predictable from neighbors


def test_chance_matches_monte_carlo():
    y_s, y_b = rng.integers(S, size=3000), rng.integers(B, size=3000)
    i, j = rng.integers(3000, size=(2, 400_000))
    keep = i != j
    mc = np.mean((y_s[i] == y_s[j]) & (y_b[i] != y_b[j]) & keep) / keep.mean()
    assert abs(g1_chance(y_s, y_b) - mc) < 0.005
    assert abs(g1_chance(y_s, y_b) - 0.25 * 5 / 6) < 0.01


def test_g3_tie_goes_to_lower_cell_index():
    # three cells at 0, 1, -1 on a line: cell (0,0) is equidistant from (0,1) [index 1] and (1,0) [index 2]
    y_s = np.array([0, 0, 1])
    y_b = np.array([0, 1, 0])
    X = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    first = cell_layout(X, y_s, y_b)
    assert first == cell_layout(X, y_s, y_b)
    # cell 0 -> nearest is cell (0,1) (lower index): shares signal; cells 1 and 2 -> cell 0: signal / background
    assert np.isclose(first["G3_shares_signal"], 2 / 3) and np.isclose(first["G3_shares_background"], 1 / 3)
