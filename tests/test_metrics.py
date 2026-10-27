"""Tests for the Phase 4 metrics (eval/metrics.py) and the pre-registered analysis rules (common/experiment.py)."""
import numpy as np

from common.experiment import M_BS, SELECTION_RUNS, confirmation_runs, paired, run_spec, select_settings, verdict
from eval.metrics import component_stats, distance_spearman, knn_recall, pair_distance_ranks

rng = np.random.default_rng(0)


def test_knn_recall():
    X = rng.standard_normal((200, 3))
    assert knn_recall(X, X, k=10) == 1.0
    assert knn_recall(X, 3.0 * X + 1.0, k=10) == 1.0  # similarity transforms keep neighbor sets
    line = np.arange(6, dtype=float)[:, None]  # neighbors on a line: point 0 -> {1, 2}
    shuffled = line[[0, 3, 4, 5, 1, 2]]  # rows reordered, so the second space disagrees on some neighbors
    r = knn_recall(line, shuffled, k=2)
    assert 0 <= r < 1


def test_distance_spearman():
    X = rng.standard_normal((150, 4))
    ref = pair_distance_ranks(X)
    assert np.isclose(distance_spearman(X, ref), 1.0)
    assert np.isclose(distance_spearman(2.5 * X, ref), 1.0)  # scaling is monotone in every distance
    assert abs(distance_spearman(rng.standard_normal((150, 4)), ref)) < 0.2


def test_component_stats():
    blobs = np.vstack([rng.standard_normal((100, 2)), 1000 + rng.standard_normal((100, 2)),
                       -1000 + rng.standard_normal((20, 2))])
    c = component_stats(blobs, k=15, small=50)
    assert c["n_components"] == 3 and np.isclose(c["frac_in_small_components"], 20 / 220)


def _rows(values, seeds=(0, 1, 2)):
    """Fake run rows: values[run] = (acc_s, recall_s, acc_b) or a per-seed list of such tuples."""
    rows = []
    for run, v in values.items():
        for k, seed in enumerate(seeds):
            acc_s, recall_s, acc_b = v[k] if isinstance(v, list) else v
            rows.append({"run": run, "seed": seed, "acc_s": acc_s, "recall_s": recall_s, "acc_b": acc_b,
                         "spearman_s": 0.0, "flag_collapse": False, "flag_fragment": False})
    return rows


def test_run_spec():
    assert run_spec("a") == {"bg_mode": "none"} and run_spec("d") == {"bg_mode": "none", "s": 0}
    assert run_spec("b_m10") == {"bg_mode": "uniform_extra", "m_B": 10}
    assert run_spec("c_m2") == {"bg_mode": "xi_b", "m_B": 2}
    assert len(SELECTION_RUNS) == 12


def test_select_settings_and_ties():
    vals = {run: (0.3, 0.1, 1.0) for run in SELECTION_RUNS}
    vals["c_m5"] = vals["c_m10"] = (0.6, 0.2, 0.9)  # tie in acc_s -> smaller m_B
    vals["b_m20"] = (0.5, 0.1, 1.0)  # best (b) by accuracy
    vals["b_m2"] = (0.4, 0.3, 1.0)  # best (b) by recall
    sel = select_settings(_rows(vals))
    assert sel["m_star_c"] == 5 and sel["best_b"] == {"acc_s": 20, "recall_s": 2}
    assert confirmation_runs(sel) == ["a", "d", "c_m5", "b_m5", "b_m20", "b_m2"]
    sel["best_b"] = {"acc_s": 5, "recall_s": 5}
    assert confirmation_runs(sel) == ["a", "d", "c_m5", "b_m5"]  # duplicates removed
    assert set(M_BS) == {1, 2, 5, 10, 20}


def test_paired_consistency():
    rows = _rows({"x": [(0.5, 0, 0), (0.6, 0, 0), (0.7, 0, 0)], "y": [(0.4, 0, 0), (0.5, 0, 0), (0.71, 0, 0)]})
    p = paired(rows, "x", "y", "acc_s")
    assert np.allclose(p["per_seed"], [0.1, 0.1, -0.01]) and p["mean_diff"] > 0 and not p["consistent"]
    rows = _rows({"x": (0.5, 0, 0.8), "y": (0.4, 0, 0.9)})
    assert paired(rows, "x", "y", "acc_s")["consistent"]
    assert paired(rows, "x", "y", "acc_b", higher_is_better=False)["consistent"]  # lower background accuracy


def test_verdict():
    sel = {"m_star_c": 5, "best_b": {"acc_s": 20, "recall_s": 2}}
    good = {"a": (0.3, 0.1, 1.0), "d": (0.35, 0.12, 1.0), "c_m5": (0.6, 0.3, 0.8), "b_m5": (0.4, 0.15, 1.0),
            "b_m20": (0.45, 0.16, 1.0), "b_m2": (0.35, 0.2, 1.0)}
    assert verdict(_rows(good, seeds=(3, 4, 5)), sel)["success"]
    matched = dict(good, b_m20=(0.6, 0.3, 0.8))  # the best control matches (c): not a success
    assert not verdict(_rows(matched, seeds=(3, 4, 5)), sel)["success"]
    rows = _rows(good, seeds=(3, 4, 5))
    for r in rows:
        if r["run"] == "c_m5":
            r["flag_fragment"] = True  # flagged runs fail the criterion but stay in the results
    v = verdict(rows, sel)
    assert v["c_flagged"] and not v["success"]
