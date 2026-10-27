"""Phase 4 experiment core (Plan §5): runs, scoring, health flags, and the pre-registered selection and comparison
rules.

Shared by scripts/phase4_run.py (main datasets: selection seeds 0-2, confirmation seeds 3-5), phase4_report.py and
phase4_stress.py. Ground truth (labels, S_T) is used only to score embeddings, never by the method.
"""
from dataclasses import dataclass

import numpy as np

from common.embed import N1, fit_neg, pca_init, stage2
from common.knn import knn_graph
from common.repro import setup_determinism
from data.synthetic import make_dataset
from eval.metrics import component_stats, distance_spearman, knn_accuracy, knn_recall, pair_distance_ranks
from method.bg_term import BackgroundTerm
from method.xi_b import build_xi_b

K_GRAPH = 15
M_BS = [1, 2, 5, 10, 20]
SELECTION_RUNS = ["a"] + [f"b_m{m}" for m in M_BS] + [f"c_m{m}" for m in M_BS] + ["d"]
PRIMARY = ("acc_s", "recall_s")
# pre-registered flags (Plan §5, rule 5); they mark runs for review and never drop them
COLLAPSE_STD_RATIO = 0.5
FRAGMENT_SMALL, FRAGMENT_FRAC = 50, 0.05
MAX_STD_GROWTH = 10.0  # Plan §6 stop condition
VALIDITY = {"min_emb_acc_b": 0.90, "max_emb_acc_s": 0.50, "min_oracle_acc_s": 0.95}  # Phase 1 criterion

ROW_FIELDS = ["level", "stage", "seed", "run", "group", "bg_mode", "m_B", "s", "acc_s", "acc_b", "recall_s",
              "spearman_s", "mean_phi", "sum_phi_est", "embd_std", "std_ratio_vs_a", "bg_share_final", "n_components",
              "frac_in_small_components", "flag_collapse", "flag_fragment", "nonfinite_batches", "max_std_growth",
              "runtime_s"]


def run_spec(label):
    """Stage 2 keyword arguments for a run label: a, d, b_m<m> (uniform_extra), c_m<m> (xi_b)."""
    if label == "a":
        return {"bg_mode": "none"}
    if label == "d":
        return {"bg_mode": "none", "s": 0}
    group, m = label.split("_m")
    return {"bg_mode": {"b": "uniform_extra", "c": "xi_b"}[group], "m_B": int(m)}


@dataclass
class Context:
    level: str
    seed: int
    ds: object
    G: object
    xi: object
    s_ranks: np.ndarray


def prepare(level, seed, rho, k_B=15, L=30):
    """Dataset, exact target kNN graph, xi_B, and the S_T distance ranks used by every run of this (level, seed)."""
    setup_determinism(seed)
    ds = make_dataset(level, seed=seed, signal_to_background=rho)
    return Context(level=level, seed=seed, ds=ds, G=knn_graph(ds.X_T, k=K_GRAPH),
                   xi=build_xi_b(ds.X_T, ds.X_B, k_B=k_B, L=L, seed=seed), s_ranks=pair_distance_ranks(ds.truth["S_T"]))


def score(X, ctx):
    """Signal/background kNN accuracy, signal kNN recall, signal distance Spearman, and kNN-graph components."""
    t = ctx.ds.truth
    return {"acc_s": knn_accuracy(X, t["y_s"]), "acc_b": knn_accuracy(X, t["y_b"]),
            "recall_s": knn_recall(X, t["S_T"], k=K_GRAPH), "spearman_s": distance_spearman(X, ctx.s_ranks),
            **component_stats(X, k=K_GRAPH, small=FRAGMENT_SMALL)}


def base_row(ctx, stage, run, group, **extra):
    row = dict.fromkeys(ROW_FIELDS, "")
    row.update({"level": ctx.level, "stage": stage, "seed": ctx.seed, "run": run, "group": group, **extra})
    return row


def reference_row(ctx, stage):
    """High-D X_T as a reference: where the signal stands before any embedding."""
    row = base_row(ctx, stage, "highD_X_T", "ref")
    row.update(score(ctx.ds.X_T, ctx))
    return row


def run_stage1(ctx):
    return fit_neg(pca_init(ctx.ds.X_T), ctx.G, ctx.seed, batch_hook=BackgroundTerm("none"), n_epochs=N1)


def run_stage2(ctx, stage1, label):
    return stage2(stage1["embd"], ctx.G, ctx.seed, xi=ctx.xi, **run_spec(label))


def fit_row(ctx, stage, run, group, fit, a_std=None):
    """Score a fit and record its scale, health and review flags."""
    embd, scale = fit["embd"], fit["scale"]
    std = np.array(scale["embd_std"])
    log = fit.get("log", {})
    bg = fit["params"].get("background", {})
    row = base_row(ctx, stage, run, group, bg_mode=bg.get("bg_mode", ""), m_B=bg.get("m_B", ""),
                   s="" if fit["params"]["s"] is None else fit["params"]["s"])
    row.update(score(embd, ctx))
    row.update({
        "mean_phi": scale["mean_phi"][-1], "sum_phi_est": scale["sum_phi_est"][-1], "embd_std": float(std[-1]),
        "bg_share_final": log["bg_share"][-1] if log else "",
        "nonfinite_batches": int(sum(log.get("n_nonfinite_batches", [0]))) + int(not all(scale["finite"])),
        "max_std_growth": float(std.max() / std[0]), "runtime_s": fit["runtime_s"],
        "flag_fragment": row["frac_in_small_components"] > FRAGMENT_FRAC,
    })
    if a_std is not None:
        row["std_ratio_vs_a"] = float(std[-1] / a_std)
        row["flag_collapse"] = row["std_ratio_vs_a"] < COLLAPSE_STD_RATIO
    return row


def healthy(row):
    """Plan §6: finite throughout and no exploding scale."""
    return row["nonfinite_batches"] == 0 and row["max_std_growth"] <= MAX_STD_GROWTH


def epoch_arrays(fit):
    """Per-epoch logs of a fit as flat numpy arrays (for the gitignored .npz)."""
    out = {f"scale_{k}": np.asarray(v) for k, v in fit["scale"].items()}
    out.update({f"log_{k}": np.asarray(v) for k, v in fit.get("log", {}).items()})
    return out


# ---- pre-registered analysis (Plan §5, rules 2-4) ----

def _mean(rows, run, metric):
    vals = [float(r[metric]) for r in rows if r["run"] == run]
    return float(np.mean(vals)) if vals else float("nan")


def select_settings(rows):
    """Rule 3 on the selection seeds of one level: m*_c = argmax mean acc_s over (c); best (b) per primary metric.
    Ties go to the smaller m_B."""
    means = {run: {m: _mean(rows, run, m) for m in ("acc_s", "acc_b", "recall_s", "spearman_s")}
             for run in SELECTION_RUNS}

    def best(group, metric):
        return max(M_BS, key=lambda m: (means[f"{group}_m{m}"][metric], -m))

    return {"m_star_c": best("c", "acc_s"), "best_b": {metric: best("b", metric) for metric in PRIMARY},
            "rule": "highest mean over selection seeds 0-2 (acc_s for c; per primary metric for b); "
                    "ties -> smaller m_B",
            "means": means}


def confirmation_runs(selection):
    """(a), (d), (c) at m*_c, (b) at m*_c, and (b) at each per-metric best m_B, without duplicates."""
    m = selection["m_star_c"]
    labels = ["a", "d", f"c_m{m}", f"b_m{m}"] + [f"b_m{selection['best_b'][k]}" for k in PRIMARY]
    return list(dict.fromkeys(labels))


def paired(rows, x, y, metric, higher_is_better=True):
    """Rule 2: per-seed paired differences x - y (sign-flipped if lower is better). 'consistent' means the mean
    difference is > 0 and every per-seed difference is > 0."""
    by_seed = {}
    for r in rows:
        if r["run"] in (x, y):
            by_seed.setdefault(int(r["seed"]), {})[r["run"]] = float(r[metric])
    seeds = sorted(s for s, v in by_seed.items() if x in v and y in v)
    sign = 1.0 if higher_is_better else -1.0
    diffs = [sign * (by_seed[s][x] - by_seed[s][y]) for s in seeds]
    mean = float(np.mean(diffs)) if diffs else float("nan")
    return {"x": x, "y": y, "metric": metric, "higher_is_better": higher_is_better, "seeds": seeds,
            "per_seed": diffs, "mean_diff": mean, "consistent": bool(diffs) and mean > 0 and all(d > 0 for d in diffs)}


def verdict(rows, selection):
    """Rule 4 on the confirmation seeds of one level: (c) at m*_c must improve both primary metrics over (a), (b) at
    m*_c, the per-metric best (b), and (d); background accuracy must drop vs (a); (c) runs must not be flagged."""
    c = f"c_m{selection['m_star_c']}"
    comparisons = []
    for metric in PRIMARY:
        for y in dict.fromkeys(["a", f"b_m{selection['m_star_c']}", f"b_m{selection['best_b'][metric]}", "d"]):
            comparisons.append(paired(rows, c, y, metric))
    comparisons.append(paired(rows, c, "a", "acc_b", higher_is_better=False))
    c_rows = [r for r in rows if r["run"] == c]
    flagged = any(str(r["flag_collapse"]) == "True" or str(r["flag_fragment"]) == "True" for r in c_rows)
    return {"c_run": c, "comparisons": comparisons, "c_flagged": flagged,
            "success": all(cmp["consistent"] for cmp in comparisons) and not flagged}
