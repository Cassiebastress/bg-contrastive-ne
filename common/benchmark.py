"""Phase 5a core: the J and K datasets, scoring, the cPCA runs (e2) and (e3), and the validity criteria V1-V4.

Shared by scripts/phase5a_calibrate.py, phase5a_run.py and phase5a_report.py. Ground truth (labels, signal_T) is used
only to score embeddings, calibrate rho and define the oracle controls, never by the method.
"""
import numpy as np

from baselines import cpca
from baselines.cpca_embed import embed_e2, embed_e3
from common.embed import pca_init, two_stage
from common.experiment import VALIDITY
from common.knn import knn_graph
from common.repro import setup_determinism
from data.synthetic_j import GAINS, PROJECTIONS, SIGNAL_TOTAL_VAR, THETAS, make_j
from data.synthetic_k import make_k
from eval.benchmark_metrics import coherence, within_cluster_accuracy
from eval.global_metrics import global_scores
from eval.metrics import knn_accuracy

DESIGNS = ("J", "K")
EXTRA_COMPONENTS = {"J": (), "K": (20, 40)}  # (e2-k20), (e2-k40): K only, descriptive in calibration
PHASE1_RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
# J's signal has unit total variance, so its rho grid is Phase 1's in units of the std of A z_s (sqrt 16.5)
RHO_GRID = {"J": [float(r * np.sqrt(SIGNAL_TOTAL_VAR)) for r in PHASE1_RHOS], "K": PHASE1_RHOS}
GRID = [{"theta": t, "projected": p, "gain": g} for t in THETAS for p in PROJECTIONS for g in GAINS]
SELECTION_SEEDS = (0, 1, 2)

V1 = {"max_acc_s": VALIDITY["max_emb_acc_s"], "min_acc_b": VALIDITY["min_emb_acc_b"]}
V2_MIN_DETECTABILITY = 0.85
V3_J = {"G1": 0.50, "G2": 0.50, "G3_shares_signal": 0.50, "acc_s": 0.80}  # frozen thresholds (Phase 5 prep)
V3_K_MAX_WC = 0.80
V4_MIN_COHERENCE = 0.5
METRICS = ["acc_s", "acc_b", "WC", "G1", "G2", "G3_shares_signal", "G3_shares_background", "G3_neither"]


def config_name(c):
    return f"t{c['theta']}_{'proj' if c['projected'] else 'unproj'}_g{c['gain']:g}"


def make(design, seed, rho, config=None):
    """One dataset. Call setup_determinism(seed) first if anything downstream draws from global RNGs."""
    if design == "J":
        return make_j(seed=seed, rho=rho, **(config or GRID[0]))
    return make_k(seed=seed, rho=rho)


def score(X, truth):
    t = truth
    g = global_scores(X, t["y_s"], t["y_b"])
    g3 = {k: g[k] for k in ("G3_shares_signal", "G3_shares_background", "G3_neither")}
    return {"acc_s": knn_accuracy(X, t["y_s"]), "acc_b": g["G2"], "WC": within_cluster_accuracy(X, t["y_s"], t["y_b"]),
            "G1": g["G1"], "G2": g["G2"], **g3}


def data_scores(ds):
    """High-D diagnostics of X_T: signal detectability (WC), coherence, and plain kNN accuracies."""
    t = ds.truth
    return {"detectability": within_cluster_accuracy(ds.X_T, t["y_s"], t["y_b"]),
            "coherence": coherence(ds.X_T, t["y_s"], t["y_b"]),
            "highd_acc_s": knn_accuracy(ds.X_T, t["y_s"]), "highd_acc_b": knn_accuracy(ds.X_T, t["y_b"])}


def fit_a(ds, seed):
    """Run (a): the two-stage Neg-t-SNE baseline on X_T (as in Phase 1)."""
    setup_determinism(seed)
    return two_stage(pca_init(ds.X_T), knn_graph(ds.X_T, k=15), seed)[1]["embd"]


def cpca_runs(ds, seed, extra_components=()):
    """(e2) and (e3) at the label-free alpha, plus the descriptive oracle alpha (10-D signal accuracy for J, 10-D WC
    for K; ties -> the smaller alpha). Each n in extra_components adds (e2-k<n>): cPCA -> Neg-t-SNE on the top n
    components at the same alpha, run after (e2) and (e3) so they are unchanged."""
    t = ds.truth
    f = cpca.fit(ds.X_T, ds.X_B)
    chosen, reps, ratios, _ = cpca.select_alpha(f, seed)
    if ds.params["design"] == "J":
        crit = [knn_accuracy(cpca.project(f, a, 10), t["y_s"]) for a in cpca.ALPHAS]
    else:
        crit = [within_cluster_accuracy(cpca.project(f, a, 10), t["y_s"], t["y_b"]) for a in cpca.ALPHAS]
    Z10 = cpca.project(f, chosen, 10)
    setup_determinism(seed)
    e2 = embed_e2(Z10, seed)
    e3 = embed_e3(Z10, seed)
    extra = {}
    for n in extra_components:
        setup_determinism(seed)
        extra[f"e2_k{n}"] = embed_e2(cpca.project(f, chosen, n), seed)
    alpha = {"selected_alpha": float(chosen), "representatives": [float(a) for a in reps],
             "variance_ratios": [float(r) for r in ratios], "oracle_alpha": float(cpca.ALPHAS[int(np.argmax(crit))]),
             "oracle_10d": float(max(crit)), "selected_10d": float(crit[int(np.flatnonzero(cpca.ALPHAS == chosen)[0])]),
             "oracle_criterion": "10-D signal accuracy" if ds.params["design"] == "J" else "10-D WC"}
    return {"e2": e2, "e3": e3, **extra}, alpha


def v1(a):
    return a["acc_s"] <= V1["max_acc_s"] and a["acc_b"] >= V1["min_acc_b"]


def v2(data):
    return data["detectability"] >= V2_MIN_DETECTABILITY


def v3_j(e2):
    """True when (e2) fails at least one frozen threshold, i.e. cPCA -> Neg-t-SNE does not solve the dataset."""
    solved = (e2["G1"] >= V3_J["G1"] and e2["G2"] <= V3_J["G2"]
              and e2["G3_shares_signal"] >= V3_J["G3_shares_signal"] and e2["acc_s"] > V3_J["acc_s"])
    return not solved


def v3_k(e2):
    return e2["WC"] <= V3_K_MAX_WC


def v4(data):
    return data["coherence"] >= V4_MIN_COHERENCE
