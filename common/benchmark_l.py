"""Phase 5b core: Design L scoring, the validity constants, and the linear and cPCA ceilings.

Ceilings are judged at the embedding level: each labeled linear projection is embedded with (a)'s two-stage
Neg-t-SNE (exact kNN graph and PCA init in the projected space, seed = dataset seed) and scored by WC. The high-D kNN
WC of the same projection (no embedding) is reported alongside as descriptive. Labels choose the projections, so
the ceilings are conservative upper bounds for "linear projection -> standard Neg-t-SNE", not fair baselines.
"""
import numpy as np

from baselines import cpca
from baselines.cpca_embed import embed_e2
from common.repro import setup_determinism
from data.synthetic_l import N_BLOCKS
from eval.benchmark_metrics import g1_k200, per_cluster_wc, within_cluster_accuracy
from eval.global_metrics import global_scores
from eval.metrics import knn_accuracy

K_LIST = (3, 5, 10, 20)
WC_BAR = 0.80  # V3: "linear methods don't solve it"; also the per-cluster "revealed" bar
V2_MIN_DETECTABILITY = 0.85
V2B_MIN_GRAPH_SIGNAL = 0.65
XI_GATES = {"min_same_b": 0.90, "max_same_s_excess": 0.10}  # Phase 2 gates


def score(X, truth):
    """WC (primary), per-cluster WC, signal and background accuracy, and the descriptive global metrics."""
    t = truth
    accs, _ = per_cluster_wc(X, t["y_s"], t["y_b"])
    g = global_scores(X, t["y_s"], t["y_b"])
    return {"WC": within_cluster_accuracy(X, t["y_s"], t["y_b"]), "acc_s": knn_accuracy(X, t["y_s"]),
            "acc_b": g["G2"], "G1": g["G1"], "G1_k200": g1_k200(X, t["y_s"], t["y_b"]), "G2": g["G2"],
            "G3_shares_signal": g["G3_shares_signal"], **{f"WC_c{i}": float(a) for i, a in enumerate(accs)}}


def wc_only(X, truth):
    accs, sizes = per_cluster_wc(X, truth["y_s"], truth["y_b"])
    return float((sizes * accs).sum() / sizes.sum()), accs


def embed(Z, seed):
    """(a)'s two-stage Neg-t-SNE on a projected representation (the (e2) embedder), seeded for reproducibility."""
    setup_determinism(seed)
    return embed_e2(np.ascontiguousarray(Z), seed)


def clean_reveals(subset):
    """L-cyclic: clusters k whose signal block Q_k is in the subset while their nuisance block Q_{k+1} is not."""
    s = set(subset)
    return sum(k in s and (k + 1) % N_BLOCKS not in s for k in range(N_BLOCKS))


def projections(ds):
    """Labeled linear projections: (name, subset, orthonormal basis). L-cyclic: U with each of the 64 subsets of the
    Q blocks. L-between: U alone, and U with V."""
    t = ds.truth
    if ds.params["variant"] == "cyclic":
        for mask in range(2 ** N_BLOCKS):
            subset = [i for i in range(N_BLOCKS) if mask >> i & 1]
            yield ("U+Q" + "".join(map(str, subset)), subset, np.hstack([t["B"]] + [t["Q"][i] for i in subset]))
    else:
        yield "U", [], t["B"]
        yield "U+V", [], np.hstack([t["B"], t["V"]])


def linear_ceilings(ds, seed):
    """Every labeled projection: high-D kNN WC (descriptive) and embedding-level WC with per-cluster values.
    Returns rows and the embeddings by name."""
    rows, embs = [], {}
    for name, subset, basis in projections(ds):
        Z = ds.X_T.astype(np.float64) @ basis
        hd, _ = wc_only(Z, ds.truth)
        E = embed(Z, seed)
        wc, accs = wc_only(E, ds.truth)
        rows.append({"kind": "linear", "name": name, "n_blocks": len(subset),
                     "clean_reveals": clean_reveals(subset) if ds.params["variant"] == "cyclic" else "",
                     "alpha": "", "k": basis.shape[1], "highd_WC": hd, "emb_WC": wc,
                     **{f"emb_WC_c{i}": float(a) for i, a in enumerate(accs)}})
        embs[name] = E
    return rows, embs


def cpca_variants(ds, seed):
    """cPCA projections: every alpha representative x k in K_LIST, plus for each k the alpha on the full grid with the
    highest high-D WC (labels; ties -> smaller alpha). Returns (variants, label-free alpha info); each variant has a
    label, alpha, k, source, the projection Z and its high-D WC."""
    f = cpca.fit(ds.X_T, ds.X_B)
    chosen, reps, ratios, _ = cpca.select_alpha(f, seed)
    variants, seen = [], set()
    for a in reps:
        for k in K_LIST:
            Z = cpca.project(f, a, k)
            variants.append({"label": f"cpca_a{a:.4g}_k{k}", "alpha": float(a), "k": k, "source": "representative",
                             "Z": Z, "highd_WC": wc_only(Z, ds.truth)[0]})
            seen.add((float(a), k))
    for k in K_LIST:
        hd = [wc_only(cpca.project(f, a, k), ds.truth)[0] for a in cpca.ALPHAS]
        a = float(cpca.ALPHAS[int(np.argmax(hd))])
        if (a, k) not in seen:
            variants.append({"label": f"cpca_or_a{a:.4g}_k{k}", "alpha": a, "k": k, "source": "oracle alpha for k",
                             "Z": cpca.project(f, a, k), "highd_WC": max(hd)})
        else:
            next(v for v in variants if v["alpha"] == a and v["k"] == k)["source"] += "; oracle alpha for k"
    info = {"selected_alpha": float(chosen), "representatives": [float(a) for a in reps],
            "variance_ratios": [float(r) for r in ratios]}
    return variants, info


def cpca_ceilings(ds, seed):
    """Embed every cPCA variant; rows with high-D and embedding-level WC (per cluster), and the embeddings."""
    variants, info = cpca_variants(ds, seed)
    rows, embs = [], {}
    for v in variants:
        E = embed(v["Z"], seed)
        wc, accs = wc_only(E, ds.truth)
        rows.append({"kind": "cpca", "name": v["label"], "n_blocks": "", "clean_reveals": "", "alpha": v["alpha"],
                     "k": v["k"], "source": v["source"], "highd_WC": v["highd_WC"], "emb_WC": wc,
                     **{f"emb_WC_c{i}": float(a) for i, a in enumerate(accs)}})
        embs[v["label"]] = E
    return rows, embs, info
