"""MNIST-on-grass Phase D: descriptive pre-checks on selection seeds 0-2 (no embedding is scored).

  --seed S          digit kNN accuracy (k = 15, stratified 5-fold) on raw X_T, on the clean digits and on PCA-50 of
                    X_T; xi_B on raw 784-D pixels (k_B = 15, L = 30): same-digit excess over chance, grass-patch
                    distance of xi_B pairs vs random target pairs vs each target's 15 grass-nearest targets, list
                    sizes, empty rows and the tie cap. Writes results/mnist_grass/prechecks/seed{S}.json.
  --tsne-check      openTSNE determinism (decision 3): T1 (library defaults, random_state = 0) on seed 0, twice;
                    the two embeddings must be bit-identical, else stop. Only hashes are kept, no scores.
  --summarize       prechecks.json and the generated table prechecks.md.

Run on the cluster (job: cluster/mnist_grass_build.sbatch). Pre-checks are descriptive: nothing here stops the phase
except the openTSNE determinism check.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "mnist_grass" / "prechecks"
SEEDS = (0, 1, 2)
K, K_B, L = 15, 15, 30
GATE_SAME_DIGIT_EXCESS = 0.10  # Phase 2 gate, shown for reference only


def load(seed):
    from data import mnist_grass as mg
    return mg.for_embedding(mg.load(mg.processed_path(seed, ROOT)))


def precheck(seed):
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    from common.knn import knn_graph, knn_indices
    from common.repro import environment_info, setup_determinism
    from eval.metrics import knn_accuracy, pair_label_agreement, uniform_pair_agreement
    from eval.xi_b_diag import weighted_row_mean
    from method.xi_b import build_xi_b

    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    y, grass = t["y"], t["grass_T"]
    rec = {"seed": seed, "digit_acc_raw": knn_accuracy(X_T, y), "digit_acc_clean": knn_accuracy(t["digits"], y),
           "digit_acc_pca50": knn_accuracy(PCA(50, svd_solver="full").fit_transform(X_T), y)}
    G = knn_graph(X_T, k=K)
    xi = build_xi_b(X_T, X_B, k_B=K_B, L=L, seed=seed)
    deg = np.diff(G.indptr).astype(float)
    P = xi.to_csr()
    heads = np.repeat(np.arange(xi.n), xi.row_lengths())
    same = pair_label_agreement(P, y, head_weights=deg)
    chance = uniform_pair_agreement(y)
    lengths = xi.row_lengths()
    nn = knn_indices(grass, K)
    rec.update({"xi_same_digit": same, "same_digit_chance": chance, "xi_same_digit_excess": same - chance,
                "gate_reference": GATE_SAME_DIGIT_EXCESS,
                "grass_dist_xi_pairs": weighted_row_mean(P, np.linalg.norm(grass[heads] - grass[xi.indices], axis=1),
                                                         deg),
                "grass_dist_random_pairs": float(pdist(grass).mean()),
                "grass_dist_grass_knn": float(np.linalg.norm(grass[:, None, :] - grass[nn], axis=2).mean()),
                "len_min": int(lengths.min()), "len_median": float(np.median(lengths)),
                "len_p90": float(np.percentile(lengths, 90)), "len_max": int(lengths.max()),
                "n_empty_rows": xi.params["n_empty_rows"],
                "frac_tie_split_rows": xi.params["n_tie_split_rows"] / xi.n,
                "frac_capped_rows": xi.params["n_capped_rows"] / xi.n, "xi_sha256": xi.sha256(),
                "environment": environment_info()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{seed}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}), flush=True)


def tsne_check():
    from openTSNE import TSNE

    from common.repro import environment_info, setup_determinism
    setup_determinism(0)
    X_T, _, _ = load(0)
    embs = [np.asarray(TSNE(random_state=0).fit(X_T)) for _ in range(2)]
    rec = {"identical": bool(np.array_equal(embs[0], embs[1])),
           "sha256": [hashlib.sha256(e.tobytes()).hexdigest() for e in embs], "environment": environment_info()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tsne_determinism.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}), flush=True)
    if not rec["identical"]:
        print("STOP (MNIST-on-grass): openTSNE with random_state is not deterministic on this node", flush=True)
        sys.exit(2)


def summarize():
    recs = [json.loads((OUT / f"seed{s}.json").read_text()) for s in SEEDS]
    tsne = json.loads((OUT / "tsne_determinism.json").read_text())
    md = ["# MNIST-on-grass pre-checks (selection seeds 0–2; descriptive)", "",
          "## Is the digit signal in the kNN graph?", "",
          "Digit kNN accuracy (k = 15, stratified 5-fold).", "",
          "| seed | raw X_T (784-D) | clean digits | PCA-50 of X_T |", "|---|---|---|---|"]
    md += [f"| {r['seed']} | {r['digit_acc_raw']:.3f} | {r['digit_acc_clean']:.3f} | {r['digit_acc_pca50']:.3f} |"
           for r in recs]
    md += ["", "## ξ_B on raw pixels (k_B = 15, L = 30)", "",
           f"Same-digit excess of ξ_B pairs over chance (Phase 2 gate value {GATE_SAME_DIGIT_EXCESS} shown for "
           "reference; not a stop). Grass distance: mean Euclidean distance between the underlying grass patches, "
           "for ξ_B pairs (heads weighted by target-graph degree), all target pairs, and each target's 15 nearest "
           "targets by grass patch.", "",
           "| seed | same digit, ξ_B | chance | excess | grass dist, ξ_B pairs | grass dist, random pairs "
           "| grass dist, grass-kNN | list length min / median / p90 / max | empty rows | tie-split rows "
           "| capped rows |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['seed']} | {r['xi_same_digit']:.3f} | {r['same_digit_chance']:.3f} | "
           f"{r['xi_same_digit_excess']:+.3f} | {r['grass_dist_xi_pairs']:.2f} | {r['grass_dist_random_pairs']:.2f} "
           f"| {r['grass_dist_grass_knn']:.2f} | {r['len_min']} / {r['len_median']:.0f} / {r['len_p90']:.0f} / "
           f"{r['len_max']} | {r['n_empty_rows']} | {r['frac_tie_split_rows']:.3f} | {r['frac_capped_rows']:.3f} |"
           for r in recs]
    md += ["", "## openTSNE determinism", "",
           f"T1 (library defaults, random_state = 0) on seed 0, run twice: "
           f"**{'bit-identical' if tsne['identical'] else 'NOT identical'}**."]
    (ROOT / "results" / "mnist_grass" / "prechecks.md").write_text("\n".join(md) + "\n")
    (ROOT / "results" / "mnist_grass" / "prechecks.json").write_text(
        json.dumps({"seeds": recs, "tsne_determinism": tsne}, indent=2))
    print("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tsne-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        return summarize()
    if args.tsne_check:
        return tsne_check()
    precheck(args.seed)


if __name__ == "__main__":
    main()
