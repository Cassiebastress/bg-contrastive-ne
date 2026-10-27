"""Data geometry of the Phase 4 main datasets (no embeddings are fit): how far apart the background clusters are, and
how many target kNN edges connect different background clusters.

For every dataset used in Phase 4 (selection seeds 0-2 and the confirmation seeds actually used, per level) it records:
- the latent separation: minimum distance between background cluster centers in the 5-D latent (simplex edge);
- the data-space separation in X_T: distances between background-cluster centroids (min, mean), the closest pair of
  points from different clusters, and the median distance from a point to its 15th nearest neighbor (for scale);
- target kNN edges (exact k = 15 graph on X_T, union-symmetrized; undirected edges) joining different background
  clusters: count and fraction.
Ground-truth labels are used for measurement only.

Run with the cne-bg env:  python scripts/phase4_data_geometry.py
Outputs: results/phase4/data_geometry.csv, results/phase4/data_geometry.md
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist, pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.knn import knn_graph, knn_indices
from data.synthetic import LEVELS, MAIN_RHO, make_dataset

OUT = ROOT / "results" / "phase4"
K = 15
FIELDS = ["level", "seed", "latent_min_center_dist", "centroid_dist_min", "centroid_dist_mean", "closest_cross_pair",
          "median_kth_nn_dist", "n_edges", "n_cross_cluster_edges", "frac_cross_cluster_edges"]


def measure(level, seed):
    ds = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level])
    X, y = ds.X_T.astype(np.float64), ds.truth["y_b"]
    labels = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in labels])
    cdists = pdist(centroids)
    closest = min(cdist(X[y == a], X[y == b]).min() for i, a in enumerate(labels) for b in labels[i + 1:])
    kth = np.linalg.norm(X - X[knn_indices(X, K)[:, -1]], axis=1)
    G = sp.triu(knn_graph(ds.X_T, k=K), k=1).tocoo()  # each undirected edge once
    cross = int((y[G.row] != y[G.col]).sum())
    return {"level": level, "seed": seed, "latent_min_center_dist": ds.params["min_center_distance_b"],
            "centroid_dist_min": float(cdists.min()), "centroid_dist_mean": float(cdists.mean()),
            "closest_cross_pair": float(closest), "median_kth_nn_dist": float(np.median(kth)),
            "n_edges": int(G.nnz), "n_cross_cluster_edges": cross, "frac_cross_cluster_edges": cross / G.nnz}


def main():
    rows = []
    for level in LEVELS:
        conf = json.loads((OUT / f"summary_{level}_confirmation.json").read_text())
        for seed in [0, 1, 2] + list(conf["seeds"]):
            rows.append(measure(level, seed))
            r = rows[-1]
            print(f"{level} seed={seed} centroid min {r['centroid_dist_min']:.1f} closest pair "
                  f"{r['closest_cross_pair']:.1f} kth-NN {r['median_kth_nn_dist']:.2f} cross edges "
                  f"{r['n_cross_cluster_edges']}/{r['n_edges']}", flush=True)
    with open(OUT / "data_geometry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def rng(vals, d=1):
        lo, hi = (f"{v:.{d}f}" for v in (min(vals), max(vals)))
        return lo if lo == hi else f"{lo}–{hi}"

    md = ["| level | seeds | latent center distance | data-space centroid distance, min (mean) | closest points of "
          "different clusters | median distance to 15th neighbor | target kNN edges joining different clusters |",
          "|---|---|---|---|---|---|---|"]
    for level in LEVELS:
        lr = [r for r in rows if r["level"] == level]
        md.append(f"| {level} | {', '.join(str(r['seed']) for r in lr)} "
                  f"| {rng([r['latent_min_center_dist'] for r in lr])} "
                  f"| {rng([r['centroid_dist_min'] for r in lr])} ({rng([r['centroid_dist_mean'] for r in lr])}) "
                  f"| {rng([r['closest_cross_pair'] for r in lr])} | {rng([r['median_kth_nn_dist'] for r in lr], 2)} "
                  f"| {sum(r['n_cross_cluster_edges'] for r in lr)} of {sum(r['n_edges'] for r in lr)} |")
    (OUT / "data_geometry.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
