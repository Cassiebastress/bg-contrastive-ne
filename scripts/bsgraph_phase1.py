"""Background-suppressed graph, Phase 1: graph quality on MNIST-on-grass (results/bsgraph/preregistration_phase1.md).

  --seed S --k-loc K    residual graphs for every r in {0, 1, 2, 5, 10} at this k_loc -> phase1/seed{S}_k{K}.json
  --seed S --reference  raw X_T, PCA-50 of X_T and (descriptive) cPCA-10 at the label-free alpha -> seed{S}_ref.json
  --make-jobs           cluster/jobs/bsgraph_phase1.txt (6 seeds x (3 k_loc + reference) = 24 processes)
  --merge               variant selection and the gate on seeds 0-2, confirmation-seed metrics (descriptive):
                        results/bsgraph/phase1.json and the generated phase1.md

Graph metrics, in the space the graph is built on: digit kNN accuracy (primary; k = 15, stratified 5-fold), edge
purity of the symmetrized kNN graph, kNN recall at 15 vs the grass patches (lower is better) and vs the clean digits.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402

OUT = ROOT / "results" / "bsgraph"
PREREG = OUT / "preregistration_phase1.md"
SEEDS, SELECTION_SEEDS = range(6), (0, 1, 2)
K_LOCS, RS = (15, 30, 100), (0, 1, 2, 5, 10)
K = 15
GATE = 0.02
TIE = 0.002


def stop(msg):
    print(f"STOP (background-suppressed graph): {msg}", flush=True)
    sys.exit(2)


def load(seed):
    """The MNIST-on-grass dataset, after checking its SHA-256 against data/MANIFEST.md."""
    from data import mnist_grass as mg
    path = mg.processed_path(seed, ROOT)
    want = re.search(rf"`{path.name}` \| `([0-9a-f]{{64}})`", (ROOT / "data" / "MANIFEST.md").read_text())
    if want is None or hashlib.sha256(path.read_bytes()).hexdigest() != want.group(1):
        stop(f"{path.name} does not match its SHA-256 in data/MANIFEST.md")
    return mg.for_embedding(mg.load(path))


def graph_metrics(space, t):
    from common.knn import knn_graph
    from eval.metrics import knn_accuracy, knn_recall
    G = knn_graph(space, k=K).tocoo()
    return {"graph_digit_acc": knn_accuracy(space, t["y"]),
            "edge_purity": float(np.mean(t["y"][G.row] == t["y"][G.col])),
            "graph_grass_recall": knn_recall(space, t["grass_T"], k=K),
            "graph_digit_recall": knn_recall(space, t["digits"], k=K)}


def write(name, rec):
    (OUT / "phase1").mkdir(parents=True, exist_ok=True)
    (OUT / "phase1" / f"{name}.json").write_text(json.dumps({**rec, "environment": environment_info()}, indent=2))
    print(json.dumps(rec), flush=True)


def residual_part(seed, k_loc):
    from method.residual_graph import local_background, residuals
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    local = local_background(X_T, X_B, k_loc)
    rows = [{"variant": f"k{k_loc}_r{r}", "k_loc": k_loc, "r": r, **graph_metrics(residuals(X_T, local, r), t)}
            for r in RS]
    write(f"seed{seed}_k{k_loc}", {"seed": seed, "rows": rows})


def reference_part(seed):
    from sklearn.decomposition import PCA

    from baselines import cpca
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    f = cpca.fit(X_T, X_B)
    alpha = cpca.select_alpha(f, seed)[0]
    spaces = {"raw": X_T, "pca50": PCA(50, svd_solver="full").fit_transform(X_T), "cpca10": cpca.project(f, alpha, 10)}
    rows = [{"variant": name, **graph_metrics(Z, t)} for name, Z in spaces.items()]
    write(f"seed{seed}_ref", {"seed": seed, "rows": rows, "cpca_alpha": alpha})


def make_jobs():
    lines = [f"$PY scripts/bsgraph_phase1.py --seed {s} --k-loc {k}" for s in SEEDS for k in K_LOCS]
    lines += [f"$PY scripts/bsgraph_phase1.py --seed {s} --reference" for s in SEEDS]
    path = ROOT / "cluster" / "jobs" / "bsgraph_phase1.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


METRICS = [("graph_digit_acc", "graph digit accuracy"), ("edge_purity", "edge purity"),
           ("graph_grass_recall", "grass recall (lower is better)"), ("graph_digit_recall", "clean-digit recall")]


def table(by, names, seeds):
    out = ["| graph | " + " | ".join(n for _, n in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name in names:
        out.append(f"| {name} | " + " | ".join(fmt([by[s][name][m] for s in seeds]) for m, _ in METRICS) + " |")
    return "\n".join(out)


def merge():
    by = {s: {} for s in SEEDS}
    for p in sorted((OUT / "phase1").glob("seed*.json")):
        rec = json.loads(p.read_text())
        for row in rec["rows"]:
            by[rec["seed"]][row["variant"]] = row
    variants = [f"k{k}_r{r}" for k in K_LOCS for r in RS]
    mean = {v: float(np.mean([by[s][v]["graph_digit_acc"] for s in SELECTION_SEEDS])) for v in variants}
    top = max(mean.values())
    tied = [v for v in variants if mean[v] >= top - TIE]
    chosen = min(tied, key=lambda v: (int(v.split("_r")[1]), int(v[1:].split("_")[0])))
    gate = {s: {"selected": by[s][chosen]["graph_digit_acc"],
                "best_reference": max(by[s]["raw"]["graph_digit_acc"], by[s]["pca50"]["graph_digit_acc"])}
            for s in SELECTION_SEEDS}
    for g in gate.values():
        g["margin"] = g["selected"] - g["best_reference"]
        g["passes"] = g["margin"] >= GATE
    passed = all(g["passes"] for g in gate.values())
    conf = [s for s in SEEDS if s not in SELECTION_SEEDS]
    grass_drop = {s: by[s][chosen]["graph_grass_recall"] - by[s]["raw"]["graph_grass_recall"] for s in SEEDS}
    result = {"selected": chosen, "selection_means": mean, "tied_within": TIE, "tied": tied, "gate_threshold": GATE,
              "gate": {str(s): g for s, g in gate.items()}, "gate_passed": passed,
              "grass_recall_selected_minus_raw": {str(s): v for s, v in grass_drop.items()},
              "environment": environment_info()}
    (OUT / "phase1.json").write_text(json.dumps(result, indent=2))
    tie_note = f"; tied: {', '.join(tied)}" if len(tied) > 1 else ""
    md = ["# Background-suppressed graph, Phase 1 (graph quality)", "",
          "## Grid, selection seeds 0–2: mean [min, max]", "", table(by, variants, SELECTION_SEEDS), "",
          "## Reference graphs, selection seeds 0–2", "", table(by, ["raw", "pca50", "cpca10"], SELECTION_SEEDS), "",
          "cpca10 (cPCA-10 at the label-free α) is descriptive only.", "",
          "## Selection and gate", "",
          f"Selected variant (highest mean graph digit accuracy on seeds 0–2; ties within {TIE} → smaller r, then "
          f"smaller k_loc): **{chosen}** (mean {mean[chosen]:.3f}{tie_note}).", "",
          "| seed | selected | max(raw, PCA-50) | margin | ≥ 0.02 |", "|---|---|---|---|---|"]
    md += [f"| {s} | {g['selected']:.3f} | {g['best_reference']:.3f} | {g['margin']:+.3f} | "
           f"{'yes' if g['passes'] else 'no'} |" for s, g in gate.items()]
    md += ["", f"**Gate: {'passed' if passed else 'failed'}.**" + ("" if passed else
           " Phase 2 (embeddings) does not run."), "",
           "## Selected variant and references on confirmation seeds 3–5 (descriptive)", "",
           table(by, [chosen, "raw", "pca50", "cpca10"], conf), "",
           "Grass recall, selected variant minus raw, per seed: "
           + ", ".join(f"{s}: {v:+.3f}" for s, v in grass_drop.items()) + "."]
    (OUT / "phase1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k-loc", type=int)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--make-jobs", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs()
    if args.merge:
        return merge()
    assert_committed(PREREG)
    reference_part(args.seed) if args.reference else residual_part(args.seed, args.k_loc)


if __name__ == "__main__":
    main()
