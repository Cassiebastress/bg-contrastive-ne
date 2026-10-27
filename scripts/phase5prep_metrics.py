"""Phase 5 prep, Part A2 (Phase_5_Plan.md §1): validate the global metrics G1-G3 before using them.

Computes G1 (cross-cluster same-signal neighbor rate), G2 (background kNN accuracy) and G3 (cell-level layout) on:
1. stored Phase 4 embeddings: (a), best (b) by signal accuracy, (c) at m*_c and (d), every Phase 4 seed and level.
   Reload check: signal and background kNN accuracy recomputed from each .npy must equal the Phase 4 CSV exactly.
2. a signal-only reference: the same two-stage Neg-t-SNE as run (a) on S_T + eps (the dataset's own noise),
   selection seeds 0-2.
3. high-D X_T, as context.
Stop condition (approved): on every level, every signal-only seed must exceed every Phase 4 run by >= 0.3 on G1 and on
G3 "shares signal".

Run with the cne-bg env:  caffeinate -i python scripts/phase5prep_metrics.py
Outputs: results/phase5prep/metric_validation.{csv,md,json} (reference embeddings are gitignored .npy files).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.embed import pca_init, two_stage
from common.knn import knn_graph
from common.repro import environment_info, setup_determinism
from data.synthetic import LEVELS, MAIN_RHO, make_dataset
from eval.global_metrics import g1_chance, global_scores
from eval.metrics import knn_accuracy

P4 = ROOT / "results" / "phase4"
OUT = ROOT / "results" / "phase5prep"
SEPARATION = 0.3
FIELDS = ["level", "seed", "stage", "group", "acc_s", "G1", "G1_chance", "G2", "G3_shares_signal",
          "G3_shares_background", "G3_neither"]
GROUP_ORDER = ["highD_X_T", "signal_only", "a", "best_b", "c", "d"]
GROUP_NAMES = {"highD_X_T": "high-D X_T (context)", "signal_only": "signal-only reference (S_T + noise)",
               "a": "(a) plain", "best_b": "best (b) uniform_extra", "c": "(c) xi_b at m*_c",
               "d": "(d) s=0 reference"}


def phase4_rows(level):
    rows = {}
    for stage in ("selection", "confirmation"):
        with open(P4 / f"runs_{level}_{stage}.csv") as f:
            for r in csv.DictReader(f):
                rows[(stage, int(r["seed"]), r["run"])] = r
    return rows


def seed_sets(level):
    conf = json.loads((P4 / f"summary_{level}_confirmation.json").read_text())
    return [("selection", s) for s in (0, 1, 2)] + [("confirmation", s) for s in conf["seeds"]]


def row(level, seed, stage, group, X, t, acc_s=None):
    scores = global_scores(X, t["y_s"], t["y_b"])
    return {"level": level, "seed": seed, "stage": stage, "group": group,
            "acc_s": knn_accuracy(X, t["y_s"]) if acc_s is None else acc_s,
            "G1": scores["G1"], "G1_chance": g1_chance(t["y_s"], t["y_b"]), "G2": scores["G2"],
            **{k: scores[k] for k in ("G3_shares_signal", "G3_shares_background", "G3_neither")}}


def fmt(vals):
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.3f} [{vals.min():.3f}, {vals.max():.3f}]"


def main():
    (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
    rows, reload_checks, separation = [], [], {}
    for level in LEVELS:
        sel = json.loads((P4 / f"selection_{level}.json").read_text())
        runs = {"a": "a", "best_b": f"b_m{sel['best_b']['acc_s']}", "c": f"c_m{sel['m_star_c']}", "d": "d"}
        p4 = phase4_rows(level)
        for stage, seed in seed_sets(level):
            setup_determinism(seed)
            ds = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level])
            t = ds.truth
            rows.append(row(level, seed, stage, "highD_X_T", ds.X_T, t))
            for group, run in runs.items():
                e = np.load(P4 / "embeddings" / f"{level}_{stage}_seed{seed}_{run}.npy")
                rec = p4[(stage, seed, run)]
                acc_s, acc_b = knn_accuracy(e, t["y_s"]), knn_accuracy(e, t["y_b"])
                ok = acc_s == float(rec["acc_s"]) and acc_b == float(rec["acc_b"])
                reload_checks.append({"level": level, "stage": stage, "seed": seed, "run": run, "match": ok})
                if not ok:
                    sys.exit(f"STOP: {level} {stage} seed {seed} {run}: stored embedding does not reproduce the "
                             f"Phase 4 accuracies ({acc_s}, {acc_b}) vs ({rec['acc_s']}, {rec['acc_b']})")
                rows.append(row(level, seed, stage, group, e, t, acc_s=acc_s))
            if stage == "selection":
                X = (t["S_T"] + (ds.X_T - t["S_T"] - t["Bg_T"])).astype(np.float32)  # S_T + the dataset's own noise
                _, s2 = two_stage(pca_init(X), knn_graph(X, k=15), seed)
                np.save(OUT / "embeddings" / f"signal_only_{level}_seed{seed}.npy", s2["embd"])
                rows.append(row(level, seed, stage, "signal_only", s2["embd"], t))
            print(f"{level} {stage} seed={seed} done", flush=True)

        lv = [r for r in rows if r["level"] == level]
        p4_runs = [r for r in lv if r["group"] in runs]
        ref = [r for r in lv if r["group"] == "signal_only"]
        separation[level] = {m: {"reference_min": min(r[m] for r in ref), "phase4_max": max(r[m] for r in p4_runs)}
                             for m in ("G1", "G3_shares_signal")}
        for m, v in separation[level].items():
            v["gap"] = v["reference_min"] - v["phase4_max"]
            v["separated"] = v["gap"] >= SEPARATION

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metric_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# Global metric validation (generated by scripts/phase5prep_metrics.py)", "",
          "Mean [min, max] over seeds. Phase 4 runs use every Phase 4 seed (selection 0-2 and the confirmation "
          "seeds); the signal-only reference uses selection seeds 0-2. G1 chance is the exact rate for random "
          "neighbors. G2 chance is about 0.17.", ""]
    for level in LEVELS:
        lv = [r for r in rows if r["level"] == level]
        md += [f"## {level}", "", "| embedding | seeds | signal acc | G1 | G2 (background acc) | G3 shares signal "
               "| G3 shares background | G3 neither |", "|---|---|---|---|---|---|---|---|"]
        for g in GROUP_ORDER:
            gr = [r for r in lv if r["group"] == g]
            md.append(f"| {GROUP_NAMES[g]} | {', '.join(str(r['seed']) for r in gr)} | {fmt([r['acc_s'] for r in gr])} "
                      f"| {fmt([r['G1'] for r in gr])} | {fmt([r['G2'] for r in gr])} "
                      f"| {fmt([r['G3_shares_signal'] for r in gr])} | {fmt([r['G3_shares_background'] for r in gr])} "
                      f"| {fmt([r['G3_neither'] for r in gr])} |")
        sep = separation[level]
        md += ["", f"G1 chance: {fmt([r['G1_chance'] for r in lv])}. Separation (signal-only minimum minus Phase 4 "
               f"maximum; must be >= {SEPARATION}): G1 {sep['G1']['gap']:+.3f}, G3 shares signal "
               f"{sep['G3_shares_signal']['gap']:+.3f}.", ""]
    (OUT / "metric_validation.md").write_text("\n".join(md))
    all_separated = all(v["separated"] for s in separation.values() for v in s.values())
    (OUT / "metric_validation.json").write_text(json.dumps(
        {"environment": environment_info(), "separation_threshold": SEPARATION, "separation": separation,
         "all_separated": all_separated, "reload_checks_all_match": all(c["match"] for c in reload_checks),
         "reload_checks": reload_checks}, indent=2))
    print(json.dumps({"all_separated": all_separated, "separation": separation}, indent=2))
    if not all_separated:
        sys.exit("STOP: the signal-only reference does not clearly separate from the Phase 4 runs")


if __name__ == "__main__":
    main()
