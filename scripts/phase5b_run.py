"""Phase 5b main runs for one Design L variant, after the committed freeze (results/phase5b/preregistration.md).

  --stage selection     seeds 0-2: Stage 1 + (a); (b) and (c) at m_B in {1, 2, 5, 10, 20, 40}; (o) at m_B in
                        {1, 2, 5, 10, 20, 40, 80}; (e2) and (e3) at the label-free alpha. The cPCA variants on these
                        seeds come from calibration (results/phase5b/ceilings_{variant}.csv). Selection rule: highest
                        mean WC per group; ties -> smaller m_B -> selection_{variant}.json
  --stage confirmation  fresh seeds from 3 (requires the committed selection): (a), best (b), best (c), best (o), (e2),
                        (e3) and every cPCA variant (alpha representative x k in {3, 5, 10, 20}, plus the per-k
                        oracle alpha). A seed failing V1, V2 or V2b (data and (a) only) is excluded and the next seed
                        drawn, until 3 valid seeds; more than 2 exclusions stops the design.
  --recheck             determinism: rerun selection seed 0 and compare rows and embeddings bit for bit

Checks (stop and report on failure): xi_B gates on every seed; every Stage 2 run finite with std growth <= 10x.

Run with the cne-bg env:  caffeinate -i python scripts/phase5b_run.py --design cyclic --stage selection
Outputs go to results/phase5b/ (embeddings and per-epoch logs are gitignored).
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.benchmark import SELECTION_SEEDS, cpca_runs
from common.benchmark_l import V2_MIN_DETECTABILITY, V2B_MIN_GRAPH_SIGNAL, XI_GATES, cpca_variants, embed, score
from common.experiment import (COLLAPSE_STD_RATIO, FRAGMENT_FRAC, FRAGMENT_SMALL, K_GRAPH, VALIDITY, epoch_arrays,
                               healthy, prepare_dataset, run_stage1, run_stage2)
from common.repro import assert_committed, environment_info, setup_determinism
from data.synthetic_l import VARIANTS, make_l
from eval.benchmark_metrics import within_cluster_accuracy
from eval.metrics import component_stats, knn_accuracy
from eval.xi_b_diag import diagnose
from method.bg_term import same_cluster_xi

OUT = ROOT / "results" / "phase5b"
CAL = OUT / "calibration.json"
PREREG = OUT / "preregistration.md"
M_GRID = {"b": [1, 2, 5, 10, 20, 40], "c": [1, 2, 5, 10, 20, 40], "o": [1, 2, 5, 10, 20, 40, 80]}
GROUPS = tuple(M_GRID)
SELECTION_RUNS = ["a"] + [f"{g}_m{m}" for g in GROUPS for m in M_GRID[g]] + ["e2", "e3"]
N_CONFIRM, FIRST_CONFIRM_SEED, MAX_EXCLUDED = 3, 3, 2
METRICS = ["WC", "acc_s", "acc_b", "G1", "G1_k200", "G2", "G3_shares_signal"] + [f"WC_c{i}" for i in range(6)]
ROW_FIELDS = ["variant", "stage", "seed", "run", "group", "m_B", "alpha", "k", "cpca_source", *METRICS, "mean_phi",
              "embd_std", "std_ratio_vs_a", "bg_share_final", "n_components", "frac_in_small_components",
              "flag_collapse", "flag_fragment", "nonfinite_batches", "max_std_growth", "runtime_s"]


def stop(msg):
    print(f"STOP (Phase 5b spec §7): {msg}", flush=True)
    sys.exit(2)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def dataset(variant, seed, cal):
    setup_determinism(seed)
    ds = make_l(variant, seed, cal["rho"][variant]["rho"])
    return prepare_dataset(ds, f"L-{variant}", seed, oracle_xi=same_cluster_xi(ds.truth["y_b"]))


def new_row(variant, stage, seed, run, **extra):
    row = dict.fromkeys(ROW_FIELDS, "")
    row.update({"variant": variant, "stage": stage, "seed": seed, "run": run,
                "group": run.split("_m")[0] if "_m" in run else run.split("_")[0], **extra})
    return row


def embedding_row(variant, stage, seed, run, X, truth, **extra):
    """A row for an embedding without a Stage 2 fit log (cPCA runs): scores, components and finiteness."""
    row = new_row(variant, stage, seed, run, **extra)
    row.update(score(X, truth))
    row.update(component_stats(X, k=K_GRAPH, small=FRAGMENT_SMALL))
    row["flag_fragment"] = row["frac_in_small_components"] > FRAGMENT_FRAC
    row["nonfinite_batches"] = int(not np.isfinite(X).all())
    return row


def fit_row(variant, stage, seed, run, fit, truth, a_std):
    """Score a Stage 2 fit and record its scale, health and review flags (as in Phases 4-5a)."""
    embd, scale, log = fit["embd"], fit["scale"], fit.get("log", {})
    std = np.array(scale["embd_std"])
    row = embedding_row(variant, stage, seed, run, embd, truth,
                        m_B=fit["params"].get("background", {}).get("m_B", ""))
    row.update({"mean_phi": scale["mean_phi"][-1], "embd_std": float(std[-1]),
                "bg_share_final": log["bg_share"][-1] if log else "",
                "nonfinite_batches": int(sum(log.get("n_nonfinite_batches", [0]))) + int(not all(scale["finite"])),
                "max_std_growth": float(std.max() / std[0]), "runtime_s": fit["runtime_s"]})
    row["std_ratio_vs_a"] = 1.0 if a_std is None else float(std[-1] / a_std)
    row["flag_collapse"] = row["std_ratio_vs_a"] < COLLAPSE_STD_RATIO
    return row


def checks_for(ctx):
    ds, t = ctx.ds, ctx.ds.truth
    diag, _ = diagnose(ds, ctx.xi, ctx.G)
    out = {"sha256_X_T": hashlib.sha256(ds.X_T.tobytes()).hexdigest(),
           "sha256_X_B": hashlib.sha256(ds.X_B.tobytes()).hexdigest(), "xi_b": ctx.xi.sha256(), "params": ds.params,
           "detectability": within_cluster_accuracy(ds.X_T, t["y_s"], t["y_b"]),
           "graph_signal": knn_accuracy(ds.X_T, t["y_s"]), "xi_b_diagnostics": diag}
    if not (diag["same_b_xi"] >= XI_GATES["min_same_b"] and diag["same_s_excess"] <= XI_GATES["max_same_s_excess"]):
        stop(f"{ctx.level} seed {ctx.seed}: xi_B gates fail ({diag['same_b_xi']:.3f}, "
             f"excess {diag['same_s_excess']:+.3f})")
    return out


def run_seed(variant, stage, seed, labels, cal, save=True, variants=False):
    """Stage 1 + the given runs (+ every cPCA variant if `variants`) on one dataset."""
    ctx = dataset(variant, seed, cal)
    t = ctx.ds.truth
    checks = checks_for(ctx)
    rows, embds = [], {}
    s1 = run_stage1(ctx)
    embds["stage1"] = s1["embd"]
    a_std = None
    for label in labels:
        if label in ("e2", "e3"):
            continue
        fit = run_stage2(ctx, s1, label)
        row = fit_row(variant, stage, seed, label, fit, t, a_std)
        if label == "a":
            a_std = row["embd_std"]
        rows.append(row)
        embds[label] = fit["embd"]
        print(f"L-{variant} {stage} seed={seed} {label:6s} WC {row['WC']:.3f} acc_s {row['acc_s']:.3f} "
              f"acc_b {row['acc_b']:.3f} ({row['runtime_s']:.0f}s)", flush=True)
        if save:
            np.savez(OUT / "logs" / f"{variant}_{stage}_seed{seed}_{label}.npz", **epoch_arrays(fit))
        if not healthy(row):
            stop(f"L-{variant} seed {seed} {label}: non-finite or exploding (std growth {row['max_std_growth']:.2f})")
        if stage == "confirmation" and label == "a":
            # exclusion rule: V1, V2, V2b from the data and (a) only, before any other run
            v = {"V1": row["acc_s"] <= VALIDITY["max_emb_acc_s"] and row["acc_b"] >= VALIDITY["min_emb_acc_b"],
                 "V2": checks["detectability"] >= V2_MIN_DETECTABILITY,
                 "V2b": checks["graph_signal"] >= V2B_MIN_GRAPH_SIGNAL}
            checks["validity"] = {"a_acc_s": row["acc_s"], "a_acc_b": row["acc_b"],
                                  "detectability": checks["detectability"], "graph_signal": checks["graph_signal"],
                                  **v, "valid": all(v.values())}
            if not checks["validity"]["valid"]:
                print(f"L-{variant} confirmation seed={seed}: EXCLUDED {checks['validity']}", flush=True)
                return rows, checks, embds
    if "e2" in labels:
        more, alpha = cpca_runs(ctx.ds, seed)
        checks["alpha"] = alpha
        for run in ("e2", "e3"):
            rows.append(embedding_row(variant, stage, seed, run, more[run], t, alpha=alpha["selected_alpha"],
                                      k=10))
            embds[run] = more[run]
            print(f"L-{variant} {stage} seed={seed} {run:6s} WC {rows[-1]['WC']:.3f} "
                  f"alpha {alpha['selected_alpha']:.3g}", flush=True)
    if variants:
        vs, info = cpca_variants(ctx.ds, seed)
        checks["cpca_variants"] = info
        for v in vs:
            E = embed(v["Z"], seed)
            rows.append(embedding_row(variant, stage, seed, v["label"], E, t, alpha=v["alpha"], k=v["k"],
                                      cpca_source=v["source"]))
            rows[-1]["group"] = "cpca"
            embds[v["label"]] = E
        best = max((r for r in rows if r["group"] == "cpca"), key=lambda r: r["WC"])
        print(f"L-{variant} {stage} seed={seed} best cPCA variant {best['run']} WC {best['WC']:.3f}", flush=True)
    if save:
        for label, e in embds.items():
            np.save(OUT / "embeddings" / f"{variant}_{stage}_seed{seed}_{label}.npy", e)
    return rows, checks, embds


def select(variant, rows):
    """Best m_B per group on the selection seeds by mean WC; ties -> smaller m_B."""
    means = {r: float(np.mean([float(x["WC"]) for x in rows if x["run"] == r])) for r in SELECTION_RUNS}
    best = {g: max(M_GRID[g], key=lambda m: (means[f"{g}_m{m}"], -m)) for g in GROUPS}
    return {"variant": variant, "metric": "WC", "best_m_B": best, "means": means,
            "rule": "highest mean WC over selection seeds 0-2 per group; ties -> smaller m_B"}


def confirmation_labels(selection):
    return ["a"] + [f"{g}_m{selection['best_m_B'][g]}" for g in GROUPS] + ["e2", "e3"]


def recheck(variant, cal):
    saved = [r for r in read_csv(OUT / f"runs_{variant}_selection.csv") if int(r["seed"]) == 0]
    rows, _, embds = run_seed(variant, "selection", 0, SELECTION_RUNS, cal, save=False)
    fields = [f for f in ROW_FIELDS if f != "runtime_s"]
    rows_ok = len(rows) == len(saved) and all(str(new[f]) == old[f] for new, old in zip(rows, saved) for f in fields)
    embd_ok = {label: bool(np.array_equal(e, np.load(OUT / "embeddings" / f"{variant}_selection_seed0_{label}.npy")))
               for label, e in embds.items()}
    result = {"environment": environment_info(), "variant": variant, "seed": 0, "rows_identical": rows_ok,
              "embeddings_identical": embd_ok, "all_identical": rows_ok and all(embd_ok.values())}
    (OUT / f"recheck_{variant}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    if not result["all_identical"]:
        stop(f"L-{variant}: rerun of selection seed 0 is not bit-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=VARIANTS, required=True)
    parser.add_argument("--stage", choices=["selection", "confirmation"])
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    assert_committed(PREREG)
    cal = json.loads(CAL.read_text())
    if not cal.get("valid", {}).get(args.design):
        stop(f"L-{args.design} is not a valid design (calibration)")
    for sub in ("embeddings", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if args.recheck:
        return recheck(args.design, cal)
    variant, stage = args.design, args.stage

    selection = None
    if stage == "confirmation":
        sel_path = OUT / f"selection_{variant}.json"
        assert_committed(sel_path)
        selection = json.loads(sel_path.read_text())
        labels = confirmation_labels(selection)
    else:
        labels = SELECTION_RUNS

    rows, checks, used, excluded = [], {}, [], []
    seed = 0 if stage == "selection" else FIRST_CONFIRM_SEED
    while len(used) < (len(SELECTION_SEEDS) if stage == "selection" else N_CONFIRM):
        seed_rows, checks[seed], _ = run_seed(variant, stage, seed, labels, cal, variants=stage == "confirmation")
        if stage == "confirmation" and not checks[seed]["validity"]["valid"]:
            excluded.append({"seed": seed, "validity": checks[seed]["validity"], "rows": seed_rows})
            if len(excluded) > MAX_EXCLUDED:
                stop(f"L-{variant}: {len(excluded)} confirmation seeds excluded before {N_CONFIRM} valid ones")
        else:
            rows += seed_rows
            used.append(seed)
            write_csv(OUT / f"runs_{variant}_{stage}.csv", rows)
        seed += 1

    summary = {"environment": environment_info(), "variant": variant, "stage": stage, "seeds": used,
               "excluded_seeds": excluded, "runs": labels, "checks": checks}
    if stage == "selection":
        selected = select(variant, rows)
        (OUT / f"selection_{variant}.json").write_text(json.dumps(selected, indent=2))
        summary["selection"] = selected
        print(json.dumps({k: v for k, v in selected.items() if k != "means"}, indent=2))
    else:
        summary["selection"] = selection
    (OUT / f"summary_{variant}_{stage}.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
