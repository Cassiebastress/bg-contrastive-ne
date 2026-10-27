"""Phase 5a main runs for one design (Phase 5a plan §5-6), after the committed freeze (results/phase5a/
preregistration.md).

  --stage selection     seeds 0-2: Stage 1 + (a), (b), (c) and the oracle same-cluster control (o) at
                        m_B in {1, 2, 5, 10, 20}, plus (e2) and (e3) at the label-free alpha (K also (e2-k20) and
                        (e2-k40): the top 20 and 40 cPCA components at the same alpha); then the pre-registered
                        selection rule (J: signal accuracy; K: WC; ties -> smaller m_B) -> selection_{design}.json
  --stage confirmation  fresh seeds from 3 (requires the committed selection): (a), best (b), best (c), best (o), (e2),
                        (e3) (K also (e2-k20), (e2-k40)). A seed failing V1, V2 (and V4 for J), judged from the data
                        and (a) only, is excluded and the next seed drawn, until 3 valid seeds; more than 2
                        exclusions stops the design.
  --recheck             determinism: rerun selection seed 0 and compare rows and embeddings bit for bit

Checks (stop and report on failure): xi_B gates on every seed; every Stage 2 run finite with std growth <= 10x.

Run with the cne-bg env:  caffeinate -i python scripts/phase5a_run.py --design J --stage selection
Outputs go to results/phase5a/ (embeddings and per-epoch logs are gitignored).
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

from common.benchmark import (DESIGNS, EXTRA_COMPONENTS, METRICS, SELECTION_SEEDS, cpca_runs, data_scores, make,
                              score, v1, v2, v4)
from common.experiment import (COLLAPSE_STD_RATIO, FRAGMENT_FRAC, FRAGMENT_SMALL, K_GRAPH, M_BS, epoch_arrays,
                               healthy, prepare_dataset, run_stage1, run_stage2)
from common.repro import assert_committed, environment_info, setup_determinism
from eval.metrics import component_stats
from eval.xi_b_diag import diagnose
from method.bg_term import same_cluster_xi

OUT = ROOT / "results" / "phase5a"
CAL = OUT / "calibration.json"
PREREG = OUT / "preregistration.md"
GROUPS = ("b", "c", "o")
SELECTION_RUNS = ["a"] + [f"{g}_m{m}" for g in GROUPS for m in M_BS] + ["e2", "e3"]
CPCA_RUNS = {d: ["e2", "e3"] + [f"e2_k{n}" for n in EXTRA_COMPONENTS[d]] for d in DESIGNS}
SELECTION_METRIC = {"J": "acc_s", "K": "WC"}
N_CONFIRM, FIRST_CONFIRM_SEED, MAX_EXCLUDED = 3, 3, 2
ROW_FIELDS = ["design", "stage", "seed", "run", "group", "m_B", *METRICS, "mean_phi", "embd_std", "std_ratio_vs_a",
              "bg_share_final", "n_components", "frac_in_small_components", "flag_collapse", "flag_fragment",
              "nonfinite_batches", "max_std_growth", "selected_alpha", "runtime_s"]


def stop(msg):
    print(f"STOP (Phase 5a plan §7): {msg}", flush=True)
    sys.exit(2)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def dataset(design, seed, cal):
    config = None
    if design == "J":
        c = cal["chosen_J"]
        config = {"theta": c["theta"], "projected": c["projected"], "gain": c["gain"]}
    setup_determinism(seed)
    ds = make(design, seed, cal["rho"][design]["rho"], config)
    return prepare_dataset(ds, design, seed, oracle_xi=same_cluster_xi(ds.truth["y_b"]))


def new_row(ctx, stage, run, **extra):
    row = dict.fromkeys(ROW_FIELDS, "")
    row.update({"design": ctx.level, "stage": stage, "seed": ctx.seed, "run": run,
                "group": run.split("_m")[0] if "_m" in run else run, **extra})
    return row


def fit_row(ctx, stage, run, fit, a_std):
    """Score a Stage 2 fit and record its scale, health and review flags (as in Phase 4)."""
    embd, scale, log = fit["embd"], fit["scale"], fit.get("log", {})
    std = np.array(scale["embd_std"])
    row = new_row(ctx, stage, run, m_B=fit["params"].get("background", {}).get("m_B", ""))
    row.update(score(embd, ctx.ds.truth))
    row.update(component_stats(embd, k=K_GRAPH, small=FRAGMENT_SMALL))
    row.update({"mean_phi": scale["mean_phi"][-1], "embd_std": float(std[-1]),
                "bg_share_final": log["bg_share"][-1] if log else "",
                "nonfinite_batches": int(sum(log.get("n_nonfinite_batches", [0]))) + int(not all(scale["finite"])),
                "max_std_growth": float(std.max() / std[0]), "runtime_s": fit["runtime_s"]})
    row["flag_fragment"] = row["frac_in_small_components"] > FRAGMENT_FRAC
    row["std_ratio_vs_a"] = 1.0 if a_std is None else float(std[-1] / a_std)
    row["flag_collapse"] = row["std_ratio_vs_a"] < COLLAPSE_STD_RATIO
    return row


def checks_for(ctx, stage):
    ds = ctx.ds
    diag, _ = diagnose(ds, ctx.xi, ctx.G)
    out = {"sha256_X_T": hashlib.sha256(ds.X_T.tobytes()).hexdigest(),
           "sha256_X_B": hashlib.sha256(ds.X_B.tobytes()).hexdigest(), "xi_b": ctx.xi.sha256(),
           "params": ds.params, **data_scores(ds), "xi_b_diagnostics": diag}
    if not (diag["gate_same_b"] and diag["gate_same_s"]):
        stop(f"{ctx.level} seed {ctx.seed}: xi_B gates fail ({diag['same_b_xi']:.3f}, "
             f"excess {diag['same_s_excess']:+.3f})")
    return out


def run_seed(design, stage, seed, labels, cal, save=True):
    """Stage 1 + the given runs on one dataset. Returns rows, checks and embeddings."""
    ctx = dataset(design, seed, cal)
    checks = checks_for(ctx, stage)
    rows, embds = [], {}
    s1 = run_stage1(ctx)
    embds["stage1"] = s1["embd"]
    a_std = None
    for label in labels:
        if label in CPCA_RUNS[design]:
            continue
        fit = run_stage2(ctx, s1, label)
        row = fit_row(ctx, stage, label, fit, a_std)
        if label == "a":
            a_std = row["embd_std"]
        rows.append(row)
        embds[label] = fit["embd"]
        print(f"{design} {stage} seed={seed} {label:6s} acc_s {row['acc_s']:.3f} WC {row['WC']:.3f} "
              f"acc_b {row['acc_b']:.3f} G1 {row['G1']:.3f} ({row['runtime_s']:.0f}s)", flush=True)
        if save:
            np.savez(OUT / "logs" / f"{design}_{stage}_seed{seed}_{label}.npz", **epoch_arrays(fit))
        if not healthy(row):
            stop(f"{design} seed {seed} {label}: non-finite or exploding (std growth {row['max_std_growth']:.2f})")
        if stage == "confirmation" and label == "a":
            # exclusion rule: V1, V2 (and V4 for J), from the data and (a) only, before any other run
            data = {k: checks[k] for k in ("detectability", "coherence")}
            v = {"V1": v1(row), "V2": v2(data), **({"V4": v4(data)} if design == "J" else {})}
            checks["validity"] = {"a_acc_s": row["acc_s"], "a_acc_b": row["acc_b"], **data, **v,
                                  "valid": all(v.values())}
            if not checks["validity"]["valid"]:
                print(f"{design} confirmation seed={seed}: EXCLUDED, not a valid test {checks['validity']}", flush=True)
                return rows, checks, embds
    if "e2" in labels:
        more, alpha = cpca_runs(ctx.ds, seed, EXTRA_COMPONENTS[design])
        checks["alpha"] = alpha
        for run in CPCA_RUNS[design]:
            row = new_row(ctx, stage, run, selected_alpha=alpha["selected_alpha"])
            row.update(score(more[run], ctx.ds.truth))
            row.update(component_stats(more[run], k=K_GRAPH, small=FRAGMENT_SMALL))
            row["flag_fragment"] = row["frac_in_small_components"] > FRAGMENT_FRAC
            row["nonfinite_batches"] = int(not np.isfinite(more[run]).all())
            rows.append(row)
            embds[run] = more[run]
            print(f"{design} {stage} seed={seed} {run:6s} acc_s {row['acc_s']:.3f} WC {row['WC']:.3f} "
                  f"acc_b {row['acc_b']:.3f} G1 {row['G1']:.3f} alpha {alpha['selected_alpha']:.3g}", flush=True)
    if save:
        for label, e in embds.items():
            np.save(OUT / "embeddings" / f"{design}_{stage}_seed{seed}_{label}.npy", e)
    return rows, checks, embds


def select(design, rows):
    """Best m_B per group on the selection seeds by the design's selection metric; ties -> smaller m_B."""
    metric = SELECTION_METRIC[design]
    means = {r: float(np.mean([float(x[metric]) for x in rows if x["run"] == r])) for r in SELECTION_RUNS}
    best = {g: max(M_BS, key=lambda m: (means[f"{g}_m{m}"], -m)) for g in GROUPS}
    return {"design": design, "metric": metric, "best_m_B": best, "means": means,
            "rule": f"highest mean {metric} over selection seeds 0-2 per group; ties -> smaller m_B"}


def confirmation_labels(design, selection):
    return ["a"] + [f"{g}_m{selection['best_m_B'][g]}" for g in GROUPS] + CPCA_RUNS[design]


def recheck(design, cal):
    saved = [r for r in read_csv(OUT / f"runs_{design}_selection.csv") if int(r["seed"]) == 0]
    rows, _, embds = run_seed(design, "selection", 0, SELECTION_RUNS + CPCA_RUNS[design][2:], cal, save=False)
    fields = [f for f in ROW_FIELDS if f != "runtime_s"]
    rows_ok = len(rows) == len(saved) and all(str(new[f]) == old[f] for new, old in zip(rows, saved) for f in fields)
    embd_ok = {label: bool(np.array_equal(e, np.load(OUT / "embeddings" / f"{design}_selection_seed0_{label}.npy")))
               for label, e in embds.items()}
    result = {"environment": environment_info(), "design": design, "seed": 0, "rows_identical": rows_ok,
              "embeddings_identical": embd_ok, "all_identical": rows_ok and all(embd_ok.values())}
    (OUT / f"recheck_{design}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    if not result["all_identical"]:
        stop(f"{design}: rerun of selection seed 0 is not bit-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=DESIGNS, required=True)
    parser.add_argument("--stage", choices=["selection", "confirmation"])
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    assert_committed(PREREG)
    cal = json.loads(CAL.read_text())
    if args.design == "J" and not cal.get("chosen_J"):
        stop("no valid Design J configuration")
    for sub in ("embeddings", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if args.recheck:
        return recheck(args.design, cal)
    design, stage = args.design, args.stage

    selection = None
    if stage == "confirmation":
        sel_path = OUT / f"selection_{design}.json"
        assert_committed(sel_path)
        selection = json.loads(sel_path.read_text())
        labels = confirmation_labels(design, selection)
    else:
        labels = SELECTION_RUNS + CPCA_RUNS[design][2:]

    rows, checks, used, excluded = [], {}, [], []
    seed = 0 if stage == "selection" else FIRST_CONFIRM_SEED
    while len(used) < (len(SELECTION_SEEDS) if stage == "selection" else N_CONFIRM):
        seed_rows, checks[seed], _ = run_seed(design, stage, seed, labels, cal)
        if stage == "confirmation" and not checks[seed]["validity"]["valid"]:
            excluded.append({"seed": seed, "validity": checks[seed]["validity"], "rows": seed_rows})
            if len(excluded) > MAX_EXCLUDED:
                stop(f"{design}: {len(excluded)} confirmation seeds excluded before {N_CONFIRM} valid ones")
        else:
            rows += seed_rows
            used.append(seed)
            write_csv(OUT / f"runs_{design}_{stage}.csv", rows)
        seed += 1

    summary = {"environment": environment_info(), "design": design, "stage": stage, "seeds": used,
               "excluded_seeds": excluded, "runs": labels, "checks": checks}
    if stage == "selection":
        selected = select(design, rows)
        (OUT / f"selection_{design}.json").write_text(json.dumps(selected, indent=2))
        summary["selection"] = selected
        print(json.dumps({k: v for k, v in selected.items() if k != "means"}, indent=2))
    else:
        summary["selection"] = selection
    (OUT / f"summary_{design}_{stage}.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
