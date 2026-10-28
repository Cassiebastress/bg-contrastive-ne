"""Phase 5b calibration and validity (selection seeds 0-2; runs (a), (e2), (e3) and the ceilings only, never (b), (c)
or the oracle control).

Steps (run in order):
  --step rho                        Phase 1 rule per variant: largest rho on the grid with V1 and signal-only kNN
                                    accuracy >= 0.95 on seed 0, confirmed on seeds 0-2.
  --step validate --variant V       data diagnostics, xi_B gates, (a), (e2), (e3), and the linear and cPCA ceilings
                                    (embedding level, with high-D kNN versions as descriptive); V1, V2, V2b, V3.
  --step summary                    calibration.md and the valid designs in calibration.json.
  --step freeze                     preregistration.md (commit before any (b), (c) or oracle-control run).

Run with the cne-bg env:  caffeinate -i python scripts/phase5b_calibrate.py --step <step>
Outputs: results/phase5b/rho_sweep.csv, validate_{variant}.csv, ceilings_{variant}.csv, calibration.json,
calibration.md, preregistration.md (embeddings are gitignored .npy files).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.benchmark import PHASE1_RHOS, SELECTION_SEEDS, cpca_runs, fit_a
from common.benchmark_l import (V2_MIN_DETECTABILITY, V2B_MIN_GRAPH_SIGNAL, WC_BAR, XI_GATES, cpca_ceilings,
                                linear_ceilings, score)
from common.experiment import VALIDITY
from common.knn import knn_graph
from common.repro import environment_info, setup_determinism
from data.synthetic_l import VARIANTS, make_l
from eval.benchmark_metrics import coherence, signal_background_geometry, within_cluster_accuracy
from eval.metrics import knn_accuracy
from eval.xi_b_diag import diagnose
from method.xi_b import build_xi_b

OUT = ROOT / "results" / "phase5b"
EMB = OUT / "embeddings"
CAL = OUT / "calibration.json"


def stop(msg):
    print(f"STOP (Phase 5b spec §7): {msg}", flush=True)
    sys.exit(2)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_cal():
    return json.loads(CAL.read_text()) if CAL.exists() else {}


def save_cal(cal):
    cal["environment"] = environment_info()
    CAL.write_text(json.dumps(cal, indent=2))


# ---- rho ----

def phase1_row(variant, rho, seed):
    setup_determinism(seed)
    ds = make_l(variant, seed, rho)
    t = ds.truth
    a = fit_a(ds, seed)
    row = {"variant": variant, "rho": rho, "seed": seed, "emb_acc_s": knn_accuracy(a, t["y_s"]),
           "emb_acc_b": knn_accuracy(a, t["y_b"]), "oracle_acc_s": knn_accuracy(t["signal_T"], t["y_s"])}
    row["valid"] = (row["emb_acc_b"] >= VALIDITY["min_emb_acc_b"] and row["emb_acc_s"] <= VALIDITY["max_emb_acc_s"]
                    and row["oracle_acc_s"] >= VALIDITY["min_oracle_acc_s"])
    print(f"{variant} rho={rho} seed={seed} emb y_s {row['emb_acc_s']:.3f} emb y_b {row['emb_acc_b']:.3f} "
          f"oracle {row['oracle_acc_s']:.3f} valid={row['valid']}", flush=True)
    return row


def step_rho():
    rows, cal = [], load_cal()
    cal["rho"] = {}
    for variant in VARIANTS:
        sweep = [phase1_row(variant, rho, 0) for rho in PHASE1_RHOS]
        rows += sweep
        write_csv(OUT / "rho_sweep.csv", rows)
        valid = [r["rho"] for r in sweep if r["valid"]]
        if not valid:
            stop(f"L-{variant}: no valid rho on seed 0")
        rho = max(valid)
        confirm = [phase1_row(variant, rho, s) for s in SELECTION_SEEDS[1:]]
        rows += confirm
        write_csv(OUT / "rho_sweep.csv", rows)
        if not all(r["valid"] for r in confirm):
            stop(f"L-{variant}: rho = {rho} valid on seed 0 but not confirmed on seeds 1-2")
        cal["rho"][variant] = {"rho": rho, "rule": "Phase 1: largest rho on the grid valid on seed 0 (emb acc_b >= "
                               "0.90, emb acc_s <= 0.50, oracle acc_s >= 0.95), confirmed on seeds 0-2",
                               "grid": PHASE1_RHOS}
        print(f"L-{variant}: rho = {rho}", flush=True)
    save_cal(cal)


# ---- validate ----

def validate_seed(variant, seed, rho):
    setup_determinism(seed)
    ds = make_l(variant, seed, rho)
    t = ds.truth
    geo = signal_background_geometry(t["signal_T"], t["B"], ds.X_B, t["y_b_background"])
    rec = {"variant": variant, "seed": seed, "rho": rho,
           "detectability": within_cluster_accuracy(ds.X_T, t["y_s"], t["y_b"]),
           "graph_signal": knn_accuracy(ds.X_T, t["y_s"]), "highd_acc_b": knn_accuracy(ds.X_T, t["y_b"]),
           "coherence": coherence(ds.X_T, t["y_s"], t["y_b"]),
           "frac_signal_var_in_U": geo["frac_signal_var_in_span_B"],
           "bg_global_local_ratio": geo["bg_global_local_ratio"]}
    if variant == "cyclic":  # fraction of each cluster's signal variance inside the other clusters' nuisance blocks
        fr = []
        for k in range(6):
            S = t["S_T"][t["y_b"] == k]
            S = S - S.mean(axis=0)
            N = np.hstack([t["Q"][(j + 1) % 6] for j in range(6) if j != k])
            fr.append(float(((S @ N) ** 2).sum() / (S ** 2).sum()))
        rec["min_frac_signal_in_other_nuisance"] = min(fr)
    G = knn_graph(ds.X_T, k=15)
    xi = build_xi_b(ds.X_T, ds.X_B, k_B=15, L=30, seed=seed)
    diag, _ = diagnose(ds, xi, G)
    rec.update({"xi_same_b": diag["same_b_xi"], "xi_same_s_excess": diag["same_s_excess"],
                "xi_gates": bool(diag["same_b_xi"] >= XI_GATES["min_same_b"]
                                 and diag["same_s_excess"] <= XI_GATES["max_same_s_excess"])})
    a = fit_a(ds, seed)
    more, alpha = cpca_runs(ds, seed)
    embs = {"a": a, **more}
    for run, X in embs.items():
        rec.update({f"{run}_{k}": v for k, v in score(X, t).items()})
    rec.update({"selected_alpha": alpha["selected_alpha"],
                "representatives": ";".join(f"{x:.6g}" for x in alpha["representatives"])})
    lin_rows, lin_embs = linear_ceilings(ds, seed)
    cp_rows, cp_embs, _ = cpca_ceilings(ds, seed)
    for r in lin_rows + cp_rows:
        r.update({"variant": variant, "seed": seed})
    for name, E in {**embs, **lin_embs, **cp_embs}.items():
        np.save(EMB / f"{variant}_cal_seed{seed}_{name}.npy", np.asarray(E, dtype=np.float32))
    best_lin = max(lin_rows, key=lambda r: r["emb_WC"])
    best_cp = max(cp_rows, key=lambda r: r["emb_WC"])
    rec.update({"linear_ceiling": best_lin["emb_WC"], "linear_ceiling_projection": best_lin["name"],
                "linear_ceiling_highd": max(r["highd_WC"] for r in lin_rows),
                "cpca_ceiling": best_cp["emb_WC"], "cpca_ceiling_variant": best_cp["name"],
                "cpca_ceiling_highd": max(r["highd_WC"] for r in cp_rows)})
    rec["V1"] = rec["a_acc_s"] <= VALIDITY["max_emb_acc_s"] and rec["a_acc_b"] >= VALIDITY["min_emb_acc_b"]
    rec["V2"] = rec["detectability"] >= V2_MIN_DETECTABILITY
    rec["V2b"] = rec["graph_signal"] >= V2B_MIN_GRAPH_SIGNAL
    rec["V3_e2"] = rec["e2_WC"] <= WC_BAR
    rec["V3_linear"] = rec["linear_ceiling"] <= WC_BAR
    rec["V3_cpca"] = rec["cpca_ceiling"] <= WC_BAR
    rec["V3"] = rec["V3_e2"] and (variant == "between" or (rec["V3_linear"] and rec["V3_cpca"]))
    print(f"L-{variant} seed={seed} det {rec['detectability']:.3f} graph {rec['graph_signal']:.3f} | a WC "
          f"{rec['a_WC']:.3f} acc_s {rec['a_acc_s']:.3f} | e2 WC {rec['e2_WC']:.3f} | linear ceiling "
          f"{rec['linear_ceiling']:.3f} ({best_lin['name']}) | cPCA ceiling {rec['cpca_ceiling']:.3f} "
          f"({best_cp['name']}) | xi {rec['xi_same_b']:.3f} {rec['xi_same_s_excess']:+.3f} | V1 {rec['V1']} V2 "
          f"{rec['V2']} V2b {rec['V2b']} V3 {rec['V3']}", flush=True)
    return rec, lin_rows + cp_rows


def step_validate(variant):
    rho = load_cal()["rho"][variant]["rho"]
    recs, ceil = [], []
    for seed in SELECTION_SEEDS:
        rec, rows = validate_seed(variant, seed, rho)
        recs.append(rec)
        ceil += rows
        write_csv(OUT / f"validate_{variant}.csv", recs)
        write_csv(OUT / f"ceilings_{variant}.csv", ceil)
    if not all(r["V2"] and r["V2b"] for r in recs):
        stop(f"L-{variant}: V2 or V2b fails (the signal is not locally present or not in the graph)")
    if variant == "cyclic" and not all(r["V3"] for r in recs):
        failing = sorted({k for r in recs for k in ("V3_e2", "V3_linear", "V3_cpca") if not r[k]})
        stop(f"L-cyclic fails V3 ({', '.join(failing)}): linear pipelines already solve it")
    if not all(r["xi_gates"] for r in recs) and all(r["V1"] and r["V3"] for r in recs):
        stop(f"L-{variant}: xi_B gates fail on a valid design")
    if variant == "between" and not all(r["V3"] for r in recs):
        print("L-between fails V3: label-free cPCA solves it (recorded; L-cyclic only)", flush=True)


# ---- summary ----

def _f(vals):
    v = np.array([float(x) for x in vals])
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def _b(x):
    return str(x) == "True"


def revealed(row, prefix):
    return ",".join(str(i) for i in range(6) if float(row[f"{prefix}{i}"]) >= WC_BAR) or "none"


def variant_section(variant, cal):
    recs = read_csv(OUT / f"validate_{variant}.csv")
    ceil = read_csv(OUT / f"ceilings_{variant}.csv")
    out = [f"## L-{variant} (ρ = {cal['rho'][variant]['rho']})", "",
           "### Per seed", "",
           "| seed | (a) WC | (a) signal acc | (a) background acc | detectability | graph signal | coherence "
           "| signal var in U | ξ_B same-y_b / excess | (e2) WC (α) | linear ceiling: embedded (projection) / high-D "
           "| cPCA ceiling: embedded (variant) / high-D | V1 | V2 | V2b | V3 |", "|" + "---|" * 16]
    for r in recs:
        out.append(
            f"| {r['seed']} | {float(r['a_WC']):.3f} | {float(r['a_acc_s']):.3f} | {float(r['a_acc_b']):.3f} "
            f"| {float(r['detectability']):.3f} | {float(r['graph_signal']):.3f} | {float(r['coherence']):.3f} "
            f"| {float(r['frac_signal_var_in_U']):.3f} | {float(r['xi_same_b']):.3f} / "
            f"{float(r['xi_same_s_excess']):+.3f} | {float(r['e2_WC']):.3f} ({float(r['selected_alpha']):.3g}) "
            f"| {float(r['linear_ceiling']):.3f} ({r['linear_ceiling_projection']}) / "
            f"{float(r['linear_ceiling_highd']):.3f} | {float(r['cpca_ceiling']):.3f} ({r['cpca_ceiling_variant']}) / "
            f"{float(r['cpca_ceiling_highd']):.3f} | {r['V1']} | {r['V2']} | {r['V2b']} | {r['V3']} |")
    out += ["", "V3 for L-cyclic: (e2) WC, the embedded linear ceiling and the embedded cPCA ceiling all ≤ "
            f"{WC_BAR:.2f}. For L-between: (e2) WC ≤ {WC_BAR:.2f} only. High-D values are descriptive.", "",
            "### Which clusters each linear pipeline reveals (per-cluster embedded WC ≥ 0.80)", "",
            "| seed | (e2) per-cluster WC | revealed | best linear subset | its per-cluster WC | revealed "
            "| best cPCA variant | its per-cluster WC | revealed |", "|" + "---|" * 9]
    for r in recs:
        lin = next(c for c in ceil if c["seed"] == r["seed"] and c["name"] == r["linear_ceiling_projection"])
        cp = next(c for c in ceil if c["seed"] == r["seed"] and c["name"] == r["cpca_ceiling_variant"])
        pc = (lambda row, pre: ", ".join(f"{float(row[f'{pre}{i}']):.2f}" for i in range(6)))
        out.append(f"| {r['seed']} | {pc(r, 'e2_WC_c')} | {revealed(r, 'e2_WC_c')} | {lin['name']} "
                   f"| {pc(lin, 'emb_WC_c')} | {revealed(lin, 'emb_WC_c')} | {cp['name']} | {pc(cp, 'emb_WC_c')} "
                   f"| {revealed(cp, 'emb_WC_c')} |")
    if variant == "cyclic":
        out += ["", "### Linear subsets by the number of clusters revealed without their nuisance", "",
                "A subset reveals cluster k \"cleanly\" if it contains Q_k (k's signal) but not Q_{k+1} "
                "(k's nuisance). "
                "Embedded WC over the subsets in each group, all seeds pooled:", "",
                "| clean reveals | subsets | embedded WC mean [min, max] | high-D WC mean [min, max] |",
                "|---|---|---|---|"]
        lin = [c for c in ceil if c["kind"] == "linear"]
        for n in sorted({int(c["clean_reveals"]) for c in lin}):
            g = [c for c in lin if int(c["clean_reveals"]) == n]
            out.append(f"| {n} | {len({c['name'] for c in g})} | {_f(c['emb_WC'] for c in g)} "
                       f"| {_f(c['highd_WC'] for c in g)} |")
    cp = [c for c in ceil if c["kind"] == "cpca"]
    out += ["", "### cPCA variants (embedded WC, mean over seeds per k)", "",
            "| k | representative α: embedded WC [min, max] | oracle α for k: embedded WC [min, max] "
            "| high-D WC, best α [min, max] |", "|---|---|---|---|"]
    for k in sorted({int(c["k"]) for c in cp}):
        reps = [c for c in cp if int(c["k"]) == k and "representative" in c["source"]]
        orc = [c for c in cp if int(c["k"]) == k and "oracle" in c["source"]]
        out.append(f"| {k} | {_f(c['emb_WC'] for c in reps)} | {_f(c['emb_WC'] for c in orc)} "
                   f"| {_f(c['highd_WC'] for c in orc)} |")
    valid = all(_b(r["V1"]) and _b(r["V2"]) and _b(r["V2b"]) and _b(r["V3"]) and _b(r["xi_gates"]) for r in recs)
    out += ["", f"**L-{variant} valid on every selection seed: {'yes' if valid else 'no'}.**"]
    return "\n".join(out), valid


def step_summary():
    cal = load_cal()
    parts, cal["valid"] = ["# Phase 5b calibration (selection seeds 0–2)", ""], {}
    for variant in VARIANTS:
        if (OUT / f"validate_{variant}.csv").exists():
            text, valid = variant_section(variant, cal)
            parts += [text, ""]
            cal["valid"][variant] = valid
    (OUT / "calibration.md").write_text("\n".join(parts) + "\n")
    save_cal(cal)
    print(cal["valid"])


def step_freeze():
    """Write results/phase5b/preregistration.md from calibration.json."""
    cal = load_cal()
    designs = [v for v, ok in cal.get("valid", {}).items() if ok]
    if not designs:
        stop("no valid Design L variant")
    rho = ", ".join(f"L-{v}: ρ = {cal['rho'][v]['rho']}" for v in VARIANTS)
    excluded = [f"L-{v} failed calibration and has no main runs" for v in VARIANTS if v not in designs]
    text = (ROOT / "scripts" / "phase5b_preregistration.template.md").read_text()
    for name, value in {"rho": rho, "designs": ", ".join("L-" + v for v in designs),
                        "excluded": ", ".join(excluded) or "none"}.items():
        text = text.replace("{{" + name + "}}", value)
    assert "{{" not in text, "unfilled marker in the pre-registration template"
    (OUT / "preregistration.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["rho", "validate", "summary", "freeze"], required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    if args.step == "rho":
        step_rho()
    elif args.step == "validate":
        step_validate(args.variant)
    elif args.step == "summary":
        step_summary()
    else:
        step_freeze()


if __name__ == "__main__":
    main()
