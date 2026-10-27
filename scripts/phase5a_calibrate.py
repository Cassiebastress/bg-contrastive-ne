"""Phase 5a calibration and validity (selection seeds 0-2; runs (a), (e2), (e3) only, never (b), (c) or the oracle
control).

Steps (run in order):
  --step rho      Phase 1 rule for rho: largest valid rho on seed 0 (J at theta = 0, and K), confirmed on seeds 0-2.
  --step grid     Design J: the theta = 0 sanity check (cPCA must solve it), then all 16 configurations.
  --step k        Design K at rho_K: V1, V2, V3-K (coherence reported).
  --step summary  grid_J.md (prediction + generated tables), validity.md, and the chosen J configuration.

Run with the cne-bg env:  caffeinate -i python scripts/phase5a_calibrate.py --step <step>
Outputs: results/phase5a/rho_sweep.csv, grid_J.csv, k_validity.csv, calibration.json, grid_J.md, validity.md.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.benchmark import (GRID, METRICS, RHO_GRID, SELECTION_SEEDS, V1, V2_MIN_DETECTABILITY, V3_J, V3_K_MAX_WC,
                              V4_MIN_COHERENCE, config_name, cpca_runs, data_scores, fit_a, make, score, v1, v2,
                              v3_j, v3_k, v4)
from common.experiment import VALIDITY
from common.repro import assert_committed, environment_info
from eval.metrics import knn_accuracy

OUT = ROOT / "results" / "phase5a"
EMB = OUT / "embeddings"
CAL = OUT / "calibration.json"
MARKER = "<!-- generated below -->"


def stop(msg):
    print(f"STOP (Phase 5a plan §7): {msg}", flush=True)
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

def phase1_row(design, rho, seed):
    ds = make(design, seed, rho)
    t = ds.truth
    a = fit_a(ds, seed)
    row = {"design": design, "rho": rho, "seed": seed, "emb_acc_s": knn_accuracy(a, t["y_s"]),
           "emb_acc_b": knn_accuracy(a, t["y_b"]), "oracle_acc_s": knn_accuracy(t["signal_T"], t["y_s"])}
    row["valid"] = (row["emb_acc_b"] >= VALIDITY["min_emb_acc_b"] and row["emb_acc_s"] <= VALIDITY["max_emb_acc_s"]
                    and row["oracle_acc_s"] >= VALIDITY["min_oracle_acc_s"])
    print(f"{design} rho={rho:.4g} seed={seed} emb y_s {row['emb_acc_s']:.3f} emb y_b {row['emb_acc_b']:.3f} "
          f"oracle {row['oracle_acc_s']:.3f} valid={row['valid']}", flush=True)
    return row


def step_rho():
    rows, cal = [], load_cal()
    cal["rho"] = {}
    for design in ("J", "K"):
        sweep = [phase1_row(design, rho, 0) for rho in RHO_GRID[design]]
        rows += sweep
        write_csv(OUT / "rho_sweep.csv", rows)
        valid = [r["rho"] for r in sweep if r["valid"]]
        if not valid:
            stop(f"{design}: no valid rho on seed 0")
        rho = max(valid)
        confirm = [phase1_row(design, rho, s) for s in SELECTION_SEEDS[1:]]
        rows += confirm
        write_csv(OUT / "rho_sweep.csv", rows)
        if not all(r["valid"] for r in confirm):
            stop(f"{design}: rho = {rho:.4g} valid on seed 0 but not confirmed on seeds 1-2")
        cal["rho"][design] = {"rho": rho, "rule": "Phase 1: largest rho on the grid valid on seed 0 (emb acc_b >= "
                              "0.90, emb acc_s <= 0.50, oracle acc_s >= 0.95), confirmed on seeds 0-2",
                              "grid": RHO_GRID[design],
                              "grid_note": "J: Phase 1 grid x sqrt(16.5), J's signal having unit total variance"
                              if design == "J" else "Phase 1 grid"}
        print(f"{design}: rho = {rho:.4g}", flush=True)
    save_cal(cal)


# ---- one dataset: data diagnostics, (a), (e2), (e3) ----

def evaluate(design, seed, rho, config=None, save_prefix=None):
    ds = make(design, seed, rho, config)
    rec = {"design": design, "seed": seed, "rho": rho}
    if config is not None:
        rec.update({"config": config_name(config), **config})
    rec.update(ds.params["map_info"])
    rec.update(data_scores(ds))
    embs = {"a": fit_a(ds, seed)}
    more, alpha = cpca_runs(ds, seed)
    embs.update(more)
    for run, X in embs.items():
        rec.update({f"{run}_{k}": v for k, v in score(X, ds.truth).items()})
        if save_prefix:
            np.save(EMB / f"{save_prefix}_{run}.npy", X.astype(np.float32))
    rec.update({k: (";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v) for k, v in alpha.items()})
    run, e2 = ({m: rec[f"{r}_{m}"] for m in METRICS} for r in ("a", "e2"))
    data = {k: rec[k] for k in ("detectability", "coherence")}
    rec.update({"V1": v1(run), "V2": v2(data), "V3": v3_j(e2) if design == "J" else v3_k(e2), "V4": v4(data)})
    print(f"{design} {rec.get('config', '')} seed={seed} coh {rec['coherence']:.3f} det {rec['detectability']:.3f} | "
          f"a acc_s {rec['a_acc_s']:.3f} acc_b {rec['a_acc_b']:.3f} | e2 acc_s {rec['e2_acc_s']:.3f} "
          f"WC {rec['e2_WC']:.3f} G1 {rec['e2_G1']:.3f} G2 {rec['e2_G2']:.3f} G3 {rec['e2_G3_shares_signal']:.3f} "
          f"alpha {rec['selected_alpha']:.3g} | V1-4 {rec['V1']} {rec['V2']} {rec['V3']} {rec['V4']}", flush=True)
    return rec


def step_grid():
    assert_committed(OUT / "grid_J.md")
    rho = load_cal()["rho"]["J"]["rho"]
    rows = []
    base = {}
    for seed in SELECTION_SEEDS:  # theta = 0: one dataset per seed, shared by its four (P, g) rows
        base[seed] = evaluate("J", seed, rho, GRID[0], save_prefix=f"J_t0_seed{seed}")
        for c in GRID:
            if c["theta"] == 0:
                rows.append({**base[seed], "config": config_name(c), **c, "shared_dataset": "t0"})
        write_csv(OUT / "grid_J.csv", rows)
    if any(base[s]["V3"] for s in SELECTION_SEEDS):
        stop("sanity: theta = 0 (additive) is not solved by cPCA -> Neg-t-SNE on every selection seed")
    for c in GRID:
        if c["theta"] == 0:
            continue
        for seed in SELECTION_SEEDS:
            rows.append(evaluate("J", seed, rho, c, save_prefix=f"J_{config_name(c)}_seed{seed}"))
            write_csv(OUT / "grid_J.csv", rows)


def step_k():
    rho = load_cal()["rho"]["K"]["rho"]
    rows = []
    for seed in SELECTION_SEEDS:
        rows.append(evaluate("K", seed, rho, save_prefix=f"K_seed{seed}"))
        write_csv(OUT / "k_validity.csv", rows)
    if not all(r["V3"] for r in rows):
        stop("K fails V3-K: cPCA -> Neg-t-SNE already reveals the signal within clusters")


# ---- summary ----

def _b(x):
    return str(x) == "True"


def _f(rows, key):
    vals = [float(r[key]) for r in rows]
    return f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]"


def _count(rows, key):
    return f"{sum(_b(r[key]) for r in rows)}/{len(rows)}"


def by_config(rows):
    out = {}
    for r in rows:
        out.setdefault(r["config"], []).append(r)
    return out


def config_valid(rs):
    return all(_b(r[v]) for r in rs for v in ("V1", "V2", "V3", "V4"))


def choose(grid):
    """Largest mean coherence among valid configurations; ties -> smaller theta, projected first, smaller gain."""
    valid = [(name, rs) for name, rs in grid.items() if config_valid(rs)]
    if not valid:
        return None
    key = (lambda item: (-np.mean([float(r["coherence"]) for r in item[1]]), int(item[1][0]["theta"]),
                         not _b(item[1][0]["projected"]), float(item[1][0]["gain"])))
    return sorted(valid, key=key)[0][0]


def grid_tables(rows):
    grid = by_config(rows)
    order = [config_name(c) for c in GRID]
    out = ["## Generated tables", "",
           "Each cell is the mean [min, max] over selection seeds 0–2, and each V column counts the seeds that "
           "meet it.",
           "θ = 0° does not depend on P or g, so its four rows are one dataset per seed.", "",
           "### Data diagnostics and validity", "",
           "| configuration | θ | P | g | coherence | detectability (high-D WC) | (a) signal acc | (a) background acc "
           "| V1 | V2 | V3 | V4 | valid |", "|" + "---|" * 13]
    for name in order:
        rs = grid[name]
        r0 = rs[0]
        proj = "proj" if _b(r0["projected"]) else "unproj"
        out.append(f"| {name} | {r0['theta']}° | {proj} | {float(r0['gain']):g} "
                   f"| {_f(rs, 'coherence')} | {_f(rs, 'detectability')} | {_f(rs, 'a_acc_s')} | {_f(rs, 'a_acc_b')} "
                   f"| {_count(rs, 'V1')} | {_count(rs, 'V2')} | {_count(rs, 'V3')} | {_count(rs, 'V4')} "
                   f"| {'yes' if config_valid(rs) else 'no'} |")
    out += ["", "### cPCA runs at the label-free α", "",
            "Signal accuracy, WC, G1, G2 and G3 \"shares signal\" for (e2) cPCA → Neg-t-SNE and (e3) cPCA → UMAP. "
            "Frozen thresholds: G1 ≥ 0.50, G2 ≤ 0.50, G3 ≥ 0.50, and V3 also counts signal accuracy ≤ 0.80 as a "
            "failure.", "",
            "| configuration | (e2) signal acc | (e2) WC | (e2) G1 | (e2) G2 | (e2) G3 | (e3) signal acc | (e3) WC "
            "| (e3) G1 | (e3) G2 | (e3) G3 |", "|" + "---|" * 11]
    for name in order:
        rs = grid[name]
        cells = [_f(rs, f"{run}_{m}") for run in ("e2", "e3") for m in ("acc_s", "WC", "G1", "G2", "G3_shares_signal")]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "### α and the oracle α (descriptive)", "",
            "The oracle α is chosen by 10-D cPCA signal accuracy. It separates \"α selection failed\" from \"no linear "
            "projection works\" and does not enter V3.", "",
            "| configuration | selected α (range) | 10-D signal acc at selected α | oracle α (range) "
            "| 10-D signal acc at oracle α | (a) WC |", "|" + "---|" * 6]
    for name in order:
        rs = grid[name]
        sel = [float(r["selected_alpha"]) for r in rs]
        orc = [float(r["oracle_alpha"]) for r in rs]
        out.append(f"| {name} | {min(sel):.3g}–{max(sel):.3g} | {_f(rs, 'selected_10d')} "
                   f"| {min(orc):.3g}–{max(orc):.3g} | {_f(rs, 'oracle_10d')} | {_f(rs, 'a_WC')} |")
    out += ["", "### The interaction map M (θ > 0)", "",
            "| configuration | median abs preactivation | fraction of abs preactivations > 2 "
            "| fraction of Δ's variance in span(B) (before projection) |", "|---|---|---|---|"]
    for name in order:
        rs = grid[name]
        if rs[0]["theta"] == "0":
            continue
        out.append(f"| {name} | {_f(rs, 'median_abs_preactivation')} | {_f(rs, 'frac_abs_preactivation_gt_2')} "
                   f"| {_f(rs, 'frac_Delta_var_in_span_B')} |")
    # prediction check
    proj_both = [n for n in order if _b(grid[n][0]["projected"]) and grid[n][0]["theta"] != "0"
                 and all(_b(r["V3"]) and _b(r["V4"]) for r in grid[n])]
    unproj_both = [n for n in order if not _b(grid[n][0]["projected"])
                   and all(_b(r["V3"]) and _b(r["V4"]) for r in grid[n])]
    per_seed = [(n, sum(_b(r["V3"]) and _b(r["V4"]) for r in grid[n])) for n in order if grid[n][0]["theta"] != "0"]
    chosen = choose(grid)
    out += ["", "### Prediction check", "",
            f"- Projected configurations with V3 and V4 met on every seed (any one contradicts the prediction): "
            f"{', '.join(proj_both) if proj_both else 'none'}. "
            f"**The prediction is {'contradicted' if proj_both else 'not contradicted'}.**",
            f"- Unprojected configurations with V3 and V4 met on every seed (supports the second claim): "
            f"{', '.join(unproj_both) if unproj_both else 'none'}.",
            "- Seeds meeting both V3 and V4, per configuration with θ > 0: "
            + "; ".join(f"{n} {k}/3" for n, k in per_seed) + ".",
            "", "### Chosen configuration", "",
            "Rule: among the configurations valid on every selection seed (V1–V4), the largest mean coherence; ties go "
            "to the smaller θ, then projected before unprojected, then the smaller g.", "",
            f"**{chosen}**" if chosen else "**No configuration satisfies V1–V4 on every selection seed.**"]
    return "\n".join(out), chosen


def first_committed_prediction():
    rel = "results/phase5a/grid_J.md"
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%H", "--", rel],
                         capture_output=True, text=True, check=True).stdout.split()[-1]
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{rel}"], capture_output=True, text=True,
                          check=True).stdout
    assert text.rstrip().endswith(MARKER), "the committed prediction must end with the marker"
    return text


def validity_md(grid_rows, k_rows, cal):
    out = ["# Phase 5a validity (selection seeds 0–2)", "",
           f"ρ: J = {cal['rho']['J']['rho']:.4g} (the Phase 1 grid × √16.5; J's signal has unit total variance), "
           f"K = {cal['rho']['K']['rho']:.4g}. Each was chosen by the Phase 1 rule and confirmed on seeds 0–2.", "",
           "| Criterion | Condition |", "|---|---|",
           f"| V1 | (a) signal acc ≤ {V1['max_acc_s']:.2f} and (a) background acc ≥ {V1['min_acc_b']:.2f} |",
           f"| V2 | high-D WC (detectability) ≥ {V2_MIN_DETECTABILITY:.2f} |",
           f"| V3 (J) | (e2) fails one of G1 ≥ {V3_J['G1']:.2f}, G2 ≤ {V3_J['G2']:.2f}, "
           f"G3 ≥ {V3_J['G3_shares_signal']:.2f}, signal acc > {V3_J['acc_s']:.2f} |",
           f"| V3-K | (e2) WC ≤ {V3_K_MAX_WC:.2f} |",
           f"| V4 (J) | coherence ≥ {V4_MIN_COHERENCE:.1f} |", ""]
    chosen = (cal.get("chosen_J") or {}).get("config")
    sections = []
    if chosen:
        sections.append((f"Design J, chosen configuration {chosen}", [r for r in grid_rows if r["config"] == chosen]))
    if k_rows:
        sections.append(("Design K", k_rows))
    for title, rs in sections:
        out += [f"## {title}", "",
                "| seed | (a) signal acc | (a) background acc | detectability | coherence | (e2) signal acc | (e2) WC "
                "| (e2) G1 | (e2) G2 | (e2) G3 | V1 | V2 | V3 | V4 |", "|" + "---|" * 14]
        for r in rs:
            out.append(f"| {r['seed']} | " + " | ".join(f"{float(r[k]):.3f}" for k in (
                "a_acc_s", "a_acc_b", "detectability", "coherence", "e2_acc_s", "e2_WC", "e2_G1", "e2_G2",
                "e2_G3_shares_signal")) + " | " + " | ".join(str(r[v]) for v in ("V1", "V2", "V3", "V4")) + " |")
        out.append("")
    out.append("For K, V4 is not a criterion. Coherence is reported because it should be about 0, which makes the "
               "global metrics G1–G3 ill-posed for K.")
    return "\n".join(out) + "\n"


def step_summary():
    cal = load_cal()
    grid_rows = read_csv(OUT / "grid_J.csv")
    tables, chosen = grid_tables(grid_rows)
    (OUT / "grid_J.md").write_text(first_committed_prediction() + "\n" + tables + "\n")
    if chosen:
        r0 = next(r for r in grid_rows if r["config"] == chosen)
        cal["chosen_J"] = {"config": chosen, "theta": int(r0["theta"]), "projected": _b(r0["projected"]),
                           "gain": float(r0["gain"])}
    else:
        cal["chosen_J"] = None
    save_cal(cal)
    k_path = OUT / "k_validity.csv"
    (OUT / "validity.md").write_text(validity_md(grid_rows, read_csv(k_path) if k_path.exists() else [], cal))
    print(f"chosen J configuration: {chosen}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["rho", "grid", "k", "summary"], required=True)
    args = parser.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    {"rho": step_rho, "grid": step_grid, "k": step_k, "summary": step_summary}[args.step]()


if __name__ == "__main__":
    main()
