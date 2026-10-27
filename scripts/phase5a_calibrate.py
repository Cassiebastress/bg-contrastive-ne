"""Phase 5a calibration and validity (selection seeds 0-2; runs (a), (e2), (e3) only, never (b), (c) or the oracle
control).

Steps (run in order):
  --step rho      Phase 1 rule for rho: largest valid rho on seed 0 (J at theta = 0, and K), confirmed on seeds 0-2.
  --step grid     Design J: the theta = 0 sanity check (cPCA must solve it), then all 16 configurations.
  --step k        Design K at rho_K: V1, V2, V3-K (coherence reported).
  --step summary  grid_J.md (prediction + generated tables), validity.md, and the chosen J configuration.
  --step freeze   preregistration.md from calibration.json (commit before any (b), (c) or oracle-control run).
  --step diagnostics  descriptive data diagnostics from the clean generator components (no embeddings).

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

from common.benchmark import (EXTRA_COMPONENTS, GRID, METRICS, RHO_GRID, SELECTION_SEEDS, V1, V2_MIN_DETECTABILITY,
                              V3_J, V3_K_MAX_WC, V4_MIN_COHERENCE, config_name, cpca_runs, data_scores, fit_a, make,
                              score, v1, v2, v3_j, v3_k, v4)
from common.experiment import VALIDITY
from common.repro import assert_committed, environment_info
from eval.benchmark_metrics import background_main_effect_share, coherence, signal_background_geometry
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
    more, alpha = cpca_runs(ds, seed, EXTRA_COMPONENTS[design])
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


def step_freeze():
    """Write results/phase5a/preregistration.md from calibration.json (commit it before any (b), (c) or (o) run)."""
    cal = load_cal()
    c = cal.get("chosen_J")
    if not c:
        stop("no valid Design J configuration: report the grid instead of freezing")
    j = (f"θ = {c['theta']}°, Δ {'projected off span(B)' if c['projected'] else 'not projected'}, MLP gain "
         f"g = {c['gain']:g} (configuration {c['config']}), ρ_J = {cal['rho']['J']['rho']:.6g}")
    k_rows = read_csv(OUT / "k_validity.csv")
    k_wc = [float(r["e2_WC"]) for r in k_rows]
    k_status = ("**Design K is stopped.** It failed V3-K on every selection seed ((e2) WC "
                f"{min(k_wc):.3f}–{max(k_wc):.3f} > {V3_K_MAX_WC:.2f}), so it has no main runs. The K rules below are "
                "kept as written but are not applied." if not any(_b(r["V3"]) for r in k_rows)
                else "Design K passed calibration.")
    jr = [r for r in read_csv(OUT / "grid_J.csv") if r["config"] == c["config"]]

    def rng(key):
        v = [float(r[key]) for r in jr]
        return f"{min(v):.3f}–{max(v):.3f}"

    j_caveat = ("**Known caveat for J** (selection seeds 0–2): the chosen configuration meets V3 only through the "
                f"G1/G2 thresholds. (e2) has G1 {rng('e2_G1')} and G2 {rng('e2_G2')}, but signal accuracy "
                f"{rng('e2_acc_s')}, WC {rng('e2_WC')} and G3 shares signal {rng('e2_G3_shares_signal')}. On seed 0, "
                "its embedding is organized by signal globally, with background sub-lobes inside each signal "
                "blob. The interaction term, averaged over the signal, depends on the background cluster: a "
                "background main effect outside span(B) that X_B never shows, so cPCA cannot remove it.")
    text = f"""# Phase 5a pre-registration (frozen before any (b), (c) or oracle-control run)

## Datasets

The rule for each design's ρ is the Phase 1 rule: the largest grid value whose plain-Neg-t-SNE run (a) is valid on
seed 0 (background accuracy ≥ 0.90, signal accuracy ≤ 0.50, signal accuracy in the signal itself ≥ 0.95), confirmed
on seeds 0–2.

| Design | Generator | Setting |
|---|---|---|
| J | `data/synthetic_j.py` | {j} |
| K | `data/synthetic_k.py` | ρ_K = {cal['rho']['K']['rho']:.6g} |

In J, Ã z_s and Δ̃ are each scaled to unit total variance per dataset; the scale factors are recorded per seed.

n_T = n_B = 5000 and D = 50. Selection seeds are 0–2; confirmation seeds start at 3.

## Status at the freeze (from the calibration CSVs)

- {k_status}
- {j_caveat}

## Metrics

All metrics use k = 15 and the true labels. kNN accuracies use the existing classifier: stratified 5-fold CV, seed 0.

| Metric | Definition |
|---|---|
| signal accuracy | kNN accuracy of y_s in the 2-D embedding |
| background accuracy (G2) | kNN accuracy of y_b in the 2-D embedding |
| WC | signal kNN accuracy computed inside each true background cluster, size-weighted over clusters |
| G1 | mean fraction of a point's 15 embedding neighbors with the same y_s and a different y_b |
| G3 "shares signal" | fraction of the 24 cells (y_s × y_b) whose nearest other cell centroid shares y_s but not y_b |
| coherence (data) | mean over g and cluster pairs k ≠ k′ of cos(d_kg, d_k′g), d_kg = cell mean − cluster mean in X_T |
| detectability (data) | WC on high-D X_T |

## Runs

- (a): plain two-stage Neg-t-SNE.
- (b): uniform extra negatives.
- (c): ξ_B negatives, built as in Phase 2.
- (o): the oracle same-cluster control. m_B extra negatives are drawn uniformly from the other members of the head's
  true background cluster. It is an oracle: it uses labels.
- (b), (c) and (o) each run at m_B ∈ {{1, 2, 5, 10, 20}} and share one Stage 1 per seed.
- (e2): cPCA → Neg-t-SNE on the top 10 components. (e3): cPCA → UMAP on the same input. Both use the label-free α of
  Phase 5 prep.
- Design K only: (e2-k20) and (e2-k40), cPCA → Neg-t-SNE on the top 20 and top 40 components at the same α as (e2).
  V3-K stays defined with the 10-component (e2).

## Selection (seeds 0–2)

For each of (b), (c) and (o), the m_B with the highest mean over the selection seeds is chosen; ties go to the smaller
m_B. The selection metric is signal accuracy for J and WC for K.

## Confirmation (fresh seeds from 3)

- **Runs:** (a), best (b), best (c), best (o), (e2) and (e3); for K also (e2-k20) and (e2-k40).
- **Exclusion:** a seed is excluded if it fails V1 ((a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90), V2
  (detectability ≥ 0.85), or, for J, V4 (coherence ≥ 0.5). These criteria use the data and (a) only, never (e2).
  The next seed is then drawn, until 3 valid seeds exist. More than 2 exclusions in a design means stop and report.
- **Listing:** excluded seeds are listed with their validity numbers and enter no comparison.

## Comparisons (confirmation seeds)

- **Rule:** "consistent across 3 paired seeds" (Phase 4): the mean paired difference is > 0 and every per-seed
  difference is > 0, with the sign flipped where lower is better. Mean and per-seed values are reported, with no
  p-values.
- **Design J:** (c) vs (e2), and (c) vs (o), each on signal accuracy, G1, G2 (lower is better) and G3 "shares
  signal", reported separately. The frozen thresholds G1 ≥ 0.50, G2 ≤ 0.50 and G3 "shares signal" ≥ 0.50 are also
  checked per seed for every run.
- **Design K:** (c) vs (e2), (c) vs (e2-k20), (c) vs (e2-k40), and (c) vs (o), on WC (primary). Signal accuracy is
  secondary. (c) "beats cPCA" only if it is consistent across 3 paired seeds against each of (e2), (e2-k20) and
  (e2-k40). G1–G3 are reported but marked ill-posed for K: same-label groups in different clusters are unrelated.
- **Descriptive only** (no verdict): (c) vs (a), (c) vs best (b), and (c) vs (e3).

## Predictions

`results/phase5a/predictions.md` (P1–P3) is reported as confirmed, not confirmed or not applicable, unrevised.

## Stop and report

- ξ_B fails the Phase 2 gates (same-y_b precision ≥ 0.90, same-y_s excess ≤ 0.10) on any seed.
- Any run is non-finite or its scale grows by more than 10×.
- The rerun of selection seed 0 is not bit-identical.
- More than 2 confirmation seeds are excluded in a design.
"""
    (OUT / "preregistration.md").write_text(text)
    print(text)


def step_diagnostics():
    """Descriptive data diagnostics from the generators' clean components (no embeddings), every J configuration and
    K, selection seeds 0-2: data_diagnostics.csv and data_diagnostics.md."""
    cal = load_cal()
    rows = []
    for design, configs in (("J", GRID), ("K", [None])):
        for c in configs:
            for seed in SELECTION_SEEDS:
                ds = make(design, seed, cal["rho"][design]["rho"], c)
                t = ds.truth
                row = {"design": design, "config": config_name(c) if c else "K", "seed": seed,
                       **signal_background_geometry(t["signal_T"], t["B"], ds.X_B, t["y_b_background"]),
                       "coherence": coherence(ds.X_T, t["y_s"], t["y_b"]),
                       "bg_main_effect_share": background_main_effect_share(t["signal_T"], t["y_b"])}
                rows.append(row)
    write_csv(OUT / "data_diagnostics.csv", rows)
    rows = read_csv(OUT / "data_diagnostics.csv")
    out = ["# Phase 5a data diagnostics (descriptive)", "",
           "Computed from the generators' clean components on selection seeds 0–2, with no embeddings. Each cell is "
           "the mean [min, max] over seeds.", "",
           "- **Signal variance in span(B):** the fraction of the clean signal's variance inside the background map's "
           "span.",
           "- **Background variance along the signal:** the clean signal's top principal directions (those explaining "
           "90% of its variance) are taken first. Along them, the global background variance over all of X_B is "
           "compared with the mean within-background-cluster variance.",
           "- **Rationale:** cPCA can only use directions with low *global* background variance, while ξ_B, built from "
           "background neighborhoods, effectively uses directions with low *local* background variance. "
           "Configurations with a large global/local ratio are where (c) is predicted to have an advantage.",
           "- **Background main effect in the signal** (added after the grid results): the share of the clean "
           "signal's variance explained by the target's background-cluster means. Coherence subtracts those means, "
           "so it cannot see this effect. In J, the interaction term carries it outside span(B), where X_B never "
           "shows it.",
           "- **Caveat on coherence:** it measures alignment in ambient coordinates, so it can call a signal "
           "incoherent even when the signal is consistent in the background's own frame (for example, with a curved "
           "background). V4 is unchanged; this caveat applies to every low-coherence result.", "",
           "| configuration | signal var in span(B) | signal directions (90%) | global bg var | mean within-cluster "
           "bg var | global / local | coherence | background main effect |", "|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == name]
        n_dirs = sorted({int(r["n_signal_directions"]) for r in rs})
        out.append(f"| {name} | {_f(rs, 'frac_signal_var_in_span_B')} | {', '.join(map(str, n_dirs))} "
                   f"| {_f(rs, 'bg_global_var')} | {_f(rs, 'bg_local_var')} | {_f(rs, 'bg_global_local_ratio')} "
                   f"| {_f(rs, 'coherence')} | {_f(rs, 'bg_main_effect_share')} |")
    (OUT / "data_diagnostics.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["rho", "grid", "k", "summary", "freeze", "diagnostics"], required=True)
    args = parser.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    {"rho": step_rho, "grid": step_grid, "k": step_k, "summary": step_summary, "freeze": step_freeze,
     "diagnostics": step_diagnostics}[args.step]()


if __name__ == "__main__":
    main()
