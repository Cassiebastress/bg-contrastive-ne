"""Phase 5a extension (requested after the main runs; labeled as an extension everywhere): the oracle same-cluster
control (o) at m_B in {40, 80} on Design J's selection seeds.

Each run starts from that seed's saved Stage 1 embedding, so it is paired with the main runs. Check first: (o) at
m_B = 20 rerun from the saved Stage 1 of selection seed 0 must reproduce the stored embedding bit for bit. The
selection rule is unchanged (highest mean signal accuracy over seeds 0-2; ties -> smaller m_B), now over
{1, 2, 5, 10, 20, 40, 80}. If the best m_B changes, (o) at the new m_B is run on the confirmation seeds and the
(c)-vs-(o) comparison is rerun.

Run with the cne-bg env:  caffeinate -i python scripts/phase5a_extend_o.py
Outputs: results/phase5a/extension_o.csv, extension_o.json, extension_o.md (embeddings gitignored).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.experiment import M_BS, healthy, paired, run_stage2
from common.repro import environment_info
from scripts.phase5a_run import CAL, OUT, ROW_FIELDS, dataset, fit_row, read_csv, stop

EXTRA_M = [40, 80]
METRICS = [("acc_s", True), ("G1", True), ("G2", False), ("G3_shares_signal", True)]


def run(stage, seed, labels, cal, a_std):
    ctx = dataset("J", seed, cal)
    s1 = {"embd": np.load(OUT / "embeddings" / f"J_{stage}_seed{seed}_stage1.npy")}
    rows, embds = [], {}
    for label in labels:
        fit = run_stage2(ctx, s1, label)
        row = fit_row(ctx, stage, label, fit, a_std)
        if not healthy(row):
            stop(f"J {stage} seed {seed} {label}: non-finite or exploding")
        rows.append(row)
        embds[label] = fit["embd"]
        print(f"J {stage} seed={seed} {label:6s} acc_s {row['acc_s']:.3f} WC {row['WC']:.3f} "
              f"({row['runtime_s']:.0f}s)", flush=True)
    return rows, embds


def a_std_of(rows, seed):
    return float(next(r for r in rows if r["run"] == "a" and int(r["seed"]) == seed)["embd_std"])


def main():
    cal = json.loads(CAL.read_text())
    sel_rows = read_csv(OUT / "runs_J_selection.csv")
    out = {"environment": environment_info(), "label": "extension (requested after the main runs)"}

    # reproduction check: (o) at m_B = 20 from the saved Stage 1 of selection seed 0
    _, e = run("selection", 0, ["o_m20"], cal, a_std_of(sel_rows, 0))
    same = bool(np.array_equal(e["o_m20"], np.load(OUT / "embeddings" / "J_selection_seed0_o_m20.npy")))
    out["check_o_m20_seed0_bit_identical"] = same
    if not same:
        stop("(o) rerun from the saved Stage 1 does not reproduce the stored embedding")

    new_rows = []
    for seed in (0, 1, 2):
        rows, embds = run("selection", seed, [f"o_m{m}" for m in EXTRA_M], cal, a_std_of(sel_rows, seed))
        new_rows += rows
        for label, emb in embds.items():
            np.save(OUT / "embeddings" / f"J_selection_seed{seed}_{label}.npy", emb)
    all_m = M_BS + EXTRA_M
    pool = sel_rows + [{k: str(v) for k, v in r.items()} for r in new_rows]
    means = {m: float(np.mean([float(r["acc_s"]) for r in pool if r["run"] == f"o_m{m}"])) for m in all_m}
    best = max(all_m, key=lambda m: (means[m], -m))
    previous = json.loads((OUT / "selection_J.json").read_text())["best_m_B"]["o"]
    out.update({"selection_means_acc_s": means, "best_m_B_o": best, "previous_best_m_B_o": previous,
                "rule": "highest mean acc_s over selection seeds 0-2; ties -> smaller m_B"})

    conf_rows = read_csv(OUT / "runs_J_confirmation.csv")
    if best != previous:
        for seed in sorted({int(r["seed"]) for r in conf_rows}):
            rows, embds = run("confirmation", seed, [f"o_m{best}"], cal, a_std_of(conf_rows, seed))
            new_rows += rows
            np.save(OUT / "embeddings" / f"J_confirmation_seed{seed}_o_m{best}.npy", embds[f"o_m{best}"])
        c = f"c_m{json.loads((OUT / 'selection_J.json').read_text())['best_m_B']['c']}"
        pool_c = conf_rows + [{k: str(v) for k, v in r.items()} for r in new_rows if r["stage"] == "confirmation"]
        out["comparisons"] = [paired(pool_c, c, f"o_m{best}", m, hib) for m, hib in METRICS]

    with open(OUT / "extension_o.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(new_rows)
    (OUT / "extension_o.json").write_text(json.dumps(out, indent=2))
    write_md(out, new_rows, conf_rows)


def write_md(out, new_rows, conf_rows):
    means, best = out["selection_means_acc_s"], out["best_m_B_o"]
    lines = ["# Extension: the oracle same-cluster control (o) at larger m_B (Design J)", "",
             "Requested after the main runs and labeled as an extension. Each run starts from its seed's saved "
             "Stage 1, so it is paired with the main runs. Rerunning (o) at m_B = 20 from the saved Stage 1 of "
             f"selection seed 0 reproduced the stored embedding bit for bit: "
             f"**{'yes' if out['check_o_m20_seed0_bit_identical'] else 'no'}**.", "",
             "## Selection seeds 0–2: mean signal accuracy of (o) by m_B", "",
             "| m_B | " + " | ".join(str(m) for m in means) + " |", "|---" * (len(means) + 1) + "|",
             "| mean signal acc | " + " | ".join(f"{v:.3f}" for v in means.values()) + " |", "",
             f"Best m_B (same rule as the main runs): **{best}** (previously {out['previous_best_m_B_o']})."]
    if "comparisons" in out:
        c_run = out["comparisons"][0]["x"]
        conf_new = [r for r in new_rows if r["stage"] == "confirmation"]
        lines += ["", f"## Confirmation seeds: (c) vs (o) at m_B = {best}", "",
                  "| run | signal acc | WC | G1 | G2 | G3 shares signal |", "|---|---|---|---|---|---|"]
        for name, rows in ((c_run, [r for r in conf_rows if r["run"] == c_run]), (f"o_m{best}", conf_new)):
            cells = []
            for m in ("acc_s", "WC", "G1", "G2", "G3_shares_signal"):
                v = np.array([float(r[m]) for r in rows])
                cells.append(f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "| metric | per-seed difference (c) − (o) | mean | verdict |", "|---|---|---|---|"]
        for cmp in out["comparisons"]:
            sign = 1 if cmp["higher_is_better"] else -1
            raw = [sign * d for d in cmp["per_seed"]]
            verdict = ("(c) better, consistent across 3 paired seeds" if cmp["consistent"]
                       else "(o) better, consistent across 3 paired seeds" if all(d < 0 for d in cmp["per_seed"])
                       else "not consistent")
            lines.append(f"| {cmp['metric']}{'' if cmp['higher_is_better'] else ' (lower is better)'} "
                         f"| {', '.join(f'{d + 0.0:+.3f}' for d in raw)} | {np.mean(raw) + 0.0:+.3f} | {verdict} |")
    else:
        lines += ["", "The best m_B did not change, so the main (c)-vs-(o) comparison stands."]
    (OUT / "extension_o.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
