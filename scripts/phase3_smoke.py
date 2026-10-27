"""Phase 3 smoke run: is Stage 2 with the background term stable, exact at its start, and bit-reproducible?

On the easy main dataset (seed 0), one shared Stage 1, then Stage 2 variants: none, uniform_extra and xi_b with
m_B in {5, 20}, and the (d) spectrum reference (s=0, no background term). Logged per epoch over the Stage 1 -> Stage 2
timeline: NaN checks, embedding scale (mean phi over a fixed pair sample, embedding std), loss components, and the
background term's share of the loss. Stability only: no signal/background metrics and no label-colored plots
(Phase 4 evaluates).

Run with the cne-bg env:  python scripts/phase3_smoke.py
Outputs go to results/phase3/.
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.embed import N1, N2, fit_neg, pca_init, stage2
from common.knn import knn_graph
from common.repro import environment_info, setup_determinism
from data.synthetic import MAIN_RHO, make_dataset
from method.bg_term import BackgroundTerm
from method.xi_b import build_xi_b

OUT = ROOT / "results" / "phase3"
LEVEL, SEED = "easy", 0
K_GRAPH = 15
MAX_STD_GROWTH = 10.0  # Stage 2 embedding std above this multiple of its start counts as exploding
VARIANTS = {
    "none (a)": {"bg_mode": "none"},
    "uniform_extra m_B=5": {"bg_mode": "uniform_extra", "m_B": 5},
    "uniform_extra m_B=20": {"bg_mode": "uniform_extra", "m_B": 20},
    "xi_b m_B=5": {"bg_mode": "xi_b", "m_B": 5},
    "xi_b m_B=20": {"bg_mode": "xi_b", "m_B": 20},
    "s=0 reference (d)": {"bg_mode": "none", "s": 0},
}
COLORS = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#008300", "#6250d6"]))
REPEAT = "xi_b m_B=20"


def recorded_xi_hash(level, seed):
    with open(ROOT / "results" / "phase2" / "xi_b_diagnostics.csv") as f:
        return next(r["sha256"] for r in csv.DictReader(f) if r["level"] == level and int(r["seed"]) == seed)


def checks(name, kwargs, s1, s2):
    """Plan §6 stop checks for one Stage 2 run."""
    log1, log2 = s1["log"], s2["log"]
    std2 = np.array(s2["scale"]["embd_std"])
    active = kwargs["bg_mode"] != "none" and kwargs.get("m_B", 0) > 0
    out = {
        "stage2_starts_at_stage1": bool(np.array_equal(s2["start"], s1["embd"].astype(np.float32))),
        "stage1_bg_share_all_zero": all(v == 0 for v in log1["bg_share"]),
        "stage2_bg_share": "all > 0" if active else "all == 0",
        "stage2_bg_share_ok": all(v > 0 for v in log2["bg_share"]) if active else all(v == 0 for v in log2["bg_share"]),
        "nonfinite_batches": int(sum(log1["n_nonfinite_batches"]) + sum(log2["n_nonfinite_batches"])),
        "embedding_finite_every_epoch": all(s1["scale"]["finite"]) and all(s2["scale"]["finite"]),
        "max_std_growth_stage2": float(std2.max() / std2[0]),
        "masked_heads": int(sum(log2["n_masked_heads"])),
    }
    out["stable"] = (out["nonfinite_batches"] == 0 and out["embedding_finite_every_epoch"]
                     and out["max_std_growth_stage2"] <= MAX_STD_GROWTH)
    out["all_ok"] = (out["stage2_starts_at_stage1"] and out["stage1_bg_share_all_zero"] and out["stage2_bg_share_ok"]
                     and out["stable"])
    return out


def timeline(s1, s2, key, source):
    """A per-epoch series over Stage 1 (epochs 0..N1-1) then Stage 2 (N1..N1+N2-1); scale logs skip epoch -1."""
    get = (lambda s: s["log"][key]) if source == "log" else (lambda s: s["scale"][key][1:])
    return np.arange(N1 + N2), np.concatenate([get(s1), get(s2)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_determinism(SEED)
    ds = make_dataset(LEVEL, seed=SEED, signal_to_background=MAIN_RHO[LEVEL])
    G = knn_graph(ds.X_T, k=K_GRAPH)
    xi = build_xi_b(ds.X_T, ds.X_B, seed=SEED)
    assert xi.sha256() == recorded_xi_hash(LEVEL, SEED), "xi_B differs from the Phase 2 build"

    s1 = fit_neg(pca_init(ds.X_T), G, SEED, batch_hook=BackgroundTerm("none"), n_epochs=N1)
    runs, results = {}, {}
    for name, kwargs in VARIANTS.items():
        t0 = time.time()
        s2 = stage2(s1["embd"], G, SEED, xi=xi, **kwargs)
        runs[name] = s2
        results[name] = {"params": s2["params"], "runtime_s": time.time() - t0, "checks": checks(name, kwargs, s1, s2)}
        c = results[name]["checks"]
        print(f"{name:22s} {results[name]['runtime_s']:5.1f}s  all_ok={c['all_ok']}  "
              f"std growth {c['max_std_growth_stage2']:.2f}  final bg share {s2['log']['bg_share'][-1]:.3f}",
              flush=True)

    rerun = stage2(s1["embd"], G, SEED, xi=xi, **VARIANTS[REPEAT])
    reproducible = bool(np.array_equal(rerun["embd"], runs[REPEAT]["embd"]))
    all_ok = reproducible and all(r["checks"]["all_ok"] for r in results.values())

    summary = {
        "environment": environment_info(), "dataset": {"level": LEVEL, "seed": SEED, "rho": MAIN_RHO[LEVEL]},
        "xi_b_sha256": xi.sha256(), "schedule": {"stage1_epochs": N1, "stage2_epochs": N2},
        "max_std_growth": MAX_STD_GROWTH, "rerun_bit_identical": {REPEAT: reproducible}, "all_ok": all_ok,
        "stage1": {"runtime_s": s1["runtime_s"], "log": s1["log"], "scale": s1["scale"]},
        "stage2": {name: {**results[name], "log": runs[name]["log"], "scale": runs[name]["scale"]} for name in runs},
    }
    (OUT / "smoke.json").write_text(json.dumps(summary, indent=2))

    panels = [("mean_phi", "scale", "mean phi over fixed pairs (embedding scale)", True),
              ("embd_std", "scale", "embedding std", False),
              ("bg_share", "log", "background share of the loss", False),
              ("attraction", "log", "attraction (sum per epoch)", False),
              ("uniform_repulsion", "log", "uniform repulsion (sum per epoch)", False),
              ("bg_weighted", "log", "background repulsion, gamma-weighted (sum per epoch)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for ax, (key, source, title, logy) in zip(axes.ravel(), panels):
        for name in VARIANTS:
            x, y = timeline(s1, runs[name], key, source)
            ax.plot(x, y, lw=1.2, color=COLORS[name], label=name)
        ax.axvline(N1, color="#52514e", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(f"epoch (Stage 1: 0-{N1 - 1}, Stage 2: {N1}-{N1 + N2 - 1}; dashed line = stage boundary)", fontsize=9)
    axes[0, 2].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Phase 3 smoke run, {LEVEL} seed {SEED}: stability only (no signal metrics)", fontsize=10)
    fig.savefig(OUT / "training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"all_ok": all_ok, "rerun_bit_identical": reproducible}, indent=2))


if __name__ == "__main__":
    main()
