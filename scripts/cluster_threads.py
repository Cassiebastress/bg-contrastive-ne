"""Cluster thread-layout timing (no experiment result; writes only to the --out folder).

One process = L-cyclic selection-stage dataset for --seed, Stage 1, then (c) m_B = 20, timed in parts. The thread
count comes from CNE_NUM_THREADS (set by cluster/parallel_lib.sh). cluster/threads.sbatch runs it in layouts that
fill the job's CPUs differently (processes x threads); --summarize prints per-layout throughput.

    python scripts/cluster_threads.py --seed 0 --layout 6x4 --out ~/cluster_threads/<job>
    python scripts/cluster_threads.py --summarize ~/cluster_threads/<job>
"""
import time

T_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(seed, layout, out):
    import scripts.phase5b_run as r
    from common.experiment import run_stage1, run_stage2
    from common.repro import environment_info

    t = {"imports": time.perf_counter() - T_START}
    cal = json.loads(r.CAL.read_text())
    t0 = time.perf_counter()
    ctx = r.dataset("cyclic", seed, cal)
    t["dataset_and_xi_b"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s1 = run_stage1(ctx)
    t["stage1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit = run_stage2(ctx, s1, "c_m20")
    t["stage2_c_m20"] = time.perf_counter() - t0
    t["total"] = time.perf_counter() - T_START
    env = environment_info()
    rec = {"seed": seed, "layout": layout, "threads": env["torch_num_threads"], "seconds": t,
           "embd_std": float(fit["embd"].std()), "environment": env}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{layout}_seed{seed}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}, default=str))


def summarize(out):
    recs = [json.loads(p.read_text()) for p in sorted(out.glob("*_seed*.json"))]
    print("| layout | processes | threads | mean stage 2 (s) | mean process total (s) | slowest process (s) "
          "| runs per hour |")
    print("|---|---|---|---|---|---|---|")
    for layout in dict.fromkeys(r["layout"] for r in recs):
        rr = [r for r in recs if r["layout"] == layout]
        s2 = sum(r["seconds"]["stage2_c_m20"] for r in rr) / len(rr)
        tot = [r["seconds"]["total"] for r in rr]
        print(f"| {layout} | {len(rr)} | {rr[0]['threads']} | {s2:.0f} | {sum(tot) / len(tot):.0f} | {max(tot):.0f} "
              f"| {3600 * len(rr) / max(tot):.0f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--layout")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    run(args.seed, args.layout, args.out)


if __name__ == "__main__":
    main()
