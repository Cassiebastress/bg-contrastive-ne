# Phase 5a predictions

Each prediction is reported in REPORT_phase5a.md as **confirmed**, **not confirmed** or **not applicable**. None will be revised.

## Status when this file was committed

**Design J grid:** 34 of 39 datasets had already been evaluated. Every configuration was complete except θ = 90°, unprojected (g = 2: 1 of 3 seeds; g = 6: 0 of 3).

**Not yet run:** the grid summary, Design K calibration, and every main run.

**Rule (user instruction):** a prediction whose results already existed is marked **"descriptive, added after results"**.

## P1. J projected configurations can fail V3 only where coherence (V4) is low

- **Status:** pre-registered. It was committed in `results/phase5a/grid_J.md` (commit c93c891) before any Design J dataset was evaluated, and is restated here.
- **Check:** contradicted if any projected configuration meets both V3 and V4 on every selection seed.

## P2. If any J configuration is valid (V1–V4 on every selection seed), it will be unprojected

- **Status:** **descriptive, added after results.** 34 of 39 grid datasets, including every projected configuration, existed when this was committed.
- **Check:** confirmed if every valid configuration is unprojected. Not applicable if no configuration is valid.

## P3. On Design K, and on any valid unprojected J configuration chosen for the main runs: (c) beats (e2) on WC, and neither (c) nor (e2) passes the G1 threshold

- **Status:** pre-registered. No K result and no main-run result existed.
- **"(c) beats (e2) on WC":** (c) at its selected m_B vs (e2), consistent across 3 paired confirmation seeds. The mean difference is > 0 and every per-seed difference is > 0.
- **"Passes the G1 threshold":** G1 ≥ 0.50 on every confirmation seed, the convention of the frozen Phase 5 prep criteria. "Neither passes" means each of (c) and (e2) has G1 < 0.50 on at least one confirmation seed.
- **Scope:**
  - Checked separately for K and for J.
  - For J it applies only if the chosen configuration is unprojected; otherwise it is "not applicable".
  - For K, (e2) is the original 10-component run. The additional (e2-k20) and (e2-k40) runs are covered by the separate "beats cPCA" rule in `preregistration.md`, not by P3.
