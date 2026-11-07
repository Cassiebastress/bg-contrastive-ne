# Single-stage vs two-stage (r_a): tables (generated)

## Reuse check of the stored two-stage runs

| dataset | seed | stored at | Slurm job | code unchanged since | .npy SHA-256 matches | digit accuracy reproduced |
|---|---|---|---|---|---|---|
| digits 0/1 | 0 | `9405a42` | 6034 | **no** | yes | yes |
| digits 0/1 | 1 | `9405a42` | 6034 | **no** | yes | yes |
| digits 0/1 | 2 | `9405a42` | 6034 | **no** | yes | yes |
| digits 0/1 | 3 | `1511f6c` | 6036 | **no** | yes | yes |
| digits 0/1 | 4 | `1511f6c` | 6036 | **no** | yes | yes |
| digits 0/1 | 5 | `1511f6c` | 6036 | **no** | yes | yes |
| digits 0–9 | 0 | `065e966` | 6086 | yes | yes | yes |
| digits 0–9 | 1 | `065e966` | 6086 | yes | yes | yes |
| digits 0–9 | 2 | `065e966` | 6086 | yes | yes | yes |
| digits 0–9 | 3 | `ba1f582` | 6087 | yes | yes | yes |
| digits 0–9 | 4 | `ba1f582` | 6087 | yes | yes | yes |
| digits 0–9 | 5 | `ba1f582` | 6087 | yes | yes | yes |

Seed-3 two-stage rerun, bit-identical to the stored embedding: digits 0/1 yes, digits 0–9 yes.

## Optimization-noise reference (seed 3, two-stage)

| dataset | training seed | digit kNN accuracy | recall vs grass |
|---|---|---|---|
| digits 0/1 | 3 | 0.9712 | 0.0043 |
| digits 0/1 | 1003 | 0.9710 | 0.0042 |
| digits 0/1 | 2003 | 0.9696 | 0.0044 |
| digits 0–9 | 3 | 0.5170 | 0.0034 |
| digits 0–9 | 1003 | 0.5078 | 0.0032 |
| digits 0–9 | 2003 | 0.5132 | 0.0033 |

| dataset | noise_acc | tolerance, accuracy | noise_grass | tolerance, grass recall |
|---|---|---|---|---|
| digits 0/1 | 0.0016 | 0.0050 | 0.0001 | 0.0050 |
| digits 0–9 | 0.0092 | 0.0092 | 0.0002 | 0.0050 |

Layout agreement between those two-stage runs (descriptive, added after the results, to read the agreement numbers below against):

| dataset | pair (training seeds) | kNN overlap at 15 | Procrustes disparity |
|---|---|---|---|
| digits 0/1 | 3 vs 1003 | 0.715 | 0.015 |
| digits 0/1 | 3 vs 2003 | 0.730 | 0.010 |
| digits 0/1 | 1003 vs 2003 | 0.751 | 0.007 |
| digits 0–9 | 3 vs 1003 | 0.713 | 0.005 |
| digits 0–9 | 3 vs 2003 | 0.717 | 0.005 |
| digits 0–9 | 1003 vs 2003 | 0.716 | 0.005 |

## Digits 0/1

Matches closely on seeds 3–5: yes. Single-stage seed-3 rerun bit-identical: yes.

### Confirmation seeds 3–5 (primary)

| seed | digit acc, two-stage | digit acc, single | Δ | within ±0.0050 | grass recall, two-stage | grass recall, single | Δ | within ±0.0050 | matches closely |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.9712 | 0.9704 | -0.0008 | yes | 0.0043 | 0.0043 | +0.0001 | yes | yes |
| 4 | 0.9738 | 0.9744 | +0.0006 | yes | 0.0042 | 0.0043 | +0.0001 | yes | yes |
| 5 | 0.9698 | 0.9698 | +0.0000 | yes | 0.0041 | 0.0043 | +0.0002 | yes | yes |

Mean [min, max] over seeds 3–5:

| metric | two-stage (r_a) | r_a-single | Δ (single − two-stage) |
|---|---|---|---|
| digit kNN accuracy | 0.972 [0.970, 0.974] | 0.972 [0.970, 0.974] | -0.0001 [-0.0008, +0.0006] |
| recall vs grass (lower is better) | 0.004 [0.004, 0.004] | 0.004 [0.004, 0.004] | +0.0001 [+0.0001, +0.0002] |
| recall vs clean digits | 0.055 [0.054, 0.056] | 0.055 [0.054, 0.056] | -0.0003 [-0.0011, +0.0002] |
| ARI (k-means) | 0.873 [0.856, 0.887] | 0.874 [0.858, 0.887] | +0.0015 [+0.0000, +0.0022] |
| kNN overlap at 15, single vs two-stage | | | 0.745 [0.731, 0.764] |
| Procrustes disparity (0 = same shape) | | | 0.009 [0.005, 0.015] |
| runtime, s | 63 [63, 63] | 69 [68, 70] | |

### Selection seeds 0–2 (secondary)

| seed | digit acc, two-stage | digit acc, single | Δ | within ±0.0050 | grass recall, two-stage | grass recall, single | Δ | within ±0.0050 | matches closely |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.9720 | 0.9728 | +0.0008 | yes | 0.0041 | 0.0043 | +0.0003 | yes | yes |
| 1 | 0.9736 | 0.9706 | -0.0030 | yes | 0.0037 | 0.0035 | -0.0002 | yes | yes |
| 2 | 0.9712 | 0.9712 | +0.0000 | yes | 0.0039 | 0.0039 | +0.0000 | yes | yes |

| metric | two-stage (r_a) | r_a-single | Δ (single − two-stage) |
|---|---|---|---|
| digit kNN accuracy | 0.972 [0.971, 0.974] | 0.972 [0.971, 0.973] | -0.0007 [-0.0030, +0.0008] |
| recall vs grass (lower is better) | 0.004 [0.004, 0.004] | 0.004 [0.003, 0.004] | +0.0000 [-0.0002, +0.0003] |
| recall vs clean digits | 0.059 [0.058, 0.060] | 0.059 [0.057, 0.061] | +0.0000 [-0.0007, +0.0008] |
| ARI (k-means) | 0.872 [0.866, 0.875] | 0.874 [0.873, 0.875] | +0.0022 [-0.0007, +0.0082] |
| kNN overlap at 15, single vs two-stage | | | 0.726 [0.639, 0.772] |
| Procrustes disparity (0 = same shape) | | | 0.073 [0.001, 0.215] |
| runtime, s | 109 [101, 119] | 68 [64, 71] | |

## Digits 0–9

Matches closely on seeds 3–5: yes. Single-stage seed-3 rerun bit-identical: yes.

### Confirmation seeds 3–5 (primary)

| seed | digit acc, two-stage | digit acc, single | Δ | within ±0.0092 | grass recall, two-stage | grass recall, single | Δ | within ±0.0050 | matches closely |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.5170 | 0.5084 | -0.0086 | yes | 0.0034 | 0.0035 | +0.0001 | yes | yes |
| 4 | 0.5138 | 0.5114 | -0.0024 | yes | 0.0034 | 0.0037 | +0.0003 | yes | yes |
| 5 | 0.5102 | 0.5164 | +0.0062 | yes | 0.0039 | 0.0037 | -0.0002 | yes | yes |

Mean [min, max] over seeds 3–5:

| metric | two-stage (r_a) | r_a-single | Δ (single − two-stage) |
|---|---|---|---|
| digit kNN accuracy | 0.514 [0.510, 0.517] | 0.512 [0.508, 0.516] | -0.0016 [-0.0086, +0.0062] |
| recall vs grass (lower is better) | 0.004 [0.003, 0.004] | 0.004 [0.004, 0.004] | +0.0001 [-0.0002, +0.0003] |
| recall vs clean digits | 0.050 [0.046, 0.053] | 0.050 [0.046, 0.052] | -0.0003 [-0.0007, -0.0001] |
| ARI (k-means) | 0.225 [0.205, 0.243] | 0.237 [0.225, 0.244] | +0.0120 [-0.0010, +0.0361] |
| kNN overlap at 15, single vs two-stage | | | 0.427 [0.231, 0.730] |
| Procrustes disparity (0 = same shape) | | | 0.165 [0.005, 0.379] |
| runtime, s | 73 [62, 79] | 81 [66, 89] | |

### Selection seeds 0–2 (secondary)

| seed | digit acc, two-stage | digit acc, single | Δ | within ±0.0092 | grass recall, two-stage | grass recall, single | Δ | within ±0.0050 | matches closely |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5194 | 0.5114 | -0.0080 | yes | 0.0036 | 0.0039 | +0.0002 | yes | yes |
| 1 | 0.5108 | 0.5116 | +0.0008 | yes | 0.0037 | 0.0036 | -0.0001 | yes | yes |
| 2 | 0.5540 | 0.5484 | -0.0056 | yes | 0.0036 | 0.0035 | -0.0001 | yes | yes |

| metric | two-stage (r_a) | r_a-single | Δ (single − two-stage) |
|---|---|---|---|
| digit kNN accuracy | 0.528 [0.511, 0.554] | 0.524 [0.511, 0.548] | -0.0043 [-0.0080, +0.0008] |
| recall vs grass (lower is better) | 0.004 [0.004, 0.004] | 0.004 [0.003, 0.004] | +0.0000 [-0.0001, +0.0002] |
| recall vs clean digits | 0.050 [0.046, 0.052] | 0.049 [0.045, 0.051] | -0.0005 [-0.0010, +0.0002] |
| ARI (k-means) | 0.237 [0.220, 0.260] | 0.237 [0.222, 0.262] | +0.0007 [-0.0026, +0.0027] |
| kNN overlap at 15, single vs two-stage | | | 0.625 [0.397, 0.749] |
| Procrustes disparity (0 = same shape) | | | 0.029 [0.003, 0.077] |
| runtime, s | 86 [80, 91] | 66 [65, 66] | |

**Adoption rule outcome: single-stage (r_a) becomes the core form.**

Runtimes: the two-stage times come from the stored runs (jobs listed above), the single-stage times from this check's job; both cover the residual graph and the fit, on 1 thread.
