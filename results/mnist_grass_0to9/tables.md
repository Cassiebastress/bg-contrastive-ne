# MNIST-on-grass 0–9 tables

## Confirmation seeds (3, 4, 5): mean [min, max]

Chance digit kNN accuracy is 0.10. ARI is secondary, with no verdicts.

| run | digit kNN accuracy | recall vs clean digits | recall vs grass (lower is better) | ARI (k-means, k = 10) | flagged seeds |
|---|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 5 | 0.530 [0.521, 0.544] | 0.052 [0.048, 0.054] | 0.003 [0.003, 0.003] | 0.244 [0.237, 0.248] | 0/3 |
| (r_a) residual graph, background term off | 0.514 [0.510, 0.517] | 0.050 [0.046, 0.053] | 0.004 [0.003, 0.004] | 0.225 [0.205, 0.243] | 0/3 |
| (c) raw graph + ξ_B, m_B = 20 | 0.226 [0.214, 0.236] | 0.009 [0.008, 0.009] | 0.019 [0.018, 0.020] | 0.031 [0.026, 0.035] | 0/3 |
| (a) raw graph, background term off | 0.115 [0.111, 0.119] | 0.003 [0.003, 0.003] | 0.097 [0.095, 0.099] | 0.001 [0.001, 0.002] | 0/3 |
| T1 t-SNE | 0.253 [0.252, 0.256] | 0.013 [0.013, 0.014] | 0.142 [0.140, 0.144] | 0.002 [0.001, 0.003] | 0/3 |
| T2 PCA-50 → t-SNE | 0.255 [0.251, 0.257] | 0.013 [0.013, 0.013] | 0.141 [0.140, 0.144] | 0.002 [0.002, 0.003] | 0/3 |
| T3 cPCA → t-SNE | 0.433 [0.408, 0.482] | 0.054 [0.044, 0.072] | 0.003 [0.003, 0.004] | 0.117 [0.105, 0.132] | 0/3 |

Per seed, digit kNN accuracy:

| run | seed 3 | seed 4 | seed 5 |
|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 5 | 0.521 | 0.544 | 0.525 |
| (r_a) residual graph, background term off | 0.517 | 0.514 | 0.510 |
| (c) raw graph + ξ_B, m_B = 20 | 0.214 | 0.228 | 0.236 |
| (a) raw graph, background term off | 0.111 | 0.116 | 0.119 |
| T1 t-SNE | 0.252 | 0.252 | 0.256 |
| T2 PCA-50 → t-SNE | 0.251 | 0.257 | 0.256 |
| T3 cPCA → t-SNE | 0.482 | 0.408 | 0.411 |

Per seed, ARI:

| run | seed 3 | seed 4 | seed 5 |
|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 5 | 0.237 | 0.248 | 0.248 |
| (r_a) residual graph, background term off | 0.243 | 0.205 | 0.226 |
| (c) raw graph + ξ_B, m_B = 20 | 0.026 | 0.035 | 0.031 |
| (a) raw graph, background term off | 0.002 | 0.002 | 0.001 |
| T1 t-SNE | 0.003 | 0.001 | 0.001 |
| T2 PCA-50 → t-SNE | 0.002 | 0.002 | 0.003 |
| T3 cPCA → t-SNE | 0.132 | 0.113 | 0.105 |

## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)

Differences are first run − second run; secondary columns are mean paired differences, without verdicts.

| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean difference | recall vs grass, mean difference | ARI, mean difference |
|---|---|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 5 vs T1 t-SNE | +0.269, +0.293, +0.268 | +0.277 | (r_c) better, consistent across 3 paired seeds | +0.038 | -0.139 | +0.242 |
| (r_c) residual graph + ξ_B, m_B = 5 vs T2 PCA-50 → t-SNE | +0.270, +0.287, +0.268 | +0.275 | (r_c) better, consistent across 3 paired seeds | +0.039 | -0.139 | +0.242 |
| (r_c) residual graph + ξ_B, m_B = 5 vs T3 cPCA → t-SNE | +0.039, +0.137, +0.114 | +0.097 | (r_c) better, consistent across 3 paired seeds | -0.002 | -0.001 | +0.127 |
| (r_a) residual graph, background term off vs T3 cPCA → t-SNE | +0.035, +0.106, +0.100 | +0.080 | (r_a) better, consistent across 3 paired seeds | -0.004 | +0.000 | +0.108 |
| (r_c) residual graph + ξ_B, m_B = 5 vs (c) raw graph + ξ_B, m_B = 20 | +0.307, +0.316, +0.289 | +0.304 | (r_c) better, consistent across 3 paired seeds | +0.043 | -0.017 | +0.214 |
| (r_c) residual graph + ξ_B, m_B = 5 vs (r_a) residual graph, background term off | +0.004, +0.031, +0.014 | +0.016 | (r_c) better, consistent across 3 paired seeds | +0.002 | -0.001 | +0.019 |

## Descriptive comparisons (no verdicts)

| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean difference | recall vs grass, mean difference | ARI, mean difference |
|---|---|---|---|---|---|---|
| (a) raw graph, background term off vs T1 t-SNE | -0.141, -0.136, -0.137 | -0.138 | (descriptive) T1 better, consistent across 3 paired seeds | -0.010 | -0.045 | -0.000 |
| (c) raw graph + ξ_B, m_B = 20 vs T1 t-SNE | -0.038, -0.023, -0.020 | -0.027 | (descriptive) T1 better, consistent across 3 paired seeds | -0.005 | -0.123 | +0.029 |
| (c) raw graph + ξ_B, m_B = 20 vs T2 PCA-50 → t-SNE | -0.037, -0.029, -0.021 | -0.029 | (descriptive) T2 better, consistent across 3 paired seeds | -0.004 | -0.122 | +0.028 |
| (c) raw graph + ξ_B, m_B = 20 vs T3 cPCA → t-SNE | -0.267, -0.179, -0.175 | -0.207 | (descriptive) T3 better, consistent across 3 paired seeds | -0.045 | +0.016 | -0.086 |

## Per-digit kNN accuracy (confirmation seeds, mean)

| run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 5 | 0.763 | 0.866 | 0.471 | 0.494 | 0.453 | 0.289 | 0.686 | 0.593 | 0.339 | 0.346 |
| (r_a) residual graph, background term off | 0.747 | 0.867 | 0.427 | 0.485 | 0.473 | 0.283 | 0.621 | 0.565 | 0.339 | 0.329 |
| (c) raw graph + ξ_B, m_B = 20 | 0.461 | 0.451 | 0.175 | 0.207 | 0.215 | 0.097 | 0.197 | 0.225 | 0.081 | 0.153 |
| (a) raw graph, background term off | 0.217 | 0.164 | 0.111 | 0.119 | 0.097 | 0.096 | 0.107 | 0.101 | 0.069 | 0.071 |
| T1 t-SNE | 0.462 | 0.403 | 0.243 | 0.252 | 0.200 | 0.137 | 0.286 | 0.257 | 0.140 | 0.152 |
| T2 PCA-50 → t-SNE | 0.445 | 0.421 | 0.240 | 0.252 | 0.198 | 0.148 | 0.282 | 0.270 | 0.143 | 0.151 |
| T3 cPCA → t-SNE | 0.589 | 0.685 | 0.283 | 0.529 | 0.329 | 0.230 | 0.499 | 0.480 | 0.312 | 0.397 |

## Determinism

- T1 t-SNE on seed 0 (selection) rerun: bit-identical.
- (r_c) residual graph + ξ_B, m_B = 5 on seed 3 (confirmation) rerun: bit-identical.

## T3: label-free α per seed

| seed | label-free α (T3) |
|---|---|
| 0 | 5.541 |
| 1 | 5.541 |
| 2 | 5.541 |
| 3 | 4.375 |
| 4 | 5.541 |
| 5 | 5.541 |

## Beside the 0/1 runs (descriptive; confirmation seeds 3–5, all cluster results)

0/1 values are the stored MNIST-on-grass and background-suppressed graph confirmation runs, each method at the m_B selected in its own phase. Chance is 0.50 there and 0.10 here, so accuracies are not directly comparable. The 0/1 ARI (k-means, k = 2) was computed post hoc from the saved embeddings (`ari_01_posthoc.csv`); it was not pre-registered.

| run | m_B (0–9 / 0/1) | digit kNN accuracy, 0–9 | digit kNN accuracy, 0/1 | ARI, 0–9 | ARI, 0/1 (post hoc) | recall vs grass, 0–9 | recall vs grass, 0/1 |
|---|---|---|---|---|---|---|---|
| (r_c) | 5 / 1 | 0.530 [0.521, 0.544] | 0.973 [0.972, 0.974] | 0.244 [0.237, 0.248] | 0.874 [0.857, 0.887] | 0.003 [0.003, 0.003] | 0.003 [0.003, 0.004] |
| (r_a) | — | 0.514 [0.510, 0.517] | 0.972 [0.970, 0.974] | 0.225 [0.205, 0.243] | 0.873 [0.856, 0.887] | 0.004 [0.003, 0.004] | 0.004 [0.004, 0.004] |
| (c) | 20 / 5 | 0.226 [0.214, 0.236] | 0.931 [0.926, 0.938] | 0.031 [0.026, 0.035] | 0.031 [0.020, 0.040] | 0.019 [0.018, 0.020] | 0.022 [0.021, 0.024] |
| (a) | — | 0.115 [0.111, 0.119] | 0.858 [0.838, 0.871] | 0.001 [0.001, 0.002] | 0.005 [0.002, 0.009] | 0.097 [0.095, 0.099] | 0.079 [0.077, 0.082] |
| T1 | — | 0.253 [0.252, 0.256] | 0.884 [0.872, 0.891] | 0.002 [0.001, 0.003] | 0.007 [0.003, 0.009] | 0.142 [0.140, 0.144] | 0.135 [0.133, 0.140] |
| T2 | — | 0.255 [0.251, 0.257] | 0.891 [0.883, 0.898] | 0.002 [0.002, 0.003] | 0.014 [0.004, 0.030] | 0.141 [0.140, 0.144] | 0.131 [0.127, 0.136] |
| T3 | — | 0.433 [0.408, 0.482] | 0.955 [0.941, 0.979] | 0.117 [0.105, 0.132] | 0.797 [0.733, 0.905] | 0.003 [0.003, 0.004] | 0.003 [0.003, 0.004] |

## Appendix: selection seeds 0–2, every run

Selection rule: highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B. Chosen: m_B = 20 for (c), m_B = 5 for (r_c).

| run | digit kNN accuracy | recall vs clean digits | recall vs grass (lower is better) | ARI (k-means, k = 10) | flagged seeds |
|---|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 1 | 0.525 [0.501, 0.554] | 0.050 [0.046, 0.052] | 0.003 [0.003, 0.003] | 0.237 [0.218, 0.262] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 2 | 0.534 [0.514, 0.550] | 0.050 [0.048, 0.052] | 0.003 [0.003, 0.003] | 0.238 [0.219, 0.259] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 5 | 0.534 [0.515, 0.555] | 0.050 [0.049, 0.051] | 0.003 [0.002, 0.003] | 0.239 [0.217, 0.258] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 10 | 0.526 [0.513, 0.550] | 0.048 [0.047, 0.050] | 0.003 [0.002, 0.003] | 0.239 [0.215, 0.261] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 20 | 0.526 [0.505, 0.539] | 0.047 [0.045, 0.049] | 0.002 [0.002, 0.003] | 0.231 [0.209, 0.252] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 40 | 0.524 [0.509, 0.535] | 0.045 [0.042, 0.048] | 0.003 [0.002, 0.003] | 0.228 [0.204, 0.249] | 0/3 |
| (r_a) residual graph, background term off | 0.528 [0.511, 0.554] | 0.050 [0.046, 0.052] | 0.004 [0.004, 0.004] | 0.237 [0.220, 0.260] | 0/3 |
| (c) raw graph + ξ_B, m_B = 1 | 0.168 [0.155, 0.182] | 0.005 [0.005, 0.006] | 0.075 [0.073, 0.076] | 0.005 [0.003, 0.006] | 0/3 |
| (c) raw graph + ξ_B, m_B = 2 | 0.183 [0.171, 0.196] | 0.006 [0.006, 0.007] | 0.057 [0.055, 0.058] | 0.010 [0.010, 0.010] | 0/3 |
| (c) raw graph + ξ_B, m_B = 5 | 0.208 [0.194, 0.222] | 0.007 [0.007, 0.008] | 0.034 [0.034, 0.035] | 0.018 [0.016, 0.021] | 0/3 |
| (c) raw graph + ξ_B, m_B = 10 | 0.214 [0.203, 0.225] | 0.008 [0.007, 0.008] | 0.024 [0.023, 0.025] | 0.023 [0.020, 0.026] | 0/3 |
| (c) raw graph + ξ_B, m_B = 20 | 0.217 [0.195, 0.232] | 0.008 [0.008, 0.009] | 0.019 [0.018, 0.021] | 0.029 [0.022, 0.037] | 0/3 |
| (c) raw graph + ξ_B, m_B = 40 | 0.215 [0.202, 0.225] | 0.008 [0.008, 0.009] | 0.019 [0.017, 0.019] | 0.028 [0.023, 0.034] | 0/3 |
| (a) raw graph, background term off | 0.114 [0.106, 0.121] | 0.003 [0.003, 0.004] | 0.097 [0.094, 0.099] | 0.000 [0.000, 0.001] | 0/3 |
| T1 t-SNE | 0.246 [0.243, 0.250] | 0.013 [0.013, 0.013] | 0.143 [0.139, 0.146] | 0.002 [0.001, 0.003] | 0/3 |
| T2 PCA-50 → t-SNE | 0.255 [0.253, 0.257] | 0.013 [0.013, 0.014] | 0.139 [0.136, 0.141] | 0.002 [0.001, 0.005] | 0/3 |
| T3 cPCA → t-SNE | 0.418 [0.401, 0.439] | 0.047 [0.042, 0.052] | 0.003 [0.003, 0.004] | 0.109 [0.106, 0.112] | 0/3 |
