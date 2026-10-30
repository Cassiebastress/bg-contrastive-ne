# MNIST-on-grass tables

## Confirmation seeds (3, 4, 5): mean [min, max]

| run | digit kNN accuracy | recall vs clean digits | recall vs grass patches (lower is better) | flagged seeds |
|---|---|---|---|---|
| (c) ξ_B negatives, m_B = 5 | 0.931 [0.926, 0.938] | 0.013 [0.013, 0.013] | 0.022 [0.021, 0.024] | 0/3 |
| (a) Neg-t-SNE, background term off | 0.858 [0.838, 0.871] | 0.006 [0.006, 0.006] | 0.079 [0.077, 0.082] | 0/3 |
| T1 t-SNE | 0.884 [0.872, 0.891] | 0.015 [0.014, 0.016] | 0.135 [0.133, 0.140] | 0/3 |
| T2 PCA-50 → t-SNE | 0.891 [0.883, 0.898] | 0.014 [0.012, 0.015] | 0.131 [0.127, 0.136] | 0/3 |
| T3 cPCA (label-free α, 10 comp.) → t-SNE | 0.955 [0.941, 0.979] | 0.035 [0.018, 0.069] | 0.003 [0.003, 0.004] | 0/3 |

Per seed, digit kNN accuracy:

| run | seed 3 | seed 4 | seed 5 |
|---|---|---|---|
| (c) ξ_B negatives, m_B = 5 | 0.926 | 0.938 | 0.930 |
| (a) Neg-t-SNE, background term off | 0.838 | 0.871 | 0.866 |
| T1 t-SNE | 0.872 | 0.891 | 0.890 |
| T2 PCA-50 → t-SNE | 0.883 | 0.898 | 0.893 |
| T3 cPCA (label-free α, 10 comp.) → t-SNE | 0.941 | 0.979 | 0.943 |

## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)

| comparison | per-seed difference in digit kNN accuracy ((c) − other) | mean | verdict | recall vs digits, mean difference | recall vs grass, mean difference |
|---|---|---|---|---|---|
| (c) vs T1 t-SNE | +0.054, +0.046, +0.040 | +0.047 | (c) better, consistent across 3 paired seeds | -0.002 | -0.113 |
| (c) vs T2 PCA-50 → t-SNE | +0.043, +0.040, +0.038 | +0.040 | (c) better, consistent across 3 paired seeds | -0.002 | -0.109 |
| (c) vs T3 cPCA (label-free α, 10 comp.) → t-SNE | -0.015, -0.042, -0.013 | -0.023 | T3 cPCA (label-free α, 10 comp.) → t-SNE better, consistent across 3 paired seeds | -0.022 | +0.019 |
| (c) vs (a) Neg-t-SNE, background term off | +0.088, +0.066, +0.065 | +0.073 | (c) better, consistent across 3 paired seeds | +0.007 | -0.057 |

Secondary columns: mean paired difference (c) − other; for grass recall, negative means (c) keeps less grass structure.

Determinism: (c) on seed 3 rerun: bit-identical.

## T3: label-free α per seed

| seed | label-free α (T3) |
|---|---|
| 0 | 3.455 |
| 1 | 11.25 |
| 2 | 5.541 |
| 3 | 11.25 |
| 4 | 4.375 |
| 5 | 11.25 |

## Appendix: selection seeds 0–2, every run

Selection rule: highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B. Chosen: m_B = 5.

| run | digit kNN accuracy | recall vs clean digits | recall vs grass patches (lower is better) | flagged seeds |
|---|---|---|---|---|
| (c) ξ_B negatives, m_B = 1 | 0.923 [0.910, 0.931] | 0.010 [0.009, 0.011] | 0.048 [0.046, 0.052] | 0/3 |
| (c) ξ_B negatives, m_B = 2 | 0.935 [0.934, 0.936] | 0.012 [0.012, 0.012] | 0.036 [0.035, 0.037] | 0/3 |
| (c) ξ_B negatives, m_B = 5 | 0.937 [0.936, 0.939] | 0.012 [0.011, 0.013] | 0.023 [0.022, 0.024] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.930 [0.928, 0.932] | 0.012 [0.012, 0.013] | 0.018 [0.018, 0.018] | 0/3 |
| (c) ξ_B negatives, m_B = 20 | 0.924 [0.919, 0.927] | 0.012 [0.012, 0.012] | 0.014 [0.014, 0.015] | 0/3 |
| (c) ξ_B negatives, m_B = 40 | 0.922 [0.919, 0.925] | 0.012 [0.012, 0.013] | 0.015 [0.014, 0.015] | 0/3 |
| (a) Neg-t-SNE, background term off | 0.850 [0.818, 0.876] | 0.006 [0.006, 0.007] | 0.077 [0.071, 0.081] | 0/3 |
| T1 t-SNE | 0.888 [0.885, 0.895] | 0.015 [0.014, 0.016] | 0.134 [0.128, 0.137] | 0/3 |
| T2 PCA-50 → t-SNE | 0.897 [0.890, 0.909] | 0.014 [0.012, 0.016] | 0.128 [0.124, 0.131] | 0/3 |
| T3 cPCA (label-free α, 10 comp.) → t-SNE | 0.968 [0.944, 0.985] | 0.051 [0.016, 0.083] | 0.004 [0.004, 0.004] | 0/3 |

## Cluster validation

Phase B (descriptive): the largest Mac-vs-cluster difference in WC was 0.027, below the smallest per-seed margin among the Phase 5b verdicts (0.070; L-between, (c) vs local_oracle_k5): yes. Gate (two cluster runs bit-identical): passed.
