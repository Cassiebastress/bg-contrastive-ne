# Background-suppressed graph, Phase 2 tables

## Confirmation seeds (3, 4, 5): mean [min, max]

(r_a) and (r_c) are new; (a), (c), T1, T2 and T3 are the stored MNIST-on-grass confirmation runs (reuse check: code (a)/(c) depend on unchanged since `5556465`: yes; dataset hashes checked by every run).

| run | digit kNN accuracy | recall vs clean digits | recall vs grass patches (lower is better) | flagged seeds |
|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 1 | 0.973 [0.972, 0.974] | 0.056 [0.055, 0.057] | 0.003 [0.003, 0.004] | 0/3 |
| (r_a) residual graph, background term off | 0.972 [0.970, 0.974] | 0.055 [0.054, 0.056] | 0.004 [0.004, 0.004] | 0/3 |
| (c) raw graph + ξ_B, m_B = 5 | 0.931 [0.926, 0.938] | 0.013 [0.013, 0.013] | 0.022 [0.021, 0.024] | 0/3 |
| (a) raw graph, background term off | 0.858 [0.838, 0.871] | 0.006 [0.006, 0.006] | 0.079 [0.077, 0.082] | 0/3 |
| T1 t-SNE | 0.884 [0.872, 0.891] | 0.015 [0.014, 0.016] | 0.135 [0.133, 0.140] | 0/3 |
| T2 PCA-50 → t-SNE | 0.891 [0.883, 0.898] | 0.014 [0.012, 0.015] | 0.131 [0.127, 0.136] | 0/3 |
| T3 cPCA → t-SNE | 0.955 [0.941, 0.979] | 0.035 [0.018, 0.069] | 0.003 [0.003, 0.004] | 0/3 |

Per seed, digit kNN accuracy:

| run | seed 3 | seed 4 | seed 5 |
|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 1 | 0.973 | 0.974 | 0.972 |
| (r_a) residual graph, background term off | 0.971 | 0.974 | 0.970 |
| (c) raw graph + ξ_B, m_B = 5 | 0.926 | 0.938 | 0.930 |
| (a) raw graph, background term off | 0.838 | 0.871 | 0.866 |
| T1 t-SNE | 0.872 | 0.891 | 0.890 |
| T2 PCA-50 → t-SNE | 0.883 | 0.898 | 0.893 |
| T3 cPCA → t-SNE | 0.941 | 0.979 | 0.943 |

## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)

Differences are first run − second run.

| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean difference | recall vs grass, mean difference |
|---|---|---|---|---|---|
| (r_c) residual graph + ξ_B, m_B = 1 vs (c) raw graph + ξ_B, m_B = 5 | +0.046, +0.036, +0.041 | +0.041 | (r_c) residual graph + ξ_B, m_B = 1 better, consistent across 3 paired seeds | +0.044 | -0.019 |
| (r_c) residual graph + ξ_B, m_B = 1 vs T3 cPCA → t-SNE | +0.032, -0.006, +0.028 | +0.018 | not consistent | +0.021 | -0.000 |
| (r_c) residual graph + ξ_B, m_B = 1 vs T1 t-SNE | +0.101, +0.083, +0.081 | +0.088 | (r_c) residual graph + ξ_B, m_B = 1 better, consistent across 3 paired seeds | +0.041 | -0.132 |
| (r_c) residual graph + ξ_B, m_B = 1 vs T2 PCA-50 → t-SNE | +0.090, +0.076, +0.079 | +0.082 | (r_c) residual graph + ξ_B, m_B = 1 better, consistent across 3 paired seeds | +0.042 | -0.128 |
| (r_c) residual graph + ξ_B, m_B = 1 vs (r_a) residual graph, background term off | +0.001, +0.000, +0.002 | +0.001 | not consistent | +0.001 | -0.001 |

## Descriptive comparison

| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean difference | recall vs grass, mean difference |
|---|---|---|---|---|---|
| (r_a) residual graph, background term off vs (a) raw graph, background term off | +0.133, +0.103, +0.104 | +0.113 | (descriptive) (r_a) residual graph, background term off better, consistent across 3 paired seeds | +0.049 | -0.075 |

Determinism: (r_c) residual graph + ξ_B, m_B = 1 on seed 3 rerun: bit-identical.

## Appendix: selection seeds 0–2

Selection rule: highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B. Chosen: m_B = 1.

| run | digit kNN accuracy | recall vs clean digits | recall vs grass patches (lower is better) | flagged seeds |
|---|---|---|---|---|
| (r_a) residual graph, background term off | 0.972 [0.971, 0.974] | 0.059 [0.058, 0.060] | 0.004 [0.004, 0.004] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 1 | 0.973 [0.971, 0.974] | 0.058 [0.058, 0.059] | 0.003 [0.003, 0.004] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 2 | 0.972 [0.971, 0.973] | 0.056 [0.056, 0.057] | 0.003 [0.003, 0.003] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 5 | 0.972 [0.969, 0.974] | 0.054 [0.053, 0.055] | 0.003 [0.003, 0.003] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 10 | 0.970 [0.968, 0.972] | 0.050 [0.050, 0.050] | 0.003 [0.002, 0.003] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 20 | 0.969 [0.967, 0.971] | 0.047 [0.045, 0.047] | 0.003 [0.003, 0.003] | 0/3 |
| (r_c) residual graph + ξ_B, m_B = 40 | 0.968 [0.966, 0.969] | 0.045 [0.043, 0.046] | 0.003 [0.003, 0.003] | 0/3 |
