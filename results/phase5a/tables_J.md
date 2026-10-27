# Design J tables

## Confirmation seeds (3, 4, 5): mean [min, max]

| run | signal acc | WC | G1 | G2 (background acc) | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.302 [0.289, 0.316] | 0.308 [0.301, 0.317] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 20 | 0.371 [0.346, 0.400] | 0.375 [0.347, 0.403] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.858 [0.830, 0.900] | 0.858 [0.830, 0.904] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 20 | 0.472 [0.445, 0.514] | 0.474 [0.440, 0.513] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) cPCA → Neg-t-SNE | 0.984 [0.983, 0.986] | 0.984 [0.983, 0.986] | 0.454 [0.119, 0.634] | 0.602 [0.434, 0.899] | 1.000 [1.000, 1.000] | 0/3 |
| (e3) cPCA → UMAP | 0.983 [0.980, 0.986] | 0.983 [0.981, 0.986] | 0.419 [0.114, 0.594] | 0.644 [0.480, 0.907] | 1.000 [1.000, 1.000] | 0/3 |

## Frozen G1–G3 thresholds per confirmation seed

Frozen thresholds, per seed: G1 ≥ 0.50, G2 ≤ 0.50, G3 shares signal ≥ 0.50.

| run | seeds meeting G1 | seeds meeting G2 | seeds meeting G3 | seeds meeting all three |
|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0/3 | 0/3 | 0/3 | 0/3 |
| (b) uniform extra negatives, m_B = 20 | 0/3 | 0/3 | 0/3 | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0/3 | 0/3 | 0/3 | 0/3 |
| (o) oracle same-cluster negatives, m_B = 20 | 0/3 | 0/3 | 0/3 | 0/3 |
| (e2) cPCA → Neg-t-SNE | 2/3 | 2/3 | 3/3 | 2/3 |
| (e3) cPCA → UMAP | 2/3 | 1/3 | 3/3 | 1/3 |

## Pre-registered comparisons (confirmation seeds)

Primary metrics: acc_s, G1, G2, G3_shares_signal.

| comparison | metric | per-seed difference (x − y) | mean | verdict |
|---|---|---|---|---|
| (c) vs (e2) | acc_s | -0.155, -0.086, -0.138 | -0.126 | e2 better, consistent across 3 paired seeds |
| (c) vs (e2) | G1 | -0.634, -0.610, -0.119 | -0.454 | e2 better, consistent across 3 paired seeds |
| (c) vs (e2) | G2 (lower is better) | +0.566, +0.528, +0.101 | +0.398 | e2 better, consistent across 3 paired seeds |
| (c) vs (e2) | G3_shares_signal | -1.000, -1.000, -1.000 | -1.000 | e2 better, consistent across 3 paired seeds |
| (c) vs (o) | acc_s | +0.385, +0.386, +0.388 | +0.386 | c better, consistent across 3 paired seeds |
| (c) vs (o) | G1 | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (o) | G2 (lower is better) | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (o) | G3_shares_signal | +0.000, +0.000, +0.000 | +0.000 | not consistent |



## Descriptive comparisons (no verdict)

| comparison | metric | per-seed difference (x − y) | mean | verdict |
|---|---|---|---|---|
| (c) vs (a) | acc_s | +0.541, +0.584, +0.543 | +0.556 | c better, consistent across 3 paired seeds |
| (c) vs (a) | G1 | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (a) | G2 (lower is better) | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (a) | G3_shares_signal | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (b) | acc_s | +0.484, +0.500, +0.478 | +0.487 | c better, consistent across 3 paired seeds |
| (c) vs (b) | G1 | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (b) | G2 (lower is better) | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (b) | G3_shares_signal | +0.000, +0.000, +0.000 | +0.000 | not consistent |
| (c) vs (e3) | acc_s | -0.151, -0.086, -0.136 | -0.124 | e3 better, consistent across 3 paired seeds |
| (c) vs (e3) | G1 | -0.594, -0.549, -0.114 | -0.419 | e3 better, consistent across 3 paired seeds |
| (c) vs (e3) | G2 (lower is better) | +0.520, +0.456, +0.093 | +0.356 | e3 better, consistent across 3 paired seeds |
| (c) vs (e3) | G3_shares_signal | -1.000, -1.000, -1.000 | -1.000 | e3 better, consistent across 3 paired seeds |

## ξ_B diagnostics and data checks (confirmation seeds)

| seed | same-y_b precision of ξ_B | same-y_s excess | gates | coherence | detectability |
|---|---|---|---|---|---|
| 3 | 1.000 | +0.014 | pass | 0.812 | 0.855 |
| 4 | 1.000 | +0.020 | pass | 0.829 | 0.886 |
| 5 | 1.000 | +0.016 | pass | 0.790 | 0.863 |

## Excluded seeds

No confirmation seed was excluded.

## Appendix: selection seeds 0–2, every run

Selection rule: highest mean acc_s over selection seeds 0-2 per group; ties -> smaller m_B. Chosen: (b) m_B = 20, (c) m_B = 10, (o) m_B = 20.

| run | signal acc | WC | G1 | G2 (background acc) | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.297 [0.286, 0.310] | 0.299 [0.285, 0.311] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 1 | 0.303 [0.288, 0.321] | 0.305 [0.290, 0.312] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 2 | 0.300 [0.284, 0.315] | 0.301 [0.291, 0.309] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 5 | 0.302 [0.291, 0.311] | 0.309 [0.296, 0.317] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 10 | 0.334 [0.313, 0.353] | 0.334 [0.315, 0.347] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 20 | 0.358 [0.350, 0.362] | 0.361 [0.355, 0.368] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 1 | 0.587 [0.537, 0.622] | 0.587 [0.539, 0.624] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 2 | 0.774 [0.754, 0.809] | 0.773 [0.753, 0.807] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 5 | 0.849 [0.820, 0.877] | 0.849 [0.820, 0.876] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.869 [0.846, 0.891] | 0.868 [0.844, 0.889] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 20 | 0.856 [0.836, 0.878] | 0.857 [0.839, 0.878] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 1 | 0.313 [0.299, 0.322] | 0.315 [0.298, 0.326] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 2 | 0.337 [0.329, 0.341] | 0.341 [0.334, 0.347] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 5 | 0.382 [0.379, 0.387] | 0.380 [0.373, 0.390] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 10 | 0.421 [0.406, 0.429] | 0.425 [0.408, 0.434] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 20 | 0.455 [0.440, 0.463] | 0.462 [0.448, 0.471] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) cPCA → Neg-t-SNE | 0.986 [0.983, 0.988] | 0.986 [0.983, 0.988] | 0.122 [0.104, 0.145] | 0.900 [0.879, 0.917] | 1.000 [1.000, 1.000] | 0/3 |
| (e3) cPCA → UMAP | 0.984 [0.981, 0.987] | 0.984 [0.981, 0.987] | 0.121 [0.110, 0.141] | 0.907 [0.893, 0.915] | 1.000 [1.000, 1.000] | 0/3 |

### ξ_B diagnostics and data checks (selection seeds)

| seed | same-y_b precision of ξ_B | same-y_s excess | gates | coherence | detectability |
|---|---|---|---|---|---|
| 0 | 1.000 | +0.014 | pass | 0.858 | 0.887 |
| 1 | 1.000 | +0.016 | pass | 0.792 | 0.851 |
| 2 | 1.000 | +0.013 | pass | 0.818 | 0.861 |
