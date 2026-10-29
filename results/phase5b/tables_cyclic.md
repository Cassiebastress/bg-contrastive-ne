# L-cyclic tables

## Confirmation seeds (3, 4, 5): mean [min, max]

G1, G1 at k = 200, G2 and G3 are descriptive only for Design L.

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.334 [0.328, 0.344] | 0.337 [0.332, 0.347] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 40 | 0.518 [0.503, 0.538] | 0.519 [0.503, 0.543] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 20 | 0.914 [0.912, 0.916] | 0.914 [0.912, 0.917] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 80 | 0.701 [0.689, 0.721] | 0.599 [0.588, 0.613] | 0.103 [0.098, 0.112] | 0.128 [0.126, 0.129] | 0.661 [0.653, 0.667] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) label-free cPCA → Neg-t-SNE | 0.420 [0.377, 0.442] | 0.381 [0.308, 0.440] | 0.082 [0.007, 0.137] | 0.093 [0.011, 0.152] | 0.721 [0.514, 0.979] | 0.056 [0.000, 0.083] | 0/3 |
| (e3) label-free cPCA → UMAP | 0.459 [0.389, 0.515] | 0.421 [0.338, 0.513] | 0.081 [0.008, 0.138] | 0.091 [0.009, 0.150] | 0.735 [0.539, 0.979] | 0.125 [0.000, 0.292] | 0/3 |
| best cPCA variant (labels; per-seed max) | 0.530 [0.500, 0.551] | 0.518 [0.496, 0.540] | 0.014 [0.000, 0.032] | 0.019 [0.000, 0.043] | 0.958 [0.895, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best labeled linear projection (descriptive; per-seed max) | 0.624 [0.619, 0.628] | 0.622 [0.615, 0.633] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |

## Pre-registered comparisons on WC (confirmation seeds)

(c) beats best cPCA only if it exceeds the per-seed maximum over every cPCA variant on every seed.

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs (e2) label-free cPCA → Neg-t-SNE | +0.537, +0.470, +0.474 | +0.494 | (c) better, consistent across 3 paired seeds |
| (c) vs best cPCA variant (labels; per-seed max) | +0.414, +0.361, +0.378 | +0.384 | (c) better, consistent across 3 paired seeds |
| (c) vs (o) oracle same-cluster negatives, m_B = 80 | +0.193, +0.217, +0.227 | +0.213 | (c) better, consistent across 3 paired seeds |
| (c) vs (b) uniform extra negatives, m_B = 40 | +0.376, +0.408, +0.403 | +0.395 | (c) better, consistent across 3 paired seeds |

## Descriptive comparisons on WC (no verdict)

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs best labeled linear projection (descriptive; per-seed max) | +0.285, +0.293, +0.291 | +0.290 | (c) better, consistent across 3 paired seeds |
| (c) vs (a) plain Neg-t-SNE | +0.570, +0.580, +0.589 | +0.579 | (c) better, consistent across 3 paired seeds |
| (c) vs (e3) label-free cPCA → UMAP | +0.525, +0.438, +0.402 | +0.455 | (c) better, consistent across 3 paired seeds |

## Per-cluster WC (confirmation seeds)

Per-cluster WC; a cluster counts as revealed if its WC ≥ 0.80.

| seed | run | per-cluster WC (clusters 0–5) | revealed |
|---|---|---|---|
| 3 | (c) ξ_B negatives, m_B = 20 | 0.94, 0.92, 0.88, 0.88, 0.93, 0.94 | 0,1,2,3,4,5 |
| 3 | (e2) label-free cPCA → Neg-t-SNE | 0.35, 0.47, 0.48, 0.28, 0.38, 0.30 | none |
| 3 | best cPCA variant (labels; per-seed max) (cpca_a1.061_k10) | 0.42, 0.39, 0.68, 0.36, 0.79, 0.37 | none |
| 3 | best labeled linear projection (descriptive; per-seed max) (lin_U+Q135) | 0.25, 0.99, 0.29, 0.99, 0.25, 0.99 | 1,3,5 |
| 4 | (c) ξ_B negatives, m_B = 20 | 0.90, 0.94, 0.94, 0.86, 0.92, 0.92 | 0,1,2,3,4,5 |
| 4 | (e2) label-free cPCA → Neg-t-SNE | 0.35, 0.60, 0.36, 0.33, 0.66, 0.34 | none |
| 4 | best cPCA variant (labels; per-seed max) (cpca_or_a1.061_k5) | 0.52, 0.58, 0.59, 0.59, 0.57, 0.46 | none |
| 4 | best labeled linear projection (descriptive; per-seed max) (lin_U+Q135) | 0.27, 0.99, 0.24, 0.99, 0.23, 0.99 | 1,3,5 |
| 5 | (c) ξ_B negatives, m_B = 20 | 0.88, 0.91, 0.93, 0.93, 0.92, 0.92 | 0,1,2,3,4,5 |
| 5 | (e2) label-free cPCA → Neg-t-SNE | 0.42, 0.39, 0.72, 0.30, 0.43, 0.39 | none |
| 5 | best cPCA variant (labels; per-seed max) (cpca_a0.6615_k10) | 0.65, 0.61, 0.60, 0.27, 0.55, 0.56 | none |
| 5 | best labeled linear projection (descriptive; per-seed max) (lin_U+Q135) | 0.25, 0.98, 0.27, 0.98, 0.27, 1.00 | 1,3,5 |

## cPCA variants by α and k (confirmation seeds)

Each cell: the per-seed best WC among the variants in that α range at that k, then mean [min, max] over seeds. The label-free row is the selected α at each k ((e2) is its k = 10 entry, embedded separately).

| α | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|
| α < 1 (best) | 0.252 [0.238, 0.268] | 0.257 [0.245, 0.266] | 0.489 [0.454, 0.538] | 0.352 [0.341, 0.363] |
| α > 1 (best) | 0.478 [0.454, 0.497] | 0.529 [0.500, 0.551] | 0.463 [0.442, 0.500] | 0.410 [0.381, 0.429] |
| label-free α | 0.428 [0.377, 0.484] | 0.451 [0.376, 0.538] | 0.420 [0.377, 0.442] | 0.394 [0.372, 0.429] |

## ξ_B diagnostics and data checks (confirmation seeds)

| seed | ξ_B same-y_b precision | same-y_s excess | detectability | graph signal |
|---|---|---|---|---|
| 3 | 1.000 | +0.044 | 0.924 | 0.923 |
| 4 | 1.000 | +0.037 | 0.919 | 0.921 |
| 5 | 1.000 | +0.038 | 0.920 | 0.919 |

## Excluded seeds

No confirmation seed was excluded.

## Appendix: selection seeds 0–2, every run

Selection rule: highest mean WC over selection seeds 0-2 per group; ties -> smaller m_B. Chosen: (b) m_B = 40, (c) m_B = 20, (o) m_B = 80.

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.347 [0.334, 0.365] | 0.345 [0.331, 0.355] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 1 | 0.353 [0.339, 0.366] | 0.350 [0.338, 0.363] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 2 | 0.360 [0.345, 0.372] | 0.358 [0.340, 0.372] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 5 | 0.370 [0.358, 0.383] | 0.372 [0.361, 0.386] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 10 | 0.392 [0.385, 0.401] | 0.394 [0.387, 0.401] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 20 | 0.447 [0.429, 0.458] | 0.444 [0.425, 0.457] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 40 | 0.522 [0.509, 0.535] | 0.517 [0.500, 0.526] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 1 | 0.658 [0.638, 0.688] | 0.656 [0.640, 0.688] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 2 | 0.818 [0.780, 0.837] | 0.820 [0.786, 0.838] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 5 | 0.903 [0.894, 0.908] | 0.903 [0.895, 0.909] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.913 [0.900, 0.921] | 0.914 [0.898, 0.925] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 20 | 0.914 [0.913, 0.915] | 0.913 [0.911, 0.917] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 40 | 0.895 [0.876, 0.907] | 0.891 [0.871, 0.902] | 0.006 [0.000, 0.012] | 0.016 [0.006, 0.031] | 0.991 [0.987, 0.994] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 1 | 0.378 [0.364, 0.390] | 0.376 [0.363, 0.389] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 2 | 0.405 [0.388, 0.416] | 0.402 [0.384, 0.413] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 5 | 0.487 [0.472, 0.495] | 0.486 [0.474, 0.493] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 10 | 0.560 [0.550, 0.576] | 0.557 [0.545, 0.575] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 20 | 0.630 [0.617, 0.650] | 0.621 [0.609, 0.645] | 0.000 [0.000, 0.000] | 0.007 [0.006, 0.007] | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 40 | 0.668 [0.657, 0.687] | 0.643 [0.634, 0.651] | 0.027 [0.024, 0.029] | 0.055 [0.053, 0.056] | 0.916 [0.909, 0.928] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 80 | 0.679 [0.665, 0.696] | 0.578 [0.561, 0.586] | 0.098 [0.092, 0.102] | 0.126 [0.122, 0.130] | 0.660 [0.649, 0.667] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) label-free cPCA → Neg-t-SNE | 0.367 [0.261, 0.428] | 0.340 [0.251, 0.400] | 0.106 [0.017, 0.209] | 0.115 [0.023, 0.208] | 0.602 [0.179, 0.954] | 0.083 [0.000, 0.208] | 0/3 |
| (e3) label-free cPCA → UMAP | 0.397 [0.249, 0.498] | 0.381 [0.258, 0.493] | 0.104 [0.011, 0.208] | 0.113 [0.016, 0.209] | 0.606 [0.162, 0.970] | 0.153 [0.000, 0.333] | 0/3 |

### ξ_B diagnostics and data checks (selection seeds)

| seed | ξ_B same-y_b precision | same-y_s excess | detectability | graph signal |
|---|---|---|---|---|
| 0 | 1.000 | +0.035 | 0.915 | 0.909 |
| 1 | 1.000 | +0.039 | 0.908 | 0.905 |
| 2 | 1.000 | +0.042 | 0.919 | 0.920 |
