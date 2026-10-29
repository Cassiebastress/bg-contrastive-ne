# L-between tables

## Confirmation seeds (3, 4, 5): mean [min, max]

G1, G1 at k = 200, G2 and G3 are descriptive only for Design L.

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.349 [0.346, 0.352] | 0.344 [0.337, 0.349] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 40 | 0.522 [0.516, 0.529] | 0.522 [0.510, 0.534] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.914 [0.910, 0.921] | 0.915 [0.911, 0.921] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 80 | 0.691 [0.684, 0.699] | 0.583 [0.572, 0.590] | 0.097 [0.092, 0.104] | 0.127 [0.120, 0.132] | 0.654 [0.647, 0.663] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) label-free cPCA → Neg-t-SNE | 0.255 [0.244, 0.262] | 0.257 [0.249, 0.262] | 0.202 [0.190, 0.209] | 0.202 [0.193, 0.207] | 0.218 [0.166, 0.303] | 0.125 [0.042, 0.208] | 0/3 |
| (e3) label-free cPCA → UMAP | 0.249 [0.239, 0.266] | 0.255 [0.249, 0.260] | 0.201 [0.190, 0.207] | 0.202 [0.191, 0.208] | 0.232 [0.186, 0.321] | 0.181 [0.000, 0.333] | 0/3 |
| best cPCA variant (labels; per-seed max) | 0.989 [0.988, 0.992] | 0.989 [0.988, 0.992] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best labeled linear projection (descriptive; per-seed max) | 0.989 [0.988, 0.991] | 0.989 [0.988, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |

## Pre-registered comparisons on WC (confirmation seeds)

(c) beats best cPCA only if it exceeds the per-seed maximum over every cPCA variant on every seed.

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs (e2) label-free cPCA → Neg-t-SNE | +0.677, +0.653, +0.648 | +0.659 | (c) better, consistent across 3 paired seeds |
| (c) vs best cPCA variant (labels; per-seed max) | -0.070, -0.077, -0.078 | -0.075 | best cPCA variant (labels; per-seed max) better, consistent across 3 paired seeds |
| (c) vs (o) oracle same-cluster negatives, m_B = 80 | +0.222, +0.227, +0.219 | +0.223 | (c) better, consistent across 3 paired seeds |
| (c) vs (b) uniform extra negatives, m_B = 40 | +0.399, +0.395, +0.382 | +0.392 | (c) better, consistent across 3 paired seeds |

## Descriptive comparisons on WC (no verdict)

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs best labeled linear projection (descriptive; per-seed max) | -0.070, -0.077, -0.078 | -0.075 | best labeled linear projection (descriptive; per-seed max) better, consistent across 3 paired seeds |
| (c) vs (a) plain Neg-t-SNE | +0.569, +0.565, +0.562 | +0.565 | (c) better, consistent across 3 paired seeds |
| (c) vs (e3) label-free cPCA → UMAP | +0.682, +0.670, +0.644 | +0.665 | (c) better, consistent across 3 paired seeds |

## Per-cluster WC (confirmation seeds)

Per-cluster WC; a cluster counts as revealed if its WC ≥ 0.80.

| seed | run | per-cluster WC (clusters 0–5) | revealed |
|---|---|---|---|
| 3 | (c) ξ_B negatives, m_B = 10 | 0.92, 0.94, 0.92, 0.87, 0.93, 0.94 | 0,1,2,3,4,5 |
| 3 | (e2) label-free cPCA → Neg-t-SNE | 0.24, 0.24, 0.23, 0.27, 0.26, 0.22 | none |
| 3 | best cPCA variant (labels; per-seed max) (cpca_or_a0.6615_k5) | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | best labeled linear projection (descriptive; per-seed max) (lin_U) | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | (c) ξ_B negatives, m_B = 10 | 0.92, 0.91, 0.91, 0.88, 0.92, 0.94 | 0,1,2,3,4,5 |
| 4 | (e2) label-free cPCA → Neg-t-SNE | 0.23, 0.27, 0.24, 0.27, 0.28, 0.26 | none |
| 4 | best cPCA variant (labels; per-seed max) (cpca_a0_k5) | 0.99, 0.99, 0.99, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | best labeled linear projection (descriptive; per-seed max) (lin_U) | 0.99, 0.99, 0.99, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | (c) ξ_B negatives, m_B = 10 | 0.87, 0.93, 0.94, 0.92, 0.89, 0.93 | 0,1,2,3,4,5 |
| 5 | (e2) label-free cPCA → Neg-t-SNE | 0.31, 0.26, 0.28, 0.24, 0.24, 0.24 | none |
| 5 | best cPCA variant (labels; per-seed max) (cpca_a0_k5) | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | best labeled linear projection (descriptive; per-seed max) (lin_U) | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |

## cPCA variants by α and k (confirmation seeds)

Each cell: the per-seed best WC among the variants in that α range at that k, then mean [min, max] over seeds. The label-free row is the selected α at each k ((e2) is its k = 10 entry, embedded separately).

| α | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|
| α < 1 (best) | 0.988 [0.987, 0.988] | 0.989 [0.988, 0.992] | 0.352 [0.342, 0.358] | 0.354 [0.345, 0.362] |
| α > 1 (best) | 0.254 [0.244, 0.263] | 0.257 [0.246, 0.262] | 0.258 [0.252, 0.264] | 0.256 [0.252, 0.259] |
| label-free α | 0.249 [0.241, 0.258] | 0.251 [0.244, 0.262] | 0.255 [0.244, 0.262] | 0.243 [0.239, 0.245] |

## ξ_B diagnostics and data checks (confirmation seeds)

| seed | ξ_B same-y_b precision | same-y_s excess | detectability | graph signal |
|---|---|---|---|---|
| 3 | 1.000 | +0.040 | 0.924 | 0.924 |
| 4 | 1.000 | +0.037 | 0.915 | 0.915 |
| 5 | 1.000 | +0.037 | 0.919 | 0.913 |

## Excluded seeds

No confirmation seed was excluded.

## Appendix: selection seeds 0–2, every run

Selection rule: highest mean WC over selection seeds 0-2 per group; ties -> smaller m_B. Chosen: (b) m_B = 40, (c) m_B = 10, (o) m_B = 80.

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.347 [0.337, 0.355] | 0.341 [0.332, 0.352] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 1 | 0.353 [0.343, 0.362] | 0.347 [0.332, 0.357] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 2 | 0.361 [0.352, 0.367] | 0.351 [0.340, 0.358] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 5 | 0.377 [0.364, 0.392] | 0.370 [0.361, 0.375] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 10 | 0.412 [0.397, 0.422] | 0.402 [0.389, 0.409] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 20 | 0.457 [0.450, 0.463] | 0.450 [0.444, 0.461] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (b) uniform extra negatives, m_B = 40 | 0.516 [0.507, 0.532] | 0.514 [0.505, 0.531] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 1 | 0.652 [0.643, 0.664] | 0.646 [0.639, 0.657] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 2 | 0.811 [0.806, 0.822] | 0.813 [0.804, 0.825] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 5 | 0.902 [0.901, 0.904] | 0.903 [0.903, 0.904] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 10 | 0.922 [0.920, 0.925] | 0.922 [0.920, 0.926] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 20 | 0.917 [0.915, 0.918] | 0.916 [0.913, 0.920] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (c) ξ_B negatives, m_B = 40 | 0.880 [0.875, 0.888] | 0.877 [0.872, 0.884] | 0.003 [0.002, 0.004] | 0.013 [0.010, 0.015] | 0.992 [0.989, 0.994] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 1 | 0.383 [0.372, 0.400] | 0.377 [0.366, 0.383] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 2 | 0.417 [0.404, 0.424] | 0.412 [0.400, 0.419] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 5 | 0.491 [0.486, 0.501] | 0.482 [0.476, 0.490] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 10 | 0.564 [0.551, 0.571] | 0.561 [0.546, 0.574] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 20 | 0.628 [0.621, 0.635] | 0.624 [0.614, 0.634] | 0.000 [0.000, 0.000] | 0.005 [0.004, 0.005] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 40 | 0.670 [0.666, 0.674] | 0.641 [0.639, 0.642] | 0.025 [0.021, 0.031] | 0.051 [0.047, 0.056] | 0.925 [0.910, 0.935] | 0.000 [0.000, 0.000] | 0/3 |
| (o) oracle same-cluster negatives, m_B = 80 | 0.686 [0.681, 0.694] | 0.582 [0.575, 0.586] | 0.093 [0.086, 0.099] | 0.119 [0.116, 0.122] | 0.667 [0.662, 0.672] | 0.000 [0.000, 0.000] | 0/3 |
| (e2) label-free cPCA → Neg-t-SNE | 0.248 [0.244, 0.252] | 0.247 [0.245, 0.249] | 0.202 [0.196, 0.206] | 0.203 [0.198, 0.206] | 0.222 [0.193, 0.266] | 0.042 [0.000, 0.083] | 0/3 |
| (e3) label-free cPCA → UMAP | 0.255 [0.248, 0.261] | 0.250 [0.247, 0.254] | 0.203 [0.197, 0.207] | 0.203 [0.198, 0.206] | 0.211 [0.176, 0.264] | 0.042 [0.000, 0.125] | 0/3 |

### ξ_B diagnostics and data checks (selection seeds)

| seed | ξ_B same-y_b precision | same-y_s excess | detectability | graph signal |
|---|---|---|---|---|
| 0 | 1.000 | +0.036 | 0.916 | 0.912 |
| 1 | 1.000 | +0.038 | 0.912 | 0.911 |
| 2 | 1.000 | +0.043 | 0.919 | 0.916 |
