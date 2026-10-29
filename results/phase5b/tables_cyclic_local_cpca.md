# L-cyclic: local cPCA (extension, added after the main results)

## Confirmation seeds: mean [min, max]

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (c) ξ_B negatives, m_B = 20 | 0.914 [0.912, 0.916] | 0.914 [0.912, 0.917] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best cPCA variant (labels; per-seed max) | 0.530 [0.500, 0.551] | 0.518 [0.496, 0.540] | 0.014 [0.000, 0.032] | 0.019 [0.000, 0.043] | 0.958 [0.895, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best labeled linear projection (descriptive; per-seed max) | 0.624 [0.619, 0.628] | 0.622 [0.615, 0.633] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| local cPCA, label-free clusters, k = 3 | 0.989 [0.989, 0.990] | 0.990 [0.989, 0.990] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 1.000 [1.000, 1.000] | 0.028 [0.000, 0.083] | — |
| local cPCA, label-free clusters, k = 5 | 0.990 [0.989, 0.991] | 0.990 [0.989, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | — |
| local cPCA, label-free clusters, k = 10 | 0.989 [0.988, 0.990] | 0.989 [0.988, 0.990] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.014 [0.000, 0.042] | — |
| local cPCA, oracle clusters, k = 3 | 0.990 [0.989, 0.990] | 0.990 [0.989, 0.990] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | — |
| local cPCA, oracle clusters, k = 5 | 0.990 [0.989, 0.991] | 0.990 [0.989, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.028 [0.000, 0.042] | — |
| local cPCA, oracle clusters, k = 10 | 0.989 [0.989, 0.991] | 0.989 [0.989, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 1.000 [1.000, 1.000] | 0.028 [0.000, 0.083] | — |

## Pre-registered comparisons on WC

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs local cPCA, label-free clusters, k = 3 | -0.076, -0.077, -0.073 | -0.076 | local cPCA, label-free clusters, k = 3 better, consistent across 3 paired seeds |
| (c) vs local cPCA, label-free clusters, k = 5 | -0.077, -0.077, -0.073 | -0.076 | local cPCA, label-free clusters, k = 5 better, consistent across 3 paired seeds |
| (c) vs local cPCA, label-free clusters, k = 10 | -0.076, -0.077, -0.072 | -0.075 | local cPCA, label-free clusters, k = 10 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 3 | -0.077, -0.078, -0.073 | -0.076 | local cPCA, oracle clusters, k = 3 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 5 | -0.077, -0.077, -0.073 | -0.076 | local cPCA, oracle clusters, k = 5 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 10 | -0.077, -0.077, -0.072 | -0.075 | local cPCA, oracle clusters, k = 10 better, consistent across 3 paired seeds |

## Per-cluster WC

Per-cluster WC; a cluster counts as revealed if its WC ≥ 0.80.

| seed | run | per-cluster WC (clusters 0–5) | revealed |
|---|---|---|---|
| 3 | (c) ξ_B negatives, m_B = 20 | 0.94, 0.92, 0.88, 0.88, 0.93, 0.94 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 3 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 5 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 10 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 3 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 5 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 10 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | (c) ξ_B negatives, m_B = 20 | 0.90, 0.94, 0.94, 0.86, 0.92, 0.92 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 3 | 0.98, 1.00, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 5 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 10 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 3 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 5 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 10 | 0.98, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | (c) ξ_B negatives, m_B = 20 | 0.88, 0.91, 0.93, 0.93, 0.92, 0.92 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 3 | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 5 | 0.99, 0.98, 1.00, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 10 | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 3 | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 5 | 0.99, 0.98, 1.00, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 10 | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |

## Clusters and α

K and the ARI (labels used for evaluation only) of the label-free clustering; the per-cluster label-free α; the one-hot scale s; and the largest fraction of kNN edges (k = 15) crossing clusters in the joint representation (0 by construction, checked).

| seed | version | K | ARI (background) | ARI (target) | per-cluster α | s at k = 3, 5, 10 | cross-cluster kNN edges |
|---|---|---|---|---|---|---|---|
| 3 | labelfree | 6 | 1.000 | 1.000 | 2.15, 11.3, 7.02, 46.4, 28.9, 22.9 | 16.1, 16.1, 16.3 | 0.000 |
| 3 | oracle | 6 | — | — | 7.02, 2.15, 46.4, 28.9, 22.9, 11.3 | 16.1, 16.1, 16.3 | 0.000 |
| 4 | labelfree | 6 | 1.000 | 1.000 | 36.7, 28.9, 22.9, 46.4, 46.4, 7.02 | 16.2, 16.3, 16.4 | 0.000 |
| 4 | oracle | 6 | — | — | 36.7, 46.4, 7.02, 46.4, 28.9, 22.9 | 16.2, 16.3, 16.4 | 0.000 |
| 5 | labelfree | 6 | 1.000 | 1.000 | 7.02, 3.46, 18, 8.89, 28.9, 36.7 | 15.4, 15.4, 15.5 | 0.000 |
| 5 | oracle | 6 | — | — | 18, 3.46, 8.89, 36.7, 7.02, 28.9 | 15.4, 15.4, 15.5 | 0.000 |
