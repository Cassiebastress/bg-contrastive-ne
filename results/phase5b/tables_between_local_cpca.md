# L-between: local cPCA (extension, added after the main results)

## Confirmation seeds: mean [min, max]

| run | WC | signal acc | G1 | G1 at k = 200 | G2 | G3 shares signal | flagged seeds |
|---|---|---|---|---|---|---|---|
| (c) ξ_B negatives, m_B = 10 | 0.914 [0.910, 0.921] | 0.915 [0.911, 0.921] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best cPCA variant (labels; per-seed max) | 0.989 [0.988, 0.992] | 0.989 [0.988, 0.992] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| best labeled linear projection (descriptive; per-seed max) | 0.989 [0.988, 0.991] | 0.989 [0.988, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0/3 |
| local cPCA, label-free clusters, k = 3 | 0.987 [0.984, 0.991] | 0.987 [0.984, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | — |
| local cPCA, label-free clusters, k = 5 | 0.988 [0.985, 0.991] | 0.988 [0.985, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.028 [0.000, 0.042] | — |
| local cPCA, label-free clusters, k = 10 | 0.989 [0.987, 0.992] | 0.989 [0.987, 0.992] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.014 [0.000, 0.042] | — |
| local cPCA, oracle clusters, k = 3 | 0.988 [0.984, 0.991] | 0.988 [0.984, 0.991] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 1.000 [1.000, 1.000] | 0.014 [0.000, 0.042] | — |
| local cPCA, oracle clusters, k = 5 | 0.988 [0.986, 0.991] | 0.988 [0.986, 0.991] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | — |
| local cPCA, oracle clusters, k = 10 | 0.989 [0.988, 0.992] | 0.989 [0.988, 0.992] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 1.000 [1.000, 1.000] | 0.028 [0.000, 0.042] | — |

## Pre-registered comparisons on WC

| comparison | per-seed WC difference (x − y) | mean | verdict |
|---|---|---|---|
| (c) vs local cPCA, label-free clusters, k = 3 | -0.070, -0.073, -0.077 | -0.073 | local cPCA, label-free clusters, k = 3 better, consistent across 3 paired seeds |
| (c) vs local cPCA, label-free clusters, k = 5 | -0.070, -0.074, -0.078 | -0.074 | local cPCA, label-free clusters, k = 5 better, consistent across 3 paired seeds |
| (c) vs local cPCA, label-free clusters, k = 10 | -0.071, -0.076, -0.077 | -0.075 | local cPCA, label-free clusters, k = 10 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 3 | -0.070, -0.073, -0.078 | -0.074 | local cPCA, oracle clusters, k = 3 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 5 | -0.070, -0.075, -0.078 | -0.074 | local cPCA, oracle clusters, k = 5 better, consistent across 3 paired seeds |
| (c) vs local cPCA, oracle clusters, k = 10 | -0.070, -0.077, -0.077 | -0.075 | local cPCA, oracle clusters, k = 10 better, consistent across 3 paired seeds |

## Per-cluster WC

Per-cluster WC; a cluster counts as revealed if its WC ≥ 0.80.

| seed | run | per-cluster WC (clusters 0–5) | revealed |
|---|---|---|---|
| 3 | (c) ξ_B negatives, m_B = 10 | 0.92, 0.94, 0.92, 0.87, 0.93, 0.94 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 3 | 0.99, 0.99, 0.99, 1.00, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 5 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, label-free clusters, k = 10 | 0.99, 0.99, 0.99, 0.99, 1.00, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 3 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 5 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 3 | local cPCA, oracle clusters, k = 10 | 0.99, 0.99, 0.99, 0.99, 1.00, 0.99 | 0,1,2,3,4,5 |
| 4 | (c) ξ_B negatives, m_B = 10 | 0.92, 0.91, 0.91, 0.88, 0.92, 0.94 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 3 | 0.99, 0.99, 0.99, 0.97, 0.98, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 5 | 0.98, 0.99, 0.98, 0.98, 0.98, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, label-free clusters, k = 10 | 0.99, 0.99, 0.99, 0.98, 0.98, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 3 | 0.99, 0.99, 0.99, 0.97, 0.98, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 5 | 0.98, 0.99, 0.98, 0.98, 0.98, 0.99 | 0,1,2,3,4,5 |
| 4 | local cPCA, oracle clusters, k = 10 | 0.99, 0.99, 0.99, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | (c) ξ_B negatives, m_B = 10 | 0.87, 0.93, 0.94, 0.92, 0.89, 0.93 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 3 | 0.99, 0.98, 0.99, 0.98, 0.99, 0.98 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 5 | 0.99, 0.98, 1.00, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, label-free clusters, k = 10 | 0.99, 0.98, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 3 | 0.99, 0.98, 1.00, 0.98, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 5 | 0.99, 0.98, 1.00, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 5 | local cPCA, oracle clusters, k = 10 | 0.99, 0.98, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |

## Clusters and α

K and the ARI (labels used for evaluation only) of the label-free clustering; the per-cluster label-free α; the one-hot scale s; and the largest fraction of kNN edges (k = 15) crossing clusters in the joint representation (0 by construction, checked).

| seed | version | K | ARI (background) | ARI (target) | per-cluster α | s at k = 3, 5, 10 | cross-cluster kNN edges |
|---|---|---|---|---|---|---|---|
| 3 | labelfree | 6 | 1.000 | 1.000 | 5.54, 8.89, 11.3, 8.89, 1.7, 11.3 | 15.6, 15.7, 16.2 | 0.000 |
| 3 | oracle | 6 | — | — | 8.89, 5.54, 1.7, 11.3, 11.3, 8.89 | 15.6, 15.7, 16.2 | 0.000 |
| 4 | labelfree | 6 | 1.000 | 1.000 | 14.3, 58.8, 46.4, 58.8, 8.89, 14.3 | 15.1, 15.4, 15.6 | 0.000 |
| 4 | oracle | 6 | — | — | 14.3, 8.89, 14.3, 58.8, 58.8, 46.4 | 15.1, 15.4, 15.6 | 0.000 |
| 5 | labelfree | 6 | 1.000 | 1.000 | 11.3, 22.9, 28.9, 5.54, 11.3, 11.3 | 14.4, 14.4, 14.7 | 0.000 |
| 5 | oracle | 6 | — | — | 28.9, 22.9, 5.54, 11.3, 11.3, 11.3 | 14.4, 14.4, 14.7 | 0.000 |
