# Pre-checks (descriptive, before any embedding)

Cells: pre 3933, post 3965, background 1985. Median log10 total counts: pre 3.69, post 3.48, background 3.29.

## Label kNN accuracy in high dimension (k = 15, stratified 5-fold)

| space | accuracy |
|---|---|
| 500 genes | 0.951 |
| PCA-50 | 0.947 |
| cPCA-10 (alpha 307) | 0.926 |
| residuals (k_loc 30, r 2) | 0.950 |

## Background-structure retention: reference values

| graph | mean Jaccard |
|---|---|
| all pairs | 0.181 |
| 500-gene kNN graph | 0.632 |
| residual kNN graph | 0.545 |

## Label-free α per seed

| seed | α | representatives |
|---|---|---|
| 0 | 307 | 0, 3.46, 22.9, 307 |
| 1 | 307 | 0, 3.46, 22.9, 307 |
| 2 | 307 | 0, 3.46, 22.9, 307 |
| 3 | 307 | 0, 3.46, 22.9, 307 |
| 4 | 307 | 0, 3.46, 22.9, 307 |
| 5 | 307 | 0, 3.46, 22.9, 307 |
| 6 | 307 | 0, 3.46, 22.9, 307 |
| 7 | 307 | 0, 3.46, 22.9, 307 |
| 8 | 307 | 0, 3.46, 22.9, 307 |
| 9 | 307 | 0, 3.46, 22.9, 307 |

## Replication: the paper's preprocessing, cPCA-2D

Label-free α on the paper's preprocessing: 389.

| α | label kNN accuracy |
|---|---|
| 0 | 0.905 |
| 3.45511 | 0.977 |
| 28.9427 | 0.928 |
| 388.816 | 0.823 |
