# Phase 5b calibration (selection seeds 0–2)

## L-cyclic (ρ = 0.2)

### Per seed

| seed | (a) WC | (a) signal acc | (a) background acc | detectability | graph signal | coherence | signal var in U | ξ_B same-y_b / excess | (e2) WC (α) | linear ceiling: embedded (projection) / high-D | cPCA ceiling: embedded (variant) / high-D | V1 | V2 | V2b | V3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.334 | 0.331 | 1.000 | 0.915 | 0.909 | -0.008 | 0.000 | 1.000 / +0.035 | 0.428 (1.34) | 0.628 (U+Q135) / 0.918 | 0.519 (cpca_a1.061_k5) / 0.869 | True | True | True | True |
| 1 | 0.365 | 0.355 | 1.000 | 0.908 | 0.905 | 0.005 | 0.000 | 1.000 / +0.039 | 0.411 (1.06) | 0.624 (U+Q135) / 0.910 | 0.508 (cpca_a1.061_k5) / 0.868 | True | True | True | True |
| 2 | 0.343 | 0.349 | 1.000 | 0.919 | 0.920 | 0.001 | 0.000 | 1.000 / +0.042 | 0.261 (2.73) | 0.628 (U+Q024) / 0.917 | 0.523 (cpca_or_a1.061_k5) / 0.863 | True | True | True | True |

V3 for L-cyclic: (e2) WC, the embedded linear ceiling and the embedded cPCA ceiling all ≤ 0.80. For L-between: (e2) WC ≤ 0.80 only. High-D values are descriptive.

### Which clusters each linear pipeline reveals (per-cluster embedded WC ≥ 0.80)

| seed | (e2) per-cluster WC | revealed | best linear subset | its per-cluster WC | revealed | best cPCA variant | its per-cluster WC | revealed |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.32, 0.53, 0.43, 0.63, 0.29, 0.37 | none | U+Q135 | 0.29, 0.99, 0.26, 0.99, 0.26, 0.99 | 1,3,5 | cpca_a1.061_k5 | 0.52, 0.66, 0.56, 0.63, 0.34, 0.39 | none |
| 1 | 0.36, 0.43, 0.31, 0.49, 0.45, 0.43 | none | U+Q135 | 0.24, 0.99, 0.28, 0.99, 0.28, 0.98 | 1,3,5 | cpca_a1.061_k5 | 0.57, 0.30, 0.53, 0.46, 0.64, 0.55 | none |
| 2 | 0.25, 0.26, 0.26, 0.28, 0.25, 0.26 | none | U+Q024 | 0.99, 0.27, 0.99, 0.26, 0.99, 0.27 | 0,2,4 | cpca_or_a1.061_k5 | 0.48, 0.55, 0.61, 0.50, 0.56, 0.45 | none |

### Linear subsets by the number of clusters revealed without their nuisance

A subset reveals cluster k "cleanly" if it contains Q_k (k's signal) but not Q_{k+1} (k's nuisance). Embedded WC over the subsets in each group, all seeds pooled:

| clean reveals | subsets | embedded WC mean [min, max] | high-D WC mean [min, max] |
|---|---|---|---|
| 0 | 2 | 0.302 [0.235, 0.362] | 0.585 [0.253, 0.918] |
| 1 | 30 | 0.407 [0.367, 0.455] | 0.595 [0.357, 0.826] |
| 2 | 30 | 0.514 [0.488, 0.550] | 0.607 [0.478, 0.730] |
| 3 | 2 | 0.621 [0.614, 0.628] | 0.617 [0.604, 0.630] |

### cPCA variants (embedded WC, mean over seeds per k)

| k | representative α: embedded WC [min, max] | oracle α for k: embedded WC [min, max] | high-D WC, best α [min, max] |
|---|---|---|---|
| 3 | 0.299 [0.238, 0.459] | 0.458 [0.456, 0.459] | 0.495 [0.478, 0.511] |
| 5 | 0.308 [0.241, 0.519] | 0.517 [0.508, 0.523] | 0.598 [0.593, 0.608] |
| 10 | 0.397 [0.254, 0.506] | 0.469 [0.411, 0.506] | 0.755 [0.747, 0.764] |
| 20 | 0.334 [0.246, 0.424] | 0.365 [0.358, 0.372] | 0.866 [0.863, 0.869] |

**L-cyclic valid on every selection seed: yes.**

## L-between (ρ = 0.2)

### Per seed

| seed | (a) WC | (a) signal acc | (a) background acc | detectability | graph signal | coherence | signal var in U | ξ_B same-y_b / excess | (e2) WC (α) | linear ceiling: embedded (projection) / high-D | cPCA ceiling: embedded (variant) / high-D | V1 | V2 | V2b | V3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.337 | 0.332 | 1.000 | 0.916 | 0.912 | 0.968 | 1.000 | 1.000 / +0.036 | 0.248 (1.7) | 0.992 (U) / 0.991 | 0.992 (cpca_or_a0.2572_k5) / 0.991 | True | True | True | True |
| 1 | 0.355 | 0.341 | 1.000 | 0.912 | 0.911 | 0.970 | 1.000 | 1.000 / +0.038 | 0.252 (1.7) | 0.987 (U) / 0.988 | 0.987 (cpca_or_a0.5223_k5) / 0.989 | True | True | True | True |
| 2 | 0.350 | 0.352 | 1.000 | 0.919 | 0.916 | 0.973 | 1.000 | 1.000 / +0.043 | 0.244 (1.34) | 0.989 (U) / 0.989 | 0.988 (cpca_or_a0.4125_k5) / 0.989 | True | True | True | True |

V3 for L-cyclic: (e2) WC, the embedded linear ceiling and the embedded cPCA ceiling all ≤ 0.80. For L-between: (e2) WC ≤ 0.80 only. High-D values are descriptive.

### Which clusters each linear pipeline reveals (per-cluster embedded WC ≥ 0.80)

| seed | (e2) per-cluster WC | revealed | best linear subset | its per-cluster WC | revealed | best cPCA variant | its per-cluster WC | revealed |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.26, 0.21, 0.24, 0.27, 0.29, 0.23 | none | U | 1.00, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 | cpca_or_a0.2572_k5 | 1.00, 0.99, 1.00, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 1 | 0.22, 0.30, 0.26, 0.25, 0.24, 0.25 | none | U | 0.99, 0.99, 0.98, 0.99, 0.99, 0.98 | 0,1,2,3,4,5 | cpca_or_a0.5223_k5 | 0.99, 0.99, 0.99, 0.99, 0.99, 0.99 | 0,1,2,3,4,5 |
| 2 | 0.23, 0.25, 0.26, 0.24, 0.24, 0.24 | none | U | 0.99, 0.99, 0.99, 0.99, 0.98, 0.99 | 0,1,2,3,4,5 | cpca_or_a0.4125_k5 | 0.99, 0.99, 0.99, 0.99, 0.98, 0.99 | 0,1,2,3,4,5 |

### cPCA variants (embedded WC, mean over seeds per k)

| k | representative α: embedded WC [min, max] | oracle α for k: embedded WC [min, max] | high-D WC, best α [min, max] |
|---|---|---|---|
| 3 | 0.435 [0.238, 0.990] | 0.988 [0.986, 0.990] | 0.988 [0.987, 0.990] |
| 5 | 0.433 [0.239, 0.991] | 0.989 [0.987, 0.992] | 0.989 [0.989, 0.991] |
| 10 | 0.276 [0.242, 0.358] | 0.349 [0.340, 0.356] | 0.917 [0.913, 0.922] |
| 20 | 0.279 [0.249, 0.362] | 0.349 [0.339, 0.358] | 0.917 [0.911, 0.922] |

**L-between valid on every selection seed: yes.**

