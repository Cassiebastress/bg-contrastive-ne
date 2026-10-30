# MNIST-on-grass pre-checks (selection seeds 0–2; descriptive)

## Is the digit signal in the kNN graph?

Digit kNN accuracy (k = 15, stratified 5-fold).

| seed | raw X_T (784-D) | clean digits | PCA-50 of X_T |
|---|---|---|---|
| 0 | 0.903 | 0.998 | 0.934 |
| 1 | 0.905 | 0.999 | 0.938 |
| 2 | 0.903 | 0.998 | 0.932 |

## ξ_B on raw pixels (k_B = 15, L = 30)

Same-digit excess of ξ_B pairs over chance (Phase 2 gate value 0.1 shown for reference; not a stop). Grass distance: mean Euclidean distance between the underlying grass patches, for ξ_B pairs (heads weighted by target-graph degree), all target pairs, and each target's 15 nearest targets by grass patch.

| seed | same digit, ξ_B | chance | excess | grass dist, ξ_B pairs | grass dist, random pairs | grass dist, grass-kNN | list length min / median / p90 / max | empty rows | tie-split rows | capped rows |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.569 | 0.502 | +0.067 | 3.79 | 7.88 | 3.63 | 22 / 42 / 60 / 60 | 0 | 0.958 | 0.106 |
| 1 | 0.557 | 0.502 | +0.055 | 3.95 | 8.05 | 3.75 | 30 / 41 / 60 / 60 | 0 | 0.952 | 0.097 |
| 2 | 0.576 | 0.502 | +0.074 | 3.83 | 8.00 | 3.65 | 29 / 41 / 60 / 60 | 0 | 0.955 | 0.108 |

## openTSNE determinism

T1 (library defaults, random_state = 0) on seed 0, run twice: **bit-identical**.
