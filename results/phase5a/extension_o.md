# Extension: the oracle same-cluster control (o) at larger m_B (Design J)

Requested after the main runs and labeled as an extension. Each run starts from its seed's saved Stage 1, so it is paired with the main runs. Rerunning (o) at m_B = 20 from the saved Stage 1 of selection seed 0 reproduced the stored embedding bit for bit: **yes**.

## Selection seeds 0–2: mean signal accuracy of (o) by m_B

| m_B | 1 | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| mean signal acc | 0.313 | 0.337 | 0.382 | 0.421 | 0.455 | 0.484 | 0.458 |

Best m_B (same rule as the main runs): **40** (previously 20).

## Confirmation seeds: (c) vs (o) at m_B = 40

| run | signal acc | WC | G1 | G2 | G3 shares signal |
|---|---|---|---|---|---|
| c_m10 | 0.858 [0.830, 0.900] | 0.858 [0.830, 0.904] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| o_m40 | 0.495 [0.463, 0.533] | 0.498 [0.467, 0.538] | 0.012 [0.009, 0.018] | 0.961 [0.935, 0.976] | 0.000 [0.000, 0.000] |

| metric | per-seed difference (c) − (o) | mean | verdict |
|---|---|---|---|
| acc_s | +0.367, +0.367, +0.355 | +0.363 | (c) better, consistent across 3 paired seeds |
| G1 | -0.009, -0.018, -0.010 | -0.012 | (o) better, consistent across 3 paired seeds |
| G2 (lower is better) | +0.029, +0.065, +0.024 | +0.039 | (o) better, consistent across 3 paired seeds |
| G3_shares_signal | +0.000, +0.000, +0.000 | +0.000 | not consistent |
