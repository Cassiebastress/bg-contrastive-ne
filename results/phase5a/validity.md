# Phase 5a validity (selection seeds 0–2)

ρ: J = 0.8124 (the Phase 1 grid × √16.5; J's signal has unit total variance), K = 0.2. Each was chosen by the Phase 1 rule and confirmed on seeds 0–2.

| Criterion | Condition |
|---|---|
| V1 | (a) signal acc ≤ 0.50 and (a) background acc ≥ 0.90 |
| V2 | high-D WC (detectability) ≥ 0.85 |
| V3 (J) | (e2) fails one of G1 ≥ 0.50, G2 ≤ 0.50, G3 ≥ 0.50, signal acc > 0.80 |
| V3-K | (e2) WC ≤ 0.80 |
| V4 (J) | coherence ≥ 0.5 |

## Design J, chosen configuration t60_proj_g2

| seed | (a) signal acc | (a) background acc | detectability | coherence | (e2) signal acc | (e2) WC | (e2) G1 | (e2) G2 | (e2) G3 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.286 | 1.000 | 0.887 | 0.858 | 0.985 | 0.985 | 0.145 | 0.879 | 1.000 | True | True | True | True |
| 1 | 0.294 | 1.000 | 0.851 | 0.792 | 0.983 | 0.983 | 0.104 | 0.917 | 1.000 | True | True | True | True |
| 2 | 0.310 | 1.000 | 0.861 | 0.818 | 0.988 | 0.988 | 0.117 | 0.905 | 1.000 | True | True | True | True |

## Design K

| seed | (a) signal acc | (a) background acc | detectability | coherence | (e2) signal acc | (e2) WC | (e2) G1 | (e2) G2 | (e2) G3 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.301 | 1.000 | 0.893 | 0.011 | 0.963 | 0.970 | 0.020 | 0.955 | 0.542 | True | True | False | False |
| 1 | 0.299 | 1.000 | 0.876 | 0.001 | 0.956 | 0.972 | 0.013 | 0.958 | 0.375 | True | True | False | False |
| 2 | 0.314 | 1.000 | 0.883 | -0.004 | 0.970 | 0.980 | 0.010 | 0.970 | 0.125 | True | True | False | False |

For K, V4 is not a criterion. Coherence is reported because it should be about 0, which makes the global metrics G1–G3 ill-posed for K.
