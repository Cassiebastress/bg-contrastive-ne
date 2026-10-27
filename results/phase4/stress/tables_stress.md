# Stress test (graph-level hiding), not used for any choice

Stress datasets: rho = 0.1 at every level, seeds 0-2. Settings frozen in stress_settings.json from the main selection seeds before any stress fit: (a), (d), (c) at m*_c and the best (b); (b) at m*_c was not part of them, so that comparison is absent. Values are mean [min, max] over seeds.

## easy

High-D X_T signal kNN accuracy (seeds 0-2, Phase 1 record): 0.509, 0.509, 0.502. xi_B same-y_b precision: 1.000, 1.000, 1.000; same-y_s excess: +0.003, +0.005, +0.003. Unhealthy runs: none.

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.507 [0.502, 0.509] | 0.0047 [0.0043, 0.0051] | 1.000 [1.000, 1.000] | 0.0046 [0.0027, 0.0065] |  | none |
| Stage 1 only (reference) | 0.257 [0.248, 0.264] | 0.0029 [0.0028, 0.0030] | 1.000 [1.000, 1.000] | 0.0009 [0.0000, 0.0017] | 0.0711 [0.0695, 0.0721] | none |
| (a) plain | 0.257 [0.250, 0.261] | 0.0028 [0.0027, 0.0029] | 1.000 [1.000, 1.000] | 0.0010 [0.0000, 0.0018] | 0.0700 [0.0684, 0.0712] | none |
| (d) s=0 reference | 0.255 [0.250, 0.259] | 0.0031 [0.0029, 0.0033] | 1.000 [1.000, 1.000] | 0.0011 [-0.0003, 0.0023] | 0.0136 [0.0133, 0.0138] | none |
| (c) xi_b, m_B=10 | 0.301 [0.295, 0.308] | 0.0034 [0.0033, 0.0036] | 1.000 [1.000, 1.000] | -0.0005 [-0.0018, 0.0005] | 0.0121 [0.0120, 0.0123] | none |
| (b) uniform_extra, m_B=20 | 0.270 [0.264, 0.274] | 0.0029 [0.0028, 0.0030] | 1.000 [1.000, 1.000] | 0.0011 [0.0001, 0.0023] | 0.0242 [0.0237, 0.0246] | none |

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=10 vs (a) plain | acc_s | +0.0437 | +0.0572, +0.0404, +0.0334 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | acc_s | +0.0313 | +0.0436, +0.0268, +0.0234 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | acc_s | +0.0456 | +0.0574, +0.0410, +0.0384 | yes |
| (c) xi_b, m_B=10 vs (a) plain | recall_s | +0.0007 | +0.0005, +0.0006, +0.0009 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | recall_s | +0.0005 | +0.0005, +0.0005, +0.0006 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | recall_s | +0.0003 | +0.0003, +0.0003, +0.0003 | yes |
| (c) xi_b, m_B=10 vs (a) plain | acc_b (drop) | +0.0000 | +0.0000, +0.0000, +0.0000 | no |

Pre-registered criterion applied descriptively: **not met**.

## medium

High-D X_T signal kNN accuracy (seeds 0-2, Phase 1 record): 0.405, 0.403, 0.397. xi_B same-y_b precision: 1.000, 1.000, 1.000; same-y_s excess: +0.013, +0.014, +0.012. Unhealthy runs: none.

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.402 [0.397, 0.405] | 0.0041 [0.0038, 0.0044] | 1.000 [1.000, 1.000] | 0.0052 [0.0033, 0.0073] |  | none |
| Stage 1 only (reference) | 0.273 [0.268, 0.276] | 0.0031 [0.0028, 0.0032] | 1.000 [1.000, 1.000] | 0.0011 [-0.0002, 0.0020] | 0.0705 [0.0694, 0.0714] | none |
| (a) plain | 0.271 [0.264, 0.282] | 0.0033 [0.0032, 0.0034] | 1.000 [1.000, 1.000] | 0.0013 [-0.0004, 0.0023] | 0.0695 [0.0683, 0.0706] | none |
| (d) s=0 reference | 0.282 [0.273, 0.289] | 0.0032 [0.0031, 0.0034] | 1.000 [1.000, 1.000] | 0.0011 [-0.0003, 0.0025] | 0.0135 [0.0132, 0.0137] | none |
| (c) xi_b, m_B=20 | 0.281 [0.265, 0.289] | 0.0034 [0.0033, 0.0036] | 0.994 [0.992, 0.997] | 0.0005 [-0.0017, 0.0033] | 0.0077 [0.0076, 0.0078] | none |
| (b) uniform_extra, m_B=20 | 0.287 [0.285, 0.291] | 0.0032 [0.0031, 0.0033] | 1.000 [1.000, 1.000] | 0.0016 [0.0009, 0.0026] | 0.0240 [0.0238, 0.0242] | none |

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=20 vs (a) plain | acc_s | +0.0099 | +0.0254, -0.0162, +0.0206 | no |
| (c) xi_b, m_B=20 vs (b) uniform_extra, m_B=20 | acc_s | -0.0057 | +0.0046, -0.0252, +0.0034 | no |
| (c) xi_b, m_B=20 vs (d) s=0 reference | acc_s | -0.0003 | +0.0162, -0.0178, +0.0006 | no |
| (c) xi_b, m_B=20 vs (a) plain | recall_s | +0.0001 | -0.0000, +0.0003, +0.0000 | no |
| (c) xi_b, m_B=20 vs (b) uniform_extra, m_B=20 | recall_s | +0.0002 | +0.0001, +0.0005, +0.0000 | yes |
| (c) xi_b, m_B=20 vs (d) s=0 reference | recall_s | +0.0002 | +0.0002, +0.0002, +0.0001 | yes |
| (c) xi_b, m_B=20 vs (a) plain | acc_b (drop) | +0.0059 | +0.0080, +0.0034, +0.0064 | yes |

Pre-registered criterion applied descriptively: **not met**.

## hard

High-D X_T signal kNN accuracy (seeds 0-2, Phase 1 record): 0.392, 0.380, 0.379. xi_B same-y_b precision: 1.000, 1.000, 1.000; same-y_s excess: +0.002, +0.004, +0.001. Unhealthy runs: none.

| run | signal acc | signal recall | background acc | signal Spearman | mean phi | flags |
|---|---|---|---|---|---|---|
| high-D X_T (reference) | 0.384 [0.379, 0.392] | 0.0041 [0.0037, 0.0045] | 1.000 [1.000, 1.000] | 0.0008 [-0.0021, 0.0029] |  | none |
| Stage 1 only (reference) | 0.254 [0.249, 0.257] | 0.0031 [0.0029, 0.0032] | 1.000 [1.000, 1.000] | 0.0024 [-0.0000, 0.0039] | 0.0483 [0.0473, 0.0492] | none |
| (a) plain | 0.254 [0.249, 0.259] | 0.0030 [0.0029, 0.0032] | 1.000 [1.000, 1.000] | 0.0025 [0.0005, 0.0041] | 0.0471 [0.0461, 0.0480] | none |
| (d) s=0 reference | 0.259 [0.253, 0.265] | 0.0032 [0.0031, 0.0033] | 1.000 [1.000, 1.000] | 0.0025 [-0.0003, 0.0053] | 0.0096 [0.0093, 0.0098] | none |
| (c) xi_b, m_B=10 | 0.287 [0.283, 0.293] | 0.0033 [0.0031, 0.0034] | 0.996 [0.996, 0.997] | 0.0025 [0.0002, 0.0042] | 0.0081 [0.0076, 0.0083] | none |
| (b) uniform_extra, m_B=20 | 0.256 [0.253, 0.260] | 0.0032 [0.0031, 0.0034] | 1.000 [1.000, 1.000] | 0.0024 [0.0000, 0.0049] | 0.0189 [0.0187, 0.0191] | none |

| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |
|---|---|---|---|---|
| (c) xi_b, m_B=10 vs (a) plain | acc_s | +0.0337 | +0.0338, +0.0378, +0.0294 | yes |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | acc_s | +0.0312 | +0.0330, +0.0344, +0.0262 | yes |
| (c) xi_b, m_B=10 vs (d) s=0 reference | acc_s | +0.0280 | +0.0274, +0.0344, +0.0222 | yes |
| (c) xi_b, m_B=10 vs (a) plain | recall_s | +0.0003 | +0.0005, -0.0001, +0.0004 | no |
| (c) xi_b, m_B=10 vs (b) uniform_extra, m_B=20 | recall_s | +0.0000 | +0.0003, -0.0002, -0.0000 | no |
| (c) xi_b, m_B=10 vs (d) s=0 reference | recall_s | +0.0001 | +0.0003, -0.0003, +0.0003 | no |
| (c) xi_b, m_B=10 vs (a) plain | acc_b (drop) | +0.0035 | +0.0042, +0.0036, +0.0028 | yes |

Pre-registered criterion applied descriptively: **not met**.
