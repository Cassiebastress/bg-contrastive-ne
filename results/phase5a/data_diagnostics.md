# Phase 5a data diagnostics (descriptive)

Computed from the generators' clean components on selection seeds 0–2, with no embeddings. Each cell is the mean [min, max] over seeds.

- **Signal variance in span(B):** the fraction of the clean signal's variance inside the background map's span.
- **Background variance along the signal:** the clean signal's top principal directions (those explaining 90% of its variance) are taken first. Along them, the global background variance over all of X_B is compared with the mean within-background-cluster variance.
- **Rationale:** cPCA can only use directions with low *global* background variance, while ξ_B, built from background neighborhoods, effectively uses directions with low *local* background variance. Configurations with a large global/local ratio are where (c) is predicted to have an advantage.
- **Background main effect in the signal** (added after the grid results): the share of the clean signal's variance explained by the target's background-cluster means. Coherence subtracts those means, so it cannot see this effect. In J, the interaction term carries it outside span(B), where X_B never shows it.
- **Caveat on coherence:** it measures alignment in ambient coordinates, so it can call a signal incoherent even when the signal is consistent in the background's own frame (for example, with a curved background). V4 is unchanged; this caveat applies to every low-coherence result.

| configuration | signal var in span(B) | signal directions (90%) | global bg var | mean within-cluster bg var | global / local | coherence | background main effect |
|---|---|---|---|---|---|---|---|
| t0_proj_g2 | 0.000 [0.000, 0.000] | 3 | 0.030 [0.030, 0.030] | 0.030 [0.030, 0.030] | 1.002 [1.001, 1.002] | 0.962 [0.960, 0.966] | 0.001 [0.001, 0.002] |
| t0_proj_g6 | 0.000 [0.000, 0.000] | 3 | 0.030 [0.030, 0.030] | 0.030 [0.030, 0.030] | 1.002 [1.001, 1.002] | 0.962 [0.960, 0.966] | 0.001 [0.001, 0.002] |
| t0_unproj_g2 | 0.000 [0.000, 0.000] | 3 | 0.030 [0.030, 0.030] | 0.030 [0.030, 0.030] | 1.002 [1.001, 1.002] | 0.962 [0.960, 0.966] | 0.001 [0.001, 0.002] |
| t0_unproj_g6 | 0.000 [0.000, 0.000] | 3 | 0.030 [0.030, 0.030] | 0.030 [0.030, 0.030] | 1.002 [1.001, 1.002] | 0.962 [0.960, 0.966] | 0.001 [0.001, 0.002] |
| t30_proj_g2 | 0.000 [0.000, 0.000] | 3 | 0.030 [0.030, 0.030] | 0.030 [0.030, 0.030] | 1.002 [1.001, 1.002] | 0.921 [0.912, 0.934] | 0.030 [0.028, 0.031] |
| t30_proj_g6 | 0.000 [0.000, 0.000] | 4, 5 | 0.043 [0.040, 0.050] | 0.043 [0.040, 0.050] | 1.001 [1.001, 1.001] | 0.906 [0.896, 0.920] | 0.054 [0.053, 0.055] |
| t30_unproj_g2 | 0.023 [0.020, 0.026] | 3 | 3.617 [3.059, 4.337] | 0.077 [0.070, 0.087] | 46.460 [43.492, 49.929] | 0.922 [0.915, 0.935] | 0.030 [0.029, 0.034] |
| t30_unproj_g6 | 0.024 [0.020, 0.030] | 4, 5 | 18.992 [6.585, 41.856] | 0.289 [0.129, 0.585] | 59.493 [51.235, 71.546] | 0.907 [0.899, 0.922] | 0.055 [0.052, 0.058] |
| t60_proj_g2 | 0.000 [0.000, 0.000] | 7, 8 | 0.077 [0.070, 0.080] | 0.077 [0.070, 0.080] | 1.001 [1.001, 1.001] | 0.823 [0.792, 0.858] | 0.087 [0.086, 0.088] |
| t60_proj_g6 | 0.000 [0.000, 0.000] | 11, 12 | 0.117 [0.110, 0.121] | 0.117 [0.110, 0.121] | 1.001 [1.001, 1.001] | 0.753 [0.716, 0.797] | 0.159 [0.156, 0.165] |
| t60_unproj_g2 | 0.069 [0.060, 0.077] | 7, 8 | 51.840 [37.465, 73.353] | 0.756 [0.566, 1.027] | 67.975 [66.227, 71.448] | 0.825 [0.802, 0.860] | 0.089 [0.084, 0.098] |
| t60_unproj_g6 | 0.073 [0.059, 0.089] | 12, 13 | 92.589 [76.589, 111.480] | 1.339 [1.147, 1.569] | 68.946 [66.762, 71.040] | 0.755 [0.727, 0.801] | 0.163 [0.154, 0.171] |
| t90_proj_g2 | 0.000 [0.000, 0.000] | 9, 10 | 0.093 [0.090, 0.100] | 0.093 [0.089, 0.099] | 1.001 [1.000, 1.001] | 0.755 [0.706, 0.799] | 0.121 [0.115, 0.123] |
| t90_proj_g6 | 0.000 [0.000, 0.000] | 14, 15 | 0.143 [0.139, 0.149] | 0.143 [0.139, 0.149] | 1.001 [1.001, 1.001] | 0.630 [0.567, 0.690] | 0.220 [0.216, 0.223] |
| t90_unproj_g2 | 0.096 [0.080, 0.109] | 9, 10, 11 | 81.252 [57.572, 100.585] | 1.166 [0.852, 1.409] | 69.392 [67.551, 71.370] | 0.758 [0.717, 0.802] | 0.124 [0.116, 0.137] |
| t90_unproj_g6 | 0.100 [0.079, 0.123] | 14, 15 | 121.319 [99.854, 152.326] | 1.739 [1.476, 2.119] | 69.476 [67.633, 71.880] | 0.633 [0.580, 0.697] | 0.224 [0.215, 0.236] |
| K | 0.000 [0.000, 0.000] | 13, 14 | 0.133 [0.129, 0.141] | 0.133 [0.129, 0.140] | 1.001 [1.001, 1.001] | 0.002 [-0.004, 0.011] | 0.001 [0.001, 0.002] |
