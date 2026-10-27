# Post-hoc (not pre-registered as a decision rule): global metrics on the confirmation seeds

Signal kNN recall and signal distance Spearman against S_T are pre-registered metrics; this view of them for (a), best (b), (c) and (d) was added after the verdicts. Recall chance level is 15/4999 = 0.0030. Values are mean [min, max] over the confirmation seeds; differences are (c) minus the other run, per seed.

## easy (confirmation seeds 3, 4, 5)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0036 [0.0033, 0.0037] | -0.0016 [-0.0037, 0.0012] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0038 [0.0036, 0.0040] | -0.0023 [-0.0050, 0.0008] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=10 | 0.0144 [0.0139, 0.0148] | 0.0156 [0.0108, 0.0189] | 1.000 [1.000, 1.000] |
| (d) s=0 reference | 0.0036 [0.0034, 0.0038] | -0.0025 [-0.0053, 0.0003] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0102, +0.0109, +0.0115 | yes | +0.0195, +0.0178, +0.0145 | yes |
| (b) uniform_extra, m_B=20 | +0.0101, +0.0107, +0.0112 | yes | +0.0198, +0.0181, +0.0158 | yes |
| (d) s=0 reference | +0.0104, +0.0110, +0.0111 | yes | +0.0197, +0.0186, +0.0161 | yes |

## medium (confirmation seeds 3, 4, 5)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0038 [0.0036, 0.0041] | 0.0013 [-0.0004, 0.0031] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0041 [0.0040, 0.0043] | 0.0015 [0.0000, 0.0034] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=20 | 0.0060 [0.0058, 0.0064] | 0.0075 [0.0069, 0.0079] | 0.998 [0.997, 0.999] |
| (d) s=0 reference | 0.0039 [0.0035, 0.0042] | 0.0020 [0.0013, 0.0033] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0028, +0.0020, +0.0018 | yes | +0.0073, +0.0048, +0.0066 | yes |
| (b) uniform_extra, m_B=20 | +0.0024, +0.0019, +0.0016 | yes | +0.0058, +0.0045, +0.0077 | yes |
| (d) s=0 reference | +0.0029, +0.0016, +0.0019 | yes | +0.0056, +0.0047, +0.0063 | yes |

## hard (confirmation seeds 4, 5, 6)

| run | signal recall | signal Spearman | background acc |
|---|---|---|---|
| (a) plain | 0.0036 [0.0032, 0.0039] | -0.0002 [-0.0027, 0.0032] | 1.000 [1.000, 1.000] |
| (b) uniform_extra, m_B=20 | 0.0047 [0.0041, 0.0054] | -0.0004 [-0.0032, 0.0029] | 1.000 [1.000, 1.000] |
| (c) xi_b, m_B=10 | 0.0136 [0.0123, 0.0156] | 0.0146 [0.0112, 0.0176] | 1.000 [1.000, 1.000] |
| (d) s=0 reference | 0.0040 [0.0037, 0.0045] | -0.0004 [-0.0031, 0.0027] | 1.000 [1.000, 1.000] |

| (c) minus | signal recall per seed | consistent | signal Spearman per seed | consistent |
|---|---|---|---|---|
| (a) plain | +0.0091, +0.0117, +0.0091 | yes | +0.0144, +0.0177, +0.0124 | yes |
| (b) uniform_extra, m_B=20 | +0.0083, +0.0102, +0.0082 | yes | +0.0147, +0.0182, +0.0121 | yes |
| (d) s=0 reference | +0.0087, +0.0111, +0.0089 | yes | +0.0149, +0.0181, +0.0121 | yes |
