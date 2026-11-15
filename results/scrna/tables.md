## Every run, seeds 0–9: mean [min, max]

| run | label kNN accuracy | ARI | background-structure retention | fragmented seeds |
|---|---|---|---|---|
| BC-NE (r_a) | 0.944 [0.943, 0.944] | 0.758 [0.755, 0.773] | 0.457 [0.456, 0.457] | 0 of 10 |
| plain Neg-t-SNE (a) | 0.939 [0.938, 0.940] | 0.355 [0.353, 0.356] | 0.572 [0.571, 0.572] | 0 of 10 |
| t-SNE (T1) | 0.943 [0.941, 0.944] | 0.379 [0.342, 0.406] | 0.595 [0.594, 0.596] | 0 of 10 |
| PCA-50 → t-SNE (T2) | 0.937 [0.936, 0.938] | 0.390 [0.373, 0.401] | 0.564 [0.564, 0.565] | 0 of 10 |
| PCA-50 → UMAP (U2) | 0.936 [0.934, 0.937] | 0.383 [0.176, 0.456] | 0.551 [0.550, 0.552] | 0 of 10 |
| cPCA → t-SNE (T3) | 0.924 [0.922, 0.926] | 0.525 [0.005, 0.657] | 0.350 [0.348, 0.351] | 0 of 10 |
| cPCA-2D | 0.844 [0.844, 0.844] | -0.000 [-0.000, -0.000] | 0.286 [0.286, 0.286] | 0 of 10 |
| PCA-2D | 0.931 [0.931, 0.931] | 0.013 [0.013, 0.013] | 0.429 [0.429, 0.429] | 0 of 10 |

## Pre-registered comparisons (label kNN accuracy; differences are r_a − baseline)

| comparison | per-seed differences: min, max | mean | verdict |
|---|---|---|---|
| r_a vs plain Neg-t-SNE (a) | +0.0032, +0.0060 | +0.0044 | r_a better, consistent across 10 paired seeds |
| r_a vs t-SNE (T1) | -0.0005, +0.0033 | +0.0011 | not consistent |
| r_a vs PCA-50 → t-SNE (T2) | +0.0057, +0.0080 | +0.0071 | r_a better, consistent across 10 paired seeds |
| r_a vs PCA-50 → UMAP (U2) | +0.0066, +0.0094 | +0.0080 | r_a better, consistent across 10 paired seeds |
| r_a vs cPCA → t-SNE (T3) | +0.0167, +0.0217 | +0.0197 | r_a better, consistent across 10 paired seeds |
| r_a vs cPCA-2D | +0.0990, +0.1002 | +0.0997 | r_a better, consistent across 10 paired seeds |
| r_a vs PCA-2D | +0.0118, +0.0129 | +0.0125 | r_a better, consistent across 10 paired seeds |

Ceiling rule (mean T1 and mean r_a both ≥ 0.95): T1 0.943, r_a 0.944 → does not apply.

## BC-NE sensitivity (descriptive): mean label kNN accuracy

| k_loc \ r | 0 | 2 | 5 |
|---|---|---|---|
| 10 | 0.931 | 0.938 | 0.943 |
| 30 | 0.939 | 0.944 | 0.946 |
| 60 | 0.940 | 0.943 | 0.945 |

Mean ARI:

| k_loc \ r | 0 | 2 | 5 |
|---|---|---|---|
| 10 | 0.577 | 0.749 | 0.734 |
| 30 | 0.530 | 0.758 | 0.761 |
| 60 | 0.375 | 0.738 | 0.759 |

## cPCA with α chosen using labels (descriptive; an upper bound, not a fair baseline)

| run | α chosen with labels (best mean accuracy) | its mean accuracy | label-free α | its mean accuracy |
|---|---|---|---|---|
| cPCA-2D | 1.7 | 0.933 | 307 | 0.844 |
| cPCA → t-SNE (T3) | 5.54 | 0.955 | 307 | 0.924 |

## Larger background (descriptive): label kNN accuracy

| run | donor 1 (1985 cells) | donors 1 + 2 (4457 cells) |
|---|---|---|
| BC-NE (r_a) | 0.944 [0.943, 0.944] | 0.942 [0.941, 0.943] |
| cPCA → t-SNE (T3) | 0.924 [0.922, 0.926] | 0.923 [0.923, 0.924] |
| cPCA-2D | 0.844 [0.844, 0.844] | 0.879 [0.879, 0.879] |

## Checks

Seed-0 reruns bit-identical: yes, every run (no exceptions).

## Runtimes (one thread, whole process)

| process | median runtime (s) | range |
|---|---|---|
| BC-NE (r_a) | 142 | 130–144 |
| plain Neg-t-SNE (a) | 133 | 122–137 |
| t-SNE (T1) | 50 | 40–53 |
| PCA-50 → t-SNE (T2) | 46 | 45–51 |
| PCA-50 → UMAP (U2) | 37 | 35–39 |
| cPCA fit + T3 + cPCA-2D + PCA-2D | 49 | 44–53 |
