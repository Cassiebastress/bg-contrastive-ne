# Design K calibration runs (selection seeds 0–2)

(e2-k20) and (e2-k40) use the same label-free α as (e2), with the top 20 and top 40 cPCA components. V3-K is defined with the 10-component (e2) only.

| run | WC | signal acc | G1 (ill-posed for K) | G2 (ill-posed for K) |
|---|---|---|---|---|
| (a) plain Neg-t-SNE | 0.306 [0.296, 0.319] | 0.304 [0.299, 0.314] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| (e2) cPCA → Neg-t-SNE | 0.974 [0.970, 0.980] | 0.963 [0.956, 0.970] | 0.014 [0.010, 0.020] | 0.961 [0.955, 0.970] |
| (e3) cPCA → UMAP | 0.971 [0.966, 0.976] | 0.961 [0.957, 0.966] | 0.014 [0.009, 0.019] | 0.961 [0.953, 0.970] |
| (e2-k20) cPCA top 20 → Neg-t-SNE | 0.981 [0.978, 0.984] | 0.980 [0.978, 0.982] | 0.005 [0.003, 0.007] | 0.986 [0.983, 0.989] |
| (e2-k40) cPCA top 40 → Neg-t-SNE | 0.980 [0.978, 0.983] | 0.980 [0.979, 0.981] | 0.005 [0.003, 0.006] | 0.984 [0.982, 0.986] |

α selected per seed: 8.89, 11.3, 5.54. 10-D WC at the selected α: 0.985, 0.982, 0.986; at the oracle α: 0.987, 0.984, 0.988.
