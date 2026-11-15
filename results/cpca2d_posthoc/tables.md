# cPCA-2D on MNIST-on-grass, post hoc (descriptive)

The 2-D cPCA projection at T3's label-free alpha, used as the embedding. Mean [min, max] over seeds; every projection and alpha was recomputed from the data and matched the saved one bit for bit.

## Selection seeds 0, 1, 2

| data | digit kNN accuracy | ARI | recall vs grass | recall vs clean digits | T3 digit kNN accuracy (same seeds) | fragmented |
|---|---|---|---|---|---|---|
| 0/1 | 0.957 [0.942, 0.965] | 0.819 [0.771, 0.852] | 0.003 [0.003, 0.004] | 0.018 [0.012, 0.024] | 0.968 [0.944, 0.985] | 0 of 3 |
| 0-9 | 0.259 [0.248, 0.279] | 0.065 [0.056, 0.075] | 0.003 [0.003, 0.003] | 0.017 [0.016, 0.017] | 0.418 [0.401, 0.439] | 0 of 3 |

## Confirmation seeds 3, 4, 5

| data | digit kNN accuracy | ARI | recall vs grass | recall vs clean digits | T3 digit kNN accuracy (same seeds) | fragmented |
|---|---|---|---|---|---|---|
| 0/1 | 0.949 [0.939, 0.967] | 0.793 [0.742, 0.861] | 0.003 [0.003, 0.003] | 0.016 [0.010, 0.022] | 0.955 [0.941, 0.979] | 0 of 3 |
| 0-9 | 0.269 [0.262, 0.279] | 0.074 [0.071, 0.075] | 0.003 [0.003, 0.003] | 0.016 [0.015, 0.017] | 0.433 [0.408, 0.482] | 0 of 3 |

## Per seed

| data | stage | seed | α | digit kNN accuracy | ARI | recall vs grass |
|---|---|---|---|---|---|---|
| 0/1 | selection | 0 | 3.46 | 0.964 | 0.852 | 0.004 |
| 0/1 | selection | 1 | 11.3 | 0.942 | 0.771 | 0.003 |
| 0/1 | selection | 2 | 5.54 | 0.965 | 0.835 | 0.003 |
| 0/1 | confirmation | 3 | 11.3 | 0.941 | 0.774 | 0.003 |
| 0/1 | confirmation | 4 | 4.38 | 0.967 | 0.861 | 0.003 |
| 0/1 | confirmation | 5 | 11.3 | 0.939 | 0.742 | 0.003 |
| 0-9 | selection | 0 | 5.54 | 0.279 | 0.075 | 0.003 |
| 0-9 | selection | 1 | 5.54 | 0.248 | 0.056 | 0.003 |
| 0-9 | selection | 2 | 5.54 | 0.251 | 0.062 | 0.003 |
| 0-9 | confirmation | 3 | 4.38 | 0.279 | 0.075 | 0.003 |
| 0-9 | confirmation | 4 | 5.54 | 0.266 | 0.071 | 0.003 |
| 0-9 | confirmation | 5 | 5.54 | 0.262 | 0.075 | 0.003 |
