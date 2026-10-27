# Excluded confirmation runs (Plan §5 exclusion rule)

Hard confirmation seed 3: the dataset failed the validity check, judged from (a) and the oracle only: (a) signal kNN accuracy 0.5126 > 0.50, background accuracy 1.000, oracle 0.9956. Before the exclusion rule existed, the first confirmation run stopped there (Plan §6), after these runs had already finished. They are listed here only and appear in no table or comparison. Lines are verbatim from that run's log:

```
hard confirmation seed=3 a      acc_s 0.513 acc_b 1.000 recall_s 0.007 (11s)
hard confirmation seed=3 d      acc_s 0.562 acc_b 1.000 recall_s 0.008 (12s)
hard confirmation seed=3 c_m10  acc_s 0.877 acc_b 1.000 recall_s 0.015 (36s)
hard confirmation seed=3 b_m10  acc_s 0.572 acc_b 1.000 recall_s 0.007 (31s)
hard confirmation seed=3 b_m20  acc_s 0.601 acc_b 1.000 recall_s 0.008 (44s)
STOP (Plan §6): hard seed 3: fresh dataset is not a valid test {'emb_acc_b': 1.0, 'emb_acc_s': 0.5126000000000001, 'oracle_acc_s': 0.9956000000000002, 'valid': False}
```
