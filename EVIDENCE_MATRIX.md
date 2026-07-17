# Evidence matrix - 15 runs, 3 seeds, judged by trainproof v0.3-dev

## Single-run verdicts (trainproof epoch)
| config | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| healthy | PASS | PASS | PASS |
| lr_hot | FAIL | FAIL | FAIL |
| lr_zero | FAIL | FAIL | FAIL |
| fp16_nan | FAIL | FAIL | FAIL |
| bad_labels | WARN | WARN | WARN |

## Reference comparison vs same-seed healthy (trainproof compare)
| run | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| lr_hot vs healthy | FAIL | FAIL | FAIL |
| lr_zero vs healthy | FAIL | PASS | FAIL |
| fp16_nan vs healthy | FAIL | FAIL | FAIL |
| bad_labels vs healthy | FAIL | FAIL | FAIL |

## Cross-seed baseline sanity (healthy vs healthy, different seeds)
- healthy s43 vs healthy s42: PASS
- healthy s44 vs healthy s42: PASS
- healthy s44 vs healthy s43: PASS

Note: lr_zero seed 43 originally evaded the dead-run rule (batch-order
noise faked >5% improvement under lr=0) and PASSes compare (its losses
land near baseline because the pretrained model never moved). This drove
the total-zero-LR fatality rule: lr=0 on every step = the optimizer never
steps = FAIL from the lr column itself, immune to loss noise.
