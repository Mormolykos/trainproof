# The fault-injection gallery — real logs

Five real QLoRA fine-tuning runs of Qwen2.5-3B-Instruct (4-bit NF4, LoRA r=16,
300 steps, Alpaca-cleaned slice, RTX 5080), 2026-07-17. One healthy baseline,
four runs with exactly one knob deliberately broken. Each folder contains the
unmodified `trainer_state.json` written by HuggingFace Trainer plus a
`run_meta.json` with the run's wall time.

Judge any of them yourself:

```bash
trainproof epoch examples/gallery/lr_hot/trainer_state.json --format hf
```

| Folder | Sabotage |
|---|---|
| `healthy` | none |
| `lr_hot` | learning rate x100 |
| `lr_zero` | learning rate 0 |
| `fp16_nan` | fp16 + hot LR + no gradient clipping |
| `bad_labels` | labels shuffled per-sequence (see README: the honest finding) |

These logs are also the seed of the training-pathology corpus (see ROADMAP):
every future rule gets regression-tested against every archived pathology.