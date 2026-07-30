# The fault-injection gallery — real logs

Eighteen real QLoRA fine-tuning runs of Qwen2.5-3B-Instruct (4-bit NF4, LoRA
r=16, 300 steps, Alpaca-cleaned slice, RTX 5080). One healthy baseline and five
configurations with exactly one knob deliberately broken, each repeated at three
seeds — 42, 43, 44. Seed 42 sits at the config root; 43 and 44 are in `seed43/`
and `seed44/` beside it. Every folder contains the unmodified
`trainer_state.json` written by HuggingFace Trainer plus a `run_meta.json` with
the seed and wall time.

Nothing here is hardware-specific: these are logged numbers, and every rule that
reads them is pure Python with no torch import. The RTX 5080 is where they were
produced, not a requirement for judging them.

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
| `overfit` | 64 training samples over many epochs, held-out eval set — memorisation |

These logs are also the seed of the training-pathology corpus (see ROADMAP):
every future rule gets regression-tested against every archived pathology.