# web_chat_tinyllama_apr2025 — the run trainproof 0.20.0 passed

A real supervised fine-tune from **2026-04-26**, recovered off disk on
2026-09-13 while clearing 16.4 GB of dead checkpoints. It is here because of
what trainproof said about it:

```
$ trainproof epoch trainer_state.json        # v0.20.0
[PASS] All checks passed.
  [PASS] TP-PASS: No mechanical failures detected.
         Ran: divergence, flat-loss, lr, zero-loss.
exit 0
```

The run was dead by step 11 and every one of its 2.1 GB checkpoints is garbage.

The model weights and optimizer states are **deliberately not preserved** — 16.4
GB of NaN is not evidence. What is preserved is the 6 KB that records the death,
the configs that explain it, and the script that caused it. `SHA256SUMS.txt`
carries the hash and original path of every file, so "which artifact was this
derived from" stays answerable.

## What it was

TinyLlama-1.1B (`LlamaForCausalLM`, hidden 2048, 22 layers, 32 heads / 4 KV
heads, vocab 32000) fine-tuned on 480 product-FAQ records across ten languages.
480 records ÷ batch 4 × 3 epochs = **360 steps**, which is exactly `max_steps`
in `trainer_state.json` — that arithmetic is how we know this dataset is the one
that ran.

## Why it passed, and what changed

Four defects stacked. The last one is the interesting one.

**1 — `qlora_train.py` does no QLoRA.** No `peft`, no `LoraConfig`, no
`BitsAndBytesConfig`, no 4-bit anything. It full-fine-tunes all 1.1B parameters
with a plain `Trainer`. The actual QLoRA recipe — `lora_r 16`, `lora_alpha 32`,
all seven target modules, flash attention — is in
`intended_qlora_command.sh.txt`, which was saved with a `.jsonl` extension and
invokes a `train.py` that is not in the directory. The intent existed; it never
ran. **Not a trainproof rule** — no artifact trainproof accepts can distinguish
a misnamed file from a deliberate one.

**2 — the responses were never trained on.** `tokenize_function` encodes
`example["instruction"]` only, then `remove_columns=dataset.column_names` drops
`output` entirely. Every answer in the dataset was discarded before the loss.

**3 — no EOS supervision at all, and it cannot be detected downstream.**
`tokenizer_config.json` has `add_eos_token: false` while `pad_token` is `</s>`,
the EOS. So encoding appends no EOS, *and* padding shares the EOS id — a
collator that masks padding would mask a genuine EOS with it, and nothing in
`input_ids` can tell the two apart. This is **not** the same as "EOS was added
and then masked": there was never an EOS to mask. → **`TP-PRE-NO-EOS-APPEND`**
(WARN; the caller may still append one, and a false FAIL aborts correct runs).

**4 — fp16 weights trained with `fp16=False`, and the trainer hid it.** The
model loads `torch_dtype=torch.float16` while `TrainingArguments` sets
`fp16=False`, so no `GradScaler` exists. Gradients overflowed to NaN — visible
as `grad_norm: NaN` at step 10, the first logged step.

Then `TrainingArguments.logging_nan_inf_filter`, which **defaults to True**,
did this (`transformers/trainer.py`):

```python
if (self.args.logging_nan_inf_filter and ... (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
    # if loss is nan or inf simply add the average of previous logged losses
    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
```

`_tr_loss` is zeroed after each logging step, so once every loss is NaN that
expression is **exactly 0.0, forever**. The log reads:

| step | loss | grad_norm |
|---|---|---|
| 10 | 2.859 | NaN |
| 20 | **0.0** | NaN |
| … | **0.0** | NaN |
| 360 | **0.0** | NaN |

The trainer cosmetically rewrote a dead run into a log that looks like
spectacular convergence.

### Why every existing rule declined

- `TP-NAN` reads the **loss** column. The losses are 0.0, not NaN. Silent.
- `TP-ZERO-LOSS` requires **every** loss to be exactly 0.0. Step 10 is 2.859. Silent.
- `TP-ZERO-GRAD` and `TP-GRAD-SPIKE` read `valid_gns`, which filters non-finite
  values out. All 36 gradient norms are NaN, so that list is **empty** and both
  checks **skipped themselves** — "no finite gradient norms in the log". The
  worse the corruption, the quieter trainproof got.
- `TP-DEAD-RUN` takes the median of the first five losses. Four of those five
  are 0.0, so the median is 0.0, so it skipped: "starting loss is not positive".
- `TP-DIVERGE` compares the end loss to the lowest **nonzero** loss. End is 0.0.
  Nothing to diverge from.

Four checks each declined for a locally reasonable reason, **and the reasons
were caused by the defect itself.** The union reported PASS. This is the
v0.11.1 bug (see `rules.py`) one layer deeper: v0.11.1 fixed the series that is
zero on *every* step; these are the series that dies *partway*, and the series
that is *non-finite* rather than zero.

→ **`TP-ZERO-LOSS-ONSET`** (FAIL) and **`TP-NAN-GRAD`** (FAIL).

## Found while fixing it — three pre-existing crashes

The property fuzz written for the new rules (R19: hand-written regression tests
cannot certify a rejecter) found that `CheckContext` gated every metric on
`is not None` and then handed it to `math.isnan()`. A column logged as a string,
a list or a dict raised `TypeError` out of the constructor, where `CONTRACTS.md`
promises a verdict or a documented exit. `step` had the same exposure in
throughput arithmetic, and `eval_loss_steps` used `r.get("step", i)` — whose
default only covers a *missing* key, not a present-and-null one, so `None`
reached a `>=` against an int.

None of these were introduced by the new rules; all three are R19's exact
failure — **no field had a declared domain**. `_num()` now declares it: a metric
is a real number (`bool` excluded, being an `int` subclass), and a value failing
that predicate is treated exactly as absent.

## Files

| file | what it is |
|---|---|
| `trainer_state.json` | the corpse: 36 logged steps, `log_history` in full |
| `config.base.json` | the loaded base model — `bfloat16`, transformers 4.35.0 |
| `config.checkpoint-360.json` | what was saved — `float16`, transformers 4.50.3 |
| `tokenizer_config.json` | `add_eos_token: false`, `pad_token: "</s>"` |
| `special_tokens_map.json` | confirms `pad_token == eos_token` |
| `qlora_train.py` | the script that ran |
| `intended_qlora_command.sh.txt` | the QLoRA recipe that did not |
| `SHA256SUMS.txt` | hash + original path of each of the above |

**The dtype/version drift between the two configs is NOT a defect.** A base
artifact published in bfloat16 under 4.35.0, loaded as fp16 and re-saved under a
newer transformers, produces exactly this. It is recorded because it looks like
provenance drift and is not, and because the `float16` in the checkpoint config
is independent corroboration of defect 4.

## Dataset, by schema rather than by bulk

Not copied here — it is reusable material and 155 KB of it adds nothing to the
diagnosis. Recorded instead:

- `web_chat_multilingual.jsonl`, 155,145 bytes,
  sha256 `b353e29d5fd758b76bb5c783292ecb6f46d97b45700bea9e9a539ab01cc697f9`
- **480 records**, keys `instruction`, `input`, `output`, `language`
- `input` is the empty string in **all 480** rows
- 48 records each across 10 language tags: `ar`, `de`, `es`, `fr`, `ja`, `ko`,
  `no`, `ru`, `zh-CN`, and one empty tag
- `output` — the field defect 2 discards — is populated in every row
