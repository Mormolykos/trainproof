# Trainproof Rule IDs

Every finding emitted by trainproof includes a stable `TP-*` ID. This document lists all known rule IDs, their severity, and the conditions that trigger them. It is kept in step with the source by a test — a rule the code does not emit, or an ID this file does not document, fails the build.

Conditions where trainproof cannot judge a run at all (unreadable file, missing path, missing optional dependency) are not rules. They exit with code `2` and emit no verdict — see [CONTRACTS.md](CONTRACTS.md).

## Epoch / Watch (Single-Run) Rules

These rules run on a single training log (via `trainproof epoch` or `trainproof watch`).

| ID | Default Level | Description |
|---|---|---|
| `TP-NO-RECORDS` | FAIL | The log file could not be parsed or contained no valid records. Reaching this through the CLI exits `2` (cannot judge), not `1`. |
| `TP-NO-LOSS` | FAIL | No loss metric could be found in the log at all. (In `compare`, the same ID means the log has fewer than 10 valid loss points.) |
| `TP-NOT-CHECKED` | NOT-CHECKED | The number of check groups that executed is exactly zero, which requires fewer than five loss points AND a non-positive mean loss, with no other judgeable column (e.g. a short run whose loss is all zeros, the sub-threshold companion to `TP-ZERO-LOSS`). The run was not judged. Reaching this through the CLI exits `2` (cannot judge), not `1`. |
| `TP-NAN` | FAIL | The loss curve contains NaN or Infinity values. |
| `TP-ZERO-LOSS` | FAIL | Every finite loss is **exactly** 0.0 (minimum 5 points). Cross-entropy returns 0.0 when every target label is masked to `-100`, so the usual cause is the data collator's prompt masking or a response truncated out of the context window. Detected by exact equality, never a threshold: a very small loss is real convergence, which is a different thing. |
| `TP-ZERO-GRAD` | FAIL | Every finite gradient norm is **exactly** 0.0 (minimum 5 points). The backward graph is severed or every parameter is frozen; with PEFT this is usually reentrant gradient checkpointing over frozen input embeddings, which detaches the graph before it reaches the adapters. |
| `TP-FLAT` | FAIL | The loss curve is completely flat (relative variation < 0.001). The run is dead. |
| `TP-DIVERGE` | FAIL | The run is diverging: the end loss is >1.5x the lowest **nonzero** loss observed. |
| `TP-DEAD-RUN` | FAIL | The median loss of the last 5 steps has improved by less than 5% compared to the first 5 steps. |
| `TP-GRAD-SPIKE` | WARN | A gradient norm spike was detected (>10x the median gradient norm). |
| `TP-ZERO-LR` | FAIL | The learning rate is zero for >=99% of logged steps - the optimizer never steps. |
| `TP-ZERO-LR-PARTIAL` | WARN | The learning rate is zero for >10% of logged steps. |
| `TP-THROUGHPUT` | INFO | Displays the calculated steps/sec over the run. |
| `TP-STEP-CLIFF` | WARN | A step time cliff was detected: the median `step_time` of the last 20% of steps is >3x the median of the first half. |
| `TP-LOADER-BOUND` | WARN | Dataloader stall detected: median `loader_time` accounts for >50% of `step_time`. |
| `TP-GPU-UTIL` | INFO | Displays median GPU utilization (if available). |
| `TP-STALL` | WARN | (Watch only) The log file has not grown within the stall timeout period. |
| `TP-OVERFIT` | WARN | Overfitting detected: eval loss degraded significantly (>1.2x) while train loss continued falling. (Note: Does not mean run is mechanically broken, just that final checkpoint is not the best). |
| `TP-PASS` | PASS | The single-run checks passed. Its message names **which** checks ran and why each of the others was skipped - see below. |

### What `TP-PASS` promises

A passing report lists the checks that actually executed and, for every check
that did not, the reason. Reports also carry this as structured data under the
`checks` key (`ran`, and `skipped` as group → reason), so a consumer never has
to read the prose.

This exists because the opposite is dangerous. Until v0.11.1 the group list was
hardcoded, so a log whose checks had all been skipped by their own guards still
produced a `TP-PASS` naming those checks as having run. A skipped check is not a
clean check, and trainproof must never imply coverage it did not deliver.

INFO-only observations (`TP-THROUGHPUT`, `TP-GPU-UTIL`) are not accounted for in
`checks`: they report context rather than judge the run, so a missing one cannot
give a reader a false all-clear.

## Compare Rules

These rules compare a run against a baseline (via `trainproof compare`).

| ID | Default Level | Description |
|---|---|---|
| `TP-BAD-BASELINE` | WARN | The baseline itself fails single-run checks; the comparison may be meaningless. |
| `TP-CMP-UNCOMPARABLE` | FAIL | Either side has no usable loss scale (for example every logged loss is exactly 0.0, so its floor beats any baseline). No ratio rule can judge it, so trainproof refuses to issue a comparison verdict rather than report a meaningless comparison as a favorable one. |
| `TP-FLOOR-RATIO` | FAIL | The loss floor of the run is >2.0x the loss floor of the baseline. |
| `TP-END-RATIO` | FAIL | The end loss of the run is >2.0x the end loss of the baseline. |
| `TP-NEG-IMPROVE` | FAIL | The run ended with a higher loss than it started with (negative improvement). |
| `TP-IMPROVE-DEFICIT` | FAIL | The run's improvement is less than 25% of the baseline's improvement. |
| `TP-GRADNORM-RATIO` | WARN | The run's median gradient norm is >5.0x the baseline's median gradient norm. |
| `TP-CMP-ERROR` | WARN | (`doctor --baseline` only) The baseline comparison could not be run; the report judges the run on its own. Emitted instead of silently omitting the comparison the user asked for. |
| `TP-CMP-PASS` | PASS | The run compares favorably to the baseline. |

## Dataset Preflight Rules

These rules validate speech/TTS datasets (via `trainproof data`).

| ID | Default Level | Description |
|---|---|---|
| `TP-DATA-INVALID-INPUT` | FAIL | The input is neither a directory nor a valid manifest.jsonl. |
| `TP-DATA-NO-DATA` | FAIL | No audio/transcript pairs were found. |
| `TP-DATA-INCONSISTENT-SR` | FAIL | Multiple different audio sample rates were detected. |
| `TP-DATA-INCONSISTENT-CHANNELS` | FAIL | Multiple different audio channel counts were detected. |
| `TP-DATA-MISSING-AUDIO` | FAIL | The manifest references audio files that do not exist. |
| `TP-DATA-UNREADABLE-AUDIO` | WARN | Audio files could not be decoded. |
| `TP-DATA-EMPTY-TRANSCRIPT` | WARN | Transcripts are empty. |
| `TP-DATA-DURATION-LONG` | WARN | Audio duration exceeds the maximum limit (default 30s). |
| `TP-DATA-DURATION-SHORT` | WARN | Audio duration is below the minimum limit (default 0.5s). |
| `TP-DATA-CHAR-RATE-OUTLIER` | WARN | A transcript length vs audio duration outlier was detected. |
| `TP-DATA-DUPLICATES` | WARN | Duplicate audio content (identical hashes) detected. |
| `TP-DATA-CLIPPING` | WARN | Audio clipping detected. |
| `TP-DATA-SILENCE` | WARN | Excessive silence detected at the start or end. |
| `TP-DATA-UNNORMALIZED` | WARN | Unnormalized text detected (e.g. digits or dates instead of spoken words). |
| `TP-DATA-MIXED-SCRIPTS` | WARN | Transcripts contain mixed character scripts. |
| `TP-DATA-PASS` | PASS | The dataset preflight completed successfully. |

## Tokenizer Preflight Rules

These rules validate tokenizers and datasets (via `trainproof tokenizer` or `trainproof preflight`).

| ID | Default Level | Description |
|---|---|---|
| `TP-TOK-SPM-MISSING` | FAIL | The `sentencepiece` module is required but not installed. |
| `TP-TOK-LOAD-FAIL` | FAIL | Failed to load the tokenizer model. |
| `TP-TOK-NO-TRANSCRIPTS` | FAIL | Transcripts file not found. |
| `TP-TOK-HIGH-OOV` | FAIL | High Out-Of-Vocabulary rate detected (>1%). |
| `TP-TOK-LOW-COVERAGE` | WARN | Vocabulary coverage is below the recommended threshold (<99%). |
| `TP-TOK-HIGH-TPS` | WARN | High tokens per second of audio (possible sequence length blowout). Measured over the lines that declare a duration, on both sides of the rate. |
| `TP-TOK-TPS-NOT-MEASURED` | NOT-CHECKED | No line declared a duration, so tokens per second was not measured. Reported rather than skipped in silence. |
| `TP-TOK-NOT-MEASURED` | NOT-CHECKED | No tokens were produced, so OOV and vocabulary coverage were not measured. An empty file previously scored 100% coverage. |
| `TP-TOK-SUSPICIOUS-SPLIT` | WARN | Suspicious splits detected on numbers or dates. |
| `TP-TOK-PASS` | PASS | The tokenizer vocabulary coverage and splits look healthy. Its evidence names whether tokens/sec was measured. |
| `TP-PRE-EMPTY-TEXT` | FAIL | Empty or whitespace-only text found in the dataset. |
| `TP-PRE-DUPLICATE-TEXT` | WARN | Exact duplicate text found in the dataset. |
| `TP-PRE-MISSING-EOS-TOKEN`| FAIL | The tokenizer has no `eos_token`. |
| `TP-PRE-MISSING-PAD-TOKEN`| WARN | The tokenizer has no `pad_token`. |
| `TP-PRE-PAD-EQUALS-EOS` | WARN | The `pad_token_id` equals the `eos_token_id`. |
| `TP-PRE-BOS-TOKEN-INFO` | INFO | Status of the `bos_token`. |
| `TP-PRE-CONTEXT-CHECK-SKIPPED`| INFO | Context length check skipped (missing `--max-len`). |
| `TP-PRE-CONTEXT-OVERFLOW` | WARN | Records exceed the maximum context length. |
| `TP-PRE-MALFORMED-JSONL` | FAIL | JSONL parsing failed for some lines. |
| `TP-PRE-OK` | PASS | The preflight checks passed. |

## Objective Rules

These rules inspect the **loss objective** — the output layer, the ignore sentinel, and
which classes actually reach the loss as positive targets. They are deliberately
independent of the loss curve.

They exist because of a failure no curve-shaped rule in this file can see. A
VALL-E-X-derived TTS model set its end-of-sequence id and its cross-entropy
`ignore_index` to the same integer, so every stop target was discarded before the loss
and the model was never taught to stop. It trained for fifteen epochs on a healthy
curve. A minimal reproduction of the collision and its fix ended at **0.0035 and
0.0034** final loss — indistinguishable. The bug is decidable at step 0 from shapes and
one integer; it is undecidable from any number of steps of loss.

| Rule ID | Level | Meaning |
| :--- | :--- | :--- |
| `TP-OBJ-IGNORE-INDEX-COLLISION` | FAIL | `ignore_index` is a valid class in the output layer (`0 <= ignore_index < num_classes`). Every target carrying that id is dropped from the loss and can never be learned. |
| `TP-OBJ-IGNORE-INDEX-OK` | INFO | `ignore_index` equals `num_classes` — one past the last class, therefore outside the output layer and safe. Reported because the same integer is fatal one class earlier. |
| `TP-OBJ-DEAD-CLASS` | FAIL | A class exists in the output layer but never appears as a positive target, while coverage of the other classes is broad. The model has no way to learn to emit it. |
| `TP-OBJ-TARGET-OUT-OF-RANGE` | FAIL | Targets contain ids the output layer cannot represent. |
| `TP-OBJ-COVERAGE-INSUFFICIENT` | INFO | Too few distinct classes were observed to judge dead classes. Absence here means small sample, not bug. |
| `TP-OBJ-DEAD-CLASS-OK` | PASS | Every class in the output layer appeared as a training target. |
| `TP-OBJ-LABEL-SHIFT-DOUBLE` | FAIL | The labels appear to be **pre-shifted** against `input_ids` (`labels[i] == input_ids[i+1]`). A causal-LM loss shifts them again, so the model is trained to predict two tokens ahead. |
| `TP-OBJ-LABEL-SHIFT-OK` | PASS | The labels are aligned with `input_ids` (`labels[i] == input_ids[i]`), which is what the causal-LM loss expects before it applies its own shift. |
| `TP-OBJ-LABEL-SHIFT-NOT-MEASURED` | NOT-CHECKED | Alignment could not be judged: too few comparable positions, a batch/row mismatch, both hypotheses matching (repeated tokens), or labels that are not a copy of `input_ids` at all. |

### The label-alignment check, and what it deliberately does not claim

The HuggingFace causal-LM convention is that `labels` arrive **aligned** with `input_ids`
and the loss performs the shift, scoring the logits at position *i* against the label at
*i+1*. A caller who shifts the labels themselves has the shift applied **twice**, and the
model is trained to predict two tokens ahead. Nothing crashes — the shapes stay valid,
gradients flow, and the loss falls. It is the same shape of defect as the `ignore_index`
collision: invisible in the curve, decidable at step 0, which is why it lives beside it.

Two hypotheses are counted over every unmasked position, **independently rather than
exclusively**, because wherever a token repeats (`input_ids[i] == input_ids[i+1]`) a
position satisfies both:

| | condition |
|---|---|
| aligned | `labels[i] == input_ids[i]` |
| pre-shifted | `labels[i] == input_ids[i+1]` |

**Which of those is CORRECT is a property of the loss, not of the tensors**, and the tensors
cannot reveal it. That is carried by `loss_shifts`, which **defaults to `None`**:

| `loss_shifts` | aligned labels | pre-shifted labels |
|---|---|---|
| `None` — unknown *(default)* | NOT-CHECKED | **NOT-CHECKED — never FAIL** |
| `True` — the loss shifts (HF causal LM) | PASS | **FAIL** |
| `False` — the loss does not shift | **FAIL** (the model learns to emit the token it was just given) | PASS |

⚠️ **`False` is never inferred.** There is no way to confirm a negative about someone
else's loss function, so it is only ever supplied explicitly by a caller.

⚠️ **The default cannot fail anything, and that is deliberate.** A custom loop that
pre-shifts its labels *and* pairs that with a loss that does not shift is training
correctly. An earlier draft of this rule failed exactly that case, and a false FAIL under
`policy="stop_on_fail"` aborts a correct run before step 1 — the worst outcome this
library can produce.

**From `TrainproofCallback`, `loss_shifts=True` is set only when the model can be
confirmed**: a class that **`transformers` itself defines** whose name ends in
`ForCausalLM`, after unwrapping PEFT (`get_base_model`), DDP/FSDP (`.module`) and
`torch.compile` (`._orig_mod`). Only the **unwrapped** module may answer True; testing
the outer wrapper as well would widen the set of objects that can, and every widening is
a step toward failing a correct run. The module-origin half is load-bearing — a user
subclass named `MyForCausalLM` that overrides `forward` with a non-shifting loss reports
unknown, and unknown never fails. `config.is_encoder_decoder` also reports unknown,
because that family derives `decoder_input_ids` from the labels internally and
`input_ids` is the encoder's source sequence.

⚠️ **Confirming the model is not the same as confirming the loss, and this is the one
gap that remains.** A `Trainer` subclass that overrides `compute_loss` can bypass
`model(..., labels=...)` entirely and apply different conventions. That is invisible from
the model object, and the callback is never handed the trainer. **If you override
`compute_loss`, pass `loss_shifts=` to `TrainproofCallback` explicitly** —
`TrainproofCallback(loss_shifts=False)` or `True` overrides detection entirely. Both
independent adversarial reviewers raised this; it is recorded here rather than papered
over.

**The verdict is a vote over rows, not a pooled count over positions**, because positions
inside one sequence are not independent observations. A row votes if it carries at least
`LABEL_ALIGNMENT_MIN_ROW_POSITIONS` comparable positions. Rows that cannot discriminate —
repeated tokens satisfying both hypotheses, or labels that are not a copy of `input_ids` —
are **excluded from the vote rather than allowed to veto it**, and a verdict needs
`LABEL_ALIGNMENT_ROW_AGREEMENT` of the informative rows, informative rows to be at least
half of those judged, and `LABEL_ALIGNMENT_MIN_POSITIONS` positions overall.

**Sampling.** At most `LABEL_ALIGNMENT_MAX_ROWS` rows and `LABEL_ALIGNMENT_MAX_COLS`
columns are inspected, sliced **before** any device transfer so that reading a few dozen
positions never drags a whole batch off the accelerator. Columns are taken from the
**tail**: SFT masks the prompt on the left, so a head slice of a long sequence would see
nothing but `ignore_index`.

**The last position of every row is excluded**, because `input_ids[i+1]` does not exist
there and it could only ever feed the aligned hypothesis. Counting it would measure the two
hypotheses on different samples.

**Streaming dataloaders are not sampled at all.** `TrainproofCallback` takes a fresh
iterator to read the first batches; on a non-restartable `IterableDataset` that would
permanently consume batches the training epoch never sees. Detection is by class name
anywhere in the MRO plus the absence of `__len__`, and is deliberately biased toward
skipping: a missed check is recoverable, eating training data is not.

⚠️ **Not to be confused with the `bad_labels` gallery run**, which is labels *shuffled* per
sequence — a different defect, measured empirically in `examples/gallery/`, whose signature
is relative and needs `trainproof compare`. Shuffled and shifted are separate failures.

`TP-OBJ-DEAD-CLASS` fires only when coverage is already broad (see
`DEAD_CLASS_MIN_COVERAGE`) and only a small number of classes are missing (see
`DEAD_CLASS_MAX_REPORTED`). One unseen class out of 1025 with the other 1024 present is
a structural exclusion; nine hundred unseen classes is a small sample, and reporting it
would bury the finding that matters.

## Environment Preflight Rules (trainproof env)

These rules check whether the machine can start a training run at all — before a single GPU-second is spent. Import checks run in a subprocess; checkpoints are read as ZIP archives and never unpickled.

| ID | Default Level | Description |
|---|---|---|
| `TP-ENV-MEM-UNKNOWN` | NOT-CHECKED | System memory could not be determined on this platform. |
| `TP-ENV-MEM-INFO` | INFO | System memory was measured but no `--required-gb` was given to judge it. |
| `TP-ENV-MEM-INSUFFICIENT` | FAIL | Less system RAM is available than the run declares it needs. |
| `TP-ENV-MEM-TIGHT` | WARN | Available RAM exceeds the requirement but the margin is below `MIN_FREE_RAM_MARGIN_GB`. |
| `TP-ENV-MEM-OK` | PASS | System RAM headroom is sufficient for the declared requirement. |
| `TP-ENV-CWD-MISSING` | NOT-CHECKED | The working directory to probe from does not exist; no import subprocess was run. |
| `TP-ENV-IMPORT-TIMEOUT` | FAIL | The import probe did not finish within the timeout — an import that hangs hangs the training launch too. |
| `TP-ENV-PYTHON-UNUSABLE` | NOT-CHECKED | The interpreter to probe with could not be run. |
| `TP-ENV-IMPORT-OK` | PASS | The requested module imports cleanly in a subprocess. |
| `TP-ENV-IMPORT-CRASH` | FAIL | Importing the module crashed the interpreter with no Python exception (native crash / segfault). |
| `TP-ENV-IMPORT-FAIL` | FAIL | The module cannot be imported — this run cannot start. |
| `TP-ENV-CKPT-MISSING` | FAIL | The checkpoint path does not exist. |
| `TP-ENV-CKPT-EMPTY` | FAIL | The checkpoint is a zero-byte file — a save that was interrupted before writing. |
| `TP-ENV-CKPT-LEGACY` | NOT-CHECKED | The checkpoint uses the pre-1.6 torch format (a bare pickle); inspecting it would require unpickling, which executes arbitrary code, so trainproof refuses. |
| `TP-ENV-CKPT-UNREADABLE` | FAIL | The checkpoint is neither a torch ZIP archive nor a recognisable pickle, or it lacks a `data.pkl` entry. |
| `TP-ENV-CKPT-TRUNCATED` | FAIL | The checkpoint archive is incomplete — the save did not finish. |
| `TP-ENV-CKPT-CORRUPT` | FAIL | The checkpoint archive fails its own CRC check. |
| `TP-ENV-CKPT-OK` | PASS | The checkpoint is a complete, readable torch archive with CRC intact — read without unpickling. |
| `TP-ENV-DISK-UNKNOWN` | NOT-CHECKED | The output directory does not exist and neither does its parent, so free space cannot be measured. |
| `TP-ENV-DISK-INFO` | INFO | Free disk space was measured but no `--checkpoint-gb` was given to judge it. |
| `TP-ENV-DISK-INSUFFICIENT` | FAIL | Not enough disk space for the checkpoints this run will write. |
| `TP-ENV-DISK-OK` | PASS | Disk space is sufficient for the declared checkpoints. |
