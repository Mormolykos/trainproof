# Trainproof Rule IDs

Every finding emitted by trainproof includes a stable `TP-*` ID. This document lists all known rule IDs, their severity, and the conditions that trigger them. It is kept in step with the source by a test — a rule the code does not emit, or an ID this file does not document, fails the build.

Conditions where trainproof cannot judge a run at all (unreadable file, missing path, missing optional dependency) are not rules. They exit with code `2` and emit no verdict — see [CONTRACTS.md](CONTRACTS.md).

## Epoch / Watch (Single-Run) Rules

These rules run on a single training log (via `trainproof epoch` or `trainproof watch`).

| ID | Default Level | Description |
|---|---|---|
| `TP-NO-RECORDS` | FAIL | The log file could not be parsed or contained no valid records. Reaching this through the CLI exits `2` (cannot judge), not `1`. |
| `TP-NO-LOSS` | FAIL | No loss metric could be found in the log at all. (In `compare`, the same ID means the log has fewer than 10 valid loss points.) |
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
| `TP-TOK-HIGH-TPS` | WARN | High tokens per second of audio (possible sequence length blowout). |
| `TP-TOK-SUSPICIOUS-SPLIT` | WARN | Suspicious splits detected on numbers or dates. |
| `TP-TOK-PASS` | PASS | The tokenizer vocabulary coverage and splits look healthy. |
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
