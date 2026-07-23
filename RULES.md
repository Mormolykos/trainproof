# Trainproof Rule IDs

Every finding emitted by trainproof includes a stable `TP-*` ID. This document lists all known rule IDs, their severity, and the conditions that trigger them. It is kept in step with the source by a test — a rule the code does not emit, or an ID this file does not document, fails the build.

Conditions where trainproof cannot judge a run at all (unreadable file, missing path, missing optional dependency) are not rules. They exit with code `2` and emit no verdict — see [CONTRACTS.md](CONTRACTS.md).

## Epoch / Watch (Single-Run) Rules

These rules run on a single training log (via `trainproof epoch` or `trainproof watch`).

| ID | Default Level | Description |
|---|---|---|
| `TP-NO-RECORDS` | FAIL | The log file could not be parsed or contained no valid records. Reaching this through the CLI exits `2` (cannot judge), not `1`. |
| `TP-NO-LOSS` | FAIL | The log file contains fewer than 10 valid loss points. |
| `TP-NAN` | FAIL | The loss curve contains NaN or Infinity values. |
| `TP-FLAT` | FAIL | The loss curve is completely flat (variation < 0.005). The run is dead. |
| `TP-DIVERGE` | FAIL | The run is diverging: the end loss is >1.5x the minimum loss observed. |
| `TP-DEAD-RUN` | FAIL | The median loss of the last 5 steps has improved by less than 5% compared to the first 5 steps. |
| `TP-GRAD-SPIKE` | WARN | A gradient norm spike was detected (>10x the median gradient norm). |
| `TP-ZERO-LR` | FAIL | The learning rate is exactly 0.0 for 100% of the run. |
| `TP-ZERO-LR-PARTIAL` | WARN | The learning rate is exactly 0.0 for >20% of the run. |
| `TP-THROUGHPUT` | INFO | Displays the calculated steps/sec over the run. |
| `TP-STEP-CLIFF` | WARN | A step time cliff was detected (recent steps are >1.5x slower than the run average). |
| `TP-LOADER-BOUND` | WARN | Dataloader stall detected: loader time accounts for >20% of total step time. |
| `TP-GPU-UTIL` | INFO | Displays average GPU utilization (if available). |
| `TP-STALL` | WARN | (Watch only) The log file has not grown within the stall timeout period. |
| `TP-OVERFIT` | WARN | Overfitting detected: eval loss degraded significantly (>1.2x) while train loss continued falling. (Note: Does not mean run is mechanically broken, just that final checkpoint is not the best). |
| `TP-PASS` | PASS | The single-run checks passed. |

## Compare Rules

These rules compare a run against a baseline (via `trainproof compare`).

| ID | Default Level | Description |
|---|---|---|
| `TP-BAD-BASELINE` | WARN | The baseline itself fails single-run checks; the comparison may be meaningless. |
| `TP-FLOOR-RATIO` | FAIL | The loss floor of the run is >2.0x the loss floor of the baseline. |
| `TP-END-RATIO` | FAIL | The end loss of the run is >2.0x the end loss of the baseline. |
| `TP-NEG-IMPROVE` | FAIL | The run ended with a higher loss than it started with (negative improvement). |
| `TP-IMPROVE-DEFICIT` | FAIL | The run's improvement percentage is less than 50% of the baseline's improvement. |
| `TP-GRADNORM-RATIO` | WARN | The run's median gradient norm is >5.0x the baseline's median gradient norm. |
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
