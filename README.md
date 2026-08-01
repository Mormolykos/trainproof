# trainproof

[![PyPI](https://img.shields.io/pypi/v/trainproof)](https://pypi.org/project/trainproof/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A deterministic linter for ML training runs.** Point it at your dataset, your
tokenizer, or your first-epoch logs — it returns a PASS / WARN / FAIL verdict
with named findings and cited evidence, before you burn days of GPU time on a
run that was doomed at step 50.

No ML judging ML. No invented "confidence 97%". Every rule is a deterministic
threshold in [one auditable module](src/trainproof/rules.py), and every finding
cites the numbers that triggered it.

**What it is:** the reliability layer between your training code and the GPU
bill — checks that run *before* training (is this run safe to start?), *during*
training (should it keep going?), and *after* (is it reproducible; did it match
a known-good baseline?).

**The one rule it never breaks:** trainproof does not infer causes, does not
invent confidence scores, and does not guess. It reports deterministic findings
backed by evidence, or it stays silent. Every feature earns its place by
answering a single question — *if it were gone, would someone lose GPU hours?*

```bash
pip install trainproof
```

## The Doctor (v0.6)

"Why did my run diverge? Why is my loss flat? Did my dataset break?"

Run the flagship zero-config autopsy on any directory. It discovers all training logs, parses them automatically, and delivers a plain-English diagnosis.

Note what a PASS actually says: it names the checks that ran **and every check
that did not, with the reason**. This log carries no eval signal, so the
overfit check could not run — and the report says so rather than letting a
clean verdict imply it was covered.

```bash
trainproof doctor .
```

```text
============================================================
FILE   : examples/gallery/healthy/trainer_state.json
FORMAT : hf
RECORDS: 60 (steps/epochs: 5.0..300.0)
------------------------------------------------------------
VERDICT: PASS
------------------------------------------------------------
[PASS] TP-PASS: No mechanical failures detected. Ran: dead-run,
       divergence, flat-loss, grad-spike, lr, zero-grad, zero-loss.
       Skipped: loader (no loader_time/step_time pair in the log);
       overfit (no eval_loss in the log - this run has no
       generalisation signal at all); step-time (no step_time column
       in the log).
       Evidence: 60 steps analyzed.

------------------------------------------------------------
Findings: 1 PASS, 0 WARN, 0 FAIL
============================================================

What this cannot tell you
-------------------------
A passing report does not mean the run is good. These checks catch
mechanical failures (divergence, NaN, flatline, spikes) from the log
alone. They cannot detect a model learning the wrong thing - a run
trained on corrupted data can look healthy here. For that, compare
against a known-good baseline: trainproof compare <baseline> <run>
```


## See a verdict in 60 seconds

This repo ships the real logs of eighteen QLoRA fine-tuning runs (Qwen2.5-3B,
RTX 5080 — six configurations at three seeds each; see the gallery below). Judge
one right now:

```bash
trainproof epoch examples/gallery/lr_hot/trainer_state.json --format hf
```

```text
========================================
TRAINPROOF VERDICT
========================================
[FAIL] Critical checks failed:
  [FAIL] TP-DIVERGE: Loss curve is diverging.
         Evidence: End loss 7.492 vs Min loss 1.398
  [WARN] TP-GRAD-SPIKE: Gradient norm spikes detected.
         Evidence: Max gn 2649.75 > 10.0x median (0.55)
========================================
```

## Why this exists

A 9.8-hour XTTS fine-tune ended measurably worse than it had been three hours
earlier, and nothing in the stack said a word. That run's Coqui Trainer log ships
in this repo, so the verdict is reproducible instead of an anecdote:

```bash
trainproof epoch examples/real_world/xtts_diverged/trainer_0_log.txt --format coqui
```

**FAIL — diverging.** The loss reached its minimum at step 48,350, which is 66% of
the way through, and the run ended 1.62x above it.

The trainer's own bookkeeping agrees, which is the part worth checking yourself:
the last `BEST MODEL` line in that log is `best_model_49880.pth`, while the last
checkpoint written is `checkpoint_70000.pth`. The weights worth keeping had existed
for roughly 23,000 steps — about three hours of GPU time — before the run stopped.
Coqui recorded it. Nothing in the stack was asking.

## The fault-injection gallery

To validate the rules, the same QLoRA fine-tune (Qwen2.5-3B-Instruct, 4-bit,
LoRA r=16, 300 steps on Alpaca-cleaned) was run in six configurations — once
healthy, five times with exactly one knob deliberately broken — and every
configuration was repeated at three seeds (42, 43, 44). Eighteen real runs, real
logs, all shipped in [`examples/gallery/`](examples/gallery/). Seed 42 is the log
at each config root; 43 and 44 are nested beside it. The table below is seed 42;
[EVIDENCE_MATRIX.md](EVIDENCE_MATRIX.md) carries all eighteen and is generated
from the logs themselves:

| Run | Sabotage | Verdict | Key evidence |
|---|---|---|---|
| `healthy` | none | **PASS** | loss 1.52 → 0.94, stable gradients |
| `lr_hot` | LR x100 (2e-2) | **FAIL** | diverging: end 7.49 vs min 1.40; grad spike 2650 vs median 0.55 |
| `lr_zero` | LR = 0 | **FAIL** | dead run: first-5 median 1.52 vs last-5 1.49 (<5% improvement); lr=0 on 100% of steps |
| `fp16_nan` | fp16 + hot LR, no clipping | **FAIL** | diverging: end 7.21 vs min 1.09 (grad scaling absorbed the intended NaN — the run diverged instead; reported as observed) |
| `bad_labels` | labels shuffled per-sequence | **WARN only** (single-run) — caught by `trainproof compare` (v0.3) | grad spike 23.3 vs median 1.09 |
| `overfit` | 64 training samples, many epochs — pure memorisation, with a held-out eval set | **WARN** (`TP-OVERFIT`) | train 1.38 → 0.03 while eval bottoms out early and climbs to 3.76 |

### The honest finding: loss curves cannot see corrupted data

The `bad_labels` run — whose shuffled labels make real learning impossible —
*reduced its loss by 62%* (15.3 → 5.75, as trainproof measures start and end) —
while the healthy baseline improved only **14.7%**. On its own curve, the run
that cannot possibly learn looks like the *better* training run. That holds in
every seed: 62.5% / 62.8% / 61.9% for `bad_labels` against 14.7% / 23.0% / 25.1%
for healthy. The model was genuinely learning: not
the task, but the marginal token statistics of the garbage. From its own loss
curve, that is indistinguishable from healthy training (neural networks
famously fit random labels). **No single-run, loss-only rule can catch this
class of failure** — its real signature is *relative*: a loss floor ~6x higher
than a known-good run of the same task (5.59 vs 0.94).

That finding produced v0.3: `trainproof compare <baseline> <run...>` —
deterministic ratio rules against the healthy baseline you already have —
which catches `bad_labels` at a 6x loss-floor ratio, in 3 seeds out of 3.
The full study was repeated with three random seeds (18 runs, all shipped):
see [EVIDENCE_MATRIX.md](EVIDENCE_MATRIX.md) for every verdict, including the
honest miss (compare alone overlooks one lr_zero seed — the single-run
zero-LR fatality rule owns that case; the two commands cover each other's
blind spots). The gallery also improved the tool itself twice: the dead-run
rule and the total-zero-LR fatality rule both exist because runs escaped
earlier rule versions. See [ROADMAP.md](ROADMAP.md).

## The commands

```bash
# 1. Dataset preflight (speech/TTS pack): audio integrity, transcript quality,
#    duplicates, text-vs-audio duration mismatches
trainproof data /path/to/dataset_or_manifest.jsonl

# 2. Tokenizer preflight: vocabulary coverage, OOV rate, sequence blowouts,
#    suspicious splits on numbers/dates
trainproof tokenizer my_tokenizer.model transcripts.txt

# 3. Training-run verdict: NaN/divergence/dead-run detection, gradient spikes,
#    LR sanity, throughput — from log files, any framework
trainproof epoch logs/run.jsonl            # exit 1 on FAIL: CI-ready

# 4. Compare runs against a baseline (v0.6: BASELINE FIRST, then one or more runs
#    — argument order changed from <run> <baseline> in v0.5 and earlier).
#    Catch relative pathologies like the `bad_labels` run that evade single-run rules.
trainproof compare examples/gallery/healthy/trainer_state.json examples/gallery/bad_labels/trainer_state.json

# N-way: rank several runs against the same baseline in one table
trainproof compare examples/gallery/healthy/trainer_state.json examples/gallery/lr_hot/trainer_state.json examples/gallery/bad_labels/trainer_state.json
```

*(Note: As of v0.8.0, all text outputs include stable TP-* rule IDs. This is a breaking change for text-parsers; use --json instead).*

```text
========================================
TRAINPROOF VERDICT
========================================
[FAIL] Critical checks failed:
  [FAIL] TP-FLOOR-RATIO: loss floor ratio exceeded limit
         Evidence: Run floor 5.592 vs Baseline floor 0.937 (ratio 6.0x > 2.0)
  [FAIL] TP-END-RATIO: end loss ratio exceeded limit
         Evidence: Run end 5.750 vs Baseline end 1.082 (ratio 5.3x > 2.0)
========================================
```

Each command prints the verdict and sets the process exit code — so it works as a
CI gate out of the box. Nothing is written to disk unless you ask: `epoch` takes
`--html [PATH]` for a self-contained HTML report and `--sarif PATH` for SARIF.

## In CI: three exit codes and SARIF

| Code | Meaning |
|---|---|
| `0` | No FAIL findings — judged and passed (warnings possible) |
| `1` | **A FAIL verdict about your run.** Investigate. |
| `2` | **trainproof could not judge** — unreadable log, missing file, missing optional dependency. Says nothing about your run. |

That separation is the point. A corrupt log gives you `2`, never `1`:
trainproof will not tell you a run failed when it merely could not read it.

Findings can also land directly in GitHub pull-request annotations:

```bash
trainproof doctor . --sarif trainproof.sarif
# then: github/codeql-action/upload-sarif@v3 with sarif_file: trainproof.sarif
```

Pass a **relative** path (`.`), not an absolute one — the SARIF URIs mirror the
path you invoked with, and GitHub silently drops annotations it cannot map back
to a file in the repository.

The full stability contract — exit codes, JSON schema policy, rule-ID
guarantees, and what may change between releases — is in
**[CONTRACTS.md](CONTRACTS.md)**.

## Live guardian (v0.4)

Don't wait for the post-mortem — catch a doomed run *while it is still burning
GPU*. Add one line to a HuggingFace `Trainer`:

```python
from transformers import Trainer
from trainproof.integrations.hf import TrainproofCallback

trainer = Trainer(
    ...,
    callbacks=[TrainproofCallback(policy="stop_on_fail")],  # or policy="warn"
)
```

Run against a real diverging QLoRA fine-tune (learning rate 100x too high), the
guardian aborts it 20 steps into a 300-step schedule — on its own:

```text
{'loss': '1.784', 'grad_norm': '9.634',  'learning_rate': '0.007'}
{'loss': '4.282', 'grad_norm': '53.76',  'learning_rate': '0.009'}
{'loss': '10.6',  'grad_norm': '13.34',  'learning_rate': '0.011'}
{'loss': '31.67', 'grad_norm': '76.67',  'learning_rate': '0.013'}
...
TRAINPROOF ABORT - stopping training at step 20. Findings:
  [FAIL] Loss curve is diverging.
         Evidence: End loss 22.952 vs Min loss 1.358
  [FAIL] Loss never improved over the run (dead run).
         Evidence: median of first 5 losses 1.502 vs last 5 22.952

  scheduled steps : 300
  stopped at step : 20
  run saved       : 93% of the scheduled steps never ran
```

On a two-day pre-training run, that fraction is days of GPU time. Or watch a
growing log file from outside the process (CI-friendly, exits non-zero on FAIL):

```bash
trainproof watch logs/run.jsonl --interval 10 --until-fail --stall-timeout 300
# [21:37:44] warming up (5 records)
# [21:37:44] n_records=15 verdict=PASS findings=1
```

**Why is my training suddenly slow or stuck?** As of v0.7.0, the guardian telemetry captures `step_time` and (if `pynvml` is installed) `gpu_util`. Deterministic timing rules will warn you if throughput drops off a cliff or if the dataloader stalls out. Note that GPU utilization is displayed strictly as context to help you debug—trainproof will never judge your run or issue verdicts based on low utilization.

**The default is safe.** `policy="warn"` (the default) only observes and reports
— it never interrupts your run, so you can leave it on even for experiments you
expect to fail. Aborting is strictly opt-in via `policy="stop_on_fail"`, the one
mode that takes an irreversible action. trainproof does not make that decision
for you unless you ask.

The guardian applies the same deterministic rules as `trainproof epoch`, so it
inherits their documented single-run limitations.

## Pre-flight (v0.5): stop a run before it starts

The guardian saves most of a doomed run; preflight saves 100% because it never starts. Catch broken datasets and tokenizer misconfigurations instantly.

```bash
trainproof preflight data/dataset.jsonl --tokenizer mistralai/Mistral-7B-v0.1 --max-len 4096
```

```text
========================================
TRAINPROOF VERDICT
========================================
[FAIL] Critical checks failed:
  [FAIL] TP-PRE-EMPTY-TEXT: Empty or whitespace-only text found.
         Evidence: 1 records (indices [1]...)
========================================
```

*Checks: malformed JSONL, empty text, exact duplicate text, tokenizer structural checks (EOS/PAD/BOS), and context length overflows.*

The dataset checks need nothing beyond trainproof itself — drop `--tokenizer` and the
JSONL, empty-text and duplicate checks all still run. `--tokenizer` loads a real
tokenizer, so it additionally needs `pip install transformers`; without it you get
exit 2 ("could not judge") and a one-line reason, never a false verdict about your
data.

## Environment pre-flight (v0.15): can this machine start the run at all?

Dataset preflight assumes the run can start. Sometimes it cannot — and those
failures never produce a log, so no log-based tool can see them. A stack that
will not import, a checkpoint that segfaults its own loader, a first batch that
exhausts system RAM: zero steps, zero metrics, hours gone.

```bash
trainproof env --module train --cwd . --checkpoint out/last.ckpt --required-gb 20
```

```text
========================================
TRAINPROOF VERDICT
========================================
[FAIL] Critical checks failed:
  [FAIL] TP-ENV-IMPORT-FAIL: 'train' cannot be imported - this run cannot start.
         Evidence: ImportError: cannot import name 'BeamSearchScorer' from
         'transformers'  (raised at .../stream_generator.py:13)
  [FAIL] TP-ENV-MEM-INSUFFICIENT: Less system RAM is available than this run
         declares it needs.
         Evidence: 10.8 GB available, 20.0 GB required (31.1 GB total).
========================================
```

*Checks: import (in a subprocess), checkpoint integrity, system RAM headroom, free disk.*

Three things this does deliberately:

**Imports run out of process.** The failures here are violent — a segfaulting
extension, a CUDA abort, a library calling `os._exit` during import. In-process,
any of them kills the linter and you learn nothing. Out of process, a crash with
no Python exception is reported as `TP-ENV-IMPORT-CRASH` and named as a native
fault rather than misreported as an `ImportError`.

**Checkpoints are never unpickled.** `torch.load` executes arbitrary code by
design — the reason torch 2.6 flipped `weights_only` to `True`. A tool that must
run the file it inspects is not a safety tool. A checkpoint is read as the ZIP
archive it is: entry table, storage count, CRC. It distinguishes missing,
zero-byte, truncated mid-write, CRC-corrupt, legacy pre-1.6 pickle and complete.

**System RAM, not VRAM.** A GPU that runs out of memory raises a clean error and
the run fails. On Windows the driver spills to system RAM instead, and the machine
pages until the desktop stops responding — recoverable only by a hard reset.

`--cwd` matters: editable installs resolve relative to the working directory, so
probing from elsewhere reports `No module named X` for a package that imports
perfectly where training launches. Anything it cannot measure is `NOT-CHECKED`
with the reason, never a silent pass.

## Supported log formats

- HuggingFace Trainer (`trainer_state.json`)
- Coqui Trainer text logs (ANSI-colored `trainer_0_log.txt`)
- TensorBoard event files (`events.out.tfevents.*`) — PyTorch Lightning, Fish
  Speech, Coqui, and anything else that writes scalars to TensorBoard
- Generic JSONL / CSV (columns: step, loss, lr, grad_norm, time — all optional)

Auto-detected; override with `--format hf|coqui|tfevents|jsonl|csv`. Point it at
a file or a directory:

```bash
trainproof doctor results/my_run/tensorboard/version_0
```

The event reader is written from the wire format and imports no tensorflow,
tensorboard, protobuf, torch or numpy — reading a log file should not require
installing a training stack. It is validated byte-exact against TensorBoard's
own `EventAccumulator` (see `evidence/`). A directory of shards is merged, and a
file truncated by a killed run is read up to the cut rather than rejected.

Lightning console captures remain unsupported: they are TTY dumps, not logs.
The event file written beside them is the real record, and that is what to pass.

## For AI coding agents

If you are a coding agent (Claude Code, Codex, Cursor, ...) checking a
training project on a user's behalf:

- Run `trainproof doctor . --json` from the project root. It discovers
  training logs, judges them, and prints one JSON document: `schema_version`,
  `trainproof_version`, per-file reports, and `worst_verdict`.
- Every finding carries a stable `id` (e.g. `TP-DIVERGE`, `TP-DEAD-RUN`) —
  look ids up in [RULES.md](RULES.md) for what fired and what it does NOT
  mean. Parse ids, never message text.
- Exit `1` is a FAIL verdict about the run. Exit `2` means trainproof could
  not judge it — report that as a tool problem, never as a failed run.
- A `TP-PASS` verdict lists which checks ran and which were skipped, each with
  its reason — no data, too few points, or a series too degenerate to measure.
  Reports also carry this as structured data under `checks` (`ran`, and
  `skipped` as group → reason); parse that, not the prose. **Do not report
  stability of anything in the skipped list — a skipped check is not a passed
  check.**
- These verdicts are deterministic threshold rules, not model judgments.
  Relay them as measurements with their cited evidence, not as opinions.

## Philosophy

1. **Deterministic.** A rule fires or it doesn't. Thresholds live in one
   module, commented, tunable.
2. **Evidence-cited.** Every finding names the steps and values that triggered
   it.
3. **Honest about limits.** What the tool cannot detect is documented in the
   README, not discovered by the user in production.

## Family

trainproof judges training runs. Its sibling [ttsproof](https://github.com/Mormolykos/ttsproof)
judges TTS model *outputs* (structural audio checks, equivalence-aware WER/CER,
published method with DOI) — and trainproof builds on it for the speech
dataset checks.

## Author

Panagiotis (Panos) Gkilis — [portfolio](https://tts.bedvibe.studio/portfolio/) ·
[bedvibe.studio](https://bedvibe.studio/)

MIT license.