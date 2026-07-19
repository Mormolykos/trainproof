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

```bash
trainproof doctor .
```

```text
============================================================
FILE   : examples/gallery/healthy/trainer_state.json
FORMAT : hf
RECORDS: 16 (steps/epochs: 20..300)
------------------------------------------------------------
VERDICT: PASS
------------------------------------------------------------
[PASS] Loss curve shows healthy shape, grad norms are stable.
       Evidence: 16 steps analyzed.

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

This repo ships the real logs of five QLoRA fine-tuning runs (Qwen2.5-3B,
RTX 5080 — see the gallery below). Judge one right now:

```bash
trainproof epoch examples/gallery/lr_hot/trainer_state.json --format hf
```

```text
[FAIL] Critical checks failed:
  [FAIL] Loss curve is diverging.
         Evidence: End loss 7.492 vs Min loss 1.398
  [WARN] Gradient norm spikes detected.
         Evidence: Max gn 2649.75 > 10.0x median (0.55)
```

## Why this exists

The author lost a real 11-hour fine-tune to a failure nothing warned about.
Pointed retroactively at that run's 1MB Coqui Trainer log (2,501 logged steps),
trainproof's verdict: **FAIL — diverging**. The loss reached its minimum at 82%
of the run and ended 1.9x above it. Translation: the final two hours of GPU
time made the model measurably worse, and the checkpoint worth keeping had
already existed for hours. No tool in the stack said a word.

## The fault-injection gallery

To validate the rules, the same QLoRA fine-tune (Qwen2.5-3B-Instruct, 4-bit,
LoRA r=16, 300 steps on Alpaca-cleaned) was run five times — once healthy, four
times with exactly one knob deliberately broken. Real runs, real logs, all
shipped in [`examples/gallery/`](examples/gallery/):

| Run | Sabotage | Verdict | Key evidence |
|---|---|---|---|
| `healthy` | none | **PASS** | loss 1.52 → 0.94, stable gradients |
| `lr_hot` | LR x100 (2e-2) | **FAIL** | diverging: end 7.49 vs min 1.40; grad spike 2650 vs median 0.55 |
| `lr_zero` | LR = 0 | **FAIL** | dead run: first-5 median 1.52 vs last-5 1.49 (<5% improvement); lr=0 on 100% of steps |
| `fp16_nan` | fp16 + hot LR, no clipping | **FAIL** | diverging: end 7.21 vs min 1.09 (grad scaling absorbed the intended NaN — the run diverged instead; reported as observed) |
| `bad_labels` | labels shuffled per-sequence | **WARN only** (single-run) — caught by `trainproof compare` (v0.3) | grad spike 23.3 vs median 1.09 |

### The honest finding: loss curves cannot see corrupted data

The `bad_labels` run — whose shuffled labels make real learning impossible —
*reduced its loss by 62%* (18.9 → 5.75). The model was genuinely learning: not
the task, but the marginal token statistics of the garbage. From its own loss
curve, that is indistinguishable from healthy training (neural networks
famously fit random labels). **No single-run, loss-only rule can catch this
class of failure** — its real signature is *relative*: a loss floor ~6x higher
than a known-good run of the same task (5.59 vs 0.94).

That finding produced v0.3: `trainproof compare <baseline> <run...>` —
deterministic ratio rules against the healthy baseline you already have —
which catches `bad_labels` at a 6x loss-floor ratio, in 3 seeds out of 3.
The full study was repeated with three random seeds (15 runs):
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
trainproof epoch logs/run.jsonl            # exit code 1 on FAIL: CI-ready

# 4. Compare runs against a baseline (v0.6: BASELINE FIRST, then one or more runs
#    — argument order changed from <run> <baseline> in v0.5 and earlier).
#    Catch relative pathologies like the `bad_labels` run that evade single-run rules.
trainproof compare examples/gallery/healthy/trainer_state.json examples/gallery/bad_labels/trainer_state.json

# N-way: rank several runs against the same baseline in one table
trainproof compare examples/gallery/healthy/trainer_state.json examples/gallery/lr_hot/trainer_state.json examples/gallery/bad_labels/trainer_state.json
```

```text
========================================
TRAINPROOF VERDICT
========================================
[FAIL] Critical checks failed:
  [FAIL] loss floor ratio exceeded limit
         Evidence: Run floor 5.592 vs Baseline floor 0.937 (ratio 6.0x > 2.0)
  [FAIL] end loss ratio exceeded limit
         Evidence: Run end 5.750 vs Baseline end 1.082 (ratio 5.3x > 2.0)
========================================
```

Each command prints the verdict, writes a self-contained HTML report, and sets
the process exit code — so it works as a CI gate out of the box.

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
  [FAIL] Empty or whitespace-only text found.
         Evidence: 1 records (indices [1]...)
========================================
```

*Checks: malformed JSONL, empty text, exact duplicate text, tokenizer structural checks (EOS/PAD/BOS), and context length overflows.*

## Supported log formats

- HuggingFace Trainer (`trainer_state.json`)
- Coqui Trainer text logs (ANSI-colored `trainer_0_log.txt`)
- Generic JSONL / CSV (columns: step, loss, lr, grad_norm, time — all optional)

Auto-detected; override with `--format hf|coqui|jsonl|csv`. TensorBoard event
files are planned (Lightning console captures are TTY dumps, not logs, and
will not be supported).

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