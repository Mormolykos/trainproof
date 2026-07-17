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

```bash
pip install trainproof
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
| `bad_labels` | labels shuffled per-sequence | **WARN only** — see below | grad spike 23.3 vs median 1.09 |

### The honest finding: loss curves cannot see corrupted data

The `bad_labels` run — whose shuffled labels make real learning impossible —
*reduced its loss by 62%* (18.9 → 5.75). The model was genuinely learning: not
the task, but the marginal token statistics of the garbage. From its own loss
curve, that is indistinguishable from healthy training (neural networks
famously fit random labels). **No single-run, loss-only rule can catch this
class of failure** — its real signature is *relative*: a loss floor ~6x higher
than a known-good run of the same task (5.59 vs 0.94).

That finding sets the roadmap: v0.3 is `trainproof compare <run> <baseline>` —
deterministic ratio rules against the healthy baseline you already have.
See [ROADMAP.md](ROADMAP.md). (The gallery also improved v0.2 itself: the
dead-run rule exists because `lr_zero` initially escaped with only a WARN.)

## The three commands

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
```

Each command prints the verdict, writes a self-contained HTML report, and sets
the process exit code — so it works as a CI gate out of the box.

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