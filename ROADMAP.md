# trainproof — Roadmap

Updated 2026-07-17, after the first fault-injection study (five controlled QLoRA
runs: one healthy, four sabotaged). Priorities below are argued from that
evidence, not from general ML advice.

## Current state (v0.2-dev)

- Deterministic single-run linting: NaN/Inf, divergence, flat/dead loss,
  no-improvement (dead run), gradient spikes, zero learning rate, throughput.
- Log adapters: generic JSONL/CSV, HuggingFace `trainer_state.json`,
  Coqui Trainer text logs. Auto-detected, `--format` override.
- Speech domain pack: dataset + tokenizer preflight (built on ttsproof).
- Validated by fault injection: healthy run PASS; hot-LR, zero-LR, and
  fp16-overflow runs FAIL with evidence-cited findings.

## Known limitation (documented, not hidden)

A garbage-labels run (shuffled targets) reduced its loss 62% by learning the
marginal token distribution — from its own loss curve it is indistinguishable
from real learning (networks fit random labels). **No single-run, loss-curve-only
rule can catch this class.** Its detectable signature is relative: a loss floor
several times higher than a known-good run of the same task.

## v0.3 — Reference-based comparison  ← next

`trainproof compare <run> <baseline>`: judge a run against a known-good
baseline of the same task. Deterministic ratio rules (loss floor ratio,
improvement-rate ratio, grad-norm distribution ratio), same PASS/WARN/FAIL
verdict form, evidence-cited. Directly closes the documented limitation using
data every team already has: their last good run.

## v0.4 — Live guardian

- `trainproof watch <logfile>`: tail a growing log, re-judge on an interval,
  alert on FAIL while the run is still burning GPU.
- HuggingFace `TrainerCallback` integration: one line in a training script;
  policies `warn` / `stop_on_fail` ("this run is dead — aborting at step 60").
- Ships after v0.3 so the guardian judges with reference-aware eyes, not just
  absolute rules.

## Later

- TensorBoard event-file adapter (unlocks Lightning/fairseq-style runs).
- LLM fine-tune domain pack: chat-template validation, sequence-length
  blowouts, duplicate/contamination checks (the speech pack's sibling) —
  dataset preflight is the correct layer for catching corrupted labels
  BEFORE training, complementing (not replacing) reference comparison.
- Evaluation plugins: trainproof will never run benchmarks itself, but it may
  CONSUME evaluation results produced by dedicated tools, to correlate "run
  looked healthy" with "model actually improved". Integration surface only.
- Training-pathology corpus: grow the labeled gallery (real runs + known fault
  + verdict) into a standard regression suite — every new rule must be tested
  against every archived pathology. The five v0.2 gallery runs are its seed
  (shipped in `examples/gallery/`).
- Multi-seed fault-injection study (3 seeds x 5 configs) as a citable
  technical report.

## Non-goals (deliberate)

- Model-quality evaluation (benchmarks, win-rates): that is the model's
  quality, not the run's health — mature tools exist. trainproof judges runs.
  (In this family, output quality is ttsproof's job.)
- Invented confidence percentages or probabilistic verdicts. Rules are
  deterministic; every finding cites its evidence. Locked.
- Lightning console TTY captures as an input format: they are terminal dumps,
  not logs.