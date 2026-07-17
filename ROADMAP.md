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

## v0.3 — Reference-based comparison (implemented)  ← next

`trainproof compare <run> <baseline>`: judge a run against a known-good
baseline of the same task. Deterministic ratio rules (loss floor ratio,
improvement-rate ratio, grad-norm distribution ratio), same PASS/WARN/FAIL
verdict form, evidence-cited. Directly closes the documented limitation using
data every team already has: their last good run.

## v0.4 — Live Guardian (implemented)

- `trainproof watch <logfile>`: tail a growing log, re-judge on an interval,
  alert on FAIL while the run is still burning GPU.
- HuggingFace `TrainerCallback` integration: one line in a training script;
  policies `warn` / `stop_on_fail` ("this run is dead — aborting at step 60").
- Ships after v0.3 so the guardian judges with reference-aware eyes, not just
  absolute rules.

## v0.5 — Pre-flight LLM dataset + tokenizer linter (implemented)

`trainproof preflight <dataset.jsonl> [--tokenizer NAME] [--max-len N]`: the
"before training" tier — catch broken data BEFORE a single GPU-second is billed
(the guardian saves most of a doomed run; preflight saves 100% because it never
starts). The deterministic answer to the v0.2 `bad_labels` finding: loss curves
can't see corrupted data, so inspect the data itself. Engine is IO-free and
imports transformers zero times; the CLI loads a tokenizer lazily. Every finding
carries a stable machine-readable id (for future `--ignore` / `--fail-on` / CI
annotations).

Checks shipped (all deterministic, universally-true facts):
- Dataset: malformed JSONL (FAIL, with line number), empty/whitespace text
  (FAIL), exact-duplicate text (WARN).
- Tokenizer: missing eos_token (FAIL), missing pad_token (WARN), pad==eos (WARN),
  bos presence (INFO only — never warns; many model families have no bos).
- Context: samples exceeding `--max-len` will be silently truncated (WARN).

Deliberately CUT from v0.5 (future work, NOT built): chat-template validation
and attention-mask correctness — every model family (ChatML/Alpaca/Llama3/Qwen/
Gemma/...) has different conventions; that is weeks of work and would violate the
"universally-true, no guessing" rule if rushed.

Market note: a text-level competitor (Parallelogram) already has community
traction — proof of demand.

## Later

- TensorBoard event-file adapter (unlocks Lightning/fairseq-style runs).
- Checkpoint-resume integrity verifier: on resume, assert current_lr and
  scheduler state align with global_step (a famous silent HF/DeepSpeed bug).
  Testable on a single GPU — in scope.
- Evaluation plugins: trainproof will never run benchmarks itself, but it may
  CONSUME evaluation results produced by dedicated tools, to correlate "run
  looked healthy" with "model actually improved". Integration surface only.
- Training-pathology corpus: grow the labeled gallery (real runs + known fault
  + verdict) into a standard regression suite — every new rule must be tested
  against every archived pathology. The five v0.2 gallery runs are its seed
  (shipped in `examples/gallery/`).
- Multi-seed fault-injection study (done for v0.3; 3 seeds x 5 configs, see
  EVIDENCE_MATRIX.md) — extend toward a citable technical report.

## Non-goals (deliberate)

- Model-quality evaluation (benchmarks, win-rates): that is the model's
  quality, not the run's health — mature tools exist. trainproof judges runs.
  (In this family, output quality is ttsproof's job.)
- Invented confidence percentages, probabilistic verdicts, or "fingerprint"
  match-scores. Rules are deterministic; every finding cites its evidence. Locked.
- **Hyperscale / multi-node features that cannot be tested on the author's own
  hardware** (DeepSpeed ZeRO shape assertors, NCCL straggler detection,
  stable-rank collapse predictors, hot-ID gradient monitors, 1k-16k-GPU
  cluster diagnostics). trainproof's credibility is that every rule is
  dogfooded on real runs; shipping untested rules for hardware we don't have
  would betray that. Scope stays single-GPU / small-cluster fine-tuning — where
  the tool's audience actually works.
- Lightning console TTY captures as an input format: they are terminal dumps,
  not logs.