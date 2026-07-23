# Changelog

All notable changes to trainproof are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning follows
[SemVer](https://semver.org/).

## [0.10.0] — 2026-07-23 — the contract release

Nothing about how trainproof judges a run changed in this release. No rule, no
threshold, and no verdict moved — every gallery verdict is locked in
`tests/golden/` and byte-identical to v0.9. What changed is what trainproof
*promises*, now written down in [CONTRACTS.md](CONTRACTS.md).

### Added
- **[CONTRACTS.md](CONTRACTS.md)**: exit codes, JSON schema policy, rule-ID
  stability, SARIF mapping, verdict-stability guarantee, and the pre-1.0
  breaking-change policy.
- **SARIF 2.1.0 output** via `--sarif PATH` on `data`, `tokenizer`, `epoch`,
  `doctor`, `compare` and `preflight` — findings become GitHub PR annotations.
  Works independently of `--json`.
- `--json` on `data`, `tokenizer` and `preflight` (previously `epoch`,
  `doctor` and `compare` only).
- Locked gallery snapshots in `tests/golden/`: verdict plus the *complete* rule
  ID set for all six runs and seven baseline comparisons. A rule that stops
  firing and one that starts firing spuriously both now fail the build.
- Test enforcing that rule IDs in source and in `RULES.md` match in both
  directions, and a test that the two declared version strings agree.

### Changed
- **Breaking — exit codes.** `2` now means "trainproof could not judge",
  covering unreadable logs, missing files, no parsed records and missing
  optional dependencies. Previously several of these exited `1`, which is
  reserved for a FAIL verdict about your run. CI that treats any non-zero as
  failure is unaffected; anything distinguishing `1` from `2` should be
  reviewed.
- **Breaking — `schema_version` is now `2`.** `doctor` no longer emits a
  separate `compare_findings` key: single-run and baseline findings live in one
  `findings` array, each tagged with `source` (`single_run` or `compare`). The
  envelope gained an `error` key.
- "Cannot judge" messages now go to **stderr**, leaving stdout parseable.
- An unreadable log no longer reports `worst_verdict: "FAIL"` in JSON. It
  reports `worst_verdict: null` with a populated `error` and exits `2`.
- A missing `transformers` install is no longer a FAIL verdict on your dataset
  (rule `TP-PRE-TRANSFORMERS-MISSING` removed); it is a tool error, exit `2`.

### Fixed
- `doctor --baseline` printed `[FAIL]` comparison findings and still exited
  `0`, because the exit code was computed from single-run verdicts alone. A
  failing baseline comparison now fails the run, as the printed output always
  claimed. Found by the contract work: `bad_labels` against `healthy` is the
  reproducing case.
- Uncaught internal errors previously fell through to Python's default exit
  code `1` and were indistinguishable from a FAIL verdict. A top-level handler
  now reports them as `2`.
- `RULES.md` no longer carries a stale version stamp.
- The v0.5.0 entry below described a `coroner` command that was never
  implemented; corrected to `epoch`.

## [0.9.0] — 2026-07-20 — the eval-aware release

### Added
- `TP-OVERFIT` (WARN): deterministic overfitting detection — eval_loss rising
  past 1.2x of its own minimum while train_loss keeps falling (needs >= 4 eval
  points). Documented in `RULES.md`. Grounded in a real Qwen2.5-3B QLoRA run
  shipped in `examples/gallery/overfit/`: eval_loss bottomed at 1.25 (step 30)
  and climbed to 3.76 (step 300) while train_loss fell 1.38 -> 0.03.

### Fixed
- HuggingFace `trainer_state.json` ends with a training-summary entry whose
  `train_loss` is the run *average*, not a per-step loss. It was leaking into
  the per-step loss series and could fire a false `TP-DIVERGE` on any
  steeply-converging run. The HF adapter now drops that summary entry.

No existing rule thresholds changed; all five original gallery verdicts are
identical. 57 tests passing.

## [0.8.0] — 2026-07-19 — the trust release

### Added
- Stable rule IDs on every finding (`TP-DIVERGE`, `TP-DEAD-RUN`, `TP-ZERO-LR`,
  `TP-STEP-CLIFF`, …), documented in `RULES.md` with threshold and scope.
- `--json` output on `epoch` / `doctor` / `compare`: one JSON document with
  `schema_version: 1`, `trainproof_version`, full reports with rule IDs and
  worst verdict.
- Honest `TP-PASS`: states which check groups ran vs. were skipped for lack of
  data, instead of implying stability of anything unmeasured.
- README section "For AI coding agents" describing `trainproof doctor .
  --json` for agent use.
- HF adapter now captures `eval_loss` records (unused by any rule until v0.9).

### Changed
- **Breaking for text-parsers:** output lines now include the rule ID
  (`[FAIL] TP-DIVERGE: ...`). Parse `--json` instead. Preflight rule IDs
  renamed to `TP-PRE-*`.
- Canonical column mapping: adapters map log columns to canonical keys by
  exact name only, never substring guessing — `eval_loss` can no longer be
  mistaken for training loss. `--map CANON=COLUMN` overrides the mapping;
  `doctor` prints the source column used for generic logs.

No rule threshold changes; all gallery verdicts identical to v0.7. 52 tests
passing.

## [0.7.0] — 2026-07-19 — guardian telemetry

### Added
- Live step-time telemetry in `TrainproofCallback`: wall-clock seconds per
  step, feeding two new deterministic rules that only fire when timing data
  exists — step-time cliff (WARN, recent median > 3x early median) and
  dataloader-bound (WARN, loader_time/step_time median > 50%).
- Optional GPU-utilization capture via `pynvml`, shown as display-only
  context — never judged (low utilization is not a failure).
- `trainproof watch <log> --stall-timeout 300`: one factual warning per
  episode if a growing log file stops growing.
- `--version` flag.

### Changed
- Log-format detection unified into the adapters; duplicate CLI heuristic
  removed.

No rule threshold changes. 43 tests passing.

## [0.6.0] — 2026-07-19 — the doctor release

### Added
- `trainproof doctor` (flagship): zero-config autopsy of a file or a whole
  directory — auto-discovers logs (HF, Coqui, JSONL/CSV), prints a
  triage-sorted summary (failures first), per-log findings with cited
  evidence, and a fixed "what this cannot tell you" footer. `diagnose` is an
  alias. Optional `--baseline` adds a VS-BASELINE section per log.
- N-way `compare`: rank several runs against one baseline in a single table.

### Changed
- **Breaking:** `compare` argument order is now `compare <baseline>
  <run...>` (previously `<run> <baseline>`). Update scripts/CI accordingly.

No engine or rule changes; all thresholds and verdicts identical to v0.5. 38
tests passing.

## [0.5.0] — 2026-07-17 — pre-flight

### Added
- `trainproof preflight <dataset.jsonl> [--tokenizer NAME] [--max-len N]`:
  deterministic checks before a single GPU-second is spent — malformed JSONL
  (FAIL, with line number), empty/whitespace samples (FAIL), exact-duplicate
  samples (WARN), missing `eos_token` (FAIL), missing `pad_token` (WARN),
  `pad==eos` (WARN), samples exceeding `--max-len` (WARN). Exits non-zero on
  FAIL for CI use before GPUs are provisioned.
- Completes the training-reliability lifecycle: preflight (before) · guardian
  (during) · epoch (after) · compare (vs. baseline).

Deliberately not built: chat-template and attention-mask validation — every
model family has different conventions; deferred rather than rushed.

## [0.4.0] — 2026-07-17 — the Live Guardian

### Added
- `TrainproofCallback` for the HuggingFace `Trainer`: re-runs trainproof's
  deterministic rules live during training. `policy="warn"` (default) only
  observes and reports; `policy="stop_on_fail"` (opt-in) aborts the run on a
  FAIL verdict.
- `trainproof watch <logfile>`: tails a growing log from outside the process,
  re-judges on an interval, exits non-zero on FAIL (`--until-fail`).

**Live proof:** armed with `stop_on_fail` against a real diverging QLoRA
fine-tune (Qwen2.5-3B, RTX 5080, learning rate 100x too high), the guardian
aborted the run at step 20 of 300 — 93% of the scheduled steps never ran.

## [0.3.0] — 2026-07-17 — compare engine

### Added
- `trainproof compare <run> <baseline>`: deterministic ratio rules —
  loss-floor ratio, end-loss ratio, improvement deficit, grad-norm ratio,
  baseline sanity check. Catches the shuffled-labels run that single-run
  rules cannot see, at a 6x floor ratio across 3 seeds.
- Total-zero-LR fatality rule: `lr=0` on every step is now a FAIL from the LR
  column itself.
- `EVIDENCE_MATRIX.md`: the fault-injection study repeated across 3 random
  seeds (15 real QLoRA runs), including the one honest miss (`compare` alone
  overlooks one lr_zero seed that the single-run rules catch).

## [0.2.0] — 2026-07-17 — the fault-injection gallery

### Added
- `examples/gallery/`: five real QLoRA runs (Qwen2.5-3B, RTX 5080) with
  unmodified `trainer_state.json` logs — one healthy, four with exactly one
  knob broken (LR x100, LR 0, fp16 overflow, shuffled labels).
- Dead-run rule: loss that never improves now FAILs (found via the gallery
  itself — the zero-LR run had been escaping with only a WARN).
- HuggingFace + Coqui log-format adapters, auto-detected, `--format` override.
- `ROADMAP.md`.

**Documented limitation:** the shuffled-labels run reduced its loss 62% by
learning the marginal token distribution — indistinguishable from real
learning on its own loss curve. No single-run rule can catch this class; its
signature is relative, which defines v0.3's `compare` command.

## [0.1.0] — 2026-07-17 — a linter for ML training runs

### Added
- `trainproof data <dir|manifest>`: speech/TTS dataset preflight
  (sample-rate/channel consistency, clipping, silence, duration outliers,
  duplicates, unnormalized transcripts, text/audio mismatch).
- `trainproof tokenizer <model> <transcripts>`: SentencePiece coverage, OOV
  rate, sequence-length blowout, suspicious number splits.
- `trainproof epoch <logfile>`: first-epoch verdict — NaN/Inf, divergence,
  dead runs, gradient spikes, LR sanity, throughput. Generic JSONL/CSV,
  HuggingFace `trainer_state.json`, and Coqui Trainer logs, auto-detected.

All verdict thresholds are deterministic, defined in one module (`rules.py`).
Exit codes make it a CI gate.

**Field test:** pointed at a real 11-hour XTTS fine-tune's log, flagged it as
diverging — final loss 1.9x above its minimum, reached at 82% of the run; the
last two hours of GPU time had made the model worse.

[0.9.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.9.0
[0.8.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.8.0
[0.7.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.7.0
[0.6.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.6.0
[0.5.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.5.0
[0.4.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.4.0
[0.3.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.3.0
[0.2.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.2.0
[0.1.0]: https://github.com/Mormolykos/trainproof/releases/tag/v0.1.0
