# trainproof — Roadmap

The official product roadmap. **Every future feature decision is evaluated
against this document.** If a proposed feature is not on the roadmap and is not
justified by the decision rule below, it does not ship.

Updated 2026-08-01, after shipping v0.15 (environment preflight). The previous
revision of this file still described v0.8 as current, seven releases after the
fact — recorded here because a roadmap that drifts is the same failure this
project exists to catch, and because `EVIDENCE_MATRIX.md` is generated precisely
so it cannot happen to the evidence.

---

## Decision rule (apply to every proposed feature)

A feature ships only if all of these hold:

1. **It is deterministic.** A rule fires or it does not; no model judges the run,
   no probability, no score.
2. **It is backed by a real, reproducible failure case** committed to the
   gallery or to `evidence/`. No evidence, no rule.
3. **It is dogfoodable on the author's own hardware** (single GPU / small
   fine-tune). Untested rules for hardware we do not have are forbidden.
4. **It cites its evidence.** Every finding shows the exact numbers that
   triggered it and carries a stable `TP-*` id.
5. **It answers "would someone lose GPU hours (or trust) without it?"** If not,
   it is scope creep.

---

## Current state — v0.15 (shipped)

Eight commands covering all three phases of a run, 84 stable rule IDs,
228 tests, `schema_version` 3.

**Before the run**

- **`env`** (v0.15) — imports the training stack **in a subprocess** and reports
  the exact exception, message and raising file; distinguishes an `ImportError`
  from a native crash (`TP-ENV-IMPORT-CRASH`) and from a hang. Inspects
  checkpoints as the ZIP archives they are — **never unpickled**, because
  `torch.load` executes arbitrary code by design. Measures system RAM headroom
  and free disk.
- **`preflight` / `data` / `tokenizer`** — malformed JSONL, empty rows,
  duplicates, missing eos/pad, context overflow; audio-corpus and tokenizer
  checks.

**During the run**

- **Live guardian** — HuggingFace `TrainerCallback` with step-time telemetry and
  opt-in auto-abort; `watch` with `--stall-timeout`.

**After the run**

- **`doctor`** (flagship) — zero-config autopsy of a file or a directory;
  discovers logs, triage-sorted summary, per-log findings.
- **`epoch`** — divergence / dead-run / NaN / flatline / spike / LR / overfit /
  throughput verdicts from a finished log.
- **`compare`** — N-way ratio rules against a known-good baseline.

**Inputs.** Five log formats: HuggingFace `trainer_state.json`, Coqui trainer
text logs, **TensorBoard event files**, JSONL, CSV. Auto-detected.

**Trust surface.** Stable rule IDs + `RULES.md` (drift-tested in both
directions), `CONTRACTS.md` (exit codes, schema policy, rule-ID and verdict
stability), `--json` and SARIF 2.1.0, honest PASS listing ran-vs-skipped checks,
and a `NOT-CHECKED` verdict that exits 2 so "could not judge" is never reported
as "clean".

**Validation.** A 6-config × 3-seed fault-injection gallery, plus `evidence/`:
real fine-tunes on an RTX 5080 from **three frameworks** — HuggingFace, Coqui
XTTS v2, and PyTorch Lightning (Fish Speech). One XTTS run ships as *both* a
text log and the TensorBoard event file written by the same training; two
independent parsers produce the identical verdict, and `EVIDENCE_MATRIX.md`
computes that agreement rather than asserting it.

### Documented limitations (not hidden)

- A shuffled-labels run reduced its loss 62% by learning the marginal token
  distribution — from its own loss curve, indistinguishable from real learning.
  No single-run, loss-only rule can catch this class; its signature is relative,
  which is why `compare` exists.
- The checkpoint check reads **structure**, not tensors. It cannot report NaN
  weights, dead layers, or a shape mismatch against a model definition.
- The memory check compares against a requirement the user supplies. trainproof
  does not predict how much memory a model, batch size or sequence length will
  need.

---

## Shipped since v0.8

| Version | Theme | What it bought |
|---|---|---|
| v0.9 | eval-aware | `TP-OVERFIT` from the eval curve |
| v0.10 | contract | `CONTRACTS.md`, SARIF 2.1.0, exit code 2 = "cannot judge" |
| v0.11 | evidence | all three seeds shipped; `EVIDENCE_MATRIX.md` generated, not written |
| v0.12 | honest verdict | degenerate-series rules; PASS lists only checks that ran |
| v0.13 | third state | `NOT-CHECKED` — "nothing could be checked" ≠ "checked and clean" |
| v0.14 | third framework | TensorBoard reader, zero dependencies; `TP-ZERO-GRAD` false positive fixed |
| v0.15 | before the GPU | environment preflight: import, checkpoint, RAM, disk |

Two things landed **better** than this roadmap planned. The TensorBoard adapter
was scheduled for v1.0 as an optional `[tb]` extra; it shipped in v0.14 with **no
optional dependency at all**, decoded from the wire format and validated
byte-exact against `tensorboard`'s own `EventAccumulator`. And v0.14's
`TP-ZERO-GRAD` fix came from running the shipped rules against a real 125,000-step
Coqui run, not from a test — the first time the tool caught its own false
positive on somebody's real training.

---

## v0.16 — checkpoint tensors (next)

v0.15 proved a checkpoint can be inspected without executing it. The archive it
already opens also contains the tensor storages, so the same ZIP read answers
questions that currently require loading the model:

- **NaN / Inf in saved weights** — counted from raw storage bytes, no torch.
- **All-zero or dead tensors** — a layer that never learned.
- **Missing optimizer state** — the run will resume, but Adam's moments restart
  from zero and the loss jumps. Silent today, brutal on a long fine-tune.
- **Shape / key mismatch between two checkpoints** — "can I resume this run into
  this architecture", answered before the GPU is reserved.

Why it qualifies under the decision rule: deterministic, evidence exists on
disk (real 5.6 GB XTTS and 4.8 GB Fish checkpoints), dogfoodable, and the
failure it prevents is a resume that silently discards optimizer state.

**The safety property is the differentiator.** `torch.load` unpickles, and
unpickling executes arbitrary code — a documented RCE vector, and the reason
torch 2.6 flipped `weights_only` to `True`. Reading tensors from the archive
without unpickling means an untrusted checkpoint can be inspected safely.
Nothing else in this space offers that.

---

## v1.0 — the stability contract (promises, not features)

1.0 is the moment the interface becomes a promise. Most of it already exists;
what remains is explicitly listed so 1.0 is a decision, not a milestone that
drifts.

Already in place: rule IDs frozen, `schema_version` policy, documented exit
codes, SemVer + CHANGELOG, `RULES.md` complete and drift-tested, the
fault-injection gallery serving as a regression suite, and the TensorBoard
adapter (delivered early, without the optional extra it was planned as).

Still required for 1.0:

- **Adapter registry** — so a new format is a plug-in, not core surgery. The
  five formats currently dispatch from one `if/elif` chain; that is fine at five
  and wrong at fifteen.
- **`[tool.trainproof]` config in `pyproject.toml`** — select/ignore rules by id,
  per-project threshold overrides. CI teams will not adopt a linter they cannot
  tune per-repo. Planned for v0.10 and never built.
- **A lint gate** (see known gaps).

v1.0 is the Show HN moment — a year of evidence behind it.

---

## Post-1.0 — exactly three lanes

1. **Adapter ecosystem** — Axolotl, Unsloth, community adapters via the
   registry, with credit. Growth without core bloat.
2. **The agent lane** — `watch` as an MCP server so a coding agent stays
   connected during a long run and interrupts at the moment of `TP-STEP-CLIFF`.
   Dogfoodable; nobody else offers it.
3. **The research lane** — a fault-injection methodology paper (Zenodo): the
   multi-seed matrix + the overfit extension, formalizing evidence-driven
   linter development.

Also acceptable (single-GPU-testable, deferred, not committed):

- **Checkpoint-resume integrity verifier** — on resume, assert `current_lr` and
  scheduler state align with `global_step` (a real silent HF/DeepSpeed bug).
  Partly subsumed by v0.16's optimizer-state check.
- **Chat-template / attention-mask validation** for preflight — *deliberately
  cut from v0.5, not abandoned.* Every model family has different conventions;
  doing it right is weeks of work, and rushing it would violate the
  "universally-true, no guessing" rule. Revisit only when it can be made
  deterministic per-family with evidence.
- **Consuming external evaluation results** to correlate "run looked healthy"
  with "model actually improved" — integration surface only; trainproof never
  runs benchmarks itself.

Market signal: a text-level dataset linter (Parallelogram) already has community
traction — proof of demand for the preflight lane; trainproof differentiates by
going deeper (tensor/tokenizer level, full lifecycle) rather than text-only.

---

## Known gaps — disclosed rather than forgotten

1. ~~**A PASS verdict is still possible when no check executed.**~~ **FIXED in
   v0.13.0** by the `NOT-CHECKED` verdict, which exits 2.

   The repro originally stated here was wrong, and is corrected for the record:
   a log carrying a loss column and fewer than five points does **not** skip
   every group. `divergence` and `flat-loss` are guarded by loss *positivity*,
   not by point count, so both execute and the resulting PASS is honest. Zero
   groups execute only when there are fewer than five finite loss points **and**
   the mean loss is non-positive — a short run whose loss is all zeros — with no
   gradient-norm, learning-rate, step-time or eval_loss column.

   Recorded because the entry asserted a behaviour the code did not exhibit, and
   v0.13.0's rule prose was written from it before anyone ran the repro. Same
   class as the six drifted `RULES.md` thresholds found in v0.12.0: documentation
   describing a trigger condition needs either generation from the code or a test
   that executes the documented repro and asserts the documented outcome.

2. **Partial zero-loss is not detected.** `TP-ZERO-LOSS` requires *every* finite
   loss to be exactly 0.0. A run where only some steps collapse to zero — the
   more common shape, since truncation and masking bugs often hit a subset of
   samples — is not flagged. The `TP-ZERO-LR` / `TP-ZERO-LR-PARTIAL` pair is the
   precedent. Needs a gallery fixture first, per the decision rule.

3. **`cli.py` still has one `try/except/pass` in `doctor`'s candidate
   discovery.** The handler that swallowed a failed baseline comparison was
   fixed in v0.12.0 (`TP-CMP-ERROR`); this one silently drops a file during the
   directory walk, before any log is judged. Lower impact — an undiscovered file
   never enters the report — but the same class.

4. **Repo-wide lint is unconfigured.** No `[tool.ruff]` section in
   `pyproject.toml`, ruff is not a dependency, and lint has never been part of
   the gate. Running it with an inherited config reports ~79 findings, almost
   all import ordering, unused unpacked variables, and the blind-except pattern.
   Adopting a lint gate is a decision to take deliberately, on its own, and never
   inside a correctness release.

5. **TensorBoard CRC footers are not verified.** `crc32c` is not in the standard
   library, and adding a dependency to checksum a file that is only being read
   would contradict the zero-dependency guarantee. A record that fails to decode
   is skipped rather than reported as corruption. *(v0.14)*

---

## Never (locked — each already refused at least once)

- **No ML judging ML.** No model scores a run.
- **No confidence scores / health percentages** (e.g. "92/100"). Fake certainty
  in a deterministic costume. Findings are counts + cited evidence.
- **No AI-generated diagnoses or hallucinated fixes / prescriptive advice.** The
  moment it advises, it can lie.
- **No dashboard-first product / web UI / experiment tracking.** That is
  wandb/MLflow's domain; trainproof answers "should I trust this run," not
  "show me metrics."
- **No SaaS / cloud / telemetry collection.** Local tool; the user's data stays
  theirs.
- **No hyperscale / multi-node features undogfoodable on one GPU** (DeepSpeed
  ZeRO shape assertors, NCCL straggler detection, stable-rank collapse
  predictors, hot-ID gradient monitors, 1k–16k-GPU cluster diagnostics).
  Relabelling this as "research" does not exempt it. Scope stays
  single-GPU / small fine-tune — where the audience works.
- **No auto-mutating the user's files, scripts, datasets, or configs.** A linter
  reports; it never rewrites your training code or your data.
- **No W&B / proprietary-API adapters** that require a network client or binary
  format. Reads plain logs only; users can `wandb export` to CSV and feed that.
- **No new rule without a real reproducible failure case.**
- **No Lightning console TTY captures as input** — terminal dumps, not logs. The
  event file Lightning writes beside them *is* supported, since v0.14.
- **No executing a user's checkpoint to inspect it.** Unpickling runs arbitrary
  code; a tool that must run the thing it audits is not a safety tool. *(v0.15)*

---

## The one sentence this roadmap protects

*If trainproof says FAIL, stop the run and investigate.* Everything on the Never
list stays off so that sentence stays true.
