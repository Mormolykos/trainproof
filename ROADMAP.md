# trainproof — Roadmap

The official product roadmap. **Every future feature decision is evaluated
against this document.** If a proposed feature is not on the roadmap and is not
justified by the decision rule below, it does not ship.

Updated 2026-07-20, after shipping v0.8 and producing the overfit evidence run.

---

## Decision rule (apply to every proposed feature)

A feature ships only if all of these hold:

1. **It is deterministic.** A rule fires or it does not; no model judges the run,
   no probability, no score.
2. **It is backed by a real, reproducible failure case** committed to the
   gallery. No evidence, no rule.
3. **It is dogfoodable on the author's own hardware** (single GPU / small
   fine-tune). Untested rules for hardware we do not have are forbidden.
4. **It cites its evidence.** Every finding shows the exact numbers that
   triggered it and carries a stable `TP-*` id.
5. **It answers "would someone lose GPU hours (or trust) without it?"** If not,
   it is scope creep.

---

## Current state — v0.8 (shipped)

The lifecycle is complete and the trust layer is in place:

- **`doctor`** (flagship, v0.6) — zero-config autopsy of a file or a whole
  directory; auto-discovers logs, triage-sorted summary, per-log findings.
- **preflight** — dataset + tokenizer checks before a GPU-second is billed.
- **Live guardian** — HuggingFace `TrainerCallback` with step-time telemetry
  (v0.7) and opt-in auto-abort; `watch` with `--stall-timeout`.
- **coroner** (`epoch`) — divergence / dead-run / NaN / spike / LR verdicts
  from a finished log.
- **compare** — N-way ratio rules against a known-good baseline.
- **Trust (v0.8)** — canonical exact-match column mapping (no substring
  guessing; `eval_loss` can never be judged as loss), stable rule IDs +
  `RULES.md`, honest PASS (lists ran vs skipped checks), `--json` output for
  CI and AI coding agents.
- Log adapters: HuggingFace `trainer_state.json`, Coqui, generic JSONL/CSV.
- Validation: 5-config × 3-seed fault-injection gallery (`examples/gallery/`,
  `EVIDENCE_MATRIX.md`), 52 tests.

### Documented limitation (not hidden)

A shuffled-labels run reduced its loss 62% by learning the marginal token
distribution — from its own loss curve, indistinguishable from real learning.
No single-run, loss-only rule can catch this class; its signature is relative,
which is why `compare` exists.

---

## v0.9 — the eval-aware release (detection)

One theme: teach trainproof to read the eval curve.

- **Fix (first): the HF training-summary leak.** HuggingFace `trainer_state.json`
  ends with a summary entry whose `train_loss` is the run *average*, not a
  per-step loss; it was leaking into the per-step loss series and could fire a
  false `TP-DIVERGE` on any steeply-converging run. The adapter now drops that
  entry. Regression-locked against the gallery.
- **`TP-OVERFIT` — deterministic overfitting detection.** eval_loss rising past
  a ratio of its own minimum while train_loss keeps falling. Grounded in a real
  overfit run: eval_loss min 1.25 @step30 → 3.76 @step300 (3.0x) while train
  fell 1.38 → 0.03. Fires WARN (early stopping is the user's choice; it flags
  that the best checkpoint was earlier). The failure practitioners most fear,
  and perfectly deterministic.
- **Evidence** — a real overfit run added to the gallery
  (`examples/gallery/overfit/`), with eval logging. No rule without a run it
  catches.

## v0.10 — configuration + CI integration

The adoption cluster, built on the v0.8 `--json` report (no new detection):

- **`[tool.trainproof]` config in pyproject.toml** — select/ignore rules by id,
  per-project threshold overrides. CI teams will not adopt a linter they cannot
  tune per-repo.
- **SARIF output** (priority) — GitHub renders findings inline on pull
  requests. Real distribution.
- **JUnit XML output** — nice-to-have; lands after SARIF.

## v1.0 — the stability contract (promises, not features)

1.0 is the moment the interface becomes a promise:

- Freeze: rule IDs (already permanent), JSON `schema_version` policy,
  documented exit codes, SemVer + CHANGELOG, complete `RULES.md`.
- **Publish the fault-injection gallery as the permanent regression suite** —
  every future rule tested against every archived pathology.
- **One earned feature: the TensorBoard events adapter** (optional `[tb]`
  extra), landing together with an **adapter registry** so future adapters are
  plug-ins, not core surgery. This unlocks the Lightning/fairseq audience.
- v1.0 is the Show HN moment — a year of evidence behind it.

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
  Testable on one GPU, in scope.
- **Chat-template / attention-mask validation** for preflight — *deliberately
  cut from v0.5, not abandoned.* Deferred because every model family
  (ChatML / Alpaca / Llama3 / Qwen / Gemma / …) has different conventions;
  doing it right is weeks of work, and rushing it would violate the
  "universally-true, no guessing" rule. Revisit only when it can be made
  deterministic per-family with evidence.
- **Consuming external evaluation results** to correlate "run looked healthy"
  with "model actually improved" — integration surface only; trainproof never
  runs benchmarks itself.

Market signal (kept from prior analysis): a text-level dataset linter
(Parallelogram) already has community traction — proof of demand for the
preflight lane; trainproof differentiates by going deeper (tensor/tokenizer
level, full lifecycle) rather than text-only.

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
- **No new rule without a real reproducible failure case in the gallery.**
- **No Lightning console TTY captures as input** — terminal dumps, not logs.

---

## The one sentence this roadmap protects

*If trainproof says FAIL, stop the run and investigate.* Everything on the Never
list stays off so that sentence stays true.
