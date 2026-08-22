# Changelog

All notable changes to trainproof are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning follows
[SemVer](https://semver.org/).

## 0.18.1 - unreleased

### Fixed

- **The core no longer requires numpy, soundfile or ttsproof.** All three were
  hard runtime dependencies, used by exactly one module - `speech/data.py` - and
  `cli.py` imported it at module scope. So `pip install trainproof` pulled an
  audio stack, and `trainproof epoch` on a text log would not start without
  libsndfile, the C library behind soundfile. For a tool whose tfevents reader is
  written from the wire format specifically to avoid TensorFlow, that was the
  wrong shape.

  `dependencies` is now empty. The three moved to a `trainproof[speech]` extra,
  `check_data` resolves lazily through a module `__getattr__`, and `cli.py`
  imports it inside the `data` subcommand. `check_tokenizer` is stdlib-only and
  is unaffected.

  A core install that lacks the extra and calls a speech check gets a legible
  error naming `pip install 'trainproof[speech]'`, not a bare ImportError.

### Added

- `tests/test_core_install.py`: blocks numpy, soundfile and ttsproof from the
  import system in a fresh interpreter and asserts that twelve core modules and
  the `epoch` CLI still work. A stray top-level import cannot quietly
  reintroduce the coupling.

## [0.18.0] — 2026-08-09 — the objective release

Every rule shipped so far reads the training run: the loss curve, the gradient norms, the learning rate, the timing. This release adds the first rules that read the **objective itself** — the output layer, the ignore sentinel, and which classes actually reach the loss as positive targets. They are deliberately independent of the curve, because the failure that motivated them is invisible to it.

The failure was mine. A VALL-E-X-derived TTS model set its end-of-sequence id and its cross-entropy `ignore_index` to the same integer, `NUM_AUDIO_TOKENS = 1024`, against a 1025-class output layer. `ignore_index` therefore discarded every end-of-sequence target before the loss was computed, and the model was never once shown what "stop" meant. It trained for fifteen epochs on a healthy-looking curve and produced a model that could not stop generating. A minimal 40-line reproduction — same data, same seed, same everything, one integer changed — ends at **0.0035** final loss with the collision and **0.0034** without it. The broken arm assigns the stop token a probability of 0.000001 and never terminates; the fixed arm assigns 0.996565 and terminates on every sample. **No loss-shaped check in this library can separate those two runs, and none ever will.** The bug is decidable at step 0 from two integers and undecidable from any number of steps afterwards.

### Added
- `TP-OBJ-IGNORE-INDEX-COLLISION` (FAIL) — `ignore_index` is a valid class in the output layer (`0 <= ignore_index < num_classes`). Every target carrying that id is dropped from the loss and can never be learned.
- `TP-OBJ-DEAD-CLASS` (FAIL) — a class exists in the output layer but never once appears as a positive target, while coverage of the other classes is broad. Fires on the motivating bug with no knowledge of TTS, EOS or the model family: *"class 1024 is in your output layer but never appears as a training target."*
- `TP-OBJ-TARGET-OUT-OF-RANGE` (FAIL) — targets contain ids the output layer cannot represent.
- `TP-OBJ-IGNORE-INDEX-OK` (INFO) — the sentinel sits exactly one past the last class, therefore outside the output layer and safe. Reported rather than stayed silent on, because the same integer is fatal one class earlier — the NAR stage of the same model uses `ignore_index = 1024` against **1024** classes and is correct for that reason alone.
- `TP-OBJ-COVERAGE-INSUFFICIENT` (INFO) — too few distinct classes observed to judge dead classes. Absence here means small sample, not bug.
- `TP-OBJ-DEAD-CLASS-OK` (PASS).
- `TrainproofCallback` now runs the objective checks **once, in `on_train_begin`, before step 1**, inferring the output-layer width and the sentinel from the model and sampling the first batches of labels from the dataloader. No user action, no new call site. `objective_check=False` disables it; `num_classes` and `ignore_index` can be passed explicitly when inference cannot see them. Under `stop_on_fail` a broken objective aborts the run before a single GPU-second is spent, which is the whole point.

### Notes
- `TP-OBJ-DEAD-CLASS` fires only when coverage is already broad (`DEAD_CLASS_MIN_COVERAGE`) **and** few classes are missing (`DEAD_CLASS_MAX_REPORTED`). One unseen class out of 1025 with the other 1024 present is a structural exclusion; nine hundred unseen classes is a small sample. A check that shouts on every short run gets switched off, and then it detects nothing.
- The regression fixtures are the real thing rather than invented cases: the AR stage that carried the bug and the NAR stage of the same model that did not, one `+ 1` apart in output width.
- No existing rule, threshold or verdict changed. 245 → 258 tests.

## [0.17.0] — 2026-08-02 — the lint gate, and one log that used to vanish

Lint had never been part of the gate. There was no `[tool.ruff]` section, ruff was not a dependency, and running it reported 103 findings against inherited defaults. A repository that ships a linter should not fail its own.

### Fixed
- **A log that could not be parsed disappeared entirely.** `doctor` walks a directory twice: once to discover candidates, once to judge them. The second pass already reported what it could not read — *"could not be parsed and were NOT judged"*. The first pass did not: a file that raised during discovery was swallowed by `except Exception: pass`, never became a candidate, and so never reached that note. The result was a file plainly visible on disk, absent from the report, and indistinguishable from one that passed — the exact failure `NOT-CHECKED` exists to prevent, sitting one loop earlier than anyone had looked. Discovery failures now feed the same note. Covered by a regression test that was verified to fail without the fix.
- Two other broad handlers were narrowed rather than removed. A non-JSON line in a JSONL log and a `{`-prefixed line that is not valid JSON are both expected, and skipping them is correct — but they now catch `json.JSONDecodeError` specifically, so a genuine fault in the surrounding code can no longer disguise itself as an unparseable line.

### Added
- **`[tool.ruff]` in `pyproject.toml`, and a lint gate in the release ritual.** The ruleset is chosen, not inherited: `E`, `F`, `I`, `B`, plus `RUF013` (implicit `Optional` — a type hint that lies), `RUF059`, and `S110` (try-except-pass). Every exclusion carries its reason in the file.
- `dev` extra (`pytest`, `ruff`) so the gate is installable rather than assumed.

### Notes on what was deliberately *not* adopted
- **`BLE001` (blind except).** trainproof catches broad exceptions on purpose when parsing logs it did not write and probing subprocesses that can die in ways Python cannot describe. A narrow `except` there would let an unforeseen parser error escape as a traceback instead of exit code 2, "cannot judge". `S110` is enforced instead: catching broadly is fine, catching and *passing* is not.
- **`PLW1510` (subprocess without `check`).** Every subprocess call here inspects `returncode` itself and turns it into a finding; `check=True` would raise instead, which is the opposite of the required behaviour.
- **`E501` (line length).** Most long lines are evidence and message strings — the text trainproof prints, asserted in tests and encoded byte-for-byte in the golden snapshots. Reflowing them to satisfy a column limit would risk changing the tool's output to satisfy a ruler.

### Verification
Zero ruff findings. 228 → 230 tests. All 38 golden snapshots byte-identical, and `scripts/regenerate_evidence.py --check` exits 0 — a cleanup that changes a verdict is not a cleanup.

## [0.16.0] — 2026-08-02 — the rule registry (no behaviour change)

Every single-run rule lived inside one function, `check_records()`. Adding a rule meant editing the body of several hundred lines that also computed the statistics every other rule depended on, so each new check raised the risk to the checks already there. That was the project's main structural bottleneck and it was blocking the checkpoint work planned next.

**No rule, threshold, verdict or output changed.** That claim is not asserted, it is enforced: all 38 golden snapshots are byte-identical to 0.15.0, `scripts/regenerate_evidence.py --check` exits 0, and the same 228 tests pass unmodified. A refactor of judging logic that cannot prove it changed nothing is indistinguishable from a silent regression, which is why the snapshots exist.

### Changed
- Each single-run rule is now a standalone function taking a `CheckContext` and returning findings. The context computes the shared series once — losses, gradient norms, learning rates, step times, eval losses, loader fractions — so no rule recomputes what another already derived, and a rule can be read, tested and reasoned about without reading the ones around it.
- `check_records()` is now composition rather than implementation: it builds the context and runs the registry in order. Evaluation order is preserved exactly, because the goldens encode the sequence findings appear in.
- The `checks.ran` / `checks.skipped` bookkeeping moved onto the context. Every skip reason string is unchanged, character for character, since those strings are asserted in tests and appear in golden output.

### Notes
- 84 rule IDs and 228 tests, both unchanged. `schema_version` remains 3.
- Deliberately shipped alone. Mixing a behaviour-preserving refactor with a new check would destroy the only evidence that the refactor preserved behaviour.

## [0.15.0] — 2026-08-01 — the before-the-GPU release

Every check in trainproof until now reads a log, which means the run already started and the hours are already spent. The failures that cost the most never reach a log at all: a stack that will not import, a checkpoint that cannot be deserialised, a first batch that exhausts system RAM and freezes the desktop. All three happened to the author in a single day on a run that never logged one step. This release checks them before the GPU is touched.

### Added
- **`trainproof env`**: environment preflight. Judges whether a machine can start a run, not whether a run went well. All checks are stdlib-only — no torch, no ML framework, no GPU, no network.
- **Import checks** (`TP-ENV-IMPORT-OK` / `-FAIL` / `-CRASH` / `-TIMEOUT`): imports the training entrypoint **in a subprocess** and reports the exact exception, message and raising file. Out-of-process is not a detail: the failures here are violent — a segfaulting extension, a CUDA abort, a library calling `os._exit` during import — and in-process any of them would kill trainproof and tell the user nothing. A crash with no Python exception is reported as `TP-ENV-IMPORT-CRASH` (a native fault, not an `ImportError`) rather than misdiagnosed.
- **`--cwd`**: probes from the directory training actually launches in. Editable installs and source checkouts resolve relative to the working directory, so probing from elsewhere reports "No module named X" for a package that imports perfectly — a false FAIL that blames the environment for the linter's mistake. A missing `--cwd` is `TP-ENV-CWD-MISSING` (NOT-CHECKED) and no subprocess runs.
- **Checkpoint checks** (`TP-ENV-CKPT-*`): a `.pt`/`.pth`/`.ckpt` is inspected **without deserialising it**. `torch.load` executes arbitrary code by design — the reason torch 2.6 flipped `weights_only` to True — so a linter that must run the file it inspects is not a safety tool. Checkpoints are read as the ZIP archives they are: entry table, storage count and CRC, all from the archive directory. Distinguishes missing, zero-byte, truncated, CRC-corrupt, legacy pre-1.6 pickle (reported NOT-CHECKED, because refusing to unpickle is correct behaviour rather than an error) and complete.
- **Memory checks** (`TP-ENV-MEM-*`): available versus required system RAM, with a headroom threshold. On Windows the driver spills to system RAM instead of raising a clean OOM, so exhaustion freezes the desktop rather than failing the run — this is the check that would have prevented two hard resets. Where memory cannot be measured it reports `TP-ENV-MEM-UNKNOWN` (NOT-CHECKED); an unmeasurable machine is never reported as a machine with no memory.
- **Disk checks** (`TP-ENV-DISK-*`): free space against declared checkpoint size times checkpoints kept.

### Fixed
- `TP-ENV-CKPT-TRUNCATED` now fires on an archive whose ZIP header is present but whose central directory is missing. A save killed mid-write leaves exactly that, and `zipfile.is_zipfile()` returns False for it, so the most common real checkpoint failure was being reported as "not a checkpoint at all" — the difference between resuming from the previous checkpoint and hunting for a file that was never written. Found by a test, before release.

### Notes
- 22 new rule IDs, all under the `TP-ENV-` prefix: 62 → 84. `schema_version` remains 3; no verdict, threshold or existing rule changed, and no consumer contract is broken.
- 210 → 228 tests.

## [0.14.0] — 2026-08-01 — the third-framework release

Every rule shipped so far had only ever been tested against HuggingFace `trainer_state.json` files. PyTorch Lightning, Fish Speech and most research code write their metrics to TensorBoard event files and nowhere else, so those runs were invisible: a Lightning run could overfit for three hours and trainproof had nothing to read. This release adds the missing reader and validates the rules against real runs from two more frameworks.

### Added
- **`tfevents` format**: a TensorBoard event-file reader written from the wire format — TFRecord framing plus the `Event`/`Summary`/`TensorProto` protobuf fields it needs. It imports no tensorflow, no tensorboard, no protobuf, no torch, and no numpy; trainproof's dependency-free guarantee is unchanged. Validated byte-exact against `tensorboard.backend.event_processing.EventAccumulator` on a real 2049-step Lightning run: all 13 tags, all point counts, all values.
- Scalars logged as rank-0 tensors are decoded as well as `simple_value`. Lightning uses the former, so a reader handling only the latter sees an empty run.
- Tag normalisation across frameworks: `train/loss`, `TrainIterStats/loss` and `training/loss` all resolve to `loss`; `val/loss` and `EvalStats/avg_loss` to `eval_loss`; `lr-AdamW/pg1` and `TrainIterStats/current_lr` to `lr`.
- When several tags claim one column — Coqui logs both `TrainIterStats/loss` per step and `TrainEpochStats/avg_loss` per epoch — the denser series wins, ties broken alphabetically so the choice is deterministic.
- `--format tfevents` on `epoch`, `doctor`, `compare` and `watch`; `auto` detects event files by name. Directory arguments now discover event files, and a directory of shards is merged into one series.
- Truncated event files — the normal state of a killed run — are read up to the cut instead of raising. A killed run is the run most in need of judging.
- `evidence/`: the real logs behind the claims above. A 125,000-step Coqui XTTS fine-tune (text log and event file from the same run) and a 2049-step Fish Speech LoRA fine-tune, both on an RTX 5080.

### Fixed
- **`TP-ZERO-GRAD` false positive.** The rule fired whenever every finite gradient norm was exactly `0.0`, and reported a severed backward graph. Coqui writes `TrainEpochStats/avg_grad_norm` as `0.0` when clipping is off, so a healthy 125k-step XTTS run whose loss reached `0.017` was reported **FAIL**. A run cannot both learn and receive no gradient: the check now stands down when the loss improved by more than `MIN_LOSS_IMPROVEMENT`, recording the reason as a skip, and stays armed when the loss is stuck. Found by running the shipped rules against a real run — not by a test.

### Notes
- `EVIDENCE_MATRIX.md` gains a cross-framework section, derived like the rest of the file: verdicts are computed from the evidence logs at generation time, including a check that the two independent readers of the same XTTS run agree.

## [0.13.0] — 2026-07-31 — the honest-silence release

A log that carries a loss column but executes zero check groups returned PASS with exit 0. "Checked and clean" and "nothing could be checked" shared one verdict. This release separates them so CI can tell them apart.

### Added
- **`TP-NOT-CHECKED` (NOT-CHECKED)**: New verdict and rule ID. Emitted when zero check groups executed, which requires fewer than five loss points AND a non-positive mean loss, with no other judgeable column (e.g. a short run whose loss is all zeros, the sub-threshold companion to TP-ZERO-LOSS). Its message states how many groups were considered and the skip reason for each, derived exactly from the `checks` structure added in v0.12.0.
- **Severity and exit codes are now two separate axes.** Severity ordering (for `worst_verdict` and triage sort) is FAIL > WARN > NOT-CHECKED > PASS. This means WARN outranks NOT-CHECKED in severity while exiting 0. Exit code is 1 if any FAIL; else 2 if anything could not be judged (NOT-CHECKED, TP-NO-RECORDS, TP-CMP-ERROR); else 0.
- `schema_version` bumped to 3 to reflect the new NOT-CHECKED verdict enum member and the separation of severity and exit codes. Documented in `CONTRACTS.md`.

### Changed
- `compare` against a NOT-CHECKED run (either run or baseline) yields `TP-CMP-UNCOMPARABLE`, reusing the existing uncomparable path rather than failing with `TP-NO-LOSS`.
- `doctor` reports group NOT-CHECKED logs in their triage summary strictly between WARN and PASS.

## [0.12.0] — 2026-07-30 — the honest-verdict release

No new diagnostic idea ships here. Every change closes a hole in a check that
already existed: two silent false negatives, one misdiagnosis, and one claim of
coverage the tool had not delivered.

The audit that found them started from a simple question — what happens to a run
whose loss is *exactly* zero? Every loss-shape check is guarded by `> 0` to avoid
dividing by zero, so the answer was: all of them skip, the verdict stays PASS,
and `TP-PASS` goes on to name loss-shape, divergence and dead-run as checks it
had run. A run where every label was masked to `-100` — which learns nothing at
all — passed clean and was told which three checks had cleared it.

**All 38 golden snapshots are byte-identical and `EVIDENCE_MATRIX.md` is
unchanged.** No existing verdict moved: no shipped fixture has an all-zero
series, and the lowest loss anywhere in the gallery is 0.026.

### Fixed
- **`TP-ZERO-LOSS`** (FAIL): every finite loss is exactly 0.0. Cross-entropy
  returns 0.0 when every target label is masked to `-100`, so the finding names
  the collator's prompt masking and context-window truncation as the places to
  look. Detected by exact equality, never a threshold — a very small loss is real
  convergence, which is a different condition.
- **`TP-ZERO-GRAD`** (FAIL): every finite gradient norm is exactly 0.0. This case
  was already *caught*, as `TP-DEAD-RUN` — "loss never improved" — which sends
  you hunting your data and learning rate for a day. It now names the actual
  cause: a severed backward graph, usually reentrant gradient checkpointing over
  frozen input embeddings with PEFT adapters deeper in the block.
- **A single zero loss no longer disables divergence detection.** `TP-DIVERGE`
  took its floor from `min()` over all losses, so one fully-masked batch anywhere
  in a run drove `min_loss` to zero and the `min_loss > 0` guard switched the
  check off for the *entire* run. The floor is now taken over nonzero losses.
- **`TP-PASS` now reports only checks that actually executed.** The group list was
  hardcoded, and grad-norm was registered on column *availability* while the
  spike test itself sat behind `median_gn > 0`. Groups are now registered at the
  point each check runs, every skip carries a reason, and reports expose this as
  structured data under a new `checks` key. This is the deepest fix in the patch
  and worth having with no new rules attached.
- **`TP-CMP-UNCOMPARABLE`** (FAIL): `compare` no longer reports a zero-loss run as
  favorable. Such a run has a loss floor of 0.0, which beats any baseline, so
  neither `TP-FLOOR-RATIO` nor `TP-END-RATIO` can fire; `extract_metrics`
  compounded it by substituting `0.0` for an undefined improvement, which kept
  `TP-NEG-IMPROVE` quiet too. Against a baseline that had not itself improved,
  nothing fired at all and the report read `TP-CMP-PASS`, "compares favorably".
  Either side being degenerate now refuses the comparison instead.
- **`TrainproofCallback` can finally see eval loss.** `_convert_state_to_records`
  required `"loss"` in every entry, but HF logs eval results as *separate* entries
  carrying `eval_loss` and no `loss` — and `eval_loss` was not copied even when it
  shared an entry. `TP-OVERFIT` needs four eval points, so it was structurally
  unreachable from the callback while `trainproof epoch` saw it correctly on the
  same data. **This is the only change here that can alter a live verdict:** an
  overfitting run that previously produced nothing from the callback will now
  warn. A parity test asserts the callback and the file path emit identical
  findings on `examples/gallery/overfit/`.
- **`TP-CMP-ERROR`** (WARN): `doctor --baseline` swallowed a failed comparison with
  `except Exception: pass` and printed the single-run verdict with no sign that
  the comparison the user asked for never happened. `doctor` also now reports
  logs it could not parse instead of dropping them from the report, where a
  vanished log is indistinguishable from a passing one.
- **Six drifted thresholds in `RULES.md`.** The published documentation disagreed
  with the code on `TP-FLAT` (0.005 → 0.001), `TP-ZERO-LR` (100% → >=99%),
  `TP-ZERO-LR-PARTIAL` (>20% → >10%), `TP-STEP-CLIFF` (1.5x of the run average →
  3x the median of the first half), `TP-LOADER-BOUND` (>20% → >50%) and
  `TP-IMPROVE-DEFICIT` (<50% → <25%). `TP-GPU-UTIL` said average where the code
  takes a median, and `TP-NO-LOSS` documented only its `compare` meaning. The
  rule-ID test compares ID *sets*, so threshold prose was never covered by it.

### Changed
- `parse_log_with_format_info` returns a fourth element: format-level metadata.
  For HF logs this preserves `max_steps`, `num_train_epochs`, `best_metric`,
  `best_global_step`, `best_model_checkpoint` and the interval settings — all of
  which the adapter used to discard, leaving no way to check what a run was
  configured to do against what it did. **Nothing reads it yet**; it is inert on
  purpose so this patch cannot change a verdict. `parse_log_with_format` is
  unchanged, and this function is not part of the stability contract.
- Two tests were corrected because their expectations were wrong, not because new
  code failed them. `test_convert_state_to_records` asserted that `eval_loss` was
  stripped and eval-only entries dropped — it locked in the bug above.
  `test_honest_tp_pass` matched the hardcoded group string literally; it now
  asserts against the `checks` key, since `CONTRACTS.md` states message text is
  prose and may be reworded in any release.

## [0.11.0] — 2026-07-30 — the evidence release

No rule and no threshold changed. Every verdict that existed in v0.10 is
byte-identical in `tests/golden/`. What changed is how much evidence ships, and
whether the documentation describing it can drift away from the data.

The 5x3 seed study had existed since v0.3 but only one seed per configuration was
ever committed, so "3 seeds out of 3" was a claim a reader could not check. All
eighteen runs now ship and all eighteen are judged.

### Added
- **All three seeds of every gallery configuration** (18 runs: 6 configs x seeds
  42/43/44). Seed 42 stays at each config root; 43 and 44 are nested beside it, so
  every previously documented path still resolves. `run_meta.json` now records the
  seed.
- **`overfit` at seeds 43 and 44**, completing the only configuration that had a
  single seed. `TP-OVERFIT` fires in all three.
- **21 new golden snapshots** covering every seeded run and its comparison against
  the same-seed baseline, plus a cross-seed sanity check (healthy judged against
  healthy from a *different* seed, which must stay clean — it does, 3/3).
- **`scripts/regenerate_evidence.py`**: `EVIDENCE_MATRIX.md` is now generated from
  the logs, stamped with the generating version, and carries the rule IDs that
  fired in every cell rather than a bare verdict. `--check` fails if it is stale,
  and `test_evidence_matrix_is_current` runs that check in the suite. The matrix
  had been publishing v0.3-dev verdicts against a v0.10 engine.
- **`python -m trainproof`** now works. It previously failed with "'trainproof' is
  a package and cannot be directly executed", which reads like a broken install.
  Both entry points call the same `main()`, and a parametrised contract test
  asserts they agree on exit codes 0, 1 and 2.
- The gallery guard now enforces coverage one level down: a seed log that no test
  judges fails the build, exactly as an unjudged config folder always has.
- **`examples/real_world/xtts_diverged/`**: a 9.8-hour Coqui XTTS v2 fine-tune that
  diverged on its own. Every other log in this repo is a fault injected on purpose,
  which is the right way to test a rule and a weak way to show the tool matters.
  This one broke by itself, and it is the repo's only Coqui-format fixture, so it
  regression-tests that adapter against a real 580KB log. Its verdict is locked in
  `tests/golden/` like everything else.

  The log's own bookkeeping corroborates the verdict independently: trainproof puts
  the loss minimum at step 48,350, Coqui's last `BEST MODEL` line is
  `best_model_49880.pth`, and the last checkpoint written is `checkpoint_70000.pth`.
  Roughly 23,000 steps ran after the best weights already existed.

  One local filesystem prefix is replaced with `<TTS>` in 14 lines. No number,
  timestamp or step was altered, and the verdict and evidence strings are
  byte-identical before and after; the unredacted original is retained privately as
  the provenance record. Disclosed rather than done quietly, because a modified
  evidence file that doesn't say so is worth nothing.

### Changed
- **`epoch --html [PATH]` is opt-in.** The self-contained HTML report used to be
  written into the caller's working directory on every invocation, unasked — it
  littered this project's own test runs, which is how it was noticed. Bare `--html`
  keeps the old `trainproof_report.html` filename. Not a contract change:
  CONTRACTS.md explicitly excludes the HTML report.
- **`compare` now labels rows with enough path to tell them apart.** Every
  gallery log is named `trainer_state.json`, so both rows rendered identically and
  reversing the baseline and run arguments produced a plausible, fully inverted
  verdict with nothing to flag it. The shortest distinguishing path suffix is used,
  deepening automatically when needed. Console layout is outside the contract.

### Fixed
- Documentation corrected against the data it describes: the gallery is six
  configurations and eighteen runs, not "five runs, four broken"; the `overfit`
  configuration was missing from both README tables; and the `bad_labels` example
  now quotes start and end in one measurement system (15.3 → 5.75, trainproof's
  windowed values) instead of mixing the raw first logged loss with the tool's end.

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
