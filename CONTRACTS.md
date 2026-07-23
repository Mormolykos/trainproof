# trainproof stability contract

What you may depend on, and what may change under you. If trainproof breaks
anything on this page, that is a bug or a documented breaking release — never
a silent drift.

The whole contract exists to protect one sentence:

> **If trainproof says FAIL, you should investigate.**

For that to be worth anything, FAIL has to mean a verdict about *your training
run* — never "trainproof had a problem."

## Exit codes

| Code | Meaning | Examples |
|---|---|---|
| `0` | No FAIL findings. The run was judged and passed, possibly with warnings. | PASS, WARN |
| `1` | **A FAIL verdict about your run.** Stop and investigate. | diverging loss, dead run, NaN, zero LR |
| `2` | **trainproof could not judge.** Says nothing about your run. | file not found, unreadable log, no records parsed, no logs in directory, missing optional dependency, internal error |

WARN exits `0` deliberately. Warnings are advisory, and a CI gate that goes red
on every advisory gets disabled within a week. If you want warnings to fail a
build, check the JSON for finding levels.

Two consequences worth stating plainly. A corrupt or empty log gives you `2`,
not `1` — trainproof will not tell you a run failed when it simply could not
read it. And an unexpected internal error is caught at the top level and
reported as `2`, so a crash can never impersonate a verdict.

`trainproof watch` interrupted with Ctrl-C exits with the last verdict it had
(`0` or `1`) rather than the conventional `130`, because interrupting the
guardian to collect its judgment is the intended workflow.

## JSON output (`--json`)

`schema_version` is **2**. Available on `data`, `tokenizer`, `epoch`, `doctor`,
`compare` and `preflight`. Not on `watch`, which is a streaming command with no
single terminal state.

```json
{
  "schema_version": 2,
  "trainproof_version": "0.10.0",
  "reports": [ { "verdict": "FAIL", "findings": [ ... ] } ],
  "worst_verdict": "FAIL",
  "error": null
}
```

Every report has `verdict` and `findings`. Every finding has `id`, `level`,
`message`, `evidence` and `source`. `source` is `single_run` for rules judging
one log, or `compare` for rules judging it against a baseline — one flat array,
so a consumer never special-cases by command.

When `error` is non-null, trainproof could not judge: `reports` is empty,
`worst_verdict` is `null`, and the exit code is `2`.

**Versioning policy.** A minor release may add new optional keys and new rule
IDs. Renaming a key, removing a key, changing a type, or changing the meaning
of an existing field requires a `schema_version` bump and a CHANGELOG entry.
Parse defensively: unknown keys and unknown rule IDs should be tolerated.

## Rule IDs

Every finding carries a stable `TP-*` ID, documented in [RULES.md](RULES.md).
**Parse IDs, never message text** — messages are prose and may be reworded in
any release.

A test enforces that the set of IDs in the source and the set documented in
RULES.md are identical in both directions, so an undocumented rule and an
orphaned doc entry both fail the build.

Rule IDs are not removed without a major-version note. New IDs may appear in a
minor release, which is why consumers must tolerate unknown ones.

## SARIF (`--sarif PATH`)

SARIF 2.1.0, for GitHub code scanning. Works independently of `--json`.

Levels map as FAIL → `error`, WARN → `warning`, INFO and PASS → `note`.
`tool.driver.rules[]` contains the rules a run actually referenced, and every
result's `ruleIndex` points at its own entry.

Results anchor to the log file at line 1. A training-log finding has no
meaningful line number, and GitHub requires a physical location — a synthetic
step-derived line would point at nothing you could open.

## Verdict stability

Every run in [`examples/gallery/`](examples/gallery/) has a locked snapshot in
`tests/golden/` holding its verdict and the complete set of rule IDs it emits.
A rule that stops firing *and* a rule that starts firing spuriously both fail
the build.

Changing any snapshot requires a CHANGELOG entry. This is what makes "no
verdicts changed" a checkable claim rather than a promise.

Snapshots deliberately exclude evidence numbers (which would couple the suite
to float formatting) and exit codes (asserted separately, so a deliberate
exit-code change cannot hide a verdict change).

## What this contract does not cover

Console text layout, the HTML report, rule *thresholds* (documented in
RULES.md, tunable across minor releases as evidence accumulates), and anything
under `examples/` other than the gallery snapshots.

Pre-1.0, a minor release may break a documented contract **only** with an
explicit CHANGELOG entry marked breaking. At 1.0 the contract freezes and
breaking it requires a major version.

## Known gaps

Log-format detection still runs in two places — the adapters, and a filename
filter in `doctor`'s directory walk. They agree today, but a future adapter
must update both. The plug-in registry that removes this hazard is deferred
until there is a second adapter to justify it.

`TP-TOK-SPM-MISSING` still reports a missing `sentencepiece` install as a FAIL
verdict rather than exit `2`, unlike the other missing-dependency paths. It is
in the speech pack and will move to the `2` convention in a later release.
