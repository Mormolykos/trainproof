"""Typed coverage for trainproof's skipped checks.

Since 0.12.0 every single-run report has carried `checks`: `ran`, and `skipped`
as a map of group → reason. The denominator existed and was machine-readable.
What did not exist was a **type** on the gap.

Two real skip reasons from the same check group, both landing in `skipped`:

    grad-spike: "no finite gradient norms in the log"
    grad-spike: "median gradient norm is zero - no scale to measure a spike against"

The first is a claim never intended — that log never carried the signal. The
second is one intended and impossible — the signal is there and degenerate. A
human tells them apart instantly. A CI job counting `skipped` cannot. Same
group, same field, opposite remediations.

The states, and who owns the fix, come from `notchecked`. The codes below are
trainproof's vocabulary within it, declared once so a reason cannot be free text
and cannot mean two things.

**The code is chosen by the check, at the moment it decides not to run.** It is
never reconstructed afterwards by matching the prose back to a category — that
reconstruction is the bug this replaces.

`checks.skipped` is unchanged and still emitted. `checks.coverage` is added
beside it. Per CONTRACTS.md a minor release may add optional keys, so
`schema_version` stays at 3.
"""

from __future__ import annotations

from typing import Any

from notchecked import Coverage, Reason, Record, Vocabulary

# --------------------------------------------------------------------------
# trainproof's reason codes.
#
# CALLER      the invocation did not ask for it
# DATA_TRANSIENT  this log does not carry the column; another log from the same
#                 trainer might, so it moves when the producer changes. Nothing
#                 in a training log is OUT_OF_SCOPE/DATA_PERMANENT -- that state
#                 is for artifacts that can never evidence a control at all, and
#                 it stays unused here. A domain need not use all six.
# DATA_DEGENERATE the signal is present and unusable: identically zero, non-
#                 finite, or too few points to measure against
# CHECKER_FAILED  trainproof itself could not read the target
# --------------------------------------------------------------------------

NOT_REQUESTED = Reason(
    "not_requested", Coverage.OUT_OF_SCOPE_CALLER,
    "the invocation did not supply what this check needs")

NO_COLUMN = Reason(
    "no_column", Coverage.OUT_OF_SCOPE_DATA_TRANSIENT,
    "this log does not carry the column the check reads")

NO_FINITE_VALUES = Reason(
    "no_finite_values", Coverage.NOT_CHECKED_DATA_DEGENERATE,
    "the column is present and every value is NaN or Inf")

NO_SCALE = Reason(
    "no_scale", Coverage.NOT_CHECKED_DATA_DEGENERATE,
    "the reference statistic is zero or non-positive, so there is no scale to "
    "measure against")

TOO_FEW_POINTS = Reason(
    "too_few_points", Coverage.NOT_CHECKED_DATA_DEGENERATE,
    "the signal is present but there are not enough points to judge it")

STOOD_DOWN = Reason(
    "stood_down", Coverage.NOT_CHECKED_DATA_DEGENERATE,
    "the check would fire but a stronger fact contradicts it, so it declines "
    "rather than reporting a finding it cannot defend")

UNREADABLE = Reason(
    "unreadable", Coverage.NOT_CHECKED_CHECKER_FAILED,
    "trainproof could not read or parse the target")

VOCABULARY = Vocabulary([
    NOT_REQUESTED, NO_COLUMN, NO_FINITE_VALUES, NO_SCALE, TOO_FEW_POINTS,
    STOOD_DOWN, UNREADABLE,
])


def coverage_records(ran: list[str], skipped: dict[str, str],
                     codes: dict[str, str]) -> list[dict[str, Any]]:
    """Build the typed rows for one report.

    `ran` and `skipped` are the existing 0.12.0 structures. `codes` maps a
    skipped group to the code its check chose. A skipped group with no code is a
    bug in trainproof, not in the log, so it is recorded as CHECKER_FAILED
    rather than guessed at — guessing is how a reconstruction quietly becomes
    wrong.
    """
    records: list[Record] = [
        Record(group, Coverage.CHECKED) for group in sorted(ran)
    ]
    for group in sorted(skipped):
        code = codes.get(group)
        if code is None:
            records.append(Record(
                group, Coverage.NOT_CHECKED_CHECKER_FAILED,
                reason=UNREADABLE.code,
                detail=f"skip recorded without a coverage code: {skipped[group]}"))
            continue
        reason = VOCABULARY.resolve(code, _state_for(code))
        records.append(Record(
            group, reason.state, reason=reason.code, detail=skipped[group]))
    return [r.to_dict() for r in records]


def _state_for(code: str) -> Coverage:
    return VOCABULARY.codes()[code].state
