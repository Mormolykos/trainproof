"""Typed coverage states for checks that did not run — the `notchecked` schema.

Every skip in this codebase already carried a reason. A reason is a sentence,
and a sentence cannot be counted, filtered or acted on: "no --module given" and
"every logged loss is NaN or Inf" are the same type to a machine and opposite
to-dos to a human. One says *you did not ask for this check*; the other says
*your data cannot answer it*. Only one of them will ever change by itself.

That distinction is the `notchecked` schema, which came out of a public review
with Boris Teplitsky from the infrastructure-compliance side. This module is
trainproof adopting it.

WHY THE STRINGS AND NOT THE PACKAGE. trainproof declares no runtime
dependencies at all, and adding one to emit a label would be a poor trade. More
importantly, a vocabulary that needs its own library to be spoken is not a
vocabulary. An adopter that carries the states without importing anything is
the stronger evidence that the schema is a primitive rather than one project's
enum — which is exactly the claim that had never been tested.

ADDITIVE, BY DESIGN. `skipped` keeps its shape: group -> human sentence. This
module adds a parallel `coverage` map: group -> state. Nothing that reads the
existing report changes behaviour, and CONTRACTS.md's guarantee holds.
"""
from __future__ import annotations

# The eight states, spelled exactly as the schema spells them.
CHECKED = "CHECKED"
DATA_DEGENERATE = "NOT_CHECKED/DATA_DEGENERATE"
CHECKER_FAILED = "NOT_CHECKED/CHECKER_FAILED"
WAIVED = "NOT_CHECKED/WAIVED"
PREREQUISITE_FAILED = "NOT_CHECKED/PREREQUISITE_FAILED"
OUT_OF_SCOPE_CALLER = "OUT_OF_SCOPE/CALLER"
DATA_TRANSIENT = "OUT_OF_SCOPE/DATA_TRANSIENT"
DATA_PERMANENT = "OUT_OF_SCOPE/DATA_PERMANENT"

STATES = (
    CHECKED, DATA_DEGENERATE, CHECKER_FAILED, WAIVED, PREREQUISITE_FAILED,
    OUT_OF_SCOPE_CALLER, DATA_TRANSIENT, DATA_PERMANENT,
)

# Who has to act, per state. Printed next to the state so a reader does not have
# to memorise the vocabulary to use the report.
OWNER = {
    CHECKED: "",
    DATA_DEGENERATE: "the data",
    CHECKER_FAILED: "trainproof",
    WAIVED: "a named person",
    PREREQUISITE_FAILED: "an upstream check",
    OUT_OF_SCOPE_CALLER: "the caller",
    DATA_TRANSIENT: "the training setup",
    DATA_PERMANENT: "nobody",
}


def classify(reason: str) -> str:
    """Map one skip sentence to its coverage state.

    Deliberately a pure function over the reason text rather than a new argument
    threaded through twenty-one call sites. Both were on the table; this one
    cannot half-land. A missed pattern falls through to DATA_DEGENERATE, which
    is the conservative direction: it claims the data was unusable rather than
    claiming nobody ever needed to look, and the second claim is the one that
    would let a real gap hide.

    `tests/test_coverage.py` asserts every reason string this codebase can
    actually produce is matched by an explicit rule, so the fallthrough is a
    safety net and never the normal path.
    """
    r = reason.lower()

    # The caller did not ask. Nothing is wrong; the flag was absent.
    if r.startswith("no --"):
        return OUT_OF_SCOPE_CALLER

    # A deliberate, reasoned stand-down. Checked before the absence rules
    # because its sentence also begins by describing the data.
    if "the log is reporting an" in r or "aggregate, not the true" in r:
        return WAIVED

    # ABSENT versus PRESENT-AND-UNUSABLE. These read almost identically in
    # English and are opposite to-dos:
    #
    #   "no learning-rate column in the log"  -> the field was never emitted
    #   "no finite gradient norms in the log" -> the field is there, all NaN
    #
    # The first is fixed by logging configuration, the second by the run. The
    # first draft of this function sent both to DATA_TRANSIENT, which would
    # have told someone to change their logger when their gradients were NaN.
    # The word that separates them is "finite".
    if "finite" in r:
        return DATA_DEGENERATE
    if ("column in the log" in r or "in the log" in r or "no loader_time" in r) and r.startswith("no "):
        # Not permanent: the same trainer with a different logging config emits
        # it, so this is a property of THIS run's setup and not of logs as a
        # kind. Boris Teplitsky's correction -- permanence is relative to a
        # target -- is what makes that distinction load-bearing rather than
        # cosmetic, and it is why nothing here claims DATA_PERMANENT.
        return DATA_TRANSIENT

    # Everything else: the signal is present and unusable. Too few points, all
    # NaN, a zero denominator.
    return DATA_DEGENERATE


def summarise(ran: list[str], skipped: dict[str, str]) -> dict:
    """Coverage block for a report: per-group state, plus counts by state.

    `total` is ran + skipped and is stated explicitly. A coverage figure whose
    denominator is implicit is the failure this schema exists to remove -- and
    the schema's own reference implementation shipped that bug in its renderer
    before catching it, which is why it is spelled out here.
    """
    states = {g: CHECKED for g in ran}
    for group, reason in skipped.items():
        states[group] = classify(reason)

    counts: dict[str, int] = {}
    for st in states.values():
        counts[st] = counts.get(st, 0) + 1

    total = len(states)
    return {
        "schema": "notchecked/1",
        "states": states,
        "counts": counts,
        "total": total,
        "checked": counts.get(CHECKED, 0),
        # Never "100% checked" unless it is literally all of them, and never
        # "0%" unless literally none. Rounding a near-miss to a clean number is
        # how a report says "nothing was measured" when something was.
        "checked_pct": _honest_pct(counts.get(CHECKED, 0), total),
    }


def _honest_pct(part: int, whole: int) -> float | None:
    """Percentage that refuses to round into a lie. None when there is nothing
    to divide, because 0/0 is not 0% and reporting it as 0% is an assertion."""
    if whole <= 0:
        return None
    if part == whole:
        return 100.0
    if part == 0:
        return 0.0
    pct = round(100.0 * part / whole, 1)
    if pct >= 100.0:
        return 99.9
    if pct <= 0.0:
        return 0.1
    return pct
