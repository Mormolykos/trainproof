"""The notchecked adoption: every skip this codebase can emit has a typed state.

The point of these tests is not that `classify` returns strings. It is that no
skip reason in the source falls through to the default. A fallthrough is
invisible — the report still renders, the state is still plausible — and it
would quietly file a degenerate-data gap as something nobody has to act on.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from trainproof import coverage
from trainproof.epoch import check_epoch

SRC = Path(__file__).resolve().parent.parent / "src" / "trainproof"

# The real runtime sentences, written out because the f-strings in the source
# cannot be evaluated by reading them. `test_every_source_reason_is_covered`
# below is what keeps this list honest against the code.
REASONS = {
    "no --module given": coverage.OUT_OF_SCOPE_CALLER,
    "no --checkpoint given": coverage.OUT_OF_SCOPE_CALLER,
    "no --output-dir given": coverage.OUT_OF_SCOPE_CALLER,
    "no learning-rate column in the log": coverage.DATA_TRANSIENT,
    "no step_time column in the log": coverage.DATA_TRANSIENT,
    "no loss column in the log": coverage.DATA_TRANSIENT,
    "no loader_time/step_time pair in the log": coverage.DATA_TRANSIENT,
    "no eval_loss in the log - this run has no generalisation signal at all":
        coverage.DATA_TRANSIENT,
    "no finite gradient norms in the log": coverage.DATA_DEGENERATE,
    "fewer than 6 finite loss points": coverage.DATA_DEGENERATE,
    "fewer than 6 finite gradient norms": coverage.DATA_DEGENERATE,
    "fewer than 10 step_time points": coverage.DATA_DEGENERATE,
    "fewer than 3 eval points": coverage.DATA_DEGENERATE,
    "every logged loss is NaN or Inf": coverage.DATA_DEGENERATE,
    "mean loss is not positive - relative variation is undefined": coverage.DATA_DEGENERATE,
    "median gradient norm is zero - no scale to measure a spike against":
        coverage.DATA_DEGENERATE,
    "median early step_time is zero - no baseline to compare against":
        coverage.DATA_DEGENERATE,
    "no positive loss to measure a floor against": coverage.DATA_DEGENERATE,
    "starting loss is not positive - relative improvement is undefined":
        coverage.DATA_DEGENERATE,
    "no records parsed": coverage.DATA_DEGENERATE,
    "all 12 gradient norms are 0.0 but the loss improved from 3.2100 to 0.0170 - "
    "the log is reporting an aggregate, not the true gradient norm": coverage.WAIVED,
}


@pytest.mark.parametrize(("reason", "state"), sorted(REASONS.items()))
def test_reason_maps_to_its_state(reason: str, state: str) -> None:
    assert coverage.classify(reason) == state


def test_absent_and_unusable_are_not_the_same_state():
    """The distinction the first draft got wrong.

    'no learning-rate column' and 'no finite gradient norms' are one word apart
    in English and opposite in what they ask of the reader: change your logging
    config, versus your gradients are NaN. Sending both to DATA_TRANSIENT sent
    someone to edit a logger while their run was broken.
    """
    absent = coverage.classify("no learning-rate column in the log")
    unusable = coverage.classify("no finite gradient norms in the log")
    assert absent == coverage.DATA_TRANSIENT
    assert unusable == coverage.DATA_DEGENERATE
    assert absent != unusable


def test_every_source_reason_is_covered():
    """No `ctx.no(...)` sentence in the source may hit the default branch.

    Scanning the source rather than trusting REASONS above: a new skip added
    next year will fail here rather than silently classify as degenerate data.
    """
    src = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    literals = set(re.findall(r'ctx\.no\(\s*"[^"]+",\s*\n?\s*"([^"]+)"', src))
    literals |= set(re.findall(r'skipped\["[^"]+"\]\s*=\s*"([^"]+)"', src))
    unknown = sorted(lit for lit in literals if lit not in REASONS)
    assert not unknown, (
        "skip reasons in the source that this test does not pin to a state: "
        + repr(unknown)
        + ". Add each to REASONS with the state it deserves; do not let it "
          "fall through to the default."
    )


def test_coverage_rides_along_with_every_verdict():
    for path in ("examples/gallery/healthy/trainer_state.json",
                 "examples/gallery/bad_labels/trainer_state.json",
                 "examples/gallery/fp16_nan/trainer_state.json"):
        report = check_epoch(path)
        cov = report["checks"]["coverage"]
        assert cov["schema"] == "notchecked/1"
        # coverage and verdict are orthogonal: a FAIL run still reports how much
        # of it was actually looked at.
        assert cov["total"] == len(cov["states"])
        assert cov["checked"] == sum(1 for s in cov["states"].values()
                                     if s == coverage.CHECKED)
        assert sum(cov["counts"].values()) == cov["total"]


def test_the_old_fields_are_untouched():
    """Additive means additive. CONTRACTS.md promises `ran` and `skipped`."""
    checks = check_epoch("examples/gallery/healthy/trainer_state.json")["checks"]
    assert isinstance(checks["ran"], list)
    assert isinstance(checks["skipped"], dict)
    assert all(isinstance(v, str) for v in checks["skipped"].values())
    assert set(checks["ran"]) | set(checks["skipped"]) == set(checks["coverage"]["states"])


def test_percentage_never_rounds_into_a_lie():
    """The bug notchecked's own renderer shipped, pinned here so trainproof
    cannot ship it too: 399 of 400 is not 100%, and 1 of 400 is not 0%."""
    assert coverage._honest_pct(400, 400) == 100.0
    assert coverage._honest_pct(399, 400) < 100.0
    assert coverage._honest_pct(0, 400) == 0.0
    assert coverage._honest_pct(1, 400) > 0.0
    # nothing to divide is not zero per cent; it is no answer
    assert coverage._honest_pct(0, 0) is None


def test_every_state_names_who_must_act():
    for state in coverage.STATES:
        assert state in coverage.OWNER
    assert coverage.OWNER[coverage.DATA_PERMANENT] == "nobody"
    assert coverage.OWNER[coverage.CHECKED] == ""
