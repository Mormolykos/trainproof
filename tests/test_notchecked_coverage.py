"""checks.coverage - the typed form of checks.skipped.

Added alongside `ran`/`skipped`, never replacing them, so an existing consumer
sees no change. Per CONTRACTS.md a minor release may add optional keys, so
`schema_version` stays at 3.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from notchecked import Coverage
from trainproof import coverage as cov
from trainproof.epoch import CHECK_GROUPS, check_epoch

FIXTURES = Path(__file__).parent / "fixtures"


def coverage_of(path):
    return check_epoch(str(path))["checks"]


def test_every_check_group_gets_exactly_one_row():
    c = coverage_of(FIXTURES / "healthy.jsonl")
    targets = [r["target"] for r in c["coverage"]]
    assert sorted(targets) == sorted(CHECK_GROUPS)
    assert len(targets) == len(set(targets)), "one row per group, never two"


def test_ran_and_skipped_agree_with_the_typed_rows():
    """The typed rows are derived from the same decisions, so they cannot
    disagree with the 0.12.0 structures they sit beside."""
    c = coverage_of(FIXTURES / "healthy.jsonl")
    checked = {r["target"] for r in c["coverage"] if r["coverage"] == "CHECKED"}
    gaps = {r["target"] for r in c["coverage"] if r["coverage"] != "CHECKED"}
    assert checked == set(c["ran"])
    assert gaps == set(c["skipped"])


def test_a_missing_column_and_a_degenerate_signal_are_different_states():
    """The distinction the untyped `skipped` map could not express: a column the
    log never carried is not the same as a column that is present and unusable,
    and they send the reader to different places."""
    c = coverage_of(FIXTURES / "healthy.jsonl")
    by_target = {r["target"]: r for r in c["coverage"]}
    absent = [r for r in by_target.values()
              if r["coverage"] == Coverage.OUT_OF_SCOPE_DATA_TRANSIENT.value]
    degenerate = [r for r in by_target.values()
                  if r["coverage"] == Coverage.NOT_CHECKED_DATA_DEGENERATE.value]
    assert absent, "this fixture has columns the log does not carry"
    assert degenerate, "this fixture has a signal too short to judge"
    assert {r["owner"] for r in absent} != {r["owner"] for r in degenerate}


def test_every_gap_carries_a_registered_code_and_an_owner():
    for name in ("healthy.jsonl", "flat.jsonl", "diverging.jsonl", "dead_noisy.jsonl"):
        c = coverage_of(FIXTURES / name)
        for row in c["coverage"]:
            assert row["owner"], f"{name}:{row['target']} has no owner"
            if row["coverage"] != "CHECKED":
                assert row["reason"] in cov.VOCABULARY.codes(), (
                    f"{name}:{row['target']} carries an unregistered reason "
                    f"{row['reason']!r}")


def test_an_uncoded_skip_is_reported_not_guessed():
    """A skip that reaches the report without a code is a bug in trainproof, not
    a fact about the log, so it is recorded as CHECKER_FAILED rather than
    reconstructed by matching prose back to a category."""
    rows = cov.coverage_records([], {"grad-spike": "something went wrong"}, {})
    assert rows[0]["coverage"] == Coverage.NOT_CHECKED_CHECKER_FAILED.value
    assert rows[0]["owner"] == "tooling"


def test_schema_version_did_not_move():
    """Adding an optional key is a minor change under CONTRACTS.md."""
    out = subprocess.run(
        [sys.executable, "-m", "trainproof", "epoch",
         str(FIXTURES / "healthy.jsonl"), "--json"],
        capture_output=True, text=True, check=False)
    payload = json.loads(out.stdout)
    assert payload["schema_version"] == 3
    assert "coverage" in payload["reports"][0]["checks"]
    assert "ran" in payload["reports"][0]["checks"]
    assert "skipped" in payload["reports"][0]["checks"]
