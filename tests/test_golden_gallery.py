"""Locked verdicts for every run in the fault-injection gallery.

Snapshots hold the verdict and the *complete* set of rule IDs, so a rule that
stops firing and a rule that starts firing spuriously both break the build --
neither is visible to a test that only asserts one ID is present.

Evidence numbers and exit codes are deliberately not snapshotted: evidence
strings couple the suite to float formatting, and exit codes are a separate
contract asserted in test_contracts.py.

Regenerate after an intended rule change (and record it in CHANGELOG.md):
    TRAINPROOF_UPDATE_GOLDEN=1 uv run pytest tests/test_golden_gallery.py
"""

import json
import os
from pathlib import Path

import pytest

from trainproof.epoch import check_epoch
from trainproof.compare import check_compare

GALLERY = Path(__file__).parent.parent / "examples" / "gallery"
GOLDEN = Path(__file__).parent / "golden"

RUNS = ["healthy", "lr_hot", "lr_zero", "fp16_nan", "bad_labels", "overfit"]
UPDATING = os.environ.get("TRAINPROOF_UPDATE_GOLDEN") == "1"


def log_for(name):
    return str((GALLERY / name / "trainer_state.json").resolve())


def snapshot(report):
    return {
        "verdict": report["verdict"],
        "finding_ids": sorted(f["id"] for f in report["findings"]),
    }


def assert_matches_golden(name, report):
    path = GOLDEN / f"{name}.json"
    actual = snapshot(report)

    if UPDATING:
        path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        return

    assert path.exists(), (
        f"missing snapshot {path.name} -- a deleted snapshot must fail, not "
        f"silently regenerate. Recreate with TRAINPROOF_UPDATE_GOLDEN=1."
    )
    assert actual == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", RUNS)
def test_epoch_verdict_is_locked(name):
    assert_matches_golden(f"epoch_{name}", check_epoch(log_for(name), fmt="hf"))


@pytest.mark.parametrize("name", RUNS)
def test_compare_against_healthy_is_locked(name):
    report = check_compare(log_for(name), log_for("healthy"), fmt="hf")
    assert_matches_golden(f"compare_{name}_vs_healthy", report)


def test_gallery_has_no_uncovered_runs():
    # a new gallery folder must arrive with a snapshot, not slip in unjudged
    on_disk = {p.name for p in GALLERY.iterdir() if p.is_dir()}
    assert on_disk == set(RUNS)
