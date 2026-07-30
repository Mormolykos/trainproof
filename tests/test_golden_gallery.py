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
import subprocess
import sys
from pathlib import Path

import pytest

from trainproof.compare import check_compare
from trainproof.epoch import check_epoch

GALLERY = Path(__file__).parent.parent / "examples" / "gallery"
REAL_WORLD = Path(__file__).parent.parent / "examples" / "real_world"
GOLDEN = Path(__file__).parent / "golden"

# Failures nobody injected, kept apart from the gallery because they prove a
# different thing. name -> (log filename, format)
REAL_RUNS = {"xtts_diverged": ("trainer_0_log.txt", "coqui")}

RUNS = ["healthy", "lr_hot", "lr_zero", "fp16_nan", "bad_labels", "overfit"]

# Seed 42 lives at the config root; 43 and 44 are nested beside it. Every config
# has all three seeds.
SEEDS = ["seed43", "seed44"]
SEEDED = list(RUNS)
FAULTS = [r for r in SEEDED if r != "healthy"]

# Healthy judged against healthy from a DIFFERENT seed. Comparing a run to
# itself proves nothing; this is the false-positive check. None means seed 42.
CROSS_SEED_PAIRS = [("seed43", None), ("seed44", None), ("seed44", "seed43")]

UPDATING = os.environ.get("TRAINPROOF_UPDATE_GOLDEN") == "1"


def log_for(name, seed=None):
    d = GALLERY / name if seed is None else GALLERY / name / seed
    return str((d / "trainer_state.json").resolve())


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


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", SEEDED)
def test_epoch_verdict_is_locked_per_seed(name, seed):
    assert_matches_golden(
        f"epoch_{name}_{seed}", check_epoch(log_for(name, seed), fmt="hf")
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", FAULTS)
def test_compare_against_same_seed_healthy_is_locked(name, seed):
    # Same-seed baseline: that is how the evidence matrix has always been built,
    # and it keeps batch-order noise out of the comparison.
    report = check_compare(log_for(name, seed), log_for("healthy", seed), fmt="hf")
    assert_matches_golden(f"compare_{name}_{seed}_vs_healthy", report)


@pytest.mark.parametrize("run_seed,base_seed", CROSS_SEED_PAIRS)
def test_cross_seed_healthy_baseline_is_clean(run_seed, base_seed):
    # A healthy run judged against a healthy baseline from another seed must stay
    # clean. If this ever fires, the compare rules are reading seed noise as signal.
    report = check_compare(
        log_for("healthy", run_seed), log_for("healthy", base_seed), fmt="hf"
    )
    tag = f"{run_seed}_vs_{base_seed or 'seed42'}"
    assert_matches_golden(f"compare_healthy_{tag}", report)


def test_gallery_has_no_uncovered_runs():
    # a new gallery folder must arrive with a snapshot, not slip in unjudged
    on_disk = {p.name for p in GALLERY.iterdir() if p.is_dir()}
    assert on_disk == set(RUNS)

    # the same rule one level down: a seed log no test judges is decoration
    for name in RUNS:
        seeds_on_disk = {p.name for p in (GALLERY / name).iterdir() if p.is_dir()}
        assert seeds_on_disk == (set(SEEDS) if name in SEEDED else set()), name


@pytest.mark.parametrize("name", sorted(REAL_RUNS))
def test_real_world_verdict_is_locked(name):
    # Same lock as the gallery. These are also the only non-HF-format fixtures in
    # the repo, so this doubles as a regression test for that adapter against a
    # real 580KB log rather than a synthetic one.
    filename, fmt = REAL_RUNS[name]
    log = str((REAL_WORLD / name / filename).resolve())
    assert_matches_golden(f"epoch_real_{name}", check_epoch(log, fmt=fmt))


def test_real_world_has_no_uncovered_runs():
    on_disk = {p.name for p in REAL_WORLD.iterdir() if p.is_dir()}
    assert on_disk == set(REAL_RUNS)


def test_evidence_matrix_is_current():
    # EVIDENCE_MATRIX.md is derived from these same logs, so a stale one is a
    # published document contradicting the data it describes -- which is exactly
    # what happened when v0.3-dev verdicts survived into the v0.10 era.
    script = Path(__file__).parent.parent / "scripts" / "regenerate_evidence.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        check=False,  # the exit code is the assertion
    )
    assert result.returncode == 0, result.stderr or result.stdout
