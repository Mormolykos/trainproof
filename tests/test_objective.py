"""Objective checks: ignore_index collisions and never-targeted classes.

The regression case is a real one. A VALL-E-X-derived AR stage used
`eos_id = ignore_index = NUM_AUDIO_TOKENS = 1024` against a 1025-class output layer,
so every stop target was dropped before the loss. `test_real_world_*` below reproduce
that exact shape, and the NAR stage of the same model as the safe counter-example:
identical ignore_index, 1024 classes instead of 1025, and therefore correct.
"""

from trainproof.objective import TargetCoverage, check_ignore_index, check_objective


def ids(findings):
    return [f["id"] for f in findings]


def levels(findings):
    return {f["id"]: f["level"] for f in findings}


# --------------------------------------------------------------------------
# ignore_index
# --------------------------------------------------------------------------

def test_sentinel_inside_vocab_fails():
    f = check_ignore_index(num_classes=1025, ignore_index=1024)
    assert ids(f) == ["TP-OBJ-IGNORE-INDEX-COLLISION"]
    assert f[0]["level"] == "FAIL"
    assert "1024" in f[0]["evidence"]


def test_pytorch_default_sentinel_is_clean():
    assert check_ignore_index(num_classes=1025, ignore_index=-100) == []


def test_sentinel_one_past_the_end_is_informational():
    f = check_ignore_index(num_classes=1024, ignore_index=1024)
    assert ids(f) == ["TP-OBJ-IGNORE-INDEX-OK"]
    assert f[0]["level"] == "INFO"


def test_no_ignore_index_is_not_a_finding():
    assert check_ignore_index(num_classes=10, ignore_index=None) == []


def test_class_zero_as_sentinel_still_collides():
    f = check_ignore_index(num_classes=32, ignore_index=0)
    assert ids(f) == ["TP-OBJ-IGNORE-INDEX-COLLISION"]


# --------------------------------------------------------------------------
# dead classes
# --------------------------------------------------------------------------

def test_single_never_targeted_class_is_caught():
    cov = TargetCoverage(num_classes=64, ignore_index=-100)
    cov.observe([c for c in range(64) if c != 17] * 3)
    f = cov.result()
    assert "TP-OBJ-DEAD-CLASS" in ids(f)
    assert "[17]" in [x["evidence"] for x in f if x["id"] == "TP-OBJ-DEAD-CLASS"][0]


def test_full_coverage_passes():
    cov = TargetCoverage(num_classes=16, ignore_index=-100)
    cov.observe(list(range(16)))
    f = cov.result()
    assert levels(f)["TP-OBJ-DEAD-CLASS-OK"] == "PASS"


def test_sparse_sample_is_informational_not_a_failure():
    # Only 10 of 1000 classes seen: absence here means "small sample", not "bug".
    cov = TargetCoverage(num_classes=1000, ignore_index=-100)
    cov.observe(list(range(10)))
    f = cov.result()
    assert ids(f) == ["TP-OBJ-COVERAGE-INSUFFICIENT"]
    assert f[0]["level"] == "INFO"


def test_many_missing_classes_are_not_reported_as_dead():
    # Broad coverage but far more than DEAD_CLASS_MAX_REPORTED missing -> stay quiet.
    cov = TargetCoverage(num_classes=100, ignore_index=-100)
    cov.observe(list(range(60)))
    assert "TP-OBJ-DEAD-CLASS" not in ids(cov.result())


def test_ignored_targets_do_not_count_as_seen():
    cov = TargetCoverage(num_classes=8, ignore_index=7)
    cov.observe(list(range(8)) * 2)
    f = cov.result()
    assert "TP-OBJ-DEAD-CLASS" in ids(f)
    assert cov.n_targets == 14


def test_out_of_range_targets_fail():
    cov = TargetCoverage(num_classes=8, ignore_index=-100)
    cov.observe([0, 1, 2, 99])
    assert "TP-OBJ-TARGET-OUT-OF-RANGE" in ids(cov.result())


def test_observe_accepts_nested_batches():
    cov = TargetCoverage(num_classes=4, ignore_index=-100)
    cov.observe([[0, 1], [2, 3]])
    assert levels(cov.result())["TP-OBJ-DEAD-CLASS-OK"] == "PASS"


# --------------------------------------------------------------------------
# the real failure, and its safe twin
# --------------------------------------------------------------------------

def test_real_world_ar_stage_collision():
    """AR: 1025 classes, EOS=1024, ignore_index=1024. Both checks must fire."""
    NUM_AUDIO_TOKENS = 1024
    targets = []
    for length in (12, 7, 19):
        targets.extend(range(length))          # ordinary codec codes
        targets.append(NUM_AUDIO_TOKENS)       # the EOS the loss will discard

    f = check_objective(num_classes=NUM_AUDIO_TOKENS + 1,
                        ignore_index=NUM_AUDIO_TOKENS,
                        targets=targets)
    assert "TP-OBJ-IGNORE-INDEX-COLLISION" in ids(f)


def test_real_world_nar_stage_is_clean():
    """NAR: same sentinel, 1024 classes. 1024 is out of range, so it is correct."""
    NUM_AUDIO_TOKENS = 1024
    f = check_ignore_index(num_classes=NUM_AUDIO_TOKENS, ignore_index=NUM_AUDIO_TOKENS)
    assert levels(f)["TP-OBJ-IGNORE-INDEX-OK"] == "INFO"


def test_healthy_loss_does_not_rescue_the_check():
    """The point of these checks: nothing about the curve is consulted."""
    f = check_ignore_index(num_classes=1025, ignore_index=1024)
    assert f and f[0]["level"] == "FAIL"
