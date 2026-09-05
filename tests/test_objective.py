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

# --------------------------------------------------------------------------
# label alignment (causal-LM double shift)
#
# Which arrangement is CORRECT is a property of the loss, not of the tensors.
# `loss_shifts` carries that, and it defaults to None -- so the default path can
# observe an arrangement but can never fail one. That default exists because an
# earlier draft failed a correct custom training loop, and a false FAIL under
# stop_on_fail aborts a real run.
#
# The verdict is a vote over ROWS: positions inside one sequence are not
# independent observations, and a single odd row must not veto the batch.
# --------------------------------------------------------------------------

from trainproof.objective import check_label_alignment  # noqa: E402


def _rows(k=4, n=40, start=10, step=1000):
    """Strictly increasing, non-overlapping rows: no position satisfies both."""
    return [list(range(start + r * step, start + r * step + n)) for r in range(k)]


def _preshift(rows):
    return [r[1:] + [r[-1]] for r in rows]


# --- the default never fails ------------------------------------------------

def test_default_never_fails_on_preshifted_labels():
    """The whole point of the tri-state. Correct-but-unconfirmable must not FAIL."""
    x = _rows()
    f = check_label_alignment(x, _preshift(x))
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
    assert f[0]["level"] == "NOT-CHECKED"
    assert "depends on whether this loss shifts" in f[0]["evidence"]


def test_default_never_fails_on_aligned_labels():
    x = _rows()
    f = check_label_alignment(x, [r[:] for r in x])
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]


# --- loss_shifts=True: the HuggingFace causal-LM convention ------------------

def test_shifting_loss_fails_preshifted_labels():
    x = _rows()
    f = check_label_alignment(x, _preshift(x), loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-DOUBLE"]
    assert f[0]["level"] == "FAIL"
    assert "two tokens ahead" in f[0]["message"]


def test_shifting_loss_passes_aligned_labels():
    x = _rows()
    f = check_label_alignment(x, [r[:] for r in x], loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]
    assert f[0]["level"] == "PASS"


# --- loss_shifts=False: the mirror bug, caller-asserted only -----------------

def test_non_shifting_loss_fails_aligned_labels():
    """Trained to emit the token it was just given. Only ever caller-asserted."""
    x = _rows()
    f = check_label_alignment(x, [r[:] for r in x], loss_shifts=False)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-DOUBLE"]
    assert "token it was just given" in f[0]["message"]


def test_non_shifting_loss_passes_preshifted_labels():
    x = _rows()
    f = check_label_alignment(x, _preshift(x), loss_shifts=False)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]


# --- row voting -------------------------------------------------------------

def test_one_degenerate_row_does_not_veto_the_batch():
    """Both critics raised this: unanimity let one odd row hide a real bug."""
    x = _rows(k=4)
    y = _preshift(x)
    x[0] = [7] * 40          # repeated tokens: satisfies both hypotheses
    y[0] = [7] * 40
    f = check_label_alignment(x, y, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-DOUBLE"]
    assert "3/3 informative rows" in f[0]["evidence"]


def test_rows_that_disagree_are_not_checked():
    x = _rows(k=4)
    y = _preshift(x)
    y[0] = x[0][:]           # one row aligned, three shifted -> 75% is not enough
    y[1] = x[1][:]
    f = check_label_alignment(x, y, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
    assert "rows disagree" in f[0]["evidence"]


def test_all_degenerate_rows_are_not_checked():
    f = check_label_alignment([[7] * 40] * 3, [[7] * 40] * 3, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
    assert "degenerate" in f[0]["evidence"]


def test_labels_unrelated_to_inputs_are_never_failed():
    """A classification head is correct and must not be failed by this convention."""
    f = check_label_alignment(_rows(), [[3] * 40] * 4, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
    assert "not a copy of input_ids" in f[0]["evidence"]


# --- masking and sampling ---------------------------------------------------

def test_masked_prompt_positions_are_skipped():
    x = _rows(k=2, n=60)
    y = [[-100] * 20 + r[20:] for r in x]
    f = check_label_alignment(x, y, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]


def test_long_prompt_short_completion_still_reaches_the_tail():
    """The column cap takes the TAIL. A head slice would see only ignore_index."""
    x = _rows(k=4, n=3000)
    y = [[-100] * 2900 + r[2900:] for r in x]
    f = check_label_alignment(x, y, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]


def test_row_cap_limits_what_is_converted():
    from trainproof import rules
    big = _rows(k=64, n=40)
    f = check_label_alignment(big, [r[:] for r in big], loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]
    assert f"of {rules.LABEL_ALIGNMENT_MAX_ROWS} judgeable" in f[0]["evidence"]


def test_too_few_positions_is_not_checked():
    x = _rows(k=1, n=8)
    f = check_label_alignment(x, [r[:] for r in x], loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]


def test_batch_mismatch_is_not_checked():
    f = check_label_alignment(_rows(k=1), _rows(k=2), loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
    assert "batch mismatch" in f[0]["evidence"]


def test_custom_ignore_index_is_honoured():
    x = _rows(k=2, n=60)
    y = [[999] * 20 + r[20:] for r in x]
    f = check_label_alignment(x, y, ignore_index=999, loss_shifts=True)
    assert ids(f) == ["TP-OBJ-LABEL-SHIFT-OK"]


def test_check_objective_threads_loss_shifts_through():
    x = _rows()
    without = check_objective(num_classes=99999, ignore_index=-100, targets=x)
    assert not any(i.startswith("TP-OBJ-LABEL-SHIFT") for i in ids(without))

    observed = check_objective(num_classes=99999, ignore_index=-100, targets=x, input_ids=x)
    assert "TP-OBJ-LABEL-SHIFT-NOT-MEASURED" in ids(observed)

    judged = check_objective(num_classes=99999, ignore_index=-100, targets=x,
                             input_ids=x, loss_shifts=True)
    assert "TP-OBJ-LABEL-SHIFT-OK" in ids(judged)
