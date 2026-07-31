"""v0.11.1: degenerate series, an honest TP-PASS, and the compare zero-loss hole.

Every test here pins a path that used to produce a clean verdict, a misdiagnosis,
or a claim of coverage trainproof had not delivered. None of them needed a new
diagnostic idea -- they were all holes in checks that already existed.
"""

import json
from pathlib import Path

import pytest

from trainproof.adapters import HF_STATE_META_KEYS, parse_log_with_format_info
from trainproof.compare import check_compare
from trainproof.epoch import CHECK_GROUPS, check_epoch, check_records
from trainproof.integrations.hf import _convert_state_to_records

FIXTURES = Path(__file__).parent / "fixtures"
GALLERY = Path(__file__).parent.parent / "examples" / "gallery"
REAL_WORLD = Path(__file__).parent.parent / "examples" / "real_world"

CONFIGS = ("healthy", "lr_hot", "lr_zero", "fp16_nan", "bad_labels", "overfit")
ALL_LOGS = [GALLERY / n / "trainer_state.json" for n in CONFIGS] + [
    GALLERY / n / s / "trainer_state.json" for n in CONFIGS for s in ("seed43", "seed44")
]


def ids_of(report):
    return {f["id"] for f in report["findings"]}


def label(path):
    return f"{path.parent.parent.name}-{path.parent.name}"


def pass_message(report):
    return next(f["message"] for f in report["findings"] if f["id"] == "TP-PASS")


# --- change 1: a degenerate series is the diagnosis, not a reason to skip -----

def test_all_zero_loss_is_a_finding_not_a_pass():
    # TP-FLAT, TP-DIVERGE and TP-DEAD-RUN are each guarded by `> 0`, so this run
    # skipped all three and reached TP-PASS -- which then named those same three
    # checks as having run.
    report = check_epoch(FIXTURES / "zero_loss.jsonl")
    assert report["verdict"] == "FAIL"
    assert "TP-ZERO-LOSS" in ids_of(report)
    assert "TP-PASS" not in ids_of(report)


def test_all_zero_grad_names_the_severed_graph():
    # this one was already caught -- as TP-DEAD-RUN, which sends you hunting
    # your data and learning rate. TP-ZERO-GRAD names the actual cause.
    report = check_epoch(FIXTURES / "zero_grad.jsonl")
    assert report["verdict"] == "FAIL"
    assert "TP-ZERO-GRAD" in ids_of(report)


def test_zero_grad_does_not_claim_the_spike_check_ran():
    # `median_gn > 0` skips the spike test on an all-zero gradient series. That
    # is fine; reporting it as a check that ran is not.
    report = check_epoch(FIXTURES / "zero_grad.jsonl")
    assert "grad-spike" not in report["checks"]["ran"]
    assert "grad-spike" in report["checks"]["skipped"]


def test_a_single_zero_loss_does_not_disable_divergence():
    # the floor used to be min() over ALL losses, so one 0.0 anywhere drove
    # min_loss to zero and the `min_loss > 0` guard silently disabled divergence
    # detection for the entire run
    records = [
        {"step": 1, "loss": 4.0, "lr": 1e-4},
        {"step": 2, "loss": 2.0, "lr": 1e-4},
        {"step": 3, "loss": 0.0, "lr": 1e-4},  # one fully-masked batch
        {"step": 4, "loss": 3.0, "lr": 1e-4},
        {"step": 5, "loss": 9.0, "lr": 1e-4},
    ]
    report = check_records(records)
    assert "TP-DIVERGE" in ids_of(report)
    assert "divergence" in report["checks"]["ran"]


def test_too_few_points_is_reported_as_skipped_never_as_passed():
    # four zeros is not evidence of a degenerate run. The verdict is NOT-CHECKED,
    # and the report must not imply any check found the run healthy.
    report = check_records([{"step": i, "loss": 0.0} for i in range(4)])
    assert report["verdict"] == "NOT-CHECKED"
    assert "TP-NOT-CHECKED" in ids_of(report)
    assert "TP-PASS" not in ids_of(report)
    assert report["checks"]["ran"] == []
    assert set(report["checks"]["skipped"]) == set(CHECK_GROUPS)


# --- change 2: TP-PASS reports only checks that actually executed ------------

@pytest.mark.parametrize("log", ALL_LOGS, ids=label)
def test_every_check_group_is_accounted_for(log):
    report = check_epoch(log, fmt="hf")
    ran = set(report["checks"]["ran"])
    skipped = set(report["checks"]["skipped"])
    assert ran & skipped == set(), "a group cannot both run and be skipped"
    assert ran | skipped == set(CHECK_GROUPS), "every group must be accounted for"


def test_every_skipped_group_carries_a_reason():
    report = check_epoch(FIXTURES / "healthy.jsonl")
    assert report["checks"]["skipped"]
    for group, reason in report["checks"]["skipped"].items():
        assert isinstance(reason, str) and reason.strip(), group


def test_pass_message_agrees_with_the_structured_checks():
    report = check_epoch(FIXTURES / "healthy.jsonl")
    assert report["verdict"] == "PASS"
    msg = pass_message(report)
    ran_segment = msg.split("Skipped:")[0]

    for group in report["checks"]["ran"]:
        assert group in ran_segment, f"{group} ran but is not in the Ran: list"
    for group in report["checks"]["skipped"]:
        assert group not in ran_segment, f"{group} was skipped but is in the Ran: list"
        assert group in msg, f"{group} was skipped without being named"


def test_real_world_log_without_grad_norms_says_so():
    # the XTTS log carries no gradient norms at all. Before 0.11.1 the PASS text
    # derived its group list from data availability, so this distinction was
    # made in the wrong place.
    report = check_epoch(REAL_WORLD / "xtts_diverged" / "trainer_0_log.txt", fmt="coqui")
    assert "grad-spike" in report["checks"]["skipped"]
    assert "zero-grad" in report["checks"]["skipped"]
    assert "grad-spike" not in report["checks"]["ran"]


# --- change 3: a zero-loss run must never read as favorable ------------------

def test_zero_loss_run_is_not_reported_as_favorable():
    # floor 0.0 beats any baseline, so neither ratio rule can fire, and the
    # undefined improvement used to be substituted with 0.0 -- which kept
    # TP-NEG-IMPROVE quiet too
    report = check_compare(
        str(FIXTURES / "zero_loss.jsonl"),
        str(GALLERY / "healthy" / "trainer_state.json"),
    )
    assert report["verdict"] == "FAIL"
    assert "TP-CMP-UNCOMPARABLE" in ids_of(report)
    assert "TP-CMP-PASS" not in ids_of(report)
    assert "TP-FLOOR-RATIO" not in ids_of(report)  # it structurally cannot fire


def test_zero_loss_run_against_a_baseline_that_never_improved():
    # the worst case. When the baseline's own improvement is not positive, the
    # improvement-deficit guard fails as well, and before 0.11.1 nothing at all
    # fired on this pair.
    report = check_compare(
        str(FIXTURES / "zero_loss.jsonl"),
        str(FIXTURES / "dead_noisy.jsonl"),
    )
    assert report["verdict"] == "FAIL"
    assert "TP-CMP-UNCOMPARABLE" in ids_of(report)
    assert "TP-CMP-PASS" not in ids_of(report)


def test_uncomparable_names_which_side_is_degenerate():
    report = check_compare(
        str(FIXTURES / "zero_loss.jsonl"),
        str(GALLERY / "healthy" / "trainer_state.json"),
    )
    finding = next(f for f in report["findings"] if f["id"] == "TP-CMP-UNCOMPARABLE")
    assert "run" in finding["message"]
    assert "exactly 0.0" in finding["evidence"]


# --- change 4: the adapter preserves run intent (inert, nothing reads it) ----

@pytest.mark.parametrize("log", ALL_LOGS, ids=label)
def test_hf_adapter_preserves_top_level_meta(log):
    _records, fmt, _mapping, meta = parse_log_with_format_info(log, fmt="hf")
    assert fmt == "hf"
    missing = set(HF_STATE_META_KEYS) - set(meta)
    assert not missing, f"dropped by the adapter: {sorted(missing)}"


def test_non_hf_formats_report_empty_meta():
    _records, fmt, _mapping, meta = parse_log_with_format_info(
        REAL_WORLD / "xtts_diverged" / "trainer_0_log.txt", fmt="coqui"
    )
    assert fmt == "coqui"
    assert meta == {}


# --- change 5: the live callback could not see eval loss --------------------

class _State:
    def __init__(self, history, step):
        self.log_history = history
        self.global_step = step


def test_callback_keeps_eval_entries():
    history = [
        {"loss": 1.0, "step": 10, "learning_rate": 0.01, "eval_loss": 2.0},
        {"eval_loss": 1.5, "step": 10},
        {"loss": 0.5, "step": 20, "grad_norm": 0.1},
        {"train_runtime": 12.0, "train_loss": 0.7},  # end-of-run summary
    ]
    records = _convert_state_to_records(_State(history, 20))
    assert len(records) == 3
    assert records[0]["eval_loss"] == 2.0
    assert records[1] == {"eval_loss": 1.5, "step": 10.0}


def test_callback_reaches_the_same_findings_as_the_file_path():
    # TP-OVERFIT needs 4 eval points. The callback dropped every eval entry, so
    # the rule -- three seeds of evidence behind it -- was structurally
    # unreachable there while `trainproof epoch` saw it fine on the same data.
    log = GALLERY / "overfit" / "trainer_state.json"
    history = json.loads(log.read_text(encoding="utf-8"))["log_history"]

    from_callback = check_records(_convert_state_to_records(_State(history, 300)))
    from_file = check_epoch(log, fmt="hf")

    assert "TP-OVERFIT" in ids_of(from_callback)
    assert ids_of(from_callback) == ids_of(from_file)
