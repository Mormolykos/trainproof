"""v0.21 — non-finite gradient norms, and a loss that dies partway through.

Both rules come from one real run recovered off disk in September 2026: a
TinyLlama-1.1B SFT from April 2025, 360 steps, 16 GB of checkpoints, every
one of them garbage. **trainproof 0.20.0 reported PASS and exit 0 on it.**

Its `trainer_state.json` had two properties no rule in the library could see:

  * loss 2.859 at step 10, then exactly 0.0 for the remaining 35 logged steps.
    `TP-ZERO-LOSS` requires *every* loss to be zero, so it never fired, and the
    2.859 -> 0.0 collapse reads as spectacular convergence to every ratio rule.
  * every gradient norm NaN. `valid_gns` filters non-finite values out, leaving
    it empty, so `zero-grad` and `grad-spike` skipped themselves with "no finite
    gradient norms in the log". The worse the corruption, the quieter trainproof
    got.

That is the v0.11.1 bug (see rules.py) one layer deeper: v0.11.1 fixed the
series that is zero on EVERY step; these are the series that dies partway and
the series that is non-finite rather than zero.

R19 says hand-written regression tests cannot certify a rejecter — they pin the
failures someone already thought of. So the cases below are followed by property
fuzzing over randomised logs.
"""
from __future__ import annotations

import math
import random

import pytest

from trainproof.epoch import _zero_tail_onset, check_records
from trainproof.preflight import check_preflight


def _ids(report):
    return {f["id"] for f in report["findings"]}


def healthy(n=20, grad=True):
    """A log that must always pass, used as the false-positive control."""
    out = []
    for i in range(n):
        r = {"step": i, "loss": 2.0 * (0.9**i) + 0.01, "learning_rate": 1e-4}
        if grad:
            r["grad_norm"] = 1.0 + (i % 3) * 0.1
        out.append(r)
    return out


# --------------------------------------------------------------------------
# The real run, reduced to its two structural features.
# --------------------------------------------------------------------------

def test_the_2025_corpse_no_longer_passes():
    """The exact shape that returned PASS and exit 0 before v0.21."""
    records = [{"step": 10, "loss": 2.859, "grad_norm": float("nan"), "learning_rate": 4.8e-5}]
    for i in range(2, 37):
        records.append({
            "step": i * 10, "loss": 0.0,
            "grad_norm": float("nan"),
            "learning_rate": 4.8e-5 * (1 - i / 36),
        })

    report = check_records(records)

    assert report["verdict"] == "FAIL"
    assert "TP-NAN-GRAD" in _ids(report)
    assert "TP-ZERO-LOSS-ONSET" in _ids(report)
    assert "TP-PASS" not in _ids(report)


def test_nan_grad_group_reports_as_checked_not_skipped():
    """The R19 point: a column present and wholly non-finite is a FINDING.

    Before v0.21 this exact input made two groups skip themselves, and skipping
    is how the run reached TP-PASS. `grad-finite` must say it ran.
    """
    records = [
        {"step": i, "loss": 1.0 - i * 0.01, "grad_norm": float("nan")}
        for i in range(20)
    ]
    report = check_records(records)
    assert "grad-finite" in report["checks"]["ran"]
    assert "grad-finite" not in report["checks"]["skipped"]
    assert "TP-NAN-GRAD" in _ids(report)


def test_absent_grad_column_is_not_a_finding():
    """Absence is UNKNOWN and never becomes a positive finding (R19/R1)."""
    report = check_records(healthy(grad=False))
    assert "TP-NAN-GRAD" not in _ids(report)
    assert report["checks"]["skipped"]["grad-finite"] == "no grad_norm column in the log"


def test_partial_nan_still_fails():
    """One non-finite gradient is enough — the weights are non-finite after it."""
    records = healthy(20)
    records[7]["grad_norm"] = float("inf")
    report = check_records(records)
    assert "TP-NAN-GRAD" in _ids(report)
    assert report["verdict"] == "FAIL"


# --------------------------------------------------------------------------
# Onset boundaries — where this rule must stay silent.
# --------------------------------------------------------------------------

def test_all_zero_stays_with_zero_loss_not_onset():
    """TP-ZERO-LOSS owns the all-zero series; the two must never double-report."""
    records = [{"step": i, "loss": 0.0} for i in range(20)]
    ids = _ids(check_records(records))
    assert "TP-ZERO-LOSS" in ids
    assert "TP-ZERO-LOSS-ONSET" not in ids


def test_short_zero_tail_does_not_fire():
    """Fewer than MIN_ZERO_TAIL_FOR_ONSET trailing zeros is not yet a signature."""
    records = healthy(20)
    for r in records[-3:]:
        r["loss"] = 0.0
    assert "TP-ZERO-LOSS-ONSET" not in _ids(check_records(records))


def test_small_loss_is_convergence_not_collapse():
    """Detection is exact equality. A tiny loss is a real run and must pass."""
    records = [{"step": i, "loss": max(1e-12, 1.0 * (0.5**i)), "grad_norm": 0.5} for i in range(25)]
    ids = _ids(check_records(records))
    assert "TP-ZERO-LOSS-ONSET" not in ids
    assert "TP-ZERO-LOSS" not in ids


def test_zero_tail_that_recovers_does_not_fire():
    """A zero in the middle is not an onset — only a tail that never recovers."""
    records = healthy(24)
    for r in records[10:14]:
        r["loss"] = 0.0
    assert "TP-ZERO-LOSS-ONSET" not in _ids(check_records(records))


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        ([], None),
        ([0.0, 0.0, 0.0], None),          # all zero -> TP-ZERO-LOSS
        ([1.0, 2.0, 3.0], None),          # no tail
        ([1.0, 0.0, 0.0], 1),
        ([0.0, 1.0, 0.0], 2),             # leading zero does not disqualify
    ],
)
def test_zero_tail_onset_boundaries(series, expected):
    assert _zero_tail_onset(series) == expected


# --------------------------------------------------------------------------
# Preflight: the token exists, it just never enters the sequence.
# --------------------------------------------------------------------------

class Tok:
    def __init__(self, **kw):
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token = "</s>"
        self.pad_token_id = 2
        self.bos_token = "<s>"
        for k, v in kw.items():
            setattr(self, k, v)


def test_no_eos_append_is_reported():
    """The real tokenizer_config.json: add_eos_token false, pad_token == eos."""
    res = check_preflight([{"text": "hello"}], tokenizer=Tok(add_eos_token=False))
    ids = {f["id"] for f in res["findings"]}
    assert "TP-PRE-NO-EOS-APPEND" in ids
    assert "TP-PRE-MISSING-EOS-TOKEN" not in ids  # the token exists

    finding = next(f for f in res["findings"] if f["id"] == "TP-PRE-NO-EOS-APPEND")
    assert finding["level"] == "WARN"
    # when the ids collide the evidence must say so: a pad-masking collator
    # cannot tell a genuine EOS from padding
    assert "pad_token_id == eos_token_id" in finding["evidence"]


def test_no_eos_append_without_id_collision_omits_the_escalation():
    tok = Tok(add_eos_token=False, pad_token="<pad>", pad_token_id=0)
    finding = next(
        f for f in check_preflight([{"text": "hi"}], tokenizer=tok)["findings"]
        if f["id"] == "TP-PRE-NO-EOS-APPEND"
    )
    assert "pad_token_id == eos_token_id" not in finding["evidence"]


@pytest.mark.parametrize("value", [None, True, "false", 0, 1])
def test_add_eos_token_only_literal_false_reports(value):
    """UNKNOWN never becomes a finding.

    Most tokenizer families do not define this attribute at all, and a truthy or
    non-bool value is not evidence of anything. Only a literal False reports —
    anything else, including absence, stays silent.
    """
    kw = {} if value is None else {"add_eos_token": value}
    res = check_preflight([{"text": "hello"}], tokenizer=Tok(**kw))
    assert "TP-PRE-NO-EOS-APPEND" not in {f["id"] for f in res["findings"]}


# --------------------------------------------------------------------------
# Property fuzzing (R19 corollary). State the properties, generate against them.
# --------------------------------------------------------------------------

PROPERTY_TRIALS = 400


def test_property_non_finite_gradient_never_passes():
    """P1: if any logged gradient norm is non-finite, the verdict is never PASS.

    Generated rather than enumerated, because the enumerated version is the
    same blind spot in another form.
    """
    rng = random.Random(20260913)
    for _ in range(PROPERTY_TRIALS):
        n = rng.randint(6, 40)
        records = []
        for i in range(n):
            records.append({
                "step": i,
                "loss": rng.choice([rng.uniform(0.001, 5.0), 0.0]),
                "grad_norm": rng.uniform(0.0, 50.0),
            })
        bad = rng.sample(range(n), rng.randint(1, n))
        for i in bad:
            records[i]["grad_norm"] = rng.choice([float("nan"), float("inf"), float("-inf")])

        report = check_records(records)
        assert report["verdict"] != "PASS", records
        assert "TP-NAN-GRAD" in _ids(report)


def test_property_positive_then_dead_never_passes():
    """P2: a positive loss followed by a sustained exact-zero tail never passes."""
    rng = random.Random(4321)
    for _ in range(PROPERTY_TRIALS):
        live = rng.randint(1, 15)
        tail = rng.randint(5, 30)
        records = [
            {"step": i, "loss": rng.uniform(0.05, 6.0), "learning_rate": 1e-4}
            for i in range(live)
        ]
        records += [
            {"step": live + i, "loss": 0.0, "learning_rate": 1e-4}
            for i in range(tail)
        ]
        report = check_records(records)
        assert report["verdict"] != "PASS", (live, tail)


def test_property_healthy_runs_are_never_failed_by_the_new_rules():
    """P3: no false positives. The new rules must be silent on finite, decreasing
    losses with finite gradients — a false FAIL under stop_on_fail aborts a
    correct run before step 1, the worst outcome this library can produce."""
    rng = random.Random(99)
    new_ids = {"TP-NAN-GRAD", "TP-ZERO-LOSS-ONSET"}
    for _ in range(PROPERTY_TRIALS):
        n = rng.randint(12, 40)
        loss = rng.uniform(2.0, 8.0)
        records = []
        for i in range(n):
            loss = max(1e-9, loss * rng.uniform(0.80, 0.97))
            records.append({
                "step": i,
                "loss": loss,
                "grad_norm": rng.uniform(0.01, 3.0),
                "learning_rate": 1e-4,
            })
        fired = _ids(check_records(records)) & new_ids
        assert not fired, (fired, records[:3])


NON_NUMERIC = ["x", "", [], {}, (), object(), b"1.0", set()]


@pytest.mark.parametrize("junk", NON_NUMERIC)
@pytest.mark.parametrize("column", ["loss", "grad_norm", "lr", "time",
                                    "eval_loss", "step_time", "loader_time", "gpu_util"])
def test_non_numeric_metric_is_absent_not_a_crash(column, junk):
    """P5: no column value can raise out of the constructor.

    PRE-EXISTING defect, found in v0.20.0 by probing the domain rather than the
    behaviour: every metric was gated on `is not None` and then handed straight
    to `math.isnan()`, so a string or a list crashed with TypeError where
    CONTRACTS.md promises a verdict or a documented exit. A value failing its
    predicate is now ABSENT, and absence is never a positive finding.
    """
    records = [{"step": i, "loss": 1.0 - i * 0.01} for i in range(12)]
    for r in records:
        r[column] = junk
    report = check_records(records)  # must not raise
    assert report["verdict"] in {"PASS", "WARN", "FAIL", "NOT-CHECKED"}
    assert "TP-NAN-GRAD" not in _ids(report)


def test_bool_is_not_a_measurement():
    """`True` is an int subclass and would otherwise be judged as a loss of 1.0."""
    records = [{"step": i, "loss": True} for i in range(12)]
    report = check_records(records)
    assert "TP-NO-LOSS" in _ids(report)


def test_property_random_junk_never_crashes():
    """P6: fuzz whole records from a mixed pool of legal and illegal values."""
    rng = random.Random(2026)
    pool = [0.0, 1.5, -2.0, float("nan"), float("inf"), None,
            "x", [], {}, True, False, 3, "1.0"]
    cols = ["loss", "grad_norm", "lr", "time", "eval_loss",
            "step_time", "loader_time", "gpu_util", "step"]
    for _ in range(PROPERTY_TRIALS):
        records = [
            {c: rng.choice(pool) for c in rng.sample(cols, rng.randint(1, len(cols)))}
            for _ in range(rng.randint(1, 25))
        ]
        report = check_records(records)  # must not raise
        assert report["verdict"] in {"PASS", "WARN", "FAIL", "NOT-CHECKED"}


def test_property_onset_index_is_consistent_with_the_series():
    """P4: whenever _zero_tail_onset returns an index, the series really is
    positive-somewhere before it and exactly zero from it onward."""
    rng = random.Random(7)
    for _ in range(PROPERTY_TRIALS):
        n = rng.randint(0, 30)
        series = [rng.choice([0.0, rng.uniform(0.001, 9.0)]) for _ in range(n)]
        onset = _zero_tail_onset(series)
        if onset is None:
            continue
        assert 0 < onset < len(series)
        assert all(v == 0.0 for v in series[onset:])
        assert any(v > 0.0 for v in series[:onset])
        assert not math.isnan(series[onset - 1])
