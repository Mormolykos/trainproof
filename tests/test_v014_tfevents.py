"""tfevents adapter - validated against a real PyTorch Lightning run.

The fixture is not synthetic. It is the TensorBoard event file from a Fish
Speech (Lightning) LoRA fine-tune of text2semantic, 2049 steps on an RTX 5080,
February 2026. Every expected number below was cross-checked against
tensorboard's own EventAccumulator before being written down, so this file is a
conformance test against the reference implementation, not against itself.

That run overfits - eval loss bottoms out at step 99 and then climbs to 16.19
while train loss keeps falling. Its saved checkpoints all come from after the
turn. Detecting that is the reason the adapter exists.
"""
import struct
import subprocess
from pathlib import Path

import pytest

from trainproof.adapters import parse_log_with_format_info
from trainproof.epoch import check_records
from trainproof.tfevents import (
    is_tfevents,
    normalize_tag,
    parse_tfevents,
    read_scalars,
)

EVIDENCE = (Path(__file__).resolve().parent.parent
            / "evidence" / "fish_lightning_feb2026")
EVENT_FILE = EVIDENCE / "events.out.tfevents.1772206357.InfinityGear.2772.0"

pytestmark = pytest.mark.skipif(
    not EVENT_FILE.exists(), reason="fish evidence file not present"
)


# ---------------------------------------------------------------- raw reading
def test_reads_every_tag_tensorboard_reports():
    scalars = read_scalars(EVENT_FILE)
    assert set(scalars) == {
        "hp_metric", "epoch", "lr-AdamW/pg1", "lr-AdamW/pg2",
        "train/loss", "train/base_loss", "train/semantic_loss",
        "train/top_5_accuracy", "train/grad_norm",
        "val/loss", "val/base_loss", "val/semantic_loss", "val/top_5_accuracy",
    }


@pytest.mark.parametrize("tag,count,first,last", [
    ("train/loss", 41, (49, 7.906250), (2049, 2.843750)),
    ("val/loss", 82, (24, 10.412500), (2049, 16.193750)),
    ("train/grad_norm", 41, (49, 5.156250), (2049, 6.406250)),
    ("lr-AdamW/pg1", 164, (49, 0.0001), (2049, 0.0001)),
    ("epoch", 123, (24, 0.0), (2049, 0.0)),
])
def test_values_match_tensorboard_reference(tag, count, first, last):
    points = read_scalars(EVENT_FILE)[tag]
    assert len(points) == count
    assert points[0][0] == first[0]
    assert points[0][1] == pytest.approx(first[1], rel=1e-6)
    assert points[-1][0] == last[0]
    assert points[-1][1] == pytest.approx(last[1], rel=1e-6)


def test_wall_time_is_populated():
    """TP-THROUGHPUT needs real timestamps; a zeroed wall_time silently kills it."""
    points = read_scalars(EVENT_FILE)["train/loss"]
    assert points[0][2] > 1_600_000_000        # a plausible unix time, not 0.0
    assert points[-1][2] > points[0][2]


# ------------------------------------------------------------- tag normalising
@pytest.mark.parametrize("tag,expected", [
    ("train/loss", "loss"),
    ("training/loss", "loss"),
    ("val/loss", "val_loss"),
    ("validation/loss", "val_loss"),
    ("eval/loss", "val_loss"),          # val_loss and eval_loss are one column
    ("TrainIterStats/loss", "loss"),    # Coqui
    ("EvalStats/avg_loss", "val_loss"),
    ("TrainEpochStats/avg_step_time", "step_time"),
    ("lr-AdamW/pg1", "lr"),
    ("lr/group0", "lr"),
    ("learning_rate", "lr"),
    ("train/grad_norm", "grad_norm"),
    ("some/other/thing", "some_other_thing"),
])
def test_normalize_tag(tag, expected):
    assert normalize_tag(tag) == expected


# ------------------------------------------------------------------- records
def test_records_fold_by_step():
    records, mapping = parse_tfevents(EVENT_FILE)
    assert len(records) == 82
    steps = [r["step"] for r in records]
    assert steps == sorted(steps), "records must be step-ordered"
    assert mapping["loss"] == "train/loss"
    assert mapping["eval_loss"] == "val/loss"
    assert mapping["grad_norm"] == "train/grad_norm"


def test_lr_column_does_not_flip_between_param_groups():
    """pg1 and pg2 both map to lr; the choice must be stable, not last-wins."""
    _, mapping = parse_tfevents(EVENT_FILE)
    assert mapping["lr"] == "lr-AdamW/pg1"


def test_format_autodetected_as_tfevents():
    records, fmt, _, _ = parse_log_with_format_info(EVENT_FILE, fmt="auto")
    assert fmt == "tfevents"
    assert len(records) == 82


def test_directory_of_shards_is_readable():
    records, fmt, _, _ = parse_log_with_format_info(EVIDENCE, fmt="auto")
    assert fmt == "tfevents"
    assert records


def test_is_tfevents():
    assert is_tfevents(EVENT_FILE)
    assert is_tfevents(EVIDENCE)
    assert not is_tfevents(Path(__file__))


# -------------------------------------------------------------------- verdict
def test_overfit_is_detected_on_the_real_run():
    records, _ = parse_tfevents(EVENT_FILE)
    report = check_records(records)
    ids = {f["id"] for f in report["findings"]}
    assert "TP-OVERFIT" in ids
    assert report["verdict"] == "WARN"


def test_cli_reads_tfevents_end_to_end():
    res = subprocess.run(
        ["trainproof", "doctor", str(EVENT_FILE)],
        capture_output=True, text=True,
    )
    assert "FORMAT : tfevents" in res.stdout
    assert "TP-OVERFIT" in res.stdout
    assert res.returncode == 0          # WARN is not a failing exit


# ----------------------------------------------------------------- robustness
def test_truncated_file_does_not_raise(tmp_path):
    """A killed run leaves a half-written record. That run still needs judging."""
    data = EVENT_FILE.read_bytes()
    cut = tmp_path / "events.out.tfevents.truncated"
    cut.write_bytes(data[: int(len(data) * 0.6)] + b"\x99\x99\x99")
    scalars = read_scalars(cut)
    assert scalars["train/loss"], "records before the cut must survive"
    assert len(scalars["train/loss"]) < 41


# --------------------------------------------- TP-ZERO-GRAD false positive
def test_zero_grad_not_reported_when_loss_improved():
    """A run cannot both learn and receive no gradient.

    Coqui writes avg_grad_norm as 0.0 when clipping is off. Firing TP-ZERO-GRAD
    there is a false FAIL - it happened on a real 125k-step XTTS fine-tune.
    """
    records = [{"step": i * 10, "loss": 5.0 - i * 0.4, "grad_norm": 0.0}
               for i in range(12)]
    report = check_records(records)
    ids = {f["id"] for f in report["findings"]}
    assert "TP-ZERO-GRAD" not in ids
    skipped = report["checks"]["skipped"]
    assert "zero-grad" in skipped
    assert "loss improved" in skipped["zero-grad"]


def test_zero_grad_still_fires_when_loss_is_stuck():
    """The severed-graph case must survive the guard above."""
    records = [{"step": i * 10, "loss": 5.0, "grad_norm": 0.0} for i in range(12)]
    report = check_records(records)
    ids = {f["id"] for f in report["findings"]}
    assert "TP-ZERO-GRAD" in ids
    assert report["verdict"] == "FAIL"


def test_garbage_file_yields_nothing_instead_of_crashing(tmp_path):
    junk = tmp_path / "events.out.tfevents.junk"
    junk.write_bytes(struct.pack("<Q", 10 ** 12) + b"\x00" * 64)
    assert read_scalars(junk) == {}
