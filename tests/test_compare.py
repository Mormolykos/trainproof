import pytest
from pathlib import Path
from trainproof.compare import check_compare

def get_gallery_log(name: str) -> str:
    path = Path(__file__).parent.parent / "examples" / "gallery" / name / "trainer_state.json"
    return str(path.resolve())

def test_compare_healthy_vs_healthy():
    healthy = get_gallery_log("healthy")
    res = check_compare(healthy, healthy, fmt="hf")
    assert res["verdict"] == "PASS"

def test_compare_bad_labels_vs_healthy():
    bad_labels = get_gallery_log("bad_labels")
    healthy = get_gallery_log("healthy")
    res = check_compare(bad_labels, healthy, fmt="hf")
    assert res["verdict"] == "FAIL"
    findings = [f for f in res["findings"] if "loss floor ratio" in f["message"]]
    assert len(findings) > 0

def test_compare_lr_zero_vs_healthy():
    lr_zero = get_gallery_log("lr_zero")
    healthy = get_gallery_log("healthy")
    res = check_compare(lr_zero, healthy, fmt="hf")
    assert res["verdict"] == "FAIL"
    findings = [f for f in res["findings"] if "improvement deficit" in f["message"]]
    assert len(findings) > 0

def test_compare_lr_hot_vs_healthy():
    # lr_hot's explosion happens INSIDE the first-5 window, inflating its own
    # start median — self-relative improvement looks positive (+62%). Only the
    # end-loss ratio (where the run LANDED: ~7x the baseline) catches it.
    lr_hot = get_gallery_log("lr_hot")
    healthy = get_gallery_log("healthy")
    res = check_compare(lr_hot, healthy, fmt="hf")
    assert res["verdict"] == "FAIL"
    findings = [f for f in res["findings"] if "end loss ratio" in f["message"]]
    assert len(findings) > 0

def test_compare_fp16_nan_vs_healthy():
    fp16_nan = get_gallery_log("fp16_nan")
    healthy = get_gallery_log("healthy")
    res = check_compare(fp16_nan, healthy, fmt="hf")
    assert res["verdict"] == "FAIL"
    findings = [f for f in res["findings"] if "negative improvement" in f["message"]]
    assert len(findings) > 0
