import pytest
from pathlib import Path
from trainproof.epoch import check_epoch

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def test_epoch_healthy():
    report = check_epoch(FIXTURES_DIR / "healthy.jsonl")
    assert report["verdict"] == "PASS"

def test_epoch_diverging():
    report = check_epoch(FIXTURES_DIR / "diverging.jsonl")
    assert report["verdict"] == "FAIL"
    assert any("diverging" in str(f) for f in report["findings"])

def test_epoch_nan():
    report = check_epoch(FIXTURES_DIR / "nan.jsonl")
    assert report["verdict"] == "FAIL"
    assert any("NaN" in str(f) for f in report["findings"])

def test_epoch_csv_with_text_columns(tmp_path):
    # regression: CSV logs with non-numeric columns must not crash the parser
    log = tmp_path / "run.csv"
    log.write_text(
        "step,loss,lr,phase\n"
        "1,4.0,0.001,train\n"
        "2,3.5,0.001,train\n"
        "3,3.1,0.001,train\n"
    )
    report = check_epoch(log)
    assert report["verdict"] in ("PASS", "WARN")

def test_epoch_flat():
    report = check_epoch(FIXTURES_DIR / "flat.jsonl")
    assert report["verdict"] == "FAIL"
    assert any("completely flat" in str(f) for f in report["findings"])

def test_epoch_dead_noisy():
    # noisy-but-never-improving loss with healthy lr: only the no-improvement
    # rule (v0.2) can catch this dead run
    report = check_epoch(FIXTURES_DIR / "dead_noisy.jsonl")
    assert report["verdict"] == "FAIL"
    assert any("never improved" in str(f) for f in report["findings"])
