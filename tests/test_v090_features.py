import pytest
from pathlib import Path
from trainproof.epoch import check_records, check_epoch

def test_overfit_fixture_warns_overfit_not_diverge():
    # overfit fixture -> WARN with TP-OVERFIT and NOT TP-DIVERGE.
    gallery = Path(__file__).parent.parent / "examples" / "gallery"
    fixture_path = gallery / "overfit" / "trainer_state.json"
    if not fixture_path.exists():
        pytest.skip("overfit fixture not found")
        
    report = check_epoch(fixture_path)
    
    assert report["verdict"] == "WARN"
    finding_ids = [f["id"] for f in report["findings"]]
    assert "TP-OVERFIT" in finding_ids
    assert "TP-DIVERGE" not in finding_ids

def test_healthy_eval_run_no_overfit():
    # a synthetic healthy-with-eval run (train falling, eval falling or flat) -> no TP-OVERFIT
    records = []
    for i in range(1, 61):
        r = {"step": i, "loss": 2.0 / i}
        if i % 3 == 0:
            # eval falls with it
            r["eval_loss"] = 2.0 / i + 0.1
        records.append(r)
        
    report = check_records(records)
    assert "TP-OVERFIT" not in [f["id"] for f in report["findings"]]

def test_less_than_4_eval_points():
    # a log with < 4 eval points -> no TP-OVERFIT (silent).
    records = []
    for i in range(1, 61):
        r = {"step": i, "loss": 2.0 / i}
        if i <= 3:
            r["eval_loss"] = 0.5 + i * 2.0 # even if ratio is huge, it shouldn't trigger because count < 4
        records.append(r)
        
    report = check_records(records)
    assert "TP-OVERFIT" not in [f["id"] for f in report["findings"]]

def test_train_only_log():
    # a train-only log (no eval_loss) -> no TP-OVERFIT, byte-identical behavior to v0.8.
    records = []
    for i in range(1, 61):
        records.append({"step": i, "loss": 2.0 / i})
        
    report = check_records(records)
    assert "TP-OVERFIT" not in [f["id"] for f in report["findings"]]

def test_tp_overfit_in_rules():
    # id-in-RULES.md test: TP-OVERFIT present.
    rules_md = (Path(__file__).parent.parent / "RULES.md").read_text()
    assert "`TP-OVERFIT`" in rules_md
