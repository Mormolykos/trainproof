from pathlib import Path
from trainproof.adapters import parse_log_with_format
from trainproof.epoch import check_epoch

FIXTURES = Path(__file__).parent / "fixtures"

def test_coqui_adapter():
    path = FIXTURES / "coqui_trainer_slice.txt"
    records = parse_log_with_format(path)
    assert len(records) >= 6
    for i, r in enumerate(records):
        assert "loss" in r
        assert "lr" in r
        assert "step" in r
        assert "time" in r
        if i > 0:
            assert r["time"] >= records[i-1]["time"]
            assert r["step"] > records[i-1]["step"]

def test_hf_healthy_adapter():
    path = FIXTURES / "trainer_state_healthy.json"
    records = parse_log_with_format(path)
    assert len(records) == 10  # 11 logs, 1 is eval and has no loss
    assert "loss" in records[0]
    
    result = check_epoch(path)
    assert result["verdict"] == "PASS"

def test_hf_diverging_adapter():
    path = FIXTURES / "trainer_state_diverging.json"
    result = check_epoch(path)
    assert result["verdict"] == "FAIL"
    # Find Divergence finding
    assert any("diverging" in f["message"] for f in result["findings"])

def test_format_override():
    # Test format override behavior
    path = FIXTURES / "trainer_state_healthy.json"
    # Overriding to jsonl should fail parsing since it's a JSON object, not JSONL
    records = parse_log_with_format(path, fmt="jsonl")
    assert len(records) == 0
