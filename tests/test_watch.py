import json
from pathlib import Path
from trainproof.watch import poll_once

def test_watch_poll_once(tmp_path: Path):
    log_file = tmp_path / "test.jsonl"
    
    # Phase 1: warming up (less than 10 valid losses)
    logs = []
    for i in range(5):
        logs.append(json.dumps({"loss": 5.0, "step": i, "lr": 1e-4}))
    log_file.write_text("\n".join(logs))
    
    verdict, changed, n = poll_once(log_file, "jsonl", None)
    assert verdict is None
    assert not changed
    assert n == 5
    
    # Phase 2: healthy phase -> PASS
    for i in range(5, 15):
        logs.append(json.dumps({"loss": 1.0, "step": i, "lr": 1e-4}))
    log_file.write_text("\n".join(logs))
    
    verdict, changed, n = poll_once(log_file, "jsonl", None)
    assert verdict == "PASS"
    assert changed
    assert n == 15
    
    # Check that it doesn't say "changed" if prev_verdict was PASS
    verdict2, changed2, n2 = poll_once(log_file, "jsonl", "PASS")
    assert verdict2 == "PASS"
    assert not changed2
    
    # Phase 3: append diverging losses -> verdict changes to FAIL
    for i in range(15, 25):
        logs.append(json.dumps({"loss": 20.0, "step": i, "lr": 1e-4}))
    log_file.write_text("\n".join(logs))
    
    verdict3, changed3, n3 = poll_once(log_file, "jsonl", "PASS")
    assert verdict3 == "FAIL"
    assert changed3
    assert n3 == 25

from unittest.mock import patch
from trainproof.watch import watch_loop

def test_watch_stall_warning(tmp_path, capsys):
    log_file = tmp_path / "test.jsonl"
    log_file.write_text("dummy")
    
    times = [100.0, 100.0, 500.0, 500.0]
    def mock_time(): return times.pop(0) if times else 500.0
    
    sleep_calls = [0]
    def mock_sleep(secs):
        if sleep_calls[0] == 0:
            sleep_calls[0] += 1
            return
        raise KeyboardInterrupt()
        
    with patch("time.monotonic", mock_time), patch("time.sleep", mock_sleep):
        try:
            watch_loop(log_file, interval=10, stall_timeout=300)
        except SystemExit:
            pass
            
    out, err = capsys.readouterr()
    assert "No log growth for 300s" in out
