import sys
import json
import pytest
from unittest.mock import patch
from pathlib import Path
import trainproof.cli as cli

@pytest.fixture
def run_cli():
    def _run(*args):
        with patch.object(sys, "argv", ["trainproof"] + list(args)):
            try:
                cli.main()
                return 0
            except SystemExit as e:
                return e.code
    return _run

def test_version_flag(run_cli, capsys):
    code = run_cli("--version")
    assert code == 0
    out, err = capsys.readouterr()
    assert "0.7.0" in out

def test_doctor_file_mode(tmp_path, run_cli, capsys):
    log_file = tmp_path / "trainer_state.json"
    log_file.write_text(json.dumps({"log_history": [{"step":1, "loss":2.0}, {"step":2, "loss":1.0}, {"step":3, "loss":0.5}]}))
    
    code = run_cli("doctor", str(log_file))
    assert code == 0
    out, err = capsys.readouterr()
    assert "FILE   :" in out
    assert "VERDICT: PASS" in out
    assert "What this cannot tell you" in out
    assert "Findings:" in out

def test_doctor_diagnose_alias(tmp_path, run_cli, capsys):
    log_file = tmp_path / "trainer_state.json"
    log_file.write_text(json.dumps({"log_history": [{"step":1, "loss":2.0}, {"step":2, "loss":1.0}, {"step":3, "loss":0.5}]}))
    
    code = run_cli("diagnose", str(log_file))
    assert code == 0

def test_doctor_directory_discovery(tmp_path, run_cli, capsys):
    # valid 1
    (tmp_path / "trainer_state.json").write_text(json.dumps({"log_history": [{"step":1, "loss":2.0}, {"step":2, "loss":1.0}, {"step":3, "loss":0.5}]}))
    # valid 2 (csv)
    (tmp_path / "log.csv").write_text("step,loss\n1,2.0\n2,1.0\n3,0.5\n")
    # valid 3 (will be failing)
    (tmp_path / "bad.jsonl").write_text('{"step":1,"loss":1}\n{"step":2,"loss":1}\n{"step":3,"loss":1}\n')
    # invalid (skipped)
    (tmp_path / "unparseable.txt").write_text("hello world")
    
    code = run_cli("doctor", str(tmp_path))
    assert code == 1 # because bad.jsonl is a flatline failure
    out, err = capsys.readouterr()
    
    # 3 logs discovered
    assert "SUMMARY" in out
    assert "trainer_state.json" in out
    assert "log.csv" in out
    assert "bad.jsonl" in out
    assert "unparseable.txt" not in out
    assert "VERDICT: FAIL" in out
    assert "VERDICT: PASS" in out

def test_doctor_zero_logs(tmp_path, run_cli, capsys):
    code = run_cli("doctor", str(tmp_path))
    assert code == 2
    out, err = capsys.readouterr()
    assert "No valid logs found." in out

def test_compare_n_way(tmp_path, run_cli, capsys):
    base = tmp_path / "base.csv"
    base.write_text("step,loss\n1,2.0\n2,1.0\n3,0.5\n4,0.4\n5,0.3\n6,0.3\n7,0.3\n8,0.3\n9,0.3\n10,0.3")
    run1 = tmp_path / "run1.csv"
    run1.write_text("step,loss\n1,2.0\n2,1.0\n3,0.5\n4,0.4\n5,0.3\n6,0.3\n7,0.3\n8,0.3\n9,0.3\n10,0.3")
    run2 = tmp_path / "run2.csv" # high loss -> fail
    run2.write_text("step,loss\n1,5.0\n2,4.0\n3,3.0\n4,3.0\n5,3.0\n6,3.0\n7,3.0\n8,3.0\n9,3.0\n10,3.0")
    
    code = run_cli("compare", str(base), str(run1), str(run2))
    assert code == 1
    out, err = capsys.readouterr()
    
    # Table should be printed
    assert "RUN" in out and "VERDICT vs BASE" in out
    assert "base.csv" in out
    assert "run1.csv" in out
    assert "run2.csv" in out
    
    # Not detailed output because len(runs) > 1
    assert "TRAINPROOF VERDICT" not in out

def test_compare_backward_compat(tmp_path, run_cli, capsys):
    base = tmp_path / "base.csv"
    base.write_text("step,loss\n1,2.0\n2,1.0\n3,0.5\n4,0.4\n5,0.3\n6,0.3\n7,0.3\n8,0.3\n9,0.3\n10,0.3")
    run1 = tmp_path / "run1.csv"
    run1.write_text("step,loss\n1,2.0\n2,1.0\n3,0.5\n4,0.4\n5,0.3\n6,0.3\n7,0.3\n8,0.3\n9,0.3\n10,0.3")
    
    code = run_cli("compare", str(base), str(run1))
    assert code == 0
    out, err = capsys.readouterr()
    
    # Detailed output should be present
    assert "TRAINPROOF VERDICT" in out
    assert "[PASS] All checks passed." in out
