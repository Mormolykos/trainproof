import json
import subprocess
from trainproof.cli import _get_worst_verdict
from trainproof.epoch import check_records
from trainproof.compare import check_compare

def test_four_point_log_not_checked(tmp_path):
    log = tmp_path / "four_points.jsonl"
    records = [{"step": i, "loss": 0.0} for i in range(4)]
    with open(log, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    report = check_records(records)
    assert report["verdict"] == "NOT-CHECKED"
    assert any(f["id"] == "TP-NOT-CHECKED" for f in report["findings"])
    
    res = subprocess.run(["trainproof", "epoch", "--json", str(log)], capture_output=True, text=True)
    assert res.returncode == 2
    out = json.loads(res.stdout)
    assert out["worst_verdict"] == "NOT-CHECKED"

def test_partial_coverage_still_pass(tmp_path):
    log = tmp_path / "partial.jsonl"
    records = [{"step": i, "loss": 2.0 - i*0.15} for i in range(1, 10)]
    with open(log, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    report = check_records(records)
    assert report["verdict"] == "PASS"
    assert any(f["id"] == "TP-PASS" for f in report["findings"])
    
    res = subprocess.run(["trainproof", "epoch", "--json", str(log)], capture_output=True, text=True)
    assert res.returncode == 0
    out = json.loads(res.stdout)
    assert out["worst_verdict"] == "PASS"

def test_severity_ordering():
    def get_order(verdicts):
        return _get_worst_verdict([{"verdict": v} for v in verdicts])
    
    assert get_order(["PASS", "NOT-CHECKED", "WARN", "FAIL"]) == "FAIL"
    assert get_order(["PASS", "NOT-CHECKED", "WARN"]) == "WARN"
    assert get_order(["PASS", "NOT-CHECKED"]) == "NOT-CHECKED"
    assert get_order(["PASS"]) == "PASS"

def test_directory_with_pass_and_not_checked(tmp_path):
    pass_log = tmp_path / "pass.jsonl"
    with open(pass_log, "w") as f:
        for i in range(1, 10):
            f.write(json.dumps({"step": i, "loss": 2.0 - i*0.15}) + "\n")
            
    nc_log = tmp_path / "nc.jsonl"
    with open(nc_log, "w") as f:
        for i in range(4):
            f.write(json.dumps({"step": i, "loss": 0.0}) + "\n")
            
    res = subprocess.run(["trainproof", "doctor", "--json", str(tmp_path)], capture_output=True, text=True)
    assert res.returncode == 2
    out = json.loads(res.stdout)
    assert out["worst_verdict"] == "NOT-CHECKED"

def test_compare_against_not_checked_yields_uncomparable(tmp_path):
    base_log = tmp_path / "base.jsonl"
    run_log = tmp_path / "run.jsonl"
    
    with open(base_log, "w") as f:
        for i in range(1, 10):
            f.write(json.dumps({"step": i, "loss": 2.0 - i*0.15}) + "\n")
            
    with open(run_log, "w") as f:
        for i in range(4):
            f.write(json.dumps({"step": i, "loss": 0.0}) + "\n")
            
    report = check_compare(str(run_log), str(base_log))
    assert report["verdict"] == "FAIL"
    assert any(f["id"] == "TP-CMP-UNCOMPARABLE" for f in report["findings"])

def test_print_verdict_console_not_checked(capsys):
    from trainproof.report import print_verdict_console
    print_verdict_console("NOT-CHECKED", [{"id": "TP-NOT-CHECKED", "level": "NOT-CHECKED", "message": "msg"}])
    captured = capsys.readouterr()
    assert "NOT-CHECKED" in captured.out
    assert "FAIL" not in captured.out

def test_print_doctor_autopsy_not_checked(capsys):
    from trainproof.report import print_doctor_autopsy
    print_doctor_autopsy("foo.jsonl", "jsonl", 4, "1..4", "NOT-CHECKED", [{"id": "TP-NOT-CHECKED", "level": "NOT-CHECKED", "message": "msg"}])
    captured = capsys.readouterr()
    assert "1 NOT-CHECKED" in captured.out

def test_to_sarif_not_checked():
    from trainproof.sarif import to_sarif
    reports = [{"file": "foo", "verdict": "NOT-CHECKED", "findings": [{"id": "TP-NOT-CHECKED", "level": "NOT-CHECKED", "message": "msg"}]}]
    sarif = to_sarif(reports, "0.13.0")
    assert sarif["runs"][0]["tool"]["driver"]["rules"][0]["defaultConfiguration"]["level"] == "warning"
    assert sarif["runs"][0]["results"][0]["level"] == "warning"

import pytest

@pytest.mark.parametrize("verdict", ["PASS", "WARN", "FAIL", "NOT-CHECKED"])
def test_print_verdict_console_exclusive_verdict(capsys, verdict):
    from trainproof.report import print_verdict_console
    print_verdict_console(verdict, [])
    captured = capsys.readouterr()
    assert verdict in captured.out
    for other in ["PASS", "WARN", "FAIL", "NOT-CHECKED"]:
        if other != verdict:
            assert other not in captured.out

def test_unrecognized_verdict_normalization():
    from trainproof.cli import _get_worst_verdict, _get_exit_code, _VERDICT_SEVERITY
    
    reports_garbage = [{"verdict": "GARBAGE"}]
    reports_missing = [{}]
    
    worst_garbage = _get_worst_verdict(reports_garbage)
    assert worst_garbage in _VERDICT_SEVERITY
    assert worst_garbage == "NOT-CHECKED"
    assert _get_exit_code(reports_garbage) == 2
    
    worst_missing = _get_worst_verdict(reports_missing)
    assert worst_missing in _VERDICT_SEVERITY
    assert worst_missing == "NOT-CHECKED"
    assert _get_exit_code(reports_missing) == 2
