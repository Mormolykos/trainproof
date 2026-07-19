import json
import pytest
from pathlib import Path
from trainproof.epoch import check_records
from trainproof.compare import check_compare
from trainproof.cli import main as cli_main
import sys
from unittest.mock import patch

def test_loss_vs_eval_loss_separation():
    # loss=0.5 healthy, eval_loss extreme
    records = []
    for i in range(15):
        records.append({"step": i, "loss": 0.5 - i*0.01, "eval_loss": 999.0})
    
    report = check_records(records)
    assert report["verdict"] == "PASS"

def test_eval_loss_only():
    records = []
    for i in range(15):
        records.append({"step": i, "eval_loss": 0.5 - i*0.01})
    
    report = check_records(records)
    assert report["verdict"] == "FAIL"
    assert report["findings"][0]["id"] == "TP-NO-LOSS"

def test_grad_accum_dropped():
    # Provide grad_accum which is high, but real grad_norm is healthy
    records = []
    for i in range(15):
        records.append({"step": i, "loss": 0.5 - i*0.01, "grad_norm": 1.0, "grad_accum": 5000.0})
    
    report = check_records(records)
    assert report["verdict"] == "PASS"
    assert not any(f["id"] == "TP-GRAD-SPIKE" for f in report["findings"])

def test_cli_map_works_end_to_end(tmp_path, capsys):
    log = tmp_path / "run.csv"
    log.write_text("step,custom,lr\n1,2.0,0.01\n2,1.0,0.01\n3,0.5,0.01\n")
    
    with patch.object(sys, "argv", ["trainproof", "epoch", str(log), "--map", "loss=custom"]):
        try:
            cli_main()
        except SystemExit as e:
            assert e.code == 0
            
    out, err = capsys.readouterr()
    assert "TP-PASS" in out

def test_rule_ids_literal_in_output(tmp_path, capsys):
    log = tmp_path / "run.csv"
    log.write_text("step,loss\n1,1.0\n2,2.0\n3,5.0\n")
    with patch.object(sys, "argv", ["trainproof", "epoch", str(log)]):
        try:
            cli_main()
        except SystemExit:
            pass
    out, err = capsys.readouterr()
    assert "TP-DIVERGE" in out

def test_json_output(tmp_path, capsys):
    log = tmp_path / "run.csv"
    log.write_text("step,loss\n1,1.0\n2,0.5\n3,0.1\n")
    with patch.object(sys, "argv", ["trainproof", "epoch", str(log), "--json"]):
        try:
            cli_main()
        except SystemExit:
            pass
    out, err = capsys.readouterr()
    data = json.loads(out)
    assert "schema_version" in data
    assert "trainproof_version" in data
    assert "worst_verdict" in data
    assert "reports" in data
    assert "id" in data["reports"][0]["findings"][0]

def test_honest_tp_pass():
    records = []
    for i in range(15):
        records.append({"step": i, "loss": 0.5 - i*0.01})
    report = check_records(records)
    assert report["verdict"] == "PASS"
    pass_finding = next(f for f in report["findings"] if f["id"] == "TP-PASS")
    assert "Ran: loss-shape, divergence, dead-run" in pass_finding["message"]
    assert "Skipped (no data): grad-norm, lr, timing" in pass_finding["message"]

def test_gallery_regression(capsys):
    gallery = Path(__file__).parent.parent / "examples" / "gallery"
    if not gallery.exists():
        pytest.skip("Gallery not found")
        
    with patch.object(sys, "argv", ["trainproof", "doctor", str(gallery), "--json"]):
        try:
            cli_main()
        except SystemExit:
            pass
    out, err = capsys.readouterr()
    data = json.loads(out)
    
    verdicts = {}
    for r in data["reports"]:
        name = Path(r["file"]).parent.name
        verdicts[name] = r["verdict"]
        
    assert verdicts.get("healthy") == "PASS"
    assert verdicts.get("lr_hot") == "FAIL"
    assert verdicts.get("lr_zero") == "FAIL"
    assert verdicts.get("fp16_nan") == "FAIL"

def test_all_emitted_ids_in_rules():
    rules_md = (Path(__file__).parent.parent / "RULES.md").read_text()
    
    # Extract IDs from RULES.md
    import re
    known_ids = set(re.findall(r'`(TP-[A-Z0-9-]+)`', rules_md))
    
    # We will test healthy and diverging just to ensure emitted ones are in rules
    records = []
    for i in range(15):
        records.append({"step": i, "loss": 0.5 - i*0.01})
    report = check_records(records)
    
    for f in report["findings"]:
        assert f["id"] in known_ids
