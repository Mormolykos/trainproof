"""The stability contract: exit codes, the JSON envelope, and version truth.

The load-bearing distinction is between exit 1 and exit 2. Exit 1 is a verdict
about the user's training run; exit 2 means trainproof could not judge it. If
those collapse into one code, CI cannot tell a doomed run from an unreadable
file, and "if trainproof says FAIL, investigate" stops meaning anything.
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import trainproof
from trainproof.cli import main as cli_main

REPO = Path(__file__).parent.parent
GALLERY = REPO / "examples" / "gallery"


def run_cli(argv):
    with patch.object(sys, "argv", ["trainproof"] + argv):
        with pytest.raises(SystemExit) as exc:
            cli_main()
    return exc.value.code


def test_version_declared_once():
    # tomllib is 3.11+, the package supports 3.10 -- match the literal instead
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert declared, "no version declared in pyproject.toml"
    assert declared.group(1) == trainproof.__version__


def test_rule_ids_and_rules_md_agree():
    # both directions: an undocumented rule is unusable, and a documented rule
    # that no longer fires is a promise the tool has quietly stopped keeping
    documented = set(
        re.findall(r"`(TP-[A-Z0-9-]+)`", (REPO / "RULES.md").read_text(encoding="utf-8"))
    )
    in_source = set()
    for path in (REPO / "src").rglob("*.py"):
        in_source |= set(re.findall(r'"(TP-[A-Z0-9-]+)"', path.read_text(encoding="utf-8")))

    assert in_source - documented == set(), "rule IDs missing from RULES.md"
    assert documented - in_source == set(), "RULES.md documents rules no code emits"


def test_fail_verdict_exits_1(tmp_path):
    log = tmp_path / "run.csv"
    log.write_text("step,loss\n1,1.0\n2,2.0\n3,5.0\n")
    assert run_cli(["epoch", str(log)]) == 1


def test_pass_verdict_exits_0(tmp_path):
    log = tmp_path / "run.csv"
    log.write_text("step,loss\n1,1.0\n2,0.5\n3,0.1\n")
    assert run_cli(["epoch", str(log)]) == 0


def test_missing_path_exits_2(tmp_path):
    assert run_cli(["doctor", str(tmp_path / "nope")]) == 2


def test_unreadable_log_exits_2(tmp_path):
    log = tmp_path / "run.jsonl"
    log.write_bytes(b"\x00\x01\x02 not a log at all")
    assert run_cli(["epoch", str(log), "--format", "hf"]) == 2


def test_directory_without_logs_exits_2(tmp_path):
    (tmp_path / "readme.md").write_text("nothing to judge here")
    assert run_cli(["doctor", str(tmp_path)]) == 2


def test_tool_error_is_not_reported_as_a_fail_verdict(tmp_path, capsys):
    # the regression that matters: an unreadable file used to emit
    # worst_verdict "FAIL", telling CI the training run was broken
    log = tmp_path / "run.jsonl"
    log.write_bytes(b"\x00\x01\x02 not a log at all")
    code = run_cli(["epoch", str(log), "--format", "hf", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["worst_verdict"] is None
    assert payload["error"]
    assert payload["reports"] == []


def test_every_finding_declares_its_source(capsys):
    run_cli(["epoch", str(GALLERY / "lr_hot" / "trainer_state.json"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    findings = payload["reports"][0]["findings"]

    assert findings
    assert all(f["source"] == "single_run" for f in findings)


def test_doctor_merges_baseline_findings_into_one_array(capsys):
    # v0.10 dropped the separate `compare_findings` key: one flat array, each
    # finding tagged, so a consumer never has to special-case by command
    run_cli([
        "doctor", str(GALLERY / "bad_labels" / "trainer_state.json"),
        "--baseline", str(GALLERY / "healthy" / "trainer_state.json"),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    report = payload["reports"][0]

    assert "compare_findings" not in report
    sources = {f["source"] for f in report["findings"]}
    assert sources == {"single_run", "compare"}


def test_failed_baseline_comparison_exits_1(capsys):
    # bad_labels is only WARN on its own; the baseline comparison is what FAILs
    # it. Printing [FAIL] findings and exiting 0 would make CI trust a bad run.
    code = run_cli([
        "doctor", str(GALLERY / "bad_labels" / "trainer_state.json"),
        "--baseline", str(GALLERY / "healthy" / "trainer_state.json"),
    ])
    assert code == 1


def test_preflight_supports_json(tmp_path, capsys):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "hello"}\n{"text": ""}\n', encoding="utf-8")

    code = run_cli(["preflight", str(dataset), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["schema_version"] == 2
    assert payload["reports"][0]["verdict"] == "FAIL"


def test_sarif_written_without_json_flag(tmp_path):
    out = tmp_path / "trainproof.sarif"
    code = run_cli([
        "epoch", str(GALLERY / "lr_hot" / "trainer_state.json"),
        "--format", "hf", "--sarif", str(out),
    ])
    doc = json.loads(out.read_text(encoding="utf-8"))
    run = doc["runs"][0]

    assert code == 1
    assert doc["version"] == "2.1.0"
    assert run["tool"]["driver"]["name"] == "trainproof"
    assert {r["id"] for r in run["tool"]["driver"]["rules"]} == {"TP-DIVERGE", "TP-GRAD-SPIKE"}

    diverge = next(r for r in run["results"] if r["ruleId"] == "TP-DIVERGE")
    assert diverge["level"] == "error"
    assert "Evidence:" in diverge["message"]["text"]
    assert diverge["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]


def test_sarif_rule_index_points_at_its_own_rule(tmp_path):
    out = tmp_path / "trainproof.sarif"
    run_cli([
        "doctor", str(GALLERY), "--sarif", str(out),
    ])
    run = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
    rules = run["tool"]["driver"]["rules"]

    assert run["results"]
    for result in run["results"]:
        assert rules[result["ruleIndex"]]["id"] == result["ruleId"]


def test_envelope_shape_is_stable(capsys):
    run_cli(["epoch", str(GALLERY / "healthy" / "trainer_state.json"), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == 2
    assert payload["trainproof_version"] == trainproof.__version__
    assert payload["error"] is None
    assert set(payload) == {
        "schema_version",
        "trainproof_version",
        "reports",
        "worst_verdict",
        "error",
    }
