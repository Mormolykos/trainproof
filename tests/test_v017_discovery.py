"""A log that cannot be parsed must never disappear (v0.17.0).

`doctor` walks a directory twice: once to decide which files look like logs,
then once to judge them. The second pass already reported anything it could not
read - "could not be parsed and were NOT judged". The first pass did not: a file
that raised during discovery was dropped with `except Exception: pass` and never
became a candidate, so it never reached that note either.

The result was a file visible on disk, absent from the report, and
indistinguishable from one that passed. That is the exact failure NOT-CHECKED
was introduced to prevent, sitting one loop earlier than anyone looked.
"""
import subprocess
import sys
from unittest.mock import patch

import pytest


def _write_log(path, n=6, loss_start=2.0):
    """A log that parses cleanly, so only the patched failure is under test."""
    lines = [
        f'{{"step": {i}, "loss": {loss_start - i * 0.1:.3f}, "learning_rate": 0.001}}'
        for i in range(n)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_unparseable_file_is_reported_not_silently_dropped(tmp_path, capsys):
    good = _write_log(tmp_path / "good.jsonl")
    _write_log(tmp_path / "bad.jsonl")   # readable on disk; made to raise below

    real = None

    def explode(path, *args, **kwargs):
        if str(path).endswith("bad.jsonl"):
            raise RuntimeError("simulated parser failure during discovery")
        return real(path, *args, **kwargs)

    from trainproof import cli

    real = cli.parse_log_with_format_info

    with patch.object(cli, "parse_log_with_format_info", side_effect=explode), \
         patch.object(sys, "argv", ["trainproof", "doctor", str(tmp_path)]), \
         pytest.raises(SystemExit):
        cli.main()

    out = capsys.readouterr().out
    assert "bad.jsonl" in out, (
        "a file that raised during discovery vanished from the report entirely"
    )
    assert "NOT judged" in out
    assert good.name in out or "VERDICT" in out, "the readable log was still judged"


def test_directory_scan_still_reports_readable_logs(tmp_path):
    """The fix must not turn a working directory scan into a wall of notes."""
    _write_log(tmp_path / "a.jsonl")
    _write_log(tmp_path / "b.jsonl")

    res = subprocess.run(
        [sys.executable, "-m", "trainproof", "doctor", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert "could not be parsed" not in res.stdout, res.stdout
    assert res.returncode in (0, 1), res.stdout
