"""Environment preflight tests (v0.15 — envcheck.py).

Every fixture is built from tmp_path. No GPU, no network, no real model.
"""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from trainproof.envcheck import (
    check_checkpoint,
    check_env,
    check_import,
    check_memory,
)


# ------------------------------------------------------------------- imports


def test_import_stdlib_module_succeeds():
    """A stdlib module must import cleanly and return TP-ENV-IMPORT-OK."""
    findings = check_import("json")
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-IMPORT-OK"
    assert findings[0]["level"] == "PASS"


def test_import_nonexistent_module_fails():
    """A module that does not exist must FAIL with evidence naming the error."""
    findings = check_import("no_such_module_xyz_9999")
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-IMPORT-FAIL"
    assert findings[0]["level"] == "FAIL"
    assert "ModuleNotFoundError" in findings[0]["evidence"]


def test_cwd_missing_prevents_subprocess(tmp_path):
    """If the cwd does not exist, no subprocess should run — NOT-CHECKED."""
    bad_dir = str(tmp_path / "does_not_exist")
    findings = check_import("json", cwd=bad_dir)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CWD-MISSING"
    assert findings[0]["level"] == "NOT-CHECKED"


def test_python_unusable_nonexistent_interpreter(tmp_path):
    """A non-existent interpreter path is TP-ENV-PYTHON-UNUSABLE."""
    bad_exe = str(tmp_path / "no_such_python_binary")
    findings = check_import("json", python=bad_exe)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-PYTHON-UNUSABLE"
    assert findings[0]["level"] == "NOT-CHECKED"


# --------------------------------------------------------------- checkpoints


def test_checkpoint_empty_file(tmp_path):
    """A zero-byte file is TP-ENV-CKPT-EMPTY."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    findings = check_checkpoint(ckpt)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CKPT-EMPTY"
    assert findings[0]["level"] == "FAIL"


def _make_valid_torch_zip(path: Path) -> None:
    """Create a minimal valid torch-style ZIP with data.pkl and a data/0 storage."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", b"\x80\x02.")  # minimal pickle
        z.writestr("archive/data/0", b"\x00" * 16)     # fake tensor storage
    path.write_bytes(buf.getvalue())


def test_checkpoint_valid_zip(tmp_path):
    """A ZIP containing data.pkl and data/0 is TP-ENV-CKPT-OK."""
    ckpt = tmp_path / "model.pt"
    _make_valid_torch_zip(ckpt)
    findings = check_checkpoint(ckpt)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CKPT-OK"
    assert findings[0]["level"] == "PASS"


def test_checkpoint_truncated_zip(tmp_path):
    """A ZIP truncated to 60% is TP-ENV-CKPT-TRUNCATED."""
    ckpt = tmp_path / "model.pt"
    _make_valid_torch_zip(ckpt)
    full = ckpt.read_bytes()
    ckpt.write_bytes(full[: int(len(full) * 0.6)])
    findings = check_checkpoint(ckpt)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CKPT-TRUNCATED"
    assert findings[0]["level"] == "FAIL"


def test_checkpoint_legacy_pickle(tmp_path):
    """A file starting with 0x80 is a legacy pickle — NOT-CHECKED, not FAIL.

    Refusing to unpickle is correct behaviour, not an error.
    """
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x80\x02" + b"\x00" * 100)
    findings = check_checkpoint(ckpt)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CKPT-LEGACY"
    assert findings[0]["level"] == "NOT-CHECKED"


def test_checkpoint_text_file(tmp_path):
    """A plain text file is neither a ZIP nor a pickle — TP-ENV-CKPT-UNREADABLE."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("this is not a checkpoint", encoding="utf-8")
    findings = check_checkpoint(ckpt)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-CKPT-UNREADABLE"
    assert findings[0]["level"] == "FAIL"


# -------------------------------------------------------------------- memory


def test_memory_insufficient():
    """Requesting an absurd amount of RAM must FAIL."""
    findings = check_memory(required_gb=10**6)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-ENV-MEM-INSUFFICIENT"
    assert findings[0]["level"] == "FAIL"


def test_memory_info_no_requirement():
    """With no requirement, memory is measured and reported as INFO."""
    findings = check_memory()
    assert len(findings) == 1
    # On platforms where memory can be read, we get INFO; otherwise NOT-CHECKED
    assert findings[0]["id"] in ("TP-ENV-MEM-INFO", "TP-ENV-MEM-UNKNOWN")
    if findings[0]["id"] == "TP-ENV-MEM-INFO":
        assert findings[0]["level"] == "INFO"


# --------------------------------------------------------------- check_env


def test_check_env_no_arguments():
    """With no module, no checkpoint, no output-dir, the verdict is NOT-CHECKED
    and every optional check group is listed as skipped with a reason."""
    report = check_env()
    assert report["verdict"] == "NOT-CHECKED"
    skipped = report["checks"]["skipped"]
    assert "import" in skipped
    assert "checkpoint" in skipped
    assert "disk" in skipped
    # Each skip reason should explain why
    for group, reason in skipped.items():
        assert reason, f"{group} skip has no reason"


# ----------------------------------------------------------- exit codes e2e


@pytest.mark.parametrize(
    "args,expected_exit",
    [
        # FAIL: import of a nonexistent module → exit 1
        (["env", "--module", "no_such_module_xyz_9999"], 1),
        # NOT-CHECKED: no judgeable arguments → exit 2
        (["env"], 2),
        # PASS: import of a stdlib module → exit 0
        (["env", "--module", "json"], 0),
    ],
    ids=["fail-exit-1", "not-checked-exit-2", "pass-exit-0"],
)
def test_exit_codes_end_to_end(args, expected_exit, tmp_path):
    """Exit codes via subprocess match the contract: FAIL→1, NOT-CHECKED→2, PASS→0."""
    result = subprocess.run(
        [sys.executable, "-m", "trainproof", *args],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == expected_exit, (
        f"Expected exit {expected_exit}, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ------------------------------------------------------- cwd forwarding (regression)
#
# check_env grew a `cwd` parameter because the CLI could not otherwise reach
# check_import's. The first wiring worked around that by calling check_import
# separately and re-deriving the verdict in cli.py -- a second copy of the
# verdict rule, on a branch only taken when --cwd was passed, which no test
# exercised. These tests pin the parameter so that workaround cannot return.

def test_check_env_forwards_cwd_to_import(tmp_path):
    """A bad cwd must surface as TP-ENV-CWD-MISSING through check_env."""
    report = check_env(module="json", cwd=str(tmp_path / "nope"))
    ids = {f["id"] for f in report["findings"]}
    assert "TP-ENV-CWD-MISSING" in ids
    assert "TP-ENV-IMPORT-OK" not in ids, "cwd was ignored and the import ran anyway"
    assert report["verdict"] == "NOT-CHECKED"


def test_check_env_valid_cwd_still_imports(tmp_path):
    report = check_env(module="json", cwd=str(tmp_path))
    ids = {f["id"] for f in report["findings"]}
    assert "TP-ENV-IMPORT-OK" in ids
    assert report["verdict"] == "PASS"
    assert "import" in report["checks"]["ran"]


def test_cli_cwd_flag_reaches_the_check(tmp_path):
    """End-to-end: --cwd must change the verdict, not be silently dropped."""
    res = subprocess.run(
        [sys.executable, "-m", "trainproof", "env",
         "--module", "json", "--cwd", str(tmp_path / "missing"), "--json"],
        capture_output=True, text=True,
    )
    assert res.returncode == 2, res.stdout
    assert "TP-ENV-CWD-MISSING" in res.stdout
