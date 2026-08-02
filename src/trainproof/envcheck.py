"""Environment preflight - can this machine start this run at all?

Every check trainproof shipped before 0.15.0 reads a log, which means the run
has already started and the GPU hours are already spent. The failures that cost
the most never reach a log:

  - the training stack does not import (a library moved a symbol between
    versions, or a file in the tree was overwritten)
  - the checkpoint cannot be deserialised
  - the first batch exhausts system RAM and freezes the desktop

All three happened on 2026-08-01 to the author, in one day, on a run that never
logged a single step. This module is the answer to that day.

Design constraints, in order:

1. **Never execute the user's code in-process.** Import checks run in a
   subprocess, so a segfaulting extension module or an `os._exit` in a library
   cannot take trainproof down with it.
2. **Never unpickle a checkpoint.** `torch.load` executes arbitrary code by
   design, which is why torch 2.6 flipped `weights_only` to True. A linter that
   must run the thing it is inspecting is not a safety tool. Checkpoints are
   read as the ZIP archives they are.
3. **No dependencies.** zipfile, ctypes, struct, subprocess - all stdlib.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import rules

# --------------------------------------------------------------------- memory


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory_gb() -> tuple[float, float] | None:
    """(total_gb, available_gb), or None where it cannot be determined.

    None is a real answer and must not be reported as zero: an unmeasurable
    machine is not a machine with no memory.
    """
    if sys.platform == "win32":
        try:
            s = _MEMORYSTATUSEX()
            s.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return None
            return s.ullTotalPhys / 1024**3, s.ullAvailPhys / 1024**3
        except Exception:
            return None

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            vals = {}
            for line in meminfo.read_text().splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    vals[key] = int(parts[0]) / 1024**2  # kB -> GiB
            total = vals.get("MemTotal")
            avail = vals.get("MemAvailable", vals.get("MemFree"))
            if total is not None and avail is not None:
                return total, avail
        except Exception:
            return None
    return None


def check_memory(required_gb: float | None = None) -> list[dict]:
    mem = system_memory_gb()
    if mem is None:
        return [{
            "id": "TP-ENV-MEM-UNKNOWN", "level": "NOT-CHECKED",
            "message": "System memory could not be determined on this platform.",
            "evidence": f"platform={sys.platform}",
        }]

    total, avail = mem
    if required_gb is None:
        return [{
            "id": "TP-ENV-MEM-INFO", "level": "INFO",
            "message": "System memory measured.",
            "evidence": f"{avail:.1f} GB available of {total:.1f} GB total. "
                        f"Pass --required-gb to judge it.",
        }]

    if avail < required_gb:
        return [{
            "id": "TP-ENV-MEM-INSUFFICIENT", "level": "FAIL",
            "message": "Less system RAM is available than this run declares it needs.",
            "evidence": f"{avail:.1f} GB available, {required_gb:.1f} GB required "
                        f"({total:.1f} GB total). On Windows the driver spills to "
                        f"system RAM instead of raising a clean OOM, so this "
                        f"freezes the desktop rather than failing the run.",
        }]

    margin = avail - required_gb
    if margin < rules.MIN_FREE_RAM_MARGIN_GB:
        return [{
            "id": "TP-ENV-MEM-TIGHT", "level": "WARN",
            "message": "System RAM headroom is thin.",
            "evidence": f"{avail:.1f} GB available, {required_gb:.1f} GB required, "
                        f"margin {margin:.1f} GB < {rules.MIN_FREE_RAM_MARGIN_GB} GB.",
        }]
    return [{
        "id": "TP-ENV-MEM-OK", "level": "PASS",
        "message": "System RAM headroom is sufficient.",
        "evidence": f"{avail:.1f} GB available, {required_gb:.1f} GB required, "
                    f"margin {margin:.1f} GB.",
    }]


# --------------------------------------------------------------------- imports

_IMPORT_PROBE = (
    "import importlib,sys\n"
    "try:\n"
    "    importlib.import_module(sys.argv[1])\n"
    "except BaseException as e:\n"
    "    import traceback\n"
    "    tb = traceback.extract_tb(e.__traceback__)\n"
    "    frame = tb[-1] if tb else None\n"
    "    sys.stderr.write('%s|%s|%s|%s\\n' % (\n"
    "        type(e).__name__, e,\n"
    "        frame.filename if frame else '', frame.lineno if frame else ''))\n"
    "    raise SystemExit(1)\n"
    "raise SystemExit(0)\n"
)


def check_import(module: str, python: str | None = None, timeout: float = 120.0,
                 cwd: str | None = None) -> list[dict]:
    """Import `module` in a subprocess and report why it failed.

    Out-of-process because the failure modes here are violent: a checkpoint
    unpickler segfaulting, a CUDA extension aborting, a library calling
    os._exit during import. In-process, any of those kills trainproof and the
    user learns nothing.

    `cwd` matters more than it looks. Editable installs and source checkouts
    resolve relative to the working directory, so probing from somewhere else
    reports "No module named X" for a package that imports perfectly where
    training actually launches - a false FAIL, and the worst kind, because it
    blames the environment for the linter's own mistake.
    """
    exe = python or sys.executable
    if cwd is not None and not Path(cwd).is_dir():
        return [{
            "id": "TP-ENV-CWD-MISSING", "level": "NOT-CHECKED",
            "message": "The working directory to probe from does not exist.",
            "evidence": str(cwd),
        }]
    try:
        proc = subprocess.run(
            [exe, "-c", _IMPORT_PROBE, module],
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return [{
            "id": "TP-ENV-IMPORT-TIMEOUT", "level": "FAIL",
            "message": f"Importing '{module}' did not finish.",
            "evidence": f"No result after {timeout:.0f}s. An import that hangs "
                        f"hangs the training launch too.",
        }]
    except OSError as e:
        return [{
            "id": "TP-ENV-PYTHON-UNUSABLE", "level": "NOT-CHECKED",
            "message": "The interpreter to probe with could not be run.",
            "evidence": f"{exe}: {e}",
        }]

    if proc.returncode == 0:
        return [{
            "id": "TP-ENV-IMPORT-OK", "level": "PASS",
            "message": f"'{module}' imports cleanly.",
            "evidence": f"interpreter {exe}",
        }]

    detail = (proc.stderr or "").strip().splitlines()
    exc_type = exc_msg = where = ""
    if detail:
        parts = detail[-1].split("|")
        if len(parts) >= 4:
            exc_type, exc_msg, fname, lineno = parts[0], parts[1], parts[2], parts[3]
            where = f"{fname}:{lineno}" if fname else ""

    # A crash with no Python exception is the segfault case - torch.load on an
    # older checkpoint under torch>=2.6 does exactly this, and reports nothing.
    if not exc_type:
        return [{
            "id": "TP-ENV-IMPORT-CRASH", "level": "FAIL",
            "message": f"Importing '{module}' crashed the interpreter.",
            "evidence": f"exit code {proc.returncode} with no Python exception - "
                        f"a native crash (segfault / access violation), not an "
                        f"ImportError. Re-run with `python -X faulthandler` to "
                        f"locate it.",
        }]

    return [{
        "id": "TP-ENV-IMPORT-FAIL", "level": "FAIL",
        "message": f"'{module}' cannot be imported - this run cannot start.",
        "evidence": f"{exc_type}: {exc_msg}" + (f"  (raised at {where})" if where else ""),
    }]


# ----------------------------------------------------------------- checkpoints


def check_checkpoint(path: str | Path) -> list[dict]:
    """Inspect a .pt/.pth/.ckpt without deserialising it.

    torch.save writes a ZIP: data.pkl describing the tensors, data/N holding raw
    storage bytes. Everything below is read from the archive directory alone -
    the pickle is never executed, so a hostile or corrupt checkpoint cannot run
    code here.
    """
    p = Path(path)
    if not p.exists():
        return [{
            "id": "TP-ENV-CKPT-MISSING", "level": "FAIL",
            "message": "Checkpoint does not exist.",
            "evidence": str(p),
        }]

    size_gb = p.stat().st_size / 1024**3
    if p.stat().st_size == 0:
        return [{
            "id": "TP-ENV-CKPT-EMPTY", "level": "FAIL",
            "message": "Checkpoint is a zero-byte file.",
            "evidence": f"{p} - a save that was interrupted before writing.",
        }]

    if not zipfile.is_zipfile(p):
        with open(p, "rb") as f:
            head = f.read(4)
        # A killed save leaves the local file header intact but no end-of-central
        # -directory record at the tail, so is_zipfile() says False. Checking the
        # magic separately distinguishes "the save was interrupted" from "this is
        # not a checkpoint at all" - the difference between resuming from the
        # previous checkpoint and hunting for a file that was never written.
        if head == b"PK\x03\x04":
            return [{
                "id": "TP-ENV-CKPT-TRUNCATED", "level": "FAIL",
                "message": "Checkpoint is a torch archive whose save never finished.",
                "evidence": f"{p} ({size_gb:.2f} GB): ZIP header present but the "
                            f"central directory is missing - the file was cut off "
                            f"mid-write. Resume from the previous checkpoint.",
            }]
        if head[:1] == b"\x80":  # raw pickle protocol marker
            return [{
                "id": "TP-ENV-CKPT-LEGACY", "level": "NOT-CHECKED",
                "message": "Checkpoint uses the pre-1.6 torch format (a bare pickle).",
                "evidence": f"{p} ({size_gb:.2f} GB). Inspecting it would require "
                            f"unpickling, which executes arbitrary code, so "
                            f"trainproof refuses rather than guess.",
            }]
        return [{
            "id": "TP-ENV-CKPT-UNREADABLE", "level": "FAIL",
            "message": "Checkpoint is neither a torch ZIP archive nor a pickle.",
            "evidence": f"{p} ({size_gb:.2f} GB), first bytes {head!r}.",
        }]

    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            names = z.namelist()
            storages = [n for n in names if "/data/" in n]
            has_pkl = any(n.endswith("data.pkl") for n in names)
    except zipfile.BadZipFile as e:
        return [{
            "id": "TP-ENV-CKPT-TRUNCATED", "level": "FAIL",
            "message": "Checkpoint archive is incomplete - the save did not finish.",
            "evidence": f"{p} ({size_gb:.2f} GB): {e}",
        }]

    if bad is not None:
        return [{
            "id": "TP-ENV-CKPT-CORRUPT", "level": "FAIL",
            "message": "Checkpoint archive fails its own CRC check.",
            "evidence": f"first bad entry: {bad}",
        }]

    if not has_pkl:
        return [{
            "id": "TP-ENV-CKPT-UNREADABLE", "level": "FAIL",
            "message": "Checkpoint archive contains no data.pkl - not a torch checkpoint.",
            "evidence": f"{p}: {len(names)} entries, none named data.pkl.",
        }]

    return [{
        "id": "TP-ENV-CKPT-OK", "level": "PASS",
        "message": "Checkpoint is a complete, readable torch archive.",
        "evidence": f"{size_gb:.2f} GB, {len(storages)} tensor storages, CRC intact. "
                    f"Read without unpickling - contents not executed.",
    }]


# ----------------------------------------------------------------------- disk


def check_disk(output_dir: str | Path, checkpoint_gb: float | None = None,
               keep: int = 1) -> list[dict]:
    p = Path(output_dir)
    probe = p if p.exists() else p.parent
    if not probe.exists():
        return [{
            "id": "TP-ENV-DISK-UNKNOWN", "level": "NOT-CHECKED",
            "message": "Output directory does not exist and neither does its parent.",
            "evidence": str(p),
        }]

    free_gb = shutil.disk_usage(probe).free / 1024**3
    if checkpoint_gb is None:
        return [{
            "id": "TP-ENV-DISK-INFO", "level": "INFO",
            "message": "Free disk space measured.",
            "evidence": f"{free_gb:.1f} GB free at {probe}. Pass --checkpoint-gb to judge it.",
        }]

    needed = checkpoint_gb * max(keep, 1)
    if free_gb < needed:
        return [{
            "id": "TP-ENV-DISK-INSUFFICIENT", "level": "FAIL",
            "message": "Not enough disk space for the checkpoints this run will write.",
            "evidence": f"{free_gb:.1f} GB free, {needed:.1f} GB needed "
                        f"({checkpoint_gb:.1f} GB x {keep} kept).",
        }]
    return [{
        "id": "TP-ENV-DISK-OK", "level": "PASS",
        "message": "Disk space is sufficient for the declared checkpoints.",
        "evidence": f"{free_gb:.1f} GB free, {needed:.1f} GB needed.",
    }]


# ------------------------------------------------------------------ aggregate


def check_env(module: str | None = None, checkpoint: str | None = None,
              required_gb: float | None = None, output_dir: str | None = None,
              checkpoint_gb: float | None = None, keep: int = 1,
              python: str | None = None, cwd: str | None = None) -> dict:
    """Run every requested environment check and return one report.

    This is the only place the verdict is derived from the findings. A caller
    that reconstructs it - to splice in a check this function did not run -
    creates a second copy of the rule that will drift from this one.
    """
    findings: list[dict] = []
    ran: list[str] = []
    skipped: dict[str, str] = {}

    if module:
        findings += check_import(module, python=python, cwd=cwd)
        ran.append("import")
    else:
        skipped["import"] = "no --module given"

    if checkpoint:
        findings += check_checkpoint(checkpoint)
        ran.append("checkpoint")
    else:
        skipped["checkpoint"] = "no --checkpoint given"

    findings += check_memory(required_gb)
    ran.append("memory")

    if output_dir:
        findings += check_disk(output_dir, checkpoint_gb, keep)
        ran.append("disk")
    else:
        skipped["disk"] = "no --output-dir given"

    levels = {f["level"] for f in findings}
    if "FAIL" in levels:
        verdict = "FAIL"
    elif "WARN" in levels:
        verdict = "WARN"
    elif levels <= {"NOT-CHECKED", "INFO"}:
        verdict = "NOT-CHECKED"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "findings": findings,
        "checks": {"ran": sorted(ran), "skipped": skipped},
    }
