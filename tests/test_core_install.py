"""The core must install and run without numpy, soundfile or ttsproof.

Those three are used by exactly one module - `speech/data.py` - and `cli.py`
used to import it at module scope. The result: `pip install trainproof` pulled
an audio stack, and `trainproof epoch` on a text log would not start without
libsndfile, the C library behind soundfile.

These tests block the three modules from the import system and assert the core
still works, so a stray top-level import cannot quietly reintroduce the coupling.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SPEECH_DEPS = {"numpy", "soundfile", "ttsproof"}
FIXTURES = Path(__file__).parent / "fixtures"

CORE_MODULES = [
    "trainproof",
    "trainproof.cli",
    "trainproof.epoch",
    "trainproof.rules",
    "trainproof.adapters",
    "trainproof.tfevents",
    "trainproof.report",
    "trainproof.compare",
    "trainproof.envcheck",
    "trainproof.preflight",
    "trainproof.sarif",
    "trainproof.speech.tokenizer",
]

BLOCKER = """
import sys, importlib.abc
BLOCK = {"numpy", "soundfile", "ttsproof"}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"{name} is not part of a core install")
        return None
sys.meta_path.insert(0, Blocker())
"""


def run_without_speech_deps(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter where the speech deps cannot import."""
    root = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + f"sys.path.insert(0, {str(root)!r})\n" + body],
        capture_output=True, text=True, check=False)


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports_without_speech_deps(module):
    out = run_without_speech_deps(f"import {module}\nprint('ok')")
    assert out.returncode == 0, f"{module} pulled in a speech dependency:\n{out.stderr}"


def test_core_cli_runs_without_speech_deps():
    """The whole point: linting a text log must not need libsndfile."""
    out = run_without_speech_deps(
        "from trainproof.cli import main\n"
        f"import sys; sys.argv = ['trainproof', 'epoch', {str(FIXTURES / 'healthy.jsonl')!r}]\n"
        "try:\n    main()\nexcept SystemExit as e:\n    print('exit', e.code)"
    )
    assert out.returncode == 0, out.stderr
    assert "exit" in out.stdout


def test_speech_data_says_what_to_install_when_the_extra_is_absent():
    out = run_without_speech_deps(
        "try:\n"
        "    from trainproof.speech.data import check_data\n"
        "except ImportError as e:\n"
        "    print('MSG:', e)"
    )
    assert "trainproof[speech]" in out.stdout, out.stdout + out.stderr


def test_speech_pack_still_works_when_the_extra_is_installed():
    """Guards the other direction: laziness must not have broken the pack."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    pytest.importorskip("ttsproof")

    from trainproof.speech import check_data, check_tokenizer
    assert callable(check_data)
    assert callable(check_tokenizer)

    mod = importlib.import_module("trainproof.speech.data")
    assert hasattr(mod, "check_data")


def test_lazy_attribute_does_not_hide_real_typos():
    import trainproof.speech as pack
    with pytest.raises(AttributeError):
        _ = pack.check_nothing_of_the_sort
