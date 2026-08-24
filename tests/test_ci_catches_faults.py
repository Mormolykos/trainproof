"""Fault injection for the gate itself.

A gate is only worth its runtime if it fails on the thing it exists to catch,
and "it passed" is the same output whether it checked or not. Every test here
breaks something on purpose in a temporary copy and asserts the gate says so.

Nothing in this module touches the real repository.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ci  # noqa: E402


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    """A minimal tree shaped like this project, with the gate pointed at it."""
    package = tmp_path / "src" / ci.PROJECT
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "{ci.PROJECT}"\nversion = "1.2.3"\n\n[tool.ruff]\ntarget-version = "py310"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "PACKAGE", package)
    return tmp_path


def _args(**kwargs):
    return argparse.Namespace(**kwargs)


# ------------------------------------------------------------ version drift


def test_the_gate_passes_on_a_healthy_tree(fake_repo):
    assert ci.cmd_version(_args(expect=None)) == ci.OK


def test_a_version_bumped_in_one_file_only_is_caught(fake_repo):
    """The bug this exists for: `pyproject.toml` bumped, `__init__.py` not.

    The package then reports one version to `pip` and another to
    `--version`, and the mismatch shows up in a user's bug report rather than
    in the release.
    """
    (fake_repo / "pyproject.toml").write_text(
        f'[project]\nname = "{ci.PROJECT}"\nversion = "1.2.4"\n\n[tool.ruff]\n', encoding="utf-8"
    )
    assert ci.cmd_version(_args(expect=None)) == ci.FAIL


def test_a_tag_that_does_not_match_the_source_is_caught(fake_repo):
    assert ci.cmd_version(_args(expect="v9.9.9")) == ci.FAIL
    assert ci.cmd_version(_args(expect="v1.2.3")) == ci.OK
    # A tag written without the `v` is the same tag.
    assert ci.cmd_version(_args(expect="1.2.3")) == ci.OK


# ------------------------------------------------------------- the artifact


def test_a_stale_artifact_is_caught(fake_repo):
    """The release publishes the artifact the gate built. This is the check
    that the download is the right one - a `dist/` left over from a previous
    version would otherwise be uploaded under the new tag."""
    dist = fake_repo / "dist"
    dist.mkdir()
    (dist / f"{ci.PROJECT}-1.2.2-py3-none-any.whl").write_bytes(b"")
    (dist / f"{ci.PROJECT}-1.2.2.tar.gz").write_bytes(b"")
    assert ci.cmd_artifact(_args(dist="dist")) == ci.FAIL


def test_a_missing_sdist_is_caught(fake_repo):
    dist = fake_repo / "dist"
    dist.mkdir()
    (dist / f"{ci.PROJECT}-1.2.3-py3-none-any.whl").write_bytes(b"")
    assert ci.cmd_artifact(_args(dist="dist")) == ci.FAIL


def test_a_complete_artifact_passes(fake_repo):
    dist = fake_repo / "dist"
    dist.mkdir()
    (dist / f"{ci.PROJECT}-1.2.3-py3-none-any.whl").write_bytes(b"")
    (dist / f"{ci.PROJECT}-1.2.3.tar.gz").write_bytes(b"")
    assert ci.cmd_artifact(_args(dist="dist")) == ci.OK


# ---------------------------------------------------------- the attribution


@pytest.fixture()
def fake_git_repo(tmp_path, monkeypatch):
    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "nobody@example.invalid")
    run("config", "user.name", "Nobody")
    (tmp_path / "README.md").write_text("A perfectly ordinary readme.\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "initial commit")
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    return tmp_path, run


def test_a_clean_history_passes(fake_git_repo):
    assert ci.cmd_attribution(_args()) == ci.OK


def test_a_trailer_in_a_commit_message_is_caught(fake_git_repo):
    """This is how the attribution actually arrives: appended to a message by
    a tool, in a commit nobody re-reads."""
    tmp_path, run = fake_git_repo
    (tmp_path / "feature.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "add a feature\n\nCo-Auth" + "ored-By: Someone <nobody@example.invalid>")
    assert ci.cmd_attribution(_args()) == ci.FAIL


def test_a_vendor_name_in_project_metadata_is_caught(fake_git_repo):
    """Prose may name a tool; `pyproject.toml` may not.

    In an authors field a vendor name is an authorship claim and can be
    nothing else, which is why metadata files are scanned for the names
    themselves while prose is scanned only for attribution shapes.
    """
    tmp_path, run = fake_git_repo
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nauthors = [{name = "Cl' + 'aude"}]\n', encoding="utf-8"
    )
    run("add", "-A")
    run("commit", "-q", "-m", "add packaging metadata")
    assert ci.cmd_attribution(_args()) == ci.FAIL


def test_prose_naming_a_tool_is_not_an_authorship_claim(fake_git_repo):
    """The regression that made this gate usable.

    The first version failed on a README paragraph addressed to coding agents
    and on the line in the spec stating this rule. A gate that fires on its own
    rule statement gets switched off, and then nothing is checked at all.
    """
    tmp_path, run = fake_git_repo
    (tmp_path / "README.md").write_text(
        "If you are a coding agent (Cl" + "aude Code, Codex, Cursor) reading this,\n"
        "the corpus is generated with a fixed seed, so ids are stable.\n",
        encoding="utf-8",
    )
    run("add", "-A")
    run("commit", "-q", "-m", "document the agent workflow")
    assert ci.cmd_attribution(_args()) == ci.OK


# ------------------------------------------------- cannot judge is not a fail


def test_an_unreachable_pypi_is_cannot_judge_and_never_a_failure(fake_repo, monkeypatch):
    """The rule this whole gate borrows from the tool it guards.

    A network failure means the check did not run. Reporting that as a failed
    check blocks a release for a reason that has nothing to do with the code -
    and teaches whoever is on the other end to re-run until it goes green,
    which is how a gate stops being read.
    """
    monkeypatch.setattr(ci, "_PYPI", "http://127.0.0.1:9/{name}/json")
    assert ci.cmd_pypi(_args(version=None, timeout=1.0)) == ci.UNJUDGED


def test_a_missing_git_checkout_is_cannot_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    assert ci.cmd_attribution(_args()) == ci.UNJUDGED
