#!/usr/bin/env python3
"""CI helpers for trainproof, and a local runner for the workflow itself.

Every gate the GitHub workflow enforces lives here as a subcommand, so the
workflow file contains `python scripts/ci.py <gate>` and nothing clever. Two
consequences, both deliberate:

  * The gate can be run on a laptop before pushing, executing the same code the
    runner will execute - not a re-implementation of it.
  * `ci.py run` reads .github/workflows/ci.yml and executes the `run:` steps it
    finds there. The list of things to check is DERIVED from the workflow, never
    restated here. A step added to the workflow is picked up with no edit to
    this file; a step that stops working locally is visible before it is
    visible to a reviewer.

Exit codes follow trainproof's own contract:

    0  the gate passed
    1  the gate failed - there is something to fix
    2  the gate could not be evaluated (network down, missing tool). This is
       never reported as a failure, because "I could not check" and "I checked
       and it is broken" are different facts.

Standard library only, except `ci.py run`, which needs PyYAML (a `dev` extra).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "trainproof"
PACKAGE = ROOT / "src" / PROJECT

OK, FAIL, UNJUDGED = 0, 1, 2


def _say(status: str, message: str) -> None:
    print(f"{status:<4} {message}")


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

# Vendor names are assembled from fragments at import time, never written as
# literals. This scanner reads every tracked file including its own source, so
# a literal here would make the gate fail on itself - and the only other way
# out would be an exemption for this file, which is precisely the file an
# automated edit would land in. Fragments keep the scan exemption-free.
_VENDORS = (
    "cl" + "aude",
    "anthro" + "pic",
    "antigra" + "vity",
    "gem" + "ini",
    "cop" + "ilot",
)

# What is actually forbidden is an AUTHORSHIP CLAIM, not the mention of a
# product. The first version of this gate scanned every tracked file for the
# vendor names and failed on two legitimate sentences: a README paragraph
# addressed to coding agents by name, and the line in SPEC.md that states this
# very rule. A gate that fires on its own rule statement gets switched off.
#
# So: prose is scanned for attribution SHAPES, and vendor names are scanned for
# only in the metadata files where a vendor name cannot be anything but an
# author claim.
_ATTRIBUTION = (
    r"co-auth" + r"ored[-\s]?by",
    # "generated with" alone is ordinary English - it matched
    # "The corpus is generated with a fixed seed" in ttsproof/cases.py. The
    # footer this pattern is written for names a vendor, so require one nearby.
    r"generated\s+with\b[^\n]{0,40}?\b(?:" + "|".join(_VENDORS) + r")\b",
    r"\bwritten\s+by\s+(?:" + "|".join(_VENDORS) + r")\b",
    r"\b(?:" + "|".join(_VENDORS) + r")\s+(?:code\s+)?(?:wrote|authored|generated)\b",
    r"\bassisted\s+by\s+(?:" + "|".join(_VENDORS) + r")\b",
)
_METADATA_FILES = ("pyproject.toml", "setup.cfg", "setup.py", ".zenodo.json", "CITATION.cff", "LICENSE")
_COMMIT_DEPTH = 20


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def cmd_attribution(_: argparse.Namespace) -> int:
    if _git("rev-parse", "--git-dir").returncode != 0:
        _say("SKIP", "not a git checkout - attribution scan cannot run")
        return UNJUDGED

    hits: list[str] = []

    # -I skips binary files; -i is case-insensitive; -E is a regular
    # expression; HEAD scans the committed tree rather than the working copy,
    # which is what a reviewer and a released sdist will actually see.
    for pattern in _ATTRIBUTION:
        found = _git("grep", "-I", "-i", "-n", "-E", "-e", pattern, "HEAD")
        if found.returncode == 0 and found.stdout.strip():
            hits.extend(f"file: {line}" for line in found.stdout.strip().splitlines())

    for vendor in _VENDORS:
        for name in _METADATA_FILES:
            found = _git("grep", "-I", "-i", "-n", "-e", vendor, "HEAD", "--", name)
            if found.returncode == 0 and found.stdout.strip():
                hits.extend(f"metadata: {line}" for line in found.stdout.strip().splitlines())

    # A commit message is not prose about the tool, so it gets the whole vendor
    # list: a trailer is the way this attribution actually arrives.
    log = _git("log", f"-{_COMMIT_DEPTH}", "--format=%h %s%n%b")
    if log.returncode == 0:
        for line in log.stdout.splitlines():
            low = line.lower()
            if any(vendor in low for vendor in _VENDORS) or "co-auth" + "ored" in low:
                hits.append(f"commit: {line.strip()}")

    if hits:
        _say("FAIL", f"attribution scan found {len(hits)} occurrence(s):")
        for hit in hits[:40]:
            print(f"       {hit}")
        if len(hits) > 40:
            print(f"       ... and {len(hits) - 40} more")
        return FAIL

    _say(
        "OK",
        f"attribution scan clean ({len(_ATTRIBUTION)} shapes in the tracked tree, "
        f"{len(_VENDORS)} vendors in {len(_METADATA_FILES)} metadata files, {_COMMIT_DEPTH} commits)",
    )
    return OK


# --------------------------------------------------------------------------
# version consistency
# --------------------------------------------------------------------------


def project_version() -> str:
    """Read `[project] version` from pyproject.toml.

    `tomllib` is 3.11+, and this project supports 3.10 - a script in the
    repository that cannot run on the repository's own lowest supported
    interpreter is a defect, so the 3.10 path is a scoped regex rather than a
    dependency. Scoped, because `[tool.ruff] target-version` is also a version
    key and an unanchored pattern happily returns it.
    """
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return str(tomllib.loads(source)["project"]["version"])

    section = re.search(r"^\[project\]\s*$(.*?)(?=^\[)", source, re.MULTILINE | re.DOTALL)
    if not section:
        raise SystemExit("no [project] table in pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE)
    if not match:
        raise SystemExit("no version in the [project] table of pyproject.toml")
    return match.group(1)


def package_version() -> str:
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"no __version__ in {PACKAGE / '__init__.py'}")
    return match.group(1)


def cmd_version(args: argparse.Namespace) -> int:
    pyproject, package = project_version(), package_version()
    if pyproject != package:
        _say("FAIL", f"version mismatch: pyproject={pyproject} __init__={package}")
        return FAIL

    if args.expect:
        expected = args.expect[1:] if args.expect.startswith("v") else args.expect
        if expected != pyproject:
            _say("FAIL", f"tag {args.expect} does not match project version {pyproject}")
            return FAIL
        _say("OK", f"version {pyproject} consistent with tag {args.expect}")
        return OK

    _say("OK", f"version {pyproject} consistent (pyproject == __init__)")
    return OK


# --------------------------------------------------------------------------
# PyPI availability
# --------------------------------------------------------------------------

_PYPI = "https://pypi.org/pypi/{name}/json"


def cmd_pypi(args: argparse.Namespace) -> int:
    version = args.version or project_version()
    try:
        with urllib.request.urlopen(_PYPI.format(name=PROJECT), timeout=args.timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            _say("OK", f"{PROJECT} is not on PyPI yet - {version} would be the first release")
            return OK
        _say("SKIP", f"PyPI returned HTTP {exc.code} - cannot judge availability")
        return UNJUDGED
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _say("SKIP", f"PyPI unreachable ({exc}) - cannot judge availability")
        return UNJUDGED

    released = sorted(data.get("releases", {}))
    if version in released:
        _say("FAIL", f"{PROJECT} {version} is ALREADY on PyPI - this release cannot be uploaded")
        print("       PyPI never allows a version to be replaced, not even after a delete.")
        print("       The procedure is yank + patch release; see docs/adr/002 'Rollback'.")
        print(f"       Latest on PyPI: {data['info']['version']} ({len(released)} releases)")
        return FAIL

    _say("OK", f"{PROJECT} {version} is free on PyPI (latest published: {data['info']['version']})")
    return OK


# --------------------------------------------------------------------------
# published exit-code contract
# --------------------------------------------------------------------------

# README.md and CONTRACTS.md promise three exit codes. Unit tests cover the
# functions; this covers the promise, through the console script, on the
# fixtures shipped in the repository.
_CONTRACT = [
    ("examples/gallery/healthy/trainer_state.json", 0, "a healthy run passes"),
    ("examples/gallery/fp16_nan/trainer_state.json", 1, "a NaN run is a FAIL verdict"),
    ("examples/gallery/lr_zero/trainer_state.json", 1, "a dead-LR run is a FAIL verdict"),
    ("examples/gallery/does-not-exist.json", 2, "an unreadable log is 'cannot judge', never a FAIL"),
]


def cmd_contract(args: argparse.Namespace) -> int:
    executable = args.executable or sys.executable
    failures = 0
    for relative, expected, description in _CONTRACT:
        target = ROOT / relative
        result = subprocess.run(
            [executable, "-m", "trainproof.cli", "doctor", str(target)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != expected:
            _say("FAIL", f"exit {result.returncode}, expected {expected}: {description}")
            failures += 1
        else:
            _say("OK", f"exit {expected}: {description}")
    if failures:
        _say("FAIL", f"{failures} of {len(_CONTRACT)} exit-code promises broken")
        return FAIL
    _say("OK", f"all {len(_CONTRACT)} documented exit codes hold")
    return OK


# --------------------------------------------------------------------------
# installed-artifact smoke test
# --------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    """Check the INSTALLED distribution, not the source tree.

    Run this against an interpreter that has the built wheel installed and no
    source directory on its path. It catches the packaging faults a source-tree
    test run cannot see: a module left out of the wheel, a console script that
    does not resolve, a version that disagrees with the metadata.
    """
    executable = args.executable or sys.executable
    expected = project_version()

    probe = (
        "import importlib.metadata as md, trainproof, pathlib, sys;"
        "print(trainproof.__version__);"
        "print(md.version('trainproof'));"
        "print(pathlib.Path(trainproof.__file__).parent)"
    )
    result = subprocess.run([executable, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        _say("FAIL", f"installed package does not import:\n{result.stderr.strip()}")
        return FAIL

    dunder, metadata, location = result.stdout.strip().splitlines()
    if dunder != expected or metadata != expected:
        _say("FAIL", f"installed version {dunder}/{metadata} != project version {expected}")
        return FAIL
    if str(ROOT / "src") in location:
        _say("FAIL", f"imported from the source tree ({location}) - this is not testing the artifact")
        return FAIL

    cli = subprocess.run([executable, "-m", "trainproof.cli", "--version"], capture_output=True, text=True)
    if cli.returncode != 0 or expected not in cli.stdout:
        _say("FAIL", f"console entry point broken: exit {cli.returncode} out={cli.stdout.strip()!r}")
        return FAIL

    _say("OK", f"installed {PROJECT} {expected} imports from {location} and its CLI answers")
    return OK


def cmd_noskips(args: argparse.Namespace) -> int:
    """In the environment that has every extra, nothing may be skipped.

    A skip is how a test stops running without anyone deciding it should. On a
    core install skipping is correct and deliberate; in the job that installs
    every extra there is nothing left to skip for, so a skip there means a test
    that executes in no environment at all.
    """
    executable = args.executable or sys.executable
    result = subprocess.run(
        [executable, "-m", "pytest", "-q", "-rs"], cwd=ROOT, capture_output=True, text=True
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0:
        _say("FAIL", f"the test suite did not pass: {summary}")
        return FAIL

    skipped = re.search(r"(\d+) skipped", summary)
    if skipped:
        _say("FAIL", f"{skipped.group(1)} test(s) skipped with every extra installed")
        for line in result.stdout.splitlines():
            if "SKIPPED" in line:
                print(f"       {line.strip()}")
        return FAIL

    _say("OK", f"no skips with every extra installed: {summary}")
    return OK


def cmd_wheelcheck(args: argparse.Namespace) -> int:
    """Build a throwaway environment, install the wheel, and smoke it there.

    Written as a subcommand rather than three shell lines in the workflow so
    that it runs identically on the Linux runner and on a Windows laptop -
    `venv/bin/python` and `venv\\Scripts\\python.exe` is the sort of difference
    that makes a "just run CI locally" instruction quietly untrue.
    """
    wheels = sorted((ROOT / args.dist).glob("*.whl"))
    if not wheels:
        _say("FAIL", f"no wheel in {ROOT / args.dist}")
        return FAIL
    wheel = wheels[-1]

    venv = ROOT / ".ci-local" / "artifact"
    if venv.exists():
        shutil.rmtree(venv)
    subprocess.run([sys.executable, "-m", "venv", venv.as_posix()], check=True, capture_output=True)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", str(wheel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        _say("FAIL", f"the wheel does not install:\n{install.stderr.strip()}")
        return FAIL

    _say("OK", f"installed {wheel.name} into a clean environment")
    args.executable = str(python)
    return cmd_smoke(args)


# --------------------------------------------------------------------------
# artifact identity
# --------------------------------------------------------------------------


def cmd_artifact(args: argparse.Namespace) -> int:
    """The files about to be uploaded are the ones this tag describes.

    The release job downloads the distributions the gate built rather than
    rebuilding them, so this is the step that proves the download is what it
    should be: right version, both formats, nothing extra.
    """
    version = project_version()
    directory = ROOT / args.dist
    if not directory.is_dir():
        _say("FAIL", f"{directory} does not exist - nothing was downloaded")
        return FAIL

    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    if not files:
        _say("FAIL", f"{directory} is empty")
        return FAIL

    problems = [name for name in files if version not in name]
    if problems:
        _say("FAIL", f"artifact does not carry version {version}: {problems}")
        return FAIL
    if not any(name.endswith(".whl") for name in files):
        _say("FAIL", "no wheel in the artifact")
        return FAIL
    if not any(name.endswith(".tar.gz") for name in files):
        _say("FAIL", "no sdist in the artifact")
        return FAIL

    _say("OK", f"{len(files)} artifact(s) for {version}: {', '.join(files)}")
    return OK


# --------------------------------------------------------------------------
# local workflow runner
# --------------------------------------------------------------------------

_EXPRESSION = re.compile(r"\$\{\{\s*([^}]+?)\s*\}\}")


def _load_workflow(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        print("PyYAML is required for `ci.py run`:  pip install -e \".[dev]\"", file=sys.stderr)
        raise SystemExit(UNJUDGED) from None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _substitute(text: str, context: dict[str, str]) -> tuple[str, list[str]]:
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return context[key]
        unresolved.append(key)
        return match.group(0)

    return _EXPRESSION.sub(replace, text), unresolved


def _matrix_context(job: dict, python: str) -> dict[str, str]:
    context = {"matrix.python-version": python, "matrix.os": "local", "runner.os": os.name}
    matrix = job.get("strategy", {}).get("matrix", {})
    for key, values in matrix.items():
        if isinstance(values, list) and values and f"matrix.{key}" not in context:
            context[f"matrix.{key}"] = str(values[0])
    return context


def _ensure_env(job: str, python: str, fresh: bool) -> Path:
    """A throwaway environment per JOB, so a workflow's `pip install` never
    lands in the interpreter the developer is using.

    Per job, not per run, and that is the point. GitHub gives every job a clean
    runner: `test` installs the core package and proves the dataset tests skip,
    while `speech` installs the extra and proves they pass. Share one local
    environment between them and the second job silently invalidates the first
    - the local run would go green on a repository where CI goes red.
    """
    venv = ROOT / ".ci-local" / f"{job}-py{python}"
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    if fresh and venv.exists():
        shutil.rmtree(venv)
    if not scripts.exists():
        uv = shutil.which("uv")
        print(f"[ci-local] creating {venv.name} on python {python} ...")
        if uv:
            # uv can materialise an interpreter this machine does not have,
            # which is what makes a local matrix run mean anything.
            subprocess.run([uv, "venv", str(venv), "--python", python, "--seed", "-q"], check=True)
        else:
            current = f"{sys.version_info.major}.{sys.version_info.minor}"
            if python != current:
                print(f"[ci-local] uv is not installed; falling back to python {current}, not {python}")
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return scripts


def cmd_run(args: argparse.Namespace) -> int:
    workflow_path = ROOT / ".github" / "workflows" / args.workflow
    workflow = _load_workflow(workflow_path)
    jobs = workflow.get("jobs", {})
    wanted = args.job or list(jobs)
    unknown = [name for name in wanted if name not in jobs]
    if unknown:
        print(f"no such job(s) in {workflow_path.name}: {', '.join(unknown)}", file=sys.stderr)
        return UNJUDGED

    shell = shutil.which("bash")
    if not shell:
        print("bash is required to run workflow steps locally", file=sys.stderr)
        return UNJUDGED

    python = args.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    results: list[tuple[str, str, str, float]] = []
    failed = 0

    for job_name in wanted:
        job = jobs[job_name]
        context = _matrix_context(job, python)
        scripts = _ensure_env(job_name, python, args.fresh)
        env = dict(os.environ)
        env["PATH"] = f"{scripts}{os.pathsep}{os.environ['PATH']}"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        print(f"\n=== job: {job_name} " + "=" * max(0, 60 - len(job_name)))
        for index, step in enumerate(job.get("steps", []), start=1):
            label = step.get("name") or step.get("uses") or f"step {index}"
            if "run" not in step:
                results.append((job_name, label, "NOT-LOCAL", 0.0))
                print(f"  --   {label}  (a GitHub action; nothing to run locally)")
                continue
            if "if" in step:
                results.append((job_name, label, "SKIP-IF", 0.0))
                print(f"  --   {label}  (guarded by `if:` - not evaluated locally)")
                continue

            script, unresolved = _substitute(step["run"], context)
            if unresolved:
                results.append((job_name, label, "SKIP-EXPR", 0.0))
                print(f"  --   {label}  (unresolved: {', '.join(sorted(set(unresolved)))})")
                continue

            print(f"  ->   {label}")
            if args.list:
                results.append((job_name, label, "LISTED", 0.0))
                continue
            started = time.perf_counter()
            completed = subprocess.run([shell, "-e", "-c", script], cwd=ROOT, env=env)
            elapsed = time.perf_counter() - started
            if completed.returncode == 0:
                status = "PASS"
            elif step.get("continue-on-error"):
                # Honoured rather than ignored: an advisory step that stopped
                # a local run but not a CI run would make the two disagree,
                # which is the one thing this runner exists to prevent.
                status = "FAIL-ALLOWED"
            else:
                status = "FAIL"
            results.append((job_name, label, status, elapsed))
            print(f"       {status} in {elapsed:.2f}s")
            if status == "FAIL":
                failed += 1
                if not args.keep_going:
                    break
        if failed and not args.keep_going:
            break

    print("\n" + "-" * 72)
    total = 0.0
    for job_name, label, status, elapsed in results:
        total += elapsed
        marker = {"PASS": "OK", "FAIL": "FAIL", "FAIL-ALLOWED": "warn"}.get(status, "--")
        print(f"{marker:<5} {job_name:<14} {label:<44} {elapsed:6.2f}s  {status}")
    print("-" * 72)
    executed = sum(1 for _job, _label, status, _elapsed in results if status in {"PASS", "FAIL", "FAIL-ALLOWED"})
    print(f"{executed} step(s) executed locally in {total:.2f}s, {failed} failed")
    return FAIL if failed else OK


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("attribution", help="no assistant attribution in the tree or recent commits")

    version = sub.add_parser("version", help="pyproject version == __init__ version (== tag)")
    version.add_argument("--expect", help="tag to match, with or without a leading v")

    pypi = sub.add_parser("pypi", help="refuse a version that is already published")
    pypi.add_argument("--version", help="version to check (default: the project version)")
    pypi.add_argument("--timeout", type=float, default=15.0)

    contract = sub.add_parser("contract", help="the documented exit codes, through the CLI")
    contract.add_argument("--executable", help="interpreter to test (default: this one)")

    smoke = sub.add_parser("smoke", help="check an INSTALLED distribution, not the source tree")
    smoke.add_argument("--executable", help="interpreter with the wheel installed")

    noskips = sub.add_parser("noskips", help="with every extra installed, no test may skip")
    noskips.add_argument("--executable", help="interpreter to run pytest with (default: this one)")

    wheelcheck = sub.add_parser("wheelcheck", help="install the built wheel in a clean venv and smoke it")
    wheelcheck.add_argument("--dist", default="dist", help="directory holding the distributions")
    wheelcheck.set_defaults(executable=None)

    artifact = sub.add_parser("artifact", help="the built distributions match the project version")
    artifact.add_argument("--dist", default="dist", help="directory holding the distributions")

    run = sub.add_parser("run", help="run this repository's CI workflow locally")
    run.add_argument("--workflow", default="ci.yml")
    run.add_argument("--job", action="append", help="run only these jobs (repeatable)")
    run.add_argument("--python", help="value to substitute for matrix.python-version")
    run.add_argument("--fresh", action="store_true", help="rebuild the local environment first")
    run.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
    run.add_argument("--list", action="store_true", help="show the steps without running them")

    args = parser.parse_args(argv)
    handler = {
        "attribution": cmd_attribution,
        "version": cmd_version,
        "pypi": cmd_pypi,
        "contract": cmd_contract,
        "smoke": cmd_smoke,
        "noskips": cmd_noskips,
        "wheelcheck": cmd_wheelcheck,
        "artifact": cmd_artifact,
        "run": cmd_run,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
