# ADR 001 — CI runs in GitHub Actions, and the same steps replay locally

**Status:** accepted, 2026-08-22
**Context:** trainproof 0.18.1, nineteen releases, zero CI workflows.

## The problem

trainproof has shipped nineteen releases to PyPI. Every one of them was gated
by `release.ps1`: attribution scan, pytest, ruff, version consistency, build,
upload, fresh-venv verify. It is a good ritual and it caught real problems.

It also has three faults, and they are all the same fault.

1. **It is not in the repository.** `.gitignore` line 9 lists `release.ps1`,
   because the script reads a PyPI token from a path on one laptop. The
   procedure that decides whether trainproof is fit to publish exists on one
   disk, in one operating system's shell, and a contributor cannot read it.
2. **It runs on one machine, in one environment.** `requires-python = ">=3.10"`
   is a promise about four Python versions and two operating systems. It was
   verified on Python 3.10, on Windows, on a machine that happened to have
   numpy, soundfile and ttsproof installed for other reasons.
3. **It runs at release time.** Everything it checks is checked once the
   decision to publish has already been made.

The third point is the expensive one. A gate that runs after the decision is a
receipt, not a gate.

## Decision

CI is GitHub Actions. `.github/workflows/ci.yml` runs on every push and pull
request; `.github/workflows/release.yml` runs on a tag and **calls `ci.yml`**
through `workflow_call` rather than repeating it, so a release cannot pass a
weaker gate than a pull request.

Every check is a subcommand of `scripts/ci.py`, so a workflow step reads
`python scripts/ci.py contract` and the logic is Python that can be run,
tested, and read by anyone with the repository. The one thing that is not in
`ci.py` is anything requiring the runner: checkout, interpreter installation,
artifact upload.

`python scripts/ci.py run` **reads `.github/workflows/ci.yml`** and executes the
`run:` steps it finds, in a throwaway virtual environment per job. The list of
things to run is derived from the workflow, never restated. Add a step and the
local replay picks it up with no second edit; that is the property, and it is
the reason the runner parses YAML instead of holding its own list.

## Alternatives considered

**Keep `release.ps1` and commit it.** Cheapest option and it fixes fault 1
only. It still runs on one machine at release time.

**`act` (run Actions locally in Docker).** Higher fidelity than replaying the
steps - it runs the real runner images and evaluates real expressions. Rejected
for now because it needs Docker and a multi-gigabyte runner image to answer a
question that eighteen `run:` lines answer in ninety seconds. The replay
runner is honest about the gap: it prints `NOT-LOCAL` for every `uses:` step
rather than pretending to have run it.

**A Makefile or `tox` as the shared definition, called by both.** The
conventional answer, and it would work. Rejected because it inverts what should
be true: the workflow would then be a thin caller of a local file, and the
thing that actually gates the release would be the file nobody reads. Deriving
the local run *from the workflow* puts the authority in the file GitHub
executes.

## Consequences

- The local replay is a replay, not a simulation. It does not evaluate `if:`
  conditions or GitHub expressions it was not given, and it says so per step
  instead of skipping silently.
- Each job gets its own environment locally, matching one clean runner per job.
  This is not cosmetic: `test` proves the core install works with no optional
  dependency, and `speech` installs the extra. Share one environment between
  them and the second silently invalidates the first.
- Tool versions are pinned (`ruff==0.16.1`, `mypy==2.3.1`). An unpinned linter
  turns a green repository red on a morning when nobody changed anything.
  Bumping them is a pull request with a diff.

## Proving the gate stops things — the failure drills

A gate's output is identical whether it checked or waved something through, so
each one is broken on purpose and watched.

`tests/test_ci_catches_faults.py` does this permanently, in twelve tests
against temporary copies: a version bumped in `pyproject.toml` but not in
`__init__.py`; a tag that names a version the source does not; a `dist/` left
over from the previous release; a missing sdist; an attribution trailer in a
commit message; a vendor name in `pyproject.toml`; and — the inverse, which is
the one that made the gate usable — prose naming a tool, which must pass.

Two more were run end to end through the real pipeline on 2026-08-22, injected
into the working tree and restored afterwards with a SHA-256 check:

| injected | job | result |
|---|---|---|
| a test asserting `2 + 2 == 5` | `test` | **FAIL in 7.46s**, exit 1 |
| a function returning `str` where it declares `int` | `types` | **FAIL in 0.25s**, exit 1 |
| nothing — `0.18.1` is genuinely published | `pypi` | **FAIL**, exit 1, with the yank-and-patch procedure in the message |

Reproduce the first two with `python scripts/ci.py run --job test` (or
`--job types`) after breaking something; the third needs no setup at all, which
is exactly why it is advisory in `ci.yml` and blocking in `release.yml`.

Two of the twelve tests assert the *third* exit code rather than a failure:
an unreachable PyPI and a missing git checkout return `2`, "could not judge",
never `1`. A gate that reports a network blip as a failed check blocks a
release for a reason unrelated to the code, and teaches whoever is on the other
end to re-run it until it goes green — which is how a gate stops being read.

## What the gate found the first time it ran

Written down because it is the argument for the whole ADR.

- `scripts/ci.py` imported `tomllib`, which is Python 3.11+, in a project whose
  floor is 3.10. Caught on the first local run.
- **A clean `pip install -e ".[dev]"` checkout ran zero tests.**
  `tests/test_data.py` imports the speech pack, which left the core install in
  v0.18.1, and the import fails at collection - which aborts the whole session
  rather than that file. The published "274 tests" figure was only reachable on
  a machine that happened to have numpy, soundfile and ttsproof. Now: 270
  collected on a core install with the dataset tests skipping, 274 with the
  extra, and a `noskips` gate in the `speech` job so that skip cannot become
  permanent.
- The attribution scan, written to catch an authorship trailer, failed on a
  README paragraph addressed to coding agents and on the line in `SPEC.md` that
  states the rule itself. See `scripts/ci.py`: prose is scanned for attribution
  *shapes*, vendor names only in metadata files where they can mean nothing
  else.
