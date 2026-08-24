# ADR 002 — Trusted publishing, and what rollback actually means on PyPI

**Status:** accepted, 2026-08-22

## Decision

`release.yml` publishes to PyPI with **trusted publishing (OIDC)**. The job
declares `id-token: write`, PyPI verifies the identity of the workflow, and
mints a token that lives for the length of the upload.

There is no API token in the repository, in an organisation secret, or on the
maintainer's laptop. Nothing to leak, nothing to rotate, and no token that
still works after the laptop it was typed on is gone.

Two supporting decisions:

- **The publish job holds `id-token: write` and nothing else.** Creating the
  GitHub release is a separate job with `contents: write`. The job that can
  mint a PyPI credential cannot write to the repository, and the job that can
  write to the repository cannot publish.
- **`pypa/gh-action-pypi-publish` is pinned to a full commit SHA.** It is the
  one action here that holds a credential; a moving tag on it is a moving
  credential holder. Everything else is pinned to a major version, where
  readability is worth more than the marginal risk. `tests/test_ci_pipeline.py`
  asserts the SHA pin, so it cannot be relaxed by accident.

## Setup this requires (once, by the maintainer, on PyPI)

Trusted publishing is configured on PyPI, not here. On the project's
*Publishing* settings, add a GitHub publisher: owner `Mormolykos`, repository
`trainproof`, workflow `release.yml`, environment `pypi`. For a project that
does not exist on PyPI yet, the same form under *Pending publishers* creates it
on first upload.

The `environment: pypi` line in the workflow is what makes that binding tight
- and a GitHub environment can carry a required reviewer, which turns a tag
push into a publish that a human approved.

## Rollback

**PyPI does not allow a version to be re-uploaded. Not after a delete, not
ever.** `1.2.3` is permanently spent the moment it is accepted. This is the
question interviewers ask, so it is written down rather than assumed:

1. **Yank the bad version.** `pip install trainproof` stops resolving to it,
   while `trainproof==<yanked>` still installs for anyone who pinned it. A yank
   is reversible; a delete is not, and deleting breaks builds that pinned it.
2. **Fix, and release a patch version.** `0.18.1` → `0.18.2`. There is no path
   back to the yanked number.
3. **Do not delete the release** unless it leaked a secret, in which case the
   secret is compromised regardless and rotation is the actual fix.

The gate helps before the fact: `python scripts/ci.py pypi` refuses a version
that is already published, with the yank-and-patch instruction in the failure
text. It is a hard gate in `release.yml` and advisory in `ci.yml` - between
releases the current version *is* on PyPI, which is correct, and a check that
is red by design is a check people learn to ignore.

## Consequences

- The release is triggered by a tag and by nothing else, so every published
  version is a named point in history someone can check out.
- The publish job uploads the artifact the gate built and verified, downloaded
  as a workflow artifact. Rebuilding inside the publish job would upload bytes
  that nothing tested.
- Trusted publishing binds to a workflow *filename*. Renaming `release.yml`
  silently breaks publishing until the PyPI setting is updated - a real
  cross-system contract with no compile-time warning on either side.
