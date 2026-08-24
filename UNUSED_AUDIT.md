# Unused Files Audit

| Path | Bucket | Size | Evidence |
|------|--------|------|----------|
| `.git/` | REQUIRED | - | Core version control directory |
| `.gitignore` | REQUIRED | 117 B | Git configuration |
| `.pytest_cache/` | GENERATED | - | Explicitly listed in prompt as GENERATED |
| `.ruff_cache/` | GENERATED | - | Explicitly listed in prompt as GENERATED |
| `.venv/` | GENERATED | - | Present in `.gitignore` |
| `build/` | GENERATED | - | Explicitly listed in prompt as GENERATED |
| `CHANGELOG.md` | DOCS | 33,883 B | Explicitly listed in prompt as DOCS |
| `CONTRACTS.md` | DOCS | 6,459 B | Explicitly listed in prompt as DOCS |
| `dist/` | GENERATED | - | Explicitly listed in prompt as GENERATED |
| `evidence/` | EVIDENCE | - | Explicitly listed in prompt as EVIDENCE |
| `EVIDENCE_MATRIX.md` | DOCS | 4,404 B | Explicitly listed in prompt as DOCS |
| `examples/` | EVIDENCE | - | Explicitly listed in prompt as EVIDENCE |
| `golden/` | EVIDENCE | - | Explicitly listed in prompt as EVIDENCE |
| `pyproject.toml` | REQUIRED | 3,420 B | Python project configuration |
| `README.md` | DOCS | 19,805 B | Explicitly listed in prompt as DOCS |
| `recon/` | GENERATED | - | Present in `.gitignore` |
| `release.ps1` | GENERATED | 6,125 B | Present in `.gitignore` |
| `ROADMAP.md` | ORPHAN | 16,174 B | `grep -r "ROADMAP.md" src/ pyproject.toml` returned nothing. Not required, not docs, not evidence. (Note: cannot verify if tracked due to shell restrictions) |
| `RULES.md` | DOCS | 11,684 B | Explicitly listed in prompt as DOCS |
| `scripts/` | REQUIRED | - | Referenced in tests (`test_golden_gallery.py`) and `src/` |
| `SPEC.md` | ORPHAN | 4,698 B | `grep -r "SPEC.md"` returned nothing except itself. (Note: cannot verify if tracked due to shell restrictions) |
| `SPEC_DECISIONS.md` | ORPHAN | 4,625 B | `grep -r "SPEC_DECISIONS.md"` returned nothing except `SPEC.md`. (Note: cannot verify if tracked due to shell restrictions) |
| `src/` | REQUIRED | - | Explicitly listed in prompt as REQUIRED |
| `tests/` | REQUIRED | - | Contains the test suite which is required |
| `uv.lock` | GENERATED | 93,057 B | Present in `.gitignore` |

## SAFE TO DELETE

The following files and directories belong to the GENERATED bucket.
Total size of listed GENERATED files (excluding directories): 99,182 Bytes.

- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/`
- `build/`
- `dist/`
- `recon/`
- `release.ps1`
- `uv.lock`

## NEEDS A HUMAN DECISION

The following files are ORPHAN:

- `ROADMAP.md`
- `SPEC.md`
- `SPEC_DECISIONS.md`

### Bucket Counts

- REQUIRED: 6
- GENERATED: 8
- DOCS: 5
- EVIDENCE: 3
- ORPHAN: 3
