# ADR 003 — How strict mypy is here, and why that number

**Status:** accepted, 2026-08-22

## Context

There was no type checking in this repository, or in either sibling library,
before today. The codebase is 3,261 lines written without a checker in mind.

## The measurement, taken before the decision

| setting | errors on `src/trainproof` |
|---|---|
| mypy defaults, no config | 13 |
| mypy defaults, optional imports declared | **6** |
| `--strict` | **159** |

Measured 2026-08-22 with mypy 2.3.1. Reproduce: `mypy src/trainproof` and
`mypy --strict src/trainproof`.

## Decision

**mypy's defaults, plus `warn_unused_configs`, `warn_redundant_casts` and
`warn_unused_ignores`, and it must pass clean.** Not `--strict`.

Six errors were worth fixing in an afternoon, and all six were real:

- `tfevents.py` — `_iter_fields` yields `int` for a varint and `bytes` for
  every other wire type. The union was never written down, so the checker
  inferred `int` from whichever branch came first. Writing the true type down
  propagated to eleven call sites, every one of which was already guarded by a
  `wire ==` test; the guards now say `isinstance` too, so the invariant is
  visible to a reader and to the checker at the same time.
- `epoch.py` — `self.ran` and `self.skipped` had no annotation and no value the
  checker could infer from. They are the coverage accounting this tool reports
  to CI, and their element types were nowhere in the source.
- `integrations/hf.py` — the `try: import transformers / except ImportError:`
  stand-in class, which cannot be expressed without an explicit ignore.

A hundred and fifty-nine would have been closed by turning the checker off,
which is how type checking usually dies in a codebase this age.

## The ratchet

`disallow_untyped_defs`, module by module, is the next step. Turning it on for
one module at a time in `[[tool.mypy.overrides]]` gets there without a flag
day and without a pull request nobody can review.

`warn_unused_ignores = true` is on from the start, and is the one setting that
matters most here: an `# type: ignore` that has stopped being needed is a lie
about the code, and it is exactly the comment nobody revisits.

## Optional dependencies

`numpy`, `soundfile`, `sentencepiece`, `transformers`, `pynvml` and `ttsproof`
are declared `ignore_missing_imports`. trainproof imports each of them inside
the function that needs it, precisely so a missing one becomes exit 2 ("cannot
judge") instead of an ImportError at startup. The checker has to be told the
same thing, or it reports the design as an error.
