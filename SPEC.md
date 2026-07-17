Build a new Python library called **trainproof** in `C:\Users\User\Desktop\trainproof` (empty folder — create it).

**One-line pitch:** a linter for ML training runs. Run it on your dataset, tokenizer, and first-epoch logs — it gives a deterministic PASS/WARN/FAIL verdict with named findings and suggested fixes, BEFORE you burn weeks of GPU time.

**Relationship to ttsproof:** The author's existing library ttsproof (installed via `pip install ttsproof`, source at `C:\Users\User\Desktop\ttsproof` — READ ONLY, never modify it) already contains structural audio checks (numpy/soundfile), a spoken-form text normalizer, and a self-contained HTML report generator. PREFER importing ttsproof as a dependency where its public API fits; where it doesn't fit cleanly, adapt a minimal copy into trainproof with a comment crediting the origin module. Do not fork large amounts of code.

## CLI — exactly three subcommands (argparse, same style as ttsproof)

**1. `trainproof data <dir|manifest.jsonl>`** — dataset preflight for speech/TTS corpora:
- Audio: sample-rate/channel/bit-depth consistency across corpus; clipping; near-silent files; duration distribution with flagged outliers; duplicate detection via content hash.
- Transcripts: empty/duplicate entries; unnormalized text that hurts TTS training (digits, dates, clock times, acronyms — use ttsproof's canonicalizer to detect); text-length vs audio-duration correlation with outliers flagged as likely misalignments; charset audit (mixed scripts, control characters).
- Optional metadata columns: speaker balance, emotion/style balance, language-tag consistency.

**2. `trainproof tokenizer <model> <transcripts>`** — tokenizer preflight (SentencePiece for v0.1, generic interface):
- corpus vocabulary coverage and OOV rate
- tokens-per-second-of-audio distribution (sequence-length blowout warning)
- suspicious splits on numbers/dates/acronyms

**3. `trainproof epoch <logfile>`** — first-epoch verdict from training logs:
- Input: JSONL or CSV with tolerant parsing; recognized columns: step, loss, lr, optional grad_norm, optional epoch/time. Works on ANY historical log — no framework integration.
- Analyses: loss-curve shape (healthy decay vs flat vs diverging vs oscillating vs NaN/Inf); grad-norm sanity if present (explosion, vanishing, spikes vs median); learning-rate sanity vs loss response (warmup visible? loss spikes after LR steps?); throughput + total-run ETA extrapolation.
- Output: verdict block PASS/WARN/FAIL with each finding citing its evidence (step numbers, values).

## Hard rules
- All verdict thresholds are DETERMINISTIC rules collected in ONE constants module (`src/trainproof/rules.py`) with a comment per threshold, so the author can tune them. Do not scatter magic numbers.
- NEVER output invented confidence percentages or probabilistic claims. State evidence, not theater.
- Reports: one self-contained HTML file (no CDN/external assets) + markdown summary + process exit code (0 pass / 1 fail) so it works as a CI gate.
- Tests: pytest, with synthetic log fixtures in `tests/fixtures/`: healthy run, diverging run, NaN run, flat/dead run. Synthetic fixtures ONLY — do not read or reference any real training folders on this machine.
- Console output must be ASCII-safe (Windows cp1252 console).
- Package: `src/trainproof/` layout, pyproject.toml with name `trainproof`, version 0.1.0, MIT license, requires-python >=3.10, deps: numpy, soundfile, ttsproof; author: Panagiotis (Panos) Gkilis <bedvibe@bedvibe.studio>.
- README.md: pitch, install, the three commands with example output, a table of all verdict rules, explicit non-goals.
- Save this brief as SPEC.md in the repo root.

## Explicit NON-GOALS for v0.1 — do not build any of these
- no live learning-rate auto-adjustment
- no PyTorch/Lightning callbacks or framework hooks (log files only)
- no MCP server, no LLM integration
- no dashboards/wandb-style UI — this is a linter with verdicts, not a monitor
- no extra features beyond this spec, no additional metrics tables, no telemetry

## Absolute constraints
- NO git commands of any kind (no init, no commit — the author handles all version control himself)
- work ONLY inside `C:\Users\User\Desktop\trainproof`; read-only access to `C:\Users\User\Desktop\ttsproof` for reference
- no network access except `pip install`
- no AI/Claude/assistant attribution anywhere in code, docs, or metadata
- if something in this spec is ambiguous, choose the simplest interpretation and note the decision in SPEC_DECISIONS.md rather than expanding scope

When done, output: a file tree, the README, and the output of running `trainproof epoch` on the diverging-run fixture.
