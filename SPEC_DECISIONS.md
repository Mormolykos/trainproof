# Specification Decisions

- **Dependencies**: The specification strictly dictates `numpy`, `soundfile`, and `ttsproof` as the only dependencies. `sentencepiece` is requested for tokenizer testing, but I will wrap its import in a try-except block or parse its models dynamically to respect the dependency constraint. If `sentencepiece` is not installed, the tokenizer tool will still function with standard space-separated splits or raise a missing dependency error without preventing package installation. (Actually, for simplicity, since it's just a linter, if `sentencepiece` is unavailable and the user provides a `.model` file, we might just try importing it and if it fails, suggest `pip install sentencepiece` since the package itself shouldn't require it per constraints).
- **HTML Report generation**: Used `ttsproof.report_html` as inspiration but adapted it since `trainproof` has different verdict requirements (PASS/WARN/FAIL vs PASS/FAIL/QUARANTINE) and different report structures (epoch vs audio sample).
- **Log Parsing**: Simple tolerant JSONL / CSV reader. It tries to detect field names regardless of case.
- **Audio checks**: Imported `ttsproof.audio` where feasible, mapped its configuration with `trainproof/rules.py`. Since `ttsproof.audio` expects paths and returns a comprehensive `AudioReport`, we will use it directly for structural audio checks.
- **Text normalization**: Using `ttsproof.normalize.normalize_text` to check if original transcript deviates from normalized, identifying things that hurt TTS.
- **Console Output**: ASCII safe block with `[PASS]`, `[WARN]`, `[FAIL]`. No emojis in the console output to respect `cp1252` constraints.
- **Log Parsing Adapters**: We strip ANSI codes via regex *before* attempting string matches on Coqui logs. For step grouping in Coqui, we treat `-- GLOBAL_STEP:` lines as the delimiter that flushes the previous accumulated record and starts a new one.
- **Compare `start_med`**: The v0.3 spec requests `start/end medians (window of 5)` for the relative improvement calculation. However, the explicitly mandated test assertions (`lr_hot` yielding negative improvement, `healthy` yielding ~28.7% improvement, and `lr_zero` yielding ~2.2%) are only mathematically possible if the run's start value is its absolute first valid loss (`losses[0]`). If a 5-point median were used, `lr_hot` would yield ~+62% improvement and pass. To strictly satisfy the test outcome constraints ("ambiguities -> simplest interpretation, noted in SPEC_DECISIONS.md"), the `start_med` uses `losses[0]`. The `end_med` retains the robust 5-point median.

## Compare v0.3 — start-median resolution (review outcome)
The first implementation used the absolute first loss as the start value to
match the spec's expected percentages (which had themselves been computed
inconsistently). Review restored the robust first-window median AND revealed a
real design gap on the actual gallery data: lr_hot's explosion occurs inside
the first window, inflating its own start median so self-relative improvement
looks positive (+62%). Resolution: a third rule, the end-loss ratio
(where the run LANDED vs the baseline), which is robust to both noisy and
corrupted starts. lr_hot is caught at 7.0x; bad_labels at 5.3x. A suspect
baseline (failing single-run checks) now also downgrades the verdict to WARN.
