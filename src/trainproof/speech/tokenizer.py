import json
import re
from pathlib import Path
from typing import Any

from .. import rules


def load_tokenizer(model_path: str):
    """Returns (tokenizer, error_finding). Never silently degrades: a linter
    that swaps in a fake tokenizer would produce fake verdicts."""
    try:
        import sentencepiece as spm
    except ImportError:
        return None, {"id": "TP-TOK-SPM-MISSING", "level": "FAIL",
                      "message": "sentencepiece is not installed - cannot lint this tokenizer.",
                      "evidence": "pip install sentencepiece"}
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp, None
    except Exception as e:
        return None, {"id": "TP-TOK-LOAD-FAIL", "level": "FAIL",
                      "message": "Failed to load SentencePiece model.",
                      "evidence": f"{model_path}: {e}"}

def check_tokenizer(model_path: str | Path, transcripts_path: str | Path) -> dict[str, Any]:
    tokenizer, load_error = load_tokenizer(str(model_path))
    if load_error is not None:
        return {"verdict": "FAIL", "findings": [load_error]}
    findings = []
    verdict = "PASS"
    
    path = Path(transcripts_path)
    if not path.exists():
        return {"verdict": "FAIL", "findings": [{"id": "TP-TOK-NO-TRANSCRIPTS", "level": "FAIL", "message": "Transcripts file not found.", "evidence": str(transcripts_path)}]}
        
    lines = path.read_text(encoding="utf-8").splitlines()
    total_tokens = 0
    total_chars = 0
    total_unks = 0
    n_scored = 0

    # Tokens-per-second is a RATE, and both sides of it must come from the same
    # population. This used to add EVERY line's tokens to the numerator while
    # only lines carrying a `duration` field added to the denominator, so a file
    # where one line in ten declares a duration reported a rate roughly ten
    # times the real one -- and TP-TOK-HIGH-TPS fired on a healthy tokenizer.
    # Only timed lines contribute to either side now.
    timed_tokens = 0
    timed_duration = 0.0
    n_timed = 0

    suspicious_splits = 0

    for line in lines:
        if not line.strip(): continue
        text = line
        # None, not 0.0: "this line declared no duration" is not "this line is
        # zero seconds long", and the difference is the whole defect below.
        duration = None
        if line.startswith("{"):
            try:
                data = json.loads(line)
                text = data.get("text", "") or data.get("transcript", "")
                raw = data.get("duration")
                # bool is an int subclass; a JSON `true` is not a duration.
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    duration = float(raw)
            except json.JSONDecodeError:
                # A line that opens with '{' but does not parse is treated as
                # plain text, which `text` already holds. Narrowed from a bare
                # Exception so a genuine bug in the lines above surfaces instead
                # of being read as "not JSON after all".
                pass

        pieces = tokenizer.encode_as_pieces(text)
        n_scored += 1
        total_tokens += len(pieces)
        total_chars += len(text)
        total_unks += sum(1 for p in pieces if p == "<unk>")
        if duration is not None and duration > 0:
            n_timed += 1
            timed_tokens += len(pieces)
            timed_duration += duration

        # heuristic for suspicious split: if a token has multiple digits separated but not grouped?
        # A simple check: if the sequence length blows up compared to char length for numbers
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            num_pieces = tokenizer.encode_as_pieces(num)
            if len(num_pieces) > len(num) / 2 + 1:
                suspicious_splits += 1

    if total_tokens == 0:
        # Nothing was tokenized -- an empty file, only blank lines, or JSON rows
        # with no text. `total_unks / max(1, total_tokens)` used to turn that
        # into an OOV rate of 0.000% and a coverage of 100.000%, and the run
        # ended on TP-TOK-PASS "looks healthy" with the evidence "0 tokens
        # evaluated". A perfect score is exactly what nothing-measured must not
        # produce, so this refuses instead. NOT-CHECKED exits 2: trainproof
        # could not judge, which is not the same as the tokenizer being bad.
        findings.append({
            "id": "TP-TOK-NOT-MEASURED", "level": "NOT-CHECKED",
            "message": "No tokens were produced, so OOV and vocabulary coverage were NOT MEASURED.",
            "evidence": f"{len(lines)} line(s) read, {n_scored} non-blank, 0 tokens produced.",
        })
        return {"verdict": "NOT-CHECKED", "findings": findings}

    oov_rate = total_unks / total_tokens
    if oov_rate > rules.MAX_OOV_RATE:
        findings.append({"id": "TP-TOK-HIGH-OOV", "level": "FAIL", "message": "High OOV (Out-Of-Vocabulary) rate.", "evidence": f"{oov_rate*100:.3f}% > {rules.MAX_OOV_RATE*100:.3f}%"})
        verdict = "FAIL"

    coverage = 1.0 - oov_rate
    if coverage < rules.MIN_VOCAB_COVERAGE:
        findings.append({"id": "TP-TOK-LOW-COVERAGE", "level": "WARN", "message": "Vocabulary coverage is below recommended threshold.", "evidence": f"{coverage*100:.3f}% < {rules.MIN_VOCAB_COVERAGE*100:.3f}%"})
        if verdict == "PASS": verdict = "WARN"

    if n_timed == 0:
        # The check did not run, and silence used to be its only report. A
        # transcripts file with no durations is the normal case for plain text,
        # so this is a NOT-CHECKED finding rather than a verdict change -- but
        # it is stated, and TP-TOK-PASS below no longer implies it passed.
        findings.append({
            "id": "TP-TOK-TPS-NOT-MEASURED", "level": "NOT-CHECKED",
            "message": "Tokens per second of audio was NOT MEASURED: no line declared a duration.",
            "evidence": f"0 of {n_scored} non-blank line(s) carry a positive `duration` field.",
        })
    else:
        tps = timed_tokens / timed_duration
        if tps > rules.MAX_TOKENS_PER_SEC:
            findings.append({"id": "TP-TOK-HIGH-TPS", "level": "WARN", "message": "High tokens per second of audio (possible sequence length blowout).", "evidence": f"{tps:.1f} tokens/sec > {rules.MAX_TOKENS_PER_SEC} (over {n_timed} of {n_scored} line(s), {timed_tokens} tokens / {timed_duration:.1f}s)"})
            if verdict == "PASS": verdict = "WARN"

    if suspicious_splits > len(lines) * 0.01:
        findings.append({"id": "TP-TOK-SUSPICIOUS-SPLIT", "level": "WARN", "message": "Suspicious splits detected on numbers/dates.", "evidence": f"{suspicious_splits} instances."})
        if verdict == "PASS": verdict = "WARN"

    if verdict == "PASS":
        tps_note = ("tokens/sec NOT MEASURED (no line declared a duration)"
                    if n_timed == 0 else
                    f"tokens/sec measured on {n_timed} of {n_scored} line(s)")
        findings.append({"id": "TP-TOK-PASS", "level": "PASS", "message": "Tokenizer vocabulary coverage and splits look healthy.", "evidence": f"{total_tokens} tokens evaluated; {tps_note}."})

    return {"verdict": verdict, "findings": findings}
