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
    total_duration = 0.0
    
    suspicious_splits = 0
    
    for line in lines:
        if not line.strip(): continue
        text = line
        duration = 0.0
        if line.startswith("{"):
            try:
                data = json.loads(line)
                text = data.get("text", "") or data.get("transcript", "")
                duration = data.get("duration", 0.0)
            except Exception:
                pass
                
        pieces = tokenizer.encode_as_pieces(text)
        total_tokens += len(pieces)
        total_chars += len(text)
        total_duration += duration
        total_unks += sum(1 for p in pieces if p == "<unk>")
        
        # heuristic for suspicious split: if a token has multiple digits separated but not grouped?
        # A simple check: if the sequence length blows up compared to char length for numbers
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            num_pieces = tokenizer.encode_as_pieces(num)
            if len(num_pieces) > len(num) / 2 + 1:
                suspicious_splits += 1

    oov_rate = total_unks / max(1, total_tokens)
    if oov_rate > rules.MAX_OOV_RATE:
        findings.append({"id": "TP-TOK-HIGH-OOV", "level": "FAIL", "message": "High OOV (Out-Of-Vocabulary) rate.", "evidence": f"{oov_rate*100:.3f}% > {rules.MAX_OOV_RATE*100:.3f}%"})
        verdict = "FAIL"
        
    coverage = 1.0 - oov_rate
    if coverage < rules.MIN_VOCAB_COVERAGE:
        findings.append({"id": "TP-TOK-LOW-COVERAGE", "level": "WARN", "message": "Vocabulary coverage is below recommended threshold.", "evidence": f"{coverage*100:.3f}% < {rules.MIN_VOCAB_COVERAGE*100:.3f}%"})
        if verdict == "PASS": verdict = "WARN"

    if total_duration > 0:
        tps = total_tokens / total_duration
        if tps > rules.MAX_TOKENS_PER_SEC:
            findings.append({"id": "TP-TOK-HIGH-TPS", "level": "WARN", "message": "High tokens per second of audio (possible sequence length blowout).", "evidence": f"{tps:.1f} tokens/sec > {rules.MAX_TOKENS_PER_SEC}"})
            if verdict == "PASS": verdict = "WARN"
            
    if suspicious_splits > len(lines) * 0.01:
        findings.append({"id": "TP-TOK-SUSPICIOUS-SPLIT", "level": "WARN", "message": "Suspicious splits detected on numbers/dates.", "evidence": f"{suspicious_splits} instances."})
        if verdict == "PASS": verdict = "WARN"

    if verdict == "PASS":
        findings.append({"id": "TP-TOK-PASS", "level": "PASS", "message": "Tokenizer vocabulary coverage and splits look healthy.", "evidence": f"{total_tokens} tokens evaluated."})

    return {"verdict": verdict, "findings": findings}
