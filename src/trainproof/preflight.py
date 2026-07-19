import json
from pathlib import Path

class PreflightResult:
    def __init__(self, verdict, findings):
        self.verdict = verdict
        self.findings = findings
        
    def print(self):
        from .report import print_verdict_console
        print_verdict_console(self.verdict, self.findings)
        
    def raise_on_fail(self):
        if self.verdict == "FAIL":
            fails = [f for f in self.findings if f.get("level") == "FAIL"]
            msg = "Preflight FAIL. Findings: " + "; ".join(f.get("message", "") for f in fails)
            raise RuntimeError(msg)

def load_jsonl(path):
    records = []
    malformed = []
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append((i, line))
    return records, malformed

def _extract_text(record, text_field):
    if text_field:
        return str(record.get(text_field, ""))
    for field in ["text", "output", "content", "response", "completion"]:
        if field in record:
            return str(record[field])
    return ""

def check_empty_rows(records, text_field):
    empties = []
    for i, r in enumerate(records):
        txt = _extract_text(r, text_field)
        if not txt.strip():
            empties.append(i)
    if empties:
        return [{"id": "TP-PRE-EMPTY-TEXT", "level": "FAIL", "message": "Empty or whitespace-only text found.", "evidence": f"{len(empties)} records (indices {empties[:3]}...)"}]
    return []

def check_duplicates(records, text_field):
    seen = set()
    dupes = 0
    distinct_dupes = set()
    for r in records:
        txt = _extract_text(r, text_field)
        if txt in seen:
            dupes += 1
            distinct_dupes.add(txt)
        else:
            seen.add(txt)
    if dupes > 0:
        return [{"id": "TP-PRE-DUPLICATE-TEXT", "level": "WARN", "message": "Exact duplicate text found.", "evidence": f"{dupes} duplicate records of {len(distinct_dupes)} distinct texts"}]
    return []

def check_tokenizer(tokenizer):
    findings = []
    has_eos = getattr(tokenizer, "eos_token", None) is not None
    if not has_eos:
        findings.append({"id": "TP-PRE-MISSING-EOS-TOKEN", "level": "FAIL", "message": "tokenizer has no eos_token - training has no stop signal", "evidence": ""})
    
    has_pad = getattr(tokenizer, "pad_token", None) is not None
    if not has_pad:
        findings.append({"id": "TP-PRE-MISSING-PAD-TOKEN", "level": "WARN", "message": "tokenizer has no pad_token", "evidence": ""})
        
    if has_eos and has_pad:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if eos_id is not None and pad_id is not None and eos_id == pad_id:
            findings.append({"id": "TP-PRE-PAD-EQUALS-EOS", "level": "WARN", "message": "pad_token_id equals eos_token_id", "evidence": f"Shared ID: {eos_id}"})
            
    has_bos = getattr(tokenizer, "bos_token", None) is not None
    if has_bos:
        findings.append({"id": "TP-PRE-BOS-TOKEN-INFO", "level": "INFO", "message": "bos_token is present", "evidence": str(getattr(tokenizer, "bos_token", ""))})
    else:
        findings.append({"id": "TP-PRE-BOS-TOKEN-INFO", "level": "INFO", "message": "no bos_token (normal for many model families)", "evidence": ""})
        
    return findings

def check_context_length(records, tokenizer, max_len, text_field):
    if max_len is None:
        return [{"id": "TP-PRE-CONTEXT-CHECK-SKIPPED", "level": "INFO", "message": "pass --max-len to enable", "evidence": ""}]
        
    overflows = 0
    max_found = 0
    for r in records:
        txt = _extract_text(r, text_field)
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(txt)
        elif callable(tokenizer):
            tokens = tokenizer(txt).get("input_ids", [])
        else:
            tokens = []
        n = len(tokens)
        if n > max_found:
            max_found = n
        if n > max_len:
            overflows += 1
            
    if overflows > 0:
        return [{"id": "TP-PRE-CONTEXT-OVERFLOW", "level": "WARN", "message": f"Records exceed max context length of {max_len}", "evidence": f"{overflows} records overflow. Largest token count: {max_found}"}]
    return []

def check_preflight(records, malformed=None, tokenizer=None, max_len=None, text_field=None):
    findings = []
    
    if malformed:
        findings.append({"id": "TP-PRE-MALFORMED-JSONL", "level": "FAIL", "message": "JSONL parsing failed.", "evidence": f"{len(malformed)} broken lines. First bad line number: {malformed[0][0]}"})
        
    findings.extend(check_empty_rows(records, text_field))
    findings.extend(check_duplicates(records, text_field))
    
    if tokenizer is not None:
        findings.extend(check_tokenizer(tokenizer))
        findings.extend(check_context_length(records, tokenizer, max_len, text_field))
        
    has_fail = any(f.get("level") == "FAIL" for f in findings)
    has_warn = any(f.get("level") == "WARN" for f in findings)
    
    if has_fail:
        verdict = "FAIL"
    elif has_warn:
        verdict = "WARN"
    else:
        verdict = "PASS"
        findings.append({"id": "TP-PRE-OK", "level": "PASS", "message": "Pre-flight checks passed.", "evidence": ""})
        
    return {"verdict": verdict, "findings": findings}

def preflight(source, tokenizer=None, max_len=None, text_field=None):
    if isinstance(source, (str, Path)):
        records, malformed = load_jsonl(source)
    else:
        records = list(source)
        malformed = None
        
    res = check_preflight(records, malformed, tokenizer, max_len, text_field)
    return PreflightResult(res["verdict"], res["findings"])
