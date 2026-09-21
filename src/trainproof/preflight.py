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
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed.append((i, line))
                continue
            # S-5 (2026-09-21): `123`, `null` and `[1,2]` are valid JSON and not
            # records. They used to be appended and then reached `field in record`
            # in _extract_text, raising TypeError out of preflight -- where
            # CONTRACTS.md promises a verdict or a documented exit. A line that
            # cannot be addressed by field name is malformed, which is a state
            # this function already has.
            if not isinstance(value, dict):
                malformed.append((i, line))
                continue
            records.append(value)
    return records, malformed

def _extract_text(record, text_field):
    # S-5 (2026-09-21): a JSONL line that is valid JSON but not an object -- `123`,
    # `null`, `[1,2]` -- reached `field in record` and raised TypeError out of
    # preflight, where CONTRACTS.md promises a verdict or a documented exit. A
    # record trainproof cannot address by field name is a malformed record, and
    # load_jsonl already has a state for that.
    #
    # NOT CHANGED HERE: how a *present* non-string value is judged. `{"text": null}`
    # still stringifies to "None" and passes the empty-text gate. That is a real
    # defect (audit V2-34) and correcting it alters verdicts on real datasets, so
    # it is deferred rather than folded into a crash fix.
    if not isinstance(record, dict):
        return ""
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
        
    shares_id = False
    if has_eos and has_pad:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if eos_id is not None and pad_id is not None and eos_id == pad_id:
            shares_id = True
            findings.append({"id": "TP-PRE-PAD-EQUALS-EOS", "level": "WARN", "message": "pad_token_id equals eos_token_id", "evidence": f"Shared ID: {eos_id}"})

    # Having an eos_token is not the same as putting one in the sequence.
    #
    # The domain of `add_eos_token` is bool. Most tokenizer families do not
    # define the attribute at all, and many pipelines append the EOS themselves
    # in the text or the collator - so anything other than a literal False is
    # UNKNOWN and reports nothing. Absence never becomes a finding here.
    #
    # WARN and deliberately not FAIL: trainproof cannot see the caller's
    # preprocessing, and a false FAIL under a stop-on-fail policy aborts a
    # correct run before step 1, which is the worst thing this library can do.
    if has_eos and getattr(tokenizer, "add_eos_token", None) is False:
        evidence = (
            "tokenizer has eos_token "
            f"{getattr(tokenizer, 'eos_token', '')!r} but add_eos_token is False, "
            "so encoding appends no EOS. Unless the caller adds one explicitly, "
            "no sequence carries a stop target and the model is never taught to "
            "end."
        )
        if shares_id:
            evidence += (
                " pad_token_id == eos_token_id here, so an EOS and a pad are the "
                "same integer: a collator that masks padding masks a genuine EOS "
                "with it, and no inspection of input_ids can tell them apart."
            )
        findings.append({
            "id": "TP-PRE-NO-EOS-APPEND",
            "level": "WARN",
            "message": "Tokenizer will not append eos_token - sequences may carry no stop target.",
            "evidence": evidence,
        })
            
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

def _is_unparseable(text):
    """True when this line is not valid JSON at all.

    The complement -- valid JSON that decodes to something other than an object --
    is a record-structure problem, not a parser problem, and the two must not be
    reported with the same sentence.
    """
    try:
        json.loads(text)
    except (TypeError, ValueError):   # ValueError covers json.JSONDecodeError
        return True
    return False


def _json_text(value):
    """The JSON text of a non-record value, for uniform classification.

    A value that will not serialise is rendered as a JSON string, which is still
    valid JSON and still not an object -- the fact being recorded.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def check_preflight(records, malformed=None, tokenizer=None, max_len=None, text_field=None):
    findings = []

    # S-5: preflight(<iterable>) bypasses load_jsonl, so non-record entries are
    # separated here instead. Same outcome, same reason. The entry keeps the
    # value's JSON text rather than a prose label, so the classification below
    # reads the same way for both producers.
    malformed = list(malformed or [])
    non_records = [(i, r) for i, r in enumerate(records, start=1) if not isinstance(r, dict)]
    if non_records:
        records = [r for r in records if isinstance(r, dict)]
        malformed += [(i, _json_text(r)) for i, r in non_records]

    if malformed:
        # WORDING CORRECTED 2026-09-21: this finding used to say "JSONL parsing
        # failed." with evidence "N broken lines". For `123`, `null` or `[1,2]`
        # that is false about the artifact -- json.loads SUCCEEDED; the value it
        # returned is simply not a record. One ID still covers both classes, so
        # the message states the condition they share and the evidence says which
        # of the two was actually seen. No verdict, level or exit code changes.
        bad_syntax, not_objects = [], []
        for lineno, text in malformed:
            (bad_syntax if _is_unparseable(text) else not_objects).append(lineno)
        first = malformed[0][0]
        if bad_syntax and not_objects:
            evidence = (f"{len(malformed)} unusable lines: {len(bad_syntax)} not valid JSON, "
                        f"{len(not_objects)} valid JSON but not an object. First at line {first}.")
        elif not_objects:
            n = len(not_objects)
            subject = "1 line is" if n == 1 else f"{n} lines are"
            carries = "it carries" if n == 1 else "they carry"
            evidence = (f"{subject} valid JSON but not an object, so {carries} no fields to check. "
                        f"First at line {first}.")
        else:
            n = len(bad_syntax)
            subject = "1 line is" if n == 1 else f"{n} lines are"
            evidence = f"{subject} not valid JSON. First at line {first}."
        findings.append({"id": "TP-PRE-MALFORMED-JSONL", "level": "FAIL",
                         "message": "JSONL lines could not be read as records.",
                         "evidence": evidence})
        
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
