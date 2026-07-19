from types import SimpleNamespace
from pathlib import Path
from trainproof.preflight import (
    load_jsonl, _extract_text, check_empty_rows, check_duplicates,
    check_tokenizer, check_context_length, check_preflight, preflight
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def test_load_jsonl():
    records, malformed = load_jsonl(FIXTURES_DIR / "clean.jsonl")
    assert len(records) == 3
    assert len(malformed) == 0

    records, malformed = load_jsonl(FIXTURES_DIR / "malformed.jsonl")
    assert len(records) == 2
    assert len(malformed) == 1
    assert malformed[0][0] == 2  # line 2

def test_extract_text():
    assert _extract_text({"text": "A"}, None) == "A"
    assert _extract_text({"output": "B"}, None) == "B"
    assert _extract_text({"content": "C"}, None) == "C"
    assert _extract_text({"custom": "D"}, "custom") == "D"
    assert _extract_text({"other": "E"}, None) == ""

def test_check_empty_rows():
    records, _ = load_jsonl(FIXTURES_DIR / "empties.jsonl")
    findings = check_empty_rows(records, None)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-PRE-EMPTY-TEXT"
    assert findings[0]["level"] == "FAIL"

    records_clean, _ = load_jsonl(FIXTURES_DIR / "clean.jsonl")
    assert len(check_empty_rows(records_clean, None)) == 0

def test_check_duplicates():
    records, _ = load_jsonl(FIXTURES_DIR / "dupes.jsonl")
    findings = check_duplicates(records, None)
    assert len(findings) == 1
    assert findings[0]["id"] == "TP-PRE-DUPLICATE-TEXT"
    assert findings[0]["level"] == "WARN"

    records_clean, _ = load_jsonl(FIXTURES_DIR / "clean.jsonl")
    assert len(check_duplicates(records_clean, None)) == 0

def test_check_tokenizer():
    # 1. Missing EOS
    tok1 = SimpleNamespace(eos_token=None, pad_token="<pad>", bos_token=None)
    f1 = check_tokenizer(tok1)
    assert f1[0]["id"] == "TP-PRE-MISSING-EOS-TOKEN"
    assert f1[0]["level"] == "FAIL"

    # 2. Missing PAD
    tok2 = SimpleNamespace(eos_token="<eos>", pad_token=None, bos_token="<bos>")
    f2 = check_tokenizer(tok2)
    assert len(f2) == 2
    ids = {f["id"] for f in f2}
    assert "TP-PRE-MISSING-PAD-TOKEN" in ids
    assert "TP-PRE-BOS-TOKEN-INFO" in ids

    # 3. PAD == EOS
    tok3 = SimpleNamespace(eos_token="<eos>", pad_token="<pad>", eos_token_id=5, pad_token_id=5, bos_token=None)
    f3 = check_tokenizer(tok3)
    assert len(f3) == 2
    ids = {f["id"] for f in f3}
    assert "TP-PRE-PAD-EQUALS-EOS" in ids
    assert "TP-PRE-BOS-TOKEN-INFO" in ids

def test_check_context_length():
    records, _ = load_jsonl(FIXTURES_DIR / "clean.jsonl")
    # mock tokenizer that treats each character as a token
    tok = SimpleNamespace(encode=lambda t: list(t))
    
    # Skipped
    f1 = check_context_length(records, tok, None, None)
    assert f1[0]["id"] == "TP-PRE-CONTEXT-CHECK-SKIPPED"
    assert f1[0]["level"] == "INFO"

    # Overflow
    f2 = check_context_length(records, tok, 10, None)
    assert f2[0]["id"] == "TP-PRE-CONTEXT-OVERFLOW"
    assert f2[0]["level"] == "WARN"

    # No overflow
    f3 = check_context_length(records, tok, 100, None)
    assert len(f3) == 0

def test_check_preflight():
    records, malformed = load_jsonl(FIXTURES_DIR / "malformed.jsonl")
    res = check_preflight(records, malformed)
    assert res["verdict"] == "FAIL"
    assert res["findings"][0]["id"] == "TP-PRE-MALFORMED-JSONL"

    records, malformed = load_jsonl(FIXTURES_DIR / "clean.jsonl")
    res = check_preflight(records, malformed)
    assert res["verdict"] == "PASS"
    assert res["findings"][-1]["id"] == "TP-PRE-OK"

def test_preflight_api():
    res = preflight(FIXTURES_DIR / "clean.jsonl")
    assert res.verdict == "PASS"
