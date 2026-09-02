"""`trainproof tokenizer`: a rate over one population, and no score without one.

Two defects found 2026-09-02, both the same shape — a number reported for
something that was never measured.

  tokens/sec   The numerator counted EVERY line's tokens; the denominator
               counted only the lines that declared a `duration`. A file where
               one line in ten is timed reported roughly ten times the real
               rate, and TP-TOK-HIGH-TPS fired on a healthy tokenizer. When no
               line was timed the check was skipped in silence and the report
               still said "looks healthy".

  coverage     `total_unks / max(1, total_tokens)` made an empty transcripts
               file score 0.000% OOV and 100.000% coverage — TP-TOK-PASS, with
               the evidence "0 tokens evaluated".

sentencepiece is not installed in every environment and is not the thing under
test, so `load_tokenizer` is replaced with a fake whose tokenization is fully
predictable.
"""

import json

import pytest

from trainproof import rules
from trainproof.speech import tokenizer as tok


class FakeSP:
    """One piece per whitespace-separated word. 'oov' becomes <unk>."""

    def encode_as_pieces(self, text):
        return ["<unk>" if w == "oov" else w for w in str(text).split()]


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(tok, "load_tokenizer", lambda path: (FakeSP(), None))


def write(tmp_path, lines):
    p = tmp_path / "transcripts.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def ids(report):
    return [f["id"] for f in report["findings"]]


def finding(report, rule_id):
    return next(f for f in report["findings"] if f["id"] == rule_id)


TEN_WORDS = " ".join(f"w{i}" for i in range(10))


def test_tokens_per_second_is_measured_over_one_population(tmp_path):
    """One timed line at a healthy 1 token/sec, nine untimed lines.

    Real rate: 10 tokens over 10 seconds = 1.0/sec, far under the limit of
    50.0. The old arithmetic put all 100 tokens over that same 10 seconds and
    reported 10.0/sec. Same file, an order of magnitude apart.
    """
    lines = [json.dumps({"text": TEN_WORDS, "duration": 10.0})]
    lines += [json.dumps({"text": TEN_WORDS}) for _ in range(9)]
    report = tok.check_tokenizer("model.model", write(tmp_path, lines))

    assert "TP-TOK-HIGH-TPS" not in ids(report)
    assert report["verdict"] == "PASS"
    assert "1 of 10 line(s)" in finding(report, "TP-TOK-PASS")["evidence"]


def test_a_genuinely_high_rate_still_warns(tmp_path):
    """The control: the check must still fire when the timed lines are bad."""
    lines = [json.dumps({"text": TEN_WORDS, "duration": 0.1}) for _ in range(4)]
    report = tok.check_tokenizer("model.model", write(tmp_path, lines))

    assert "TP-TOK-HIGH-TPS" in ids(report)
    assert report["verdict"] == "WARN"
    evidence = finding(report, "TP-TOK-HIGH-TPS")["evidence"]
    assert f"> {rules.MAX_TOKENS_PER_SEC}" in evidence
    assert "4 of 4 line(s)" in evidence  # the population is stated


def test_a_line_with_no_duration_is_not_a_line_of_zero_seconds(tmp_path):
    """Adding UNTIMED lines must not move the tokens-per-second verdict.

    Two timed lines at 20 tokens over 10 seconds is 2/sec. Adding sixty untimed
    lines put 620 tokens over the same 10 seconds under the old arithmetic —
    62/sec, past the limit of 50 — so the WARN fired because of lines that
    contributed no time at all.
    """
    timed = [json.dumps({"text": TEN_WORDS, "duration": 5.0}) for _ in range(2)]
    one = tok.check_tokenizer("model.model", write(tmp_path, timed))

    p2 = tmp_path / "second.jsonl"
    p2.write_text("\n".join(timed + [json.dumps({"text": TEN_WORDS})] * 60) + "\n",
                  encoding="utf-8")
    two = tok.check_tokenizer("model.model", p2)

    assert "TP-TOK-HIGH-TPS" not in ids(one)
    assert "TP-TOK-HIGH-TPS" not in ids(two), (
        "adding untimed lines changed the tokens-per-second verdict, so the "
        "numerator and the denominator are still different populations"
    )


def test_no_durations_at_all_reports_not_measured(tmp_path):
    """Plain text transcripts: the check cannot run, and says so."""
    report = tok.check_tokenizer("model.model", write(tmp_path, [TEN_WORDS] * 5))

    assert "TP-TOK-TPS-NOT-MEASURED" in ids(report)
    assert finding(report, "TP-TOK-TPS-NOT-MEASURED")["level"] == "NOT-CHECKED"
    # The verdict stays PASS -- plain text is the normal case, and this is not a
    # fault in the tokenizer -- but PASS may not imply an unmeasured axis.
    assert report["verdict"] == "PASS"
    assert "NOT MEASURED" in finding(report, "TP-TOK-PASS")["evidence"]


def test_an_empty_file_does_not_score_100_percent_coverage(tmp_path):
    """Nothing tokenized is NOT-CHECKED. It used to be PASS at 100% coverage."""
    p = tmp_path / "empty.jsonl"
    p.write_text("\n   \n\n", encoding="utf-8")
    report = tok.check_tokenizer("model.model", p)

    assert report["verdict"] == "NOT-CHECKED"
    assert ids(report) == ["TP-TOK-NOT-MEASURED"]
    assert "TP-TOK-PASS" not in ids(report)
    assert finding(report, "TP-TOK-NOT-MEASURED")["level"] == "NOT-CHECKED"


def test_oov_is_still_measured_when_there_are_tokens(tmp_path):
    """The control for the coverage half: a real OOV rate still fails."""
    lines = ["oov " + TEN_WORDS] * 3
    report = tok.check_tokenizer("model.model", write(tmp_path, lines))

    assert "TP-TOK-HIGH-OOV" in ids(report)
    assert report["verdict"] == "FAIL"
