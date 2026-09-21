"""Regression tests for the bounded v0.x truthfulness pass (2026-09-21).

Each test pins one of the seven repairs classified SAFE v0.x in
`TRAINPROOF_REPAIR_BOUNDARY.md`. Every one of them fails on the pre-repair code.

What these tests deliberately do NOT assert: any of the architectural defects the
two forensic audits established. TP-DIVERGE still reads one training series and
still cannot be overruled by an improving held-out series; checkpoint PASS still
means container integrity; coverage still cannot attribute a gap to trainproof.
Those are vNext work and nothing here should be read as evidence they are fixed.
"""
import json
import sys
from pathlib import Path

import pytest

from trainproof.epoch import check_records
from trainproof.preflight import check_preflight, load_jsonl, preflight

# --------------------------------------------------------------------- S-1

def test_missing_sentencepiece_is_not_a_verdict_on_the_tokenizer(monkeypatch):
    """A missing optional dependency is trainproof's problem, not the user's.

    Pre-repair: TP-TOK-SPM-MISSING was level FAIL, the report verdict was FAIL,
    and `_get_exit_code` therefore returned 1 -- a user-facing failure verdict
    about a tokenizer trainproof never read, emitted because trainproof was
    missing one of its own optional packages. cli.py already implemented the
    opposite policy for `transformers`.
    """
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    from trainproof.cli import _get_exit_code
    from trainproof.speech.tokenizer import check_tokenizer

    report = check_tokenizer("no-such.model", "no-such.txt")
    finding = report["findings"][0]

    assert finding["id"] == "TP-TOK-SPM-MISSING"
    assert finding["level"] == "NOT-CHECKED", "a missing dependency is not a FAIL"
    assert report["verdict"] == "NOT-CHECKED"
    assert _get_exit_code([report]) == 2, "must be 'could not judge', never 'failed'"
    assert _get_exit_code([report]) != 1


def test_an_unreadable_tokenizer_model_is_still_a_fail(monkeypatch):
    """The complement: S-1 must not soften a real artifact fault.

    A model file that exists and cannot be loaded is a finding about the
    ARTIFACT, and stays FAIL. S-1 changed one line -- the verdict mapping that
    now distinguishes a checker outcome from an artifact fault -- so that line is
    what this pins.

    `load_tokenizer` is stubbed rather than driven with a real corrupt model.
    With sentencepiece absent, `load_tokenizer` short-circuits on the
    SPM-MISSING branch and never reaches the load-failure branch, so a real
    corrupt file can only exercise this in an environment that installs a
    package no extra declares. The dependency-present path was separately
    executed against sentencepiece 0.2.2 during release-candidate verification.
    """
    from trainproof.speech import tokenizer as tok_mod

    load_fail = {
        "id": "TP-TOK-LOAD-FAIL", "level": "FAIL",
        "message": "Failed to load SentencePiece model.",
        "evidence": "broken.model: unable to parse",
    }
    monkeypatch.setattr(tok_mod, "load_tokenizer", lambda p: (None, load_fail))

    report = tok_mod.check_tokenizer("broken.model", "no-such.txt")
    assert report["findings"][0]["id"] == "TP-TOK-LOAD-FAIL"
    assert report["verdict"] == "FAIL", "an unreadable artifact is still the user's fault"


# --------------------------------------------------------------------- S-2

def test_watch_loop_accepts_and_forwards_mapping_overrides(tmp_path):
    """`watch --map` was registered, parsed, and then dropped.

    Pre-repair: `watch_loop` took no `mapping_overrides` parameter at all, so
    live monitoring silently ignored the column mapping the user asked for and
    reported "warming up" forever on a log whose loss column had a custom name.
    """
    import inspect

    from trainproof.watch import poll_once, watch_loop

    assert "mapping_overrides" in inspect.signature(watch_loop).parameters
    assert "mapping_overrides" in inspect.signature(poll_once).parameters

    log = tmp_path / "run.jsonl"
    log.write_text(
        "\n".join(json.dumps({"step": i, "total_loss": 2.0 - i * 0.05}) for i in range(20)),
        encoding="utf-8",
    )

    # Without the override the custom column is invisible: no loss, no verdict.
    verdict_without, _, _ = poll_once(log, "jsonl", None)
    assert verdict_without is None

    # With it, the column resolves and the run is judged.
    verdict_with, _, n = poll_once(log, "jsonl", None, {"loss": "total_loss"})
    assert verdict_with is not None
    assert n == 20


# --------------------------------------------------------------------- S-3

def test_negative_learning_rate_is_not_described_as_zero():
    """The evidence string asserted something false about the artifact.

    Pre-repair: a series where every logged lr was -1e-4 produced the evidence
    "100.0% of steps have lr=0". No step had lr=0.

    The PREDICATE is unchanged and this test does not touch it: `lr <= 0` still
    counts negatives, and TP-ZERO-LR still fires. Narrowing the predicate would
    change which runs FAIL, which needs calibration and is deferred.
    """
    negative = check_records([{"loss": 2.0 - i * 0.05, "lr": -1e-4} for i in range(20)])
    lr_findings = [f for f in negative["findings"] if f["id"] == "TP-ZERO-LR"]
    assert len(lr_findings) == 1, "the predicate is unchanged: this must still fire"

    evidence = lr_findings[0]["evidence"]
    assert "lr=0" not in evidence, "must not claim lr was zero when it was negative"
    assert "lr <= 0" in evidence
    assert "negative" in evidence

    # A genuinely zero series must not gain a spurious 'negative' clause.
    zeroed = check_records([{"loss": 2.0 - i * 0.05, "lr": 0.0} for i in range(20)])
    zero_evidence = [f for f in zeroed["findings"] if f["id"] == "TP-ZERO-LR"][0]["evidence"]
    assert "negative" not in zero_evidence


# --------------------------------------------------------------------- S-4

def test_manifest_relative_audio_paths_resolve_against_the_manifest(tmp_path, monkeypatch):
    """Relative paths resolved against the process CWD, not the manifest.

    Pre-repair: a valid manifest invoked from anywhere other than its own folder
    reported every file as TP-DATA-MISSING-AUDIO -- a FAIL about the user's data
    caused by where trainproof happened to be launched from.
    """
    pytest.importorskip("soundfile", reason="needs the speech extra")
    from trainproof.speech.data import check_data

    corpus = tmp_path / "corpus"
    (corpus / "wavs").mkdir(parents=True)
    # Contents do not matter: existence is tested before any decode, and a file
    # that exists but will not decode is TP-DATA-UNREADABLE-AUDIO, not MISSING.
    (corpus / "wavs" / "a.wav").write_bytes(b"\0")
    manifest = corpus / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": "wavs/a.wav", "text": "hello"}) + "\n",
        encoding="utf-8",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = check_data(manifest)
    ids = {f["id"] for f in report["findings"]}
    assert "TP-DATA-MISSING-AUDIO" not in ids, (
        "the file exists next to its manifest; only the CWD differed"
    )


def test_absolute_audio_paths_are_left_alone(tmp_path, monkeypatch):
    """S-4 must not rewrite absolute paths."""
    pytest.importorskip("soundfile", reason="needs the speech extra")
    from trainproof.speech.data import check_data

    audio = tmp_path / "abs.wav"
    audio.write_bytes(b"\0")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": str(audio), "text": "hello"}) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path.parent)

    ids = {f["id"] for f in check_data(manifest)["findings"]}
    assert "TP-DATA-MISSING-AUDIO" not in ids


def test_a_manifest_in_a_nested_directory_resolves_against_that_directory(tmp_path, monkeypatch):
    """The manifest's own directory is the base, however deep it sits."""
    pytest.importorskip("soundfile", reason="needs the speech extra")
    from trainproof.speech.data import check_data

    deep = tmp_path / "datasets" / "ljspeech" / "v2"
    (deep / "wavs").mkdir(parents=True)
    (deep / "wavs" / "a.wav").write_bytes(b"\0")
    manifest = deep / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": "wavs/a.wav", "text": "hello"}) + "\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    ids = {f["id"] for f in check_data(manifest)["findings"]}
    assert "TP-DATA-MISSING-AUDIO" not in ids


def test_a_relative_directory_input_is_not_rebased_onto_itself(tmp_path, monkeypatch):
    """`trainproof data corpus` must not check `corpus/corpus/a.wav`.

    REGRESSION (found by the final independent audit, 2026-09-21). The S-4 repair
    rebased every non-absolute record path onto the input path. For a MANIFEST that
    is right. For a DIRECTORY it is wrong twice over: `rglob` already yields paths
    carrying the input directory as their prefix, so rebasing prepended it a second
    time and a valid corpus given by relative path FAILed with
    TP-DATA-MISSING-AUDIO. The same corpus passed when given by absolute path,
    because an absolute path is exempt from rebasing -- which is exactly why the
    original S-4 tests, which all use absolute `tmp_path` inputs, did not catch it.

    Paths discovered by traversal are already correct and must not be rebased.
    """
    pytest.importorskip("soundfile", reason="needs the speech extra")
    from trainproof.speech.data import check_data

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"\0")
    (corpus / "a.txt").write_text("hello world", encoding="utf-8")

    # The input is RELATIVE -- the reported condition. Absolute already worked.
    monkeypatch.chdir(tmp_path)

    probed: list[str] = []
    real_exists = Path.exists

    def spy(self, *args, **kwargs):
        probed.append(str(self))
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", spy)
    report = check_data("corpus")

    ids = {f["id"] for f in report["findings"]}
    assert "TP-DATA-MISSING-AUDIO" not in ids, (
        "corpus/a.wav exists; a relative directory input must not manufacture a FAIL"
    )

    # Negative control: no doubled prefix may be produced anywhere.
    doubled = [p for p in probed if Path(p).parts.count("corpus") > 1]
    assert not doubled, f"input directory prepended twice: {doubled}"

    # The audio file is addressed as exactly one path, not two candidates.
    wavs = {Path(p).as_posix() for p in probed if p.endswith(".wav")}
    assert wavs == {"corpus/a.wav"}, wavs


# --------------------------------------------------------------------- S-5

@pytest.mark.parametrize("body", ["123", "null", "[1, 2, 3]", '"a string"', "true"])
def test_valid_json_that_is_not_a_record_yields_a_verdict_not_a_crash(tmp_path, body):
    """CONTRACTS.md promises a verdict or a documented exit.

    Pre-repair: `123` and `null` reached `field in record` inside `_extract_text`
    and raised TypeError out of preflight.
    """
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(body for _ in range(5)), encoding="utf-8")

    result = preflight(p)  # must not raise
    assert result.verdict in {"PASS", "WARN", "FAIL", "NOT-CHECKED"}
    assert "TP-PRE-MALFORMED-JSONL" in {f["id"] for f in result.findings}

    records, malformed = load_jsonl(p)
    assert records == []
    assert len(malformed) == 5


def test_non_record_entries_in_an_in_memory_iterable_are_also_malformed():
    """preflight(<iterable>) bypasses load_jsonl, so it is guarded separately."""
    report = check_preflight([123, None, {"text": "a real row"}])
    assert "TP-PRE-MALFORMED-JSONL" in {f["id"] for f in report["findings"]}


def _malformed_finding(report_findings):
    return [f for f in report_findings if f["id"] == "TP-PRE-MALFORMED-JSONL"][0]


@pytest.mark.parametrize("body", ["123", "null", "[1, 2]", '"a string"', "true"])
def test_valid_json_that_is_not_an_object_is_not_called_a_parsing_failure(tmp_path, body):
    """The S-5 crash fix left the diagnostic asserting something false.

    Pre-correction the finding read "JSONL parsing failed." with evidence
    "N broken lines. First bad line number: N". For `123`, `null` and `[1,2]`
    `json.loads` SUCCEEDED -- nothing was broken and no parse failed. Only the
    decoded value is not a record. Reporting a parser failure for a line the
    parser read is a false statement about the artifact.

    This asserts the user-facing sentences, not the rule ID: the ID was already
    correct and is exactly what hid the defect.
    """
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(body for _ in range(3)), encoding="utf-8")

    finding = _malformed_finding(preflight(p).findings)
    text = f"{finding['message']} {finding['evidence']}".lower()

    assert "parsing failed" not in text, "the parser succeeded on this line"
    assert "broken" not in text, "'broken' asserts a syntax failure that did not happen"
    assert "not an object" in text, "the real condition must be named"
    assert "valid json" in text, "the evidence must say the JSON itself was fine"
    assert finding["level"] == "FAIL", "the verdict is unchanged by this correction"


def test_genuinely_invalid_json_is_still_reported_as_invalid_json(tmp_path):
    """The complement: real syntax errors must still be described as such."""
    p = tmp_path / "rows.jsonl"
    p.write_text('{"text": "a valid row here"}\n{not json at all\n', encoding="utf-8")

    finding = _malformed_finding(preflight(p).findings)
    text = f"{finding['message']} {finding['evidence']}".lower()

    assert "not valid json" in text
    assert "not an object" not in text, "no structural failure occurred here"
    assert "line 2" in text, "the offending line must still be located"
    assert finding["level"] == "FAIL"


def test_both_failure_classes_in_one_file_are_reported_separately(tmp_path):
    """One ID covers two conditions, so the evidence has to distinguish them."""
    p = tmp_path / "rows.jsonl"
    p.write_text('{"text": "a valid row here"}\n{not json at all\n123\n', encoding="utf-8")

    finding = _malformed_finding(preflight(p).findings)
    evidence = finding["evidence"].lower()

    assert "1 not valid json" in evidence
    assert "1 valid json but not an object" in evidence
    assert "2 unusable lines" in evidence


def test_valid_object_records_are_untouched_by_the_wording_fix(tmp_path):
    """Negative control: a clean file must gain no malformed finding at all."""
    p = tmp_path / "rows.jsonl"
    p.write_text(
        json.dumps({"text": "the first genuine row"}) + "\n"
        + json.dumps({"text": "the second genuine row"}) + "\n",
        encoding="utf-8",
    )

    result = preflight(p)
    assert "TP-PRE-MALFORMED-JSONL" not in {f["id"] for f in result.findings}
    assert result.verdict == "PASS"


def test_s5_does_not_change_how_a_present_null_text_value_is_judged(tmp_path):
    """Scope pin, not an endorsement.

    `{"text": null}` still stringifies to "None" and still passes the empty-text
    gate. Audit finding V2-34 established that this is wrong. Correcting it
    changes verdicts on real datasets, so it is deferred to vNext rather than
    folded into a crash fix. This test exists so the deferral is deliberate and
    visible, and so a future repair has to change it on purpose.
    """
    p = tmp_path / "nulls.jsonl"
    p.write_text(
        "\n".join(json.dumps({"text": None, "output": f"real answer {i}"}) for i in range(5)),
        encoding="utf-8",
    )
    ids = {f["id"] for f in preflight(p).findings}
    assert "TP-PRE-EMPTY-TEXT" not in ids, "KNOWN DEFECT, deferred to vNext: see V2-34"


# --------------------------------------------------------------------- S-6
#
# These do NOT skip when `transformers` is absent. `integrations/hf.py` ships a
# stub TrainerCallback for that case, and the repository's own HF tests
# (tests/test_hf_callback.py:5, tests/test_hf_objective.py:16) exercise the
# callback by setting the module flag. Guarding with importorskip here would
# make the S-6 regressions permanently unrunnable, because no extra declares
# `transformers` -- so the project's "no test is skipped when every extra is
# installed" gate could never satisfy them.
#
# The dependency-present path was separately executed against transformers
# 5.17.0 during release-candidate verification: all three properties below held
# with `_HAS_TRANSFORMERS` true from the real import.

import trainproof.integrations.hf as hf_mod  # noqa: E402

hf_mod._HAS_TRANSFORMERS = True


class _Control:
    should_training_stop = False


class _Model:
    class config:
        vocab_size = 1025

    def modules(self):
        return []


def test_objective_check_is_off_by_default():
    """The callback described as 'only observes' iterated the training dataloader.

    Pre-repair: `objective_check` defaulted to True, so `on_train_begin` created
    a fresh iterator over the caller's train_dataloader. For a map-style loader
    with a random sampler that draws from a generator, doing so changes the batch
    order of the run being measured.
    """
    from trainproof.integrations.hf import TrainproofCallback

    assert TrainproofCallback().objective_check is False


def test_default_callback_does_not_touch_the_dataloader():
    from trainproof.integrations.hf import TrainproofCallback

    class ExplodingLoader:
        def __iter__(self):
            raise AssertionError("the default callback must not iterate the dataloader")

    TrainproofCallback().on_train_begin(
        None, None, _Control(), model=_Model(), train_dataloader=ExplodingLoader()
    )


def test_objective_checks_still_run_when_explicitly_requested():
    """Opting in restores the previous behaviour, sampling included."""
    from trainproof.integrations.hf import TrainproofCallback

    seen = []

    class CountingLoader:
        def __iter__(self):
            seen.append(1)
            return iter([])

    cb = TrainproofCallback(objective_check=True)
    assert cb.objective_check is True
    cb.on_train_begin(None, None, _Control(), model=_Model(), train_dataloader=CountingLoader())
    assert seen, "explicit objective_check=True must still sample the loader"


# --------------------------------------------------------------------- S-7

def test_no_dead_accumulators_remain():
    """`bit_depths` and `total_chars` were computed on every record and never read."""
    data_src = Path("src/trainproof/speech/data.py").read_text(encoding="utf-8")
    tok_src = Path("src/trainproof/speech/tokenizer.py").read_text(encoding="utf-8")
    assert "bit_depths" not in data_src
    assert "total_chars" not in tok_src
