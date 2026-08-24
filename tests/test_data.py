import json

import pytest

# The speech pack (numpy, soundfile, ttsproof) left the core install in v0.18.1
# and became the `trainproof[speech]` extra. This module is the only one that
# needs it, and a plain `import` here fails at COLLECTION, which aborts the
# whole session rather than this file: a clean `pip install -e ".[dev]"`
# checkout ran zero tests, not 274. Skipping keeps the suite runnable on a core
# install; the `speech` job in CI is where these tests actually execute.
# The skip is keyed on the third-party packages, not on
# `trainproof.speech.data`. pytest 9 narrowed `importorskip` to catch
# ModuleNotFoundError only, and data.py deliberately re-raises a plain
# ImportError carrying the install instruction - so asking for the module would
# not skip. Asking for what is actually missing works on every pytest version,
# and a genuine break inside data.py still fails loudly instead of vanishing
# into a skip.
for _dependency in ("numpy", "soundfile", "ttsproof"):
    pytest.importorskip(_dependency, reason="trainproof[speech] is not installed")

from trainproof.speech.data import check_data  # noqa: E402


def test_data_empty_manifest(tmp_path):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("")
    report = check_data(manifest)
    assert report["verdict"] == "FAIL"

def test_data_valid_manifest(tmp_path):
    # transcript-only records (no audio field) are allowed
    manifest = tmp_path / "valid.jsonl"
    manifest.write_text(json.dumps({"text": "This is a normal sentence."}) + "\n")
    report = check_data(manifest)
    assert report["verdict"] == "PASS"

def test_data_unnormalized_text(tmp_path):
    manifest = tmp_path / "unnorm.jsonl"
    manifest.write_text(json.dumps({"text": "I was born in 1999."}) + "\n")
    report = check_data(manifest)
    assert report["verdict"] == "WARN"
    assert any("Unnormalized" in str(f) for f in report["findings"])

def test_data_missing_audio_fails(tmp_path):
    # a manifest pointing at audio that does not exist is a broken dataset
    manifest = tmp_path / "missing.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "nonexistent.wav", "text": "Hello there."}) + "\n")
    report = check_data(manifest)
    assert report["verdict"] == "FAIL"
    assert any("do not exist" in str(f) for f in report["findings"])
