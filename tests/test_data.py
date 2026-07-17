import pytest
from pathlib import Path
from trainproof.speech.data import check_data
import json

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
