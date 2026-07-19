import os
import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf
from ttsproof.normalize import normalize_text, plain_token
from .. import rules

def get_audio_hash(filepath: str | Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def check_data(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    findings = []
    verdict = "PASS"
    
    records = []
    if path.is_file() and path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            records.append(json.loads(line))
    elif path.is_dir():
        for wav_path in path.rglob("*.wav"):
            txt_path = wav_path.with_suffix(".txt")
            txt = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            records.append({"audio_filepath": str(wav_path), "text": txt})
    else:
        return {"verdict": "FAIL", "findings": [{"id": "TP-DATA-INVALID-INPUT", "level": "FAIL", "message": "Input must be a directory or manifest.jsonl.", "evidence": str(path)}]}

    if not records:
        return {"verdict": "FAIL", "findings": [{"id": "TP-DATA-NO-DATA", "level": "FAIL", "message": "No data found.", "evidence": ""}]}

    # Audio corpus stats
    sample_rates = set()
    channels = set()
    bit_depths = set()
    audio_hashes = {}
    duplicates = 0
    durations = []
    
    clipping_count = 0
    silent_count = 0
    missing_audio = 0
    unreadable_audio = 0
    chars_per_sec = []  # (file, rate) pairs for text/audio alignment check
    
    # Text stats
    empty_transcripts = 0
    unnormalized_count = 0
    mixed_scripts_count = 0
    text_lengths = []
    
    for r in records:
        audio_file = Path(r.get("audio_filepath", r.get("audio", "")))
        text = r.get("text", r.get("transcript", ""))
        
        # Transcript checks
        if not text:
            empty_transcripts += 1
        else:
            text_lengths.append(len(text))
            normalized = normalize_text(text)
            # if normalization significantly changes the text, we flag it as unnormalized
            if normalized != text and plain_token(normalized) != plain_token(text):
                unnormalized_count += 1
            
            # naive charset check for mixed scripts (e.g., Cyrillic + Latin)
            has_latin = any('a' <= c.lower() <= 'z' for c in text)
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)
            has_greek = any('\u0370' <= c <= '\u03FF' for c in text)
            if sum([has_latin, has_cyrillic, has_greek]) > 1:
                mixed_scripts_count += 1
        
        # Audio checks (records without an audio field are transcript-only — allowed)
        if str(audio_file) in ("", "."):
            pass
        elif not audio_file.exists():
            missing_audio += 1
        elif audio_file.exists():
            ahash = get_audio_hash(audio_file)
            if ahash in audio_hashes:
                duplicates += 1
            else:
                audio_hashes[ahash] = str(audio_file)
                
            try:
                info = sf.info(str(audio_file))
                sample_rates.add(info.samplerate)
                channels.add(info.channels)
                bit_depths.add(info.subtype)
                duration = info.frames / info.samplerate
                durations.append(duration)
                if text and duration > 0:
                    chars_per_sec.append((str(audio_file), len(text) / duration))
                
                audio_data, sr = sf.read(str(audio_file), dtype="float32", always_2d=True)
                mono = np.mean(audio_data, axis=1)
                peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
                if peak >= rules.MAX_CLIPPING_PEAK:
                    clipping_count += 1
                    
                silent = np.abs(mono) <= rules.SILENCE_AMPLITUDE
                max_run = 0
                current = 0
                for is_silent in silent:
                    if is_silent: current += 1
                    else:
                        max_run = max(max_run, current)
                        current = 0
                max_run = max(max_run, current)
                max_silence_sec = max_run / float(sr)
                if max_silence_sec > rules.MAX_SILENCE_SEC:
                    silent_count += 1
            except Exception:
                unreadable_audio += 1

    if len(sample_rates) > 1:
        findings.append({"id": "TP-DATA-INCONSISTENT-SR", "level": "FAIL", "message": "Inconsistent sample rates found.", "evidence": f"Rates: {sample_rates}"})
        verdict = "FAIL"
        
    if len(channels) > 1:
        findings.append({"id": "TP-DATA-INCONSISTENT-CHANNELS", "level": "FAIL", "message": "Inconsistent channel counts found.", "evidence": f"Channels: {channels}"})
        verdict = "FAIL"

    if missing_audio > 0:
        findings.append({"id": "TP-DATA-MISSING-AUDIO", "level": "FAIL", "message": "Manifest references audio files that do not exist.", "evidence": f"{missing_audio} of {len(records)} records."})
        verdict = "FAIL"

    if unreadable_audio > 0:
        findings.append({"id": "TP-DATA-UNREADABLE-AUDIO", "level": "WARN", "message": "Audio files could not be read/decoded.", "evidence": f"{unreadable_audio} files."})
        if verdict == "PASS": verdict = "WARN"

    if empty_transcripts > 0:
        findings.append({"id": "TP-DATA-EMPTY-TRANSCRIPT", "level": "WARN", "message": "Empty transcripts.", "evidence": f"{empty_transcripts} of {len(records)} records."})
        if verdict == "PASS": verdict = "WARN"

    if durations:
        too_long = sum(1 for d in durations if d > rules.MAX_AUDIO_DURATION_SEC)
        too_short = sum(1 for d in durations if d < rules.MIN_AUDIO_DURATION_SEC)
        sorted_d = sorted(durations)
        dist = f"min {sorted_d[0]:.2f}s / median {sorted_d[len(sorted_d)//2]:.2f}s / max {sorted_d[-1]:.2f}s"
        if too_long > 0:
            findings.append({"id": "TP-DATA-DURATION-LONG", "level": "WARN", "message": "Audio duration outliers above MAX_AUDIO_DURATION_SEC.", "evidence": f"{too_long} files > {rules.MAX_AUDIO_DURATION_SEC}s ({dist})"})
            if verdict == "PASS": verdict = "WARN"
        if too_short > 0:
            findings.append({"id": "TP-DATA-DURATION-SHORT", "level": "WARN", "message": "Audio duration outliers below MIN_AUDIO_DURATION_SEC.", "evidence": f"{too_short} files < {rules.MIN_AUDIO_DURATION_SEC}s ({dist})"})
            if verdict == "PASS": verdict = "WARN"

    if len(chars_per_sec) >= 10:
        rates = sorted(r for _, r in chars_per_sec)
        median_rate = rates[len(rates) // 2]
        if median_rate > 0:
            outliers = [(f, r) for f, r in chars_per_sec
                        if r > median_rate * rules.CHARS_PER_SEC_OUTLIER_RATIO
                        or r < median_rate / rules.CHARS_PER_SEC_OUTLIER_RATIO]
            if outliers:
                worst = max(outliers, key=lambda x: abs(x[1] - median_rate))
                findings.append({"id": "TP-DATA-CHAR-RATE-OUTLIER", "level": "WARN", "message": "Text-length vs audio-duration outliers (possible transcript/audio mismatch).",
                                 "evidence": f"{len(outliers)} records deviate >{rules.CHARS_PER_SEC_OUTLIER_RATIO}x from median {median_rate:.1f} chars/sec; worst: {worst[0]} ({worst[1]:.1f} chars/sec)"})
                if verdict == "PASS": verdict = "WARN"

    if duplicates > 0:
        findings.append({"id": "TP-DATA-DUPLICATES", "level": "WARN", "message": "Duplicate audio content detected.", "evidence": f"{duplicates} files have identical hashes."})
        if verdict == "PASS": verdict = "WARN"

    if clipping_count > 0:
        findings.append({"id": "TP-DATA-CLIPPING", "level": "WARN", "message": "Audio clipping detected.", "evidence": f"{clipping_count} files exceed MAX_CLIPPING_PEAK ({rules.MAX_CLIPPING_PEAK})."})
        if verdict == "PASS": verdict = "WARN"
        
    if silent_count > 0:
        findings.append({"id": "TP-DATA-SILENCE", "level": "WARN", "message": "Excessive silence detected.", "evidence": f"{silent_count} files exceed MAX_SILENCE_SEC ({rules.MAX_SILENCE_SEC}s)."})
        if verdict == "PASS": verdict = "WARN"
        
    if unnormalized_count > 0:
        findings.append({"id": "TP-DATA-UNNORMALIZED", "level": "WARN", "message": "Unnormalized text that hurts TTS (digits, dates).", "evidence": f"{unnormalized_count} transcripts need normalization."})
        if verdict == "PASS": verdict = "WARN"

    if mixed_scripts_count > 0:
        findings.append({"id": "TP-DATA-MIXED-SCRIPTS", "level": "WARN", "message": "Mixed character scripts in transcripts.", "evidence": f"{mixed_scripts_count} transcripts."})
        if verdict == "PASS": verdict = "WARN"

    if verdict == "PASS":
        findings.append({"id": "TP-DATA-PASS", "level": "PASS", "message": "Dataset preflight completed successfully.", "evidence": f"Processed {len(records)} records."})

    return {"verdict": verdict, "findings": findings}
