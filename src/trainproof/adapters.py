import json
import csv
import re
from pathlib import Path
from datetime import datetime

ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

CANONICAL_ALIASES = {
    "loss": ["loss", "train_loss", "training_loss"],
    "eval_loss": ["eval_loss", "val_loss", "validation_loss", "test_loss"],
    "lr": ["lr", "learning_rate", "current_lr"],
    "step": ["step", "iter", "iteration", "global_step"],
    "grad_norm": ["grad_norm", "gnorm", "gradient_norm", "grad_l2"],
    "time": ["time", "timestamp", "elapsed"],
    "step_time": ["step_time", "seconds_per_step"],
    "loader_time": ["loader_time", "data_time"],
    "gpu_util": ["gpu_util"]
}

def _resolve_key(col_name: str, overrides: dict[str, str] = None) -> str | None:
    if overrides is not None:
        for canon, col in overrides.items():
            if col.lower() == col_name.lower():
                return canon
    for canon, aliases in CANONICAL_ALIASES.items():
        if col_name.lower() in aliases:
            return canon
    return None

def parse_hf_trainer_state(text: str, mapping_overrides: dict[str, str] = None) -> tuple[list[dict[str, float]], dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], {}
        
    history = data.get("log_history", [])
    records = []
    used_mapping = {}
    
    for entry in history:
        if "train_runtime" in entry:
            continue
        record = {}
        for col_name, v in entry.items():
            if v is None:
                continue
            canon = _resolve_key(col_name, mapping_overrides)
            if canon:
                try:
                    record[canon] = float(v)
                    used_mapping[canon] = col_name
                except (ValueError, TypeError):
                    pass
        if "loss" in record or "eval_loss" in record:
            records.append(record)
    return records, used_mapping

def parse_coqui_trainer_log(text: str, mapping_overrides: dict[str, str] = None) -> tuple[list[dict[str, float]], dict[str, str]]:
    text = ANSI_ESCAPE_PATTERN.sub('', text)
    records = []
    used_mapping = {}
    current_record = {}
    
    def set_canon(col_name, val):
        canon = _resolve_key(col_name, mapping_overrides)
        if canon:
            current_record[canon] = val
            used_mapping[canon] = col_name

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("--> TIME:") and "-- GLOBAL_STEP:" in line:
            if current_record and ("loss" in current_record or "eval_loss" in current_record):
                records.append(current_record)
            current_record = {}
            
            m_time = re.search(r'TIME:\s*([\d-]+\s[\d:]+)', line)
            if m_time:
                try:
                    dt = datetime.strptime(m_time.group(1), "%Y-%m-%d %H:%M:%S")
                    set_canon("time", dt.timestamp())
                except ValueError:
                    pass
                    
            m_step = re.search(r'GLOBAL_STEP:\s*(\d+)', line)
            if m_step:
                set_canon("step", float(m_step.group(1)))
                
        elif current_record is not None and line.startswith("| > "):
            parts = line[4:].split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                val_str = parts[1].strip().split()[0]
                try:
                    val = float(val_str)
                    set_canon(key, val)
                except ValueError:
                    pass
                        
    if current_record and ("loss" in current_record or "eval_loss" in current_record):
        records.append(current_record)
        
    return records, used_mapping

def parse_generic_log(text: str, is_csv: bool, mapping_overrides: dict[str, str] = None) -> tuple[list[dict[str, float]], dict[str, str]]:
    records = []
    used_mapping = {}
    
    if is_csv:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames:
            resolved_keys = {}
            for col in reader.fieldnames:
                canon = _resolve_key(col.strip(), mapping_overrides)
                if canon:
                    resolved_keys[col] = canon
                    used_mapping[canon] = col
                    
            for row in reader:
                norm_row = {}
                for k, v in row.items():
                    if v is None or not str(v).strip():
                        continue
                    canon = resolved_keys.get(k)
                    if canon:
                        try:
                            norm_row[canon] = float(v)
                        except (ValueError, TypeError):
                            pass
                if norm_row:
                    records.append(norm_row)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                norm_row = {}
                for col_name, v in data.items():
                    col_strip = col_name.strip()
                    canon = _resolve_key(col_strip, mapping_overrides)
                    if canon:
                        try:
                            norm_row[canon] = float(v)
                            used_mapping[canon] = col_strip
                        except (ValueError, TypeError):
                            pass
                if norm_row:
                    records.append(norm_row)
            except Exception:
                pass
    return records, used_mapping

def parse_log_with_format_info(path: str | Path, fmt: str = "auto", mapping_overrides: dict[str, str] = None) -> tuple[list[dict[str, float]], str, dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
        
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    
    if fmt == "auto":
        if path.name == "trainer_state.json":
            fmt = "hf"
        elif path.suffix == ".json" and '"log_history"' in text:
            fmt = "hf"
        elif "-- GLOBAL_STEP:" in text:
            fmt = "coqui"
        elif path.suffix == ".csv" or (text and text[0] != '{'):
            fmt = "csv"
        else:
            fmt = "jsonl"
            
    if fmt == "hf":
        records, mapping = parse_hf_trainer_state(text, mapping_overrides)
        return records, fmt, mapping
    elif fmt == "coqui":
        records, mapping = parse_coqui_trainer_log(text, mapping_overrides)
        return records, fmt, mapping
    elif fmt == "csv":
        records, mapping = parse_generic_log(text, True, mapping_overrides)
        return records, fmt, mapping
    elif fmt == "jsonl":
        records, mapping = parse_generic_log(text, False, mapping_overrides)
        return records, fmt, mapping
    else:
        raise ValueError(f"Unknown format: {fmt}")

def parse_log_with_format(path: str | Path, fmt: str = "auto", mapping_overrides: dict[str, str] = None) -> list[dict[str, float]]:
    records, _, _ = parse_log_with_format_info(path, fmt, mapping_overrides)
    return records
