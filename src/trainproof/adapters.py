import json
import csv
import re
from pathlib import Path
from datetime import datetime

ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def parse_hf_trainer_state(text: str) -> list[dict[str, float]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
        
    history = data.get("log_history", [])
    records = []
    for entry in history:
        if "loss" not in entry:
            continue
        record = {}
        for k in ["loss", "grad_norm", "step"]:
            if k in entry and entry[k] is not None:
                try:
                    record[k] = float(entry[k])
                except (ValueError, TypeError):
                    pass
        if "learning_rate" in entry and entry["learning_rate"] is not None:
            try:
                record["lr"] = float(entry["learning_rate"])
            except (ValueError, TypeError):
                pass
        records.append(record)
    return records

def parse_coqui_trainer_log(text: str) -> list[dict[str, float]]:
    text = ANSI_ESCAPE_PATTERN.sub('', text)
    records = []
    current_record = None
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("--> TIME:") and "-- GLOBAL_STEP:" in line:
            if current_record is not None and "loss" in current_record:
                records.append(current_record)
            current_record = {}
            
            m_time = re.search(r'TIME:\s*([\d-]+\s[\d:]+)', line)
            if m_time:
                try:
                    dt = datetime.strptime(m_time.group(1), "%Y-%m-%d %H:%M:%S")
                    current_record["time"] = dt.timestamp()
                except ValueError:
                    pass
                    
            m_step = re.search(r'GLOBAL_STEP:\s*(\d+)', line)
            if m_step:
                current_record["step"] = float(m_step.group(1))
                
        elif current_record is not None and line.startswith("| > "):
            parts = line[4:].split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                if key in ("loss", "current_lr"):
                    val_str = parts[1].strip().split()[0]
                    try:
                        val = float(val_str)
                        if key == "loss":
                            current_record["loss"] = val
                        elif key == "current_lr":
                            current_record["lr"] = val
                    except ValueError:
                        pass
                        
    if current_record is not None and "loss" in current_record:
        records.append(current_record)
        
    return records

def parse_generic_log(text: str, is_csv: bool) -> list[dict[str, float]]:
    records = []
    if is_csv:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames:
            for row in reader:
                norm_row = {}
                for k, v in row.items():
                    if v is None or not str(v).strip():
                        continue
                    try:
                        norm_row[k.strip().lower()] = float(v)
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
                for k, v in data.items():
                    try:
                        norm_row[k.strip().lower()] = float(v)
                    except (ValueError, TypeError):
                        pass
                records.append(norm_row)
            except Exception:
                pass
    return records

def parse_log_with_format_info(path: str | Path, fmt: str = "auto") -> tuple[list[dict[str, float]], str]:
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
        return parse_hf_trainer_state(text), fmt
    elif fmt == "coqui":
        return parse_coqui_trainer_log(text), fmt
    elif fmt == "csv":
        return parse_generic_log(text, is_csv=True), fmt
    elif fmt == "jsonl":
        return parse_generic_log(text, is_csv=False), fmt
    else:
        raise ValueError(f"Unknown format: {fmt}")

def parse_log_with_format(path: str | Path, fmt: str = "auto") -> list[dict[str, float]]:
    records, _ = parse_log_with_format_info(path, fmt)
    return records
