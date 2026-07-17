import time
import sys
from datetime import datetime
from pathlib import Path
from .adapters import parse_log_with_format
from .epoch import check_records
from .report import print_verdict_console

def poll_once(path: str | Path, fmt: str, prev_verdict: str | None) -> tuple[str | None, bool, int]:
    try:
        records = parse_log_with_format(path, fmt)
    except Exception:
        records = []
        
    n_records = len(records)
    
    valid_losses = 0
    for r in records:
        if "loss" in r and r["loss"] is not None:
            valid_losses += 1
            
    now_str = datetime.now().strftime("%H:%M:%S")
    
    if valid_losses < 10:
        print(f"[{now_str}] warming up ({n_records} records)")
        return (None, False, n_records)
        
    report = check_records(records)
    verdict = report.get("verdict", "UNKNOWN")
    n_findings = len(report.get("findings", []))
    
    changed = (verdict != prev_verdict)
    
    print(f"[{now_str}] n_records={n_records} verdict={verdict} findings={n_findings}")
    
    if changed:
        print_verdict_console(verdict, report.get("findings", []))
        
    return (verdict, changed, n_records)

def watch_loop(path: str | Path, interval: int = 10, fmt: str = "auto", until_fail: bool = False):
    prev_verdict = None
    try:
        while True:
            verdict, changed, n = poll_once(path, fmt, prev_verdict)
            if verdict is not None:
                prev_verdict = verdict
            
            if until_fail and verdict == "FAIL":
                sys.exit(1)
                
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nStopped watching. Final verdict: {prev_verdict}")
        if prev_verdict == "FAIL":
            sys.exit(1)
        sys.exit(0)
