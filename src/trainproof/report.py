import html as H
from pathlib import Path
from typing import Any

def print_verdict_console(verdict: str, findings: list[dict[str, Any]]):
    """Prints an ASCII-safe verdict summary to the console."""
    print("=" * 40)
    print("TRAINPROOF VERDICT")
    print("=" * 40)
    
    if verdict == "PASS":
        print("[PASS] All checks passed.")
    elif verdict == "WARN":
        print("[WARN] Some checks triggered warnings:")
    else:
        print("[FAIL] Critical checks failed:")
        
    for finding in findings:
        level = finding.get("level", "INFO")
        msg = finding.get("message", "")
        rid = finding.get("id", "")
        prefix = f"[{level}] {rid}: " if rid else f"[{level}] "
        evidence = finding.get("evidence", "")
        print(f"  {prefix}{msg}")
        if evidence:
            print(f"         Evidence: {evidence}")
            
    print("=" * 40)

def write_html_report(report: dict[str, Any], path: str | Path) -> None:
    """Writes a self-contained HTML report."""
    verdict = report.get("verdict", "UNKNOWN")
    findings = report.get("findings", [])
    
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Trainproof report</title>
<style>
 body{{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0d1117;color:#e6edf3;line-height:1.5}}
 .wrap{{max-width:960px;margin:0 auto;padding:28px 18px 60px}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:32px 0 10px}}
 .stat{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 18px;min-width:110px; display:inline-block;}}
 .stat b{{display:block;font-size:22px}} .stat.pass b{{color:#3fb950}} .stat.fail b{{color:#f85149}} .stat.warn b{{color:#d29922}}
 .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;margin:10px 0}}
 .level-FAIL{{color:#f85149;font-weight:bold;}} .level-WARN{{color:#d29922;font-weight:bold;}} .level-PASS{{color:#3fb950;font-weight:bold;}}
 .evidence{{color:#8b949e;font-size:13px;margin-top:4px}}
</style></head><body><div class="wrap">
<h1>Trainproof Report</h1>
<div class="stat {verdict.lower()}"><b>{verdict}</b><span>Overall Verdict</span></div>
<h2>Findings</h2>
"""
    for finding in findings:
        level = finding.get("level", "INFO")
        msg = H.escape(str(finding.get("message", "")))
        rid = H.escape(str(finding.get("id", "")))
        prefix = f"[{level}] {rid}: " if rid else f"[{level}] "
        evidence = H.escape(str(finding.get("evidence", "")))
        doc += f'<div class="card"><div class="level-{level}">{prefix}{msg}</div><div class="evidence">Evidence: {evidence}</div></div>\n'
        
    doc += "</div></body></html>"
    Path(path).write_text(doc, encoding="utf-8")

def print_doctor_autopsy(filepath: str, fmt: str, num_records: int, step_range: str, verdict: str, findings: list[dict[str, Any]]):
    print("=" * 60)
    print(f"FILE   : {filepath}")
    print(f"FORMAT : {fmt}")
    print(f"RECORDS: {num_records} (steps/epochs: {step_range})")
    print("-" * 60)
    print(f"VERDICT: {verdict}")
    print("-" * 60)
    
    passes = warns = fails = 0
    for finding in findings:
        level = finding.get("level", "INFO")
        if level == "PASS": passes += 1
        elif level == "WARN": warns += 1
        elif level == "FAIL": fails += 1
        
        msg = finding.get("message", "")
        rid = finding.get("id", "")
        prefix = f"[{level}] {rid}: " if rid else f"[{level}] "
        evidence = finding.get("evidence", "")
        print(f"{prefix}{msg}")
        if evidence:
            print(f"       Evidence: {evidence}")
        print()
        
    print("-" * 60)
    print(f"Findings: {passes} PASS, {warns} WARN, {fails} FAIL")
    print("=" * 60)
    print()

def print_doctor_footer():
    print("What this cannot tell you")
    print("-" * 25)
    print("A passing report does not mean the run is good. These checks catch")
    print("mechanical failures (divergence, NaN, flatline, spikes) from the log")
    print("alone. They cannot detect a model learning the wrong thing - a run")
    print("trained on corrupted data can look healthy here. For that, compare")
    print("against a known-good baseline: trainproof compare <baseline> <run>")
    print()

def print_compare_table(results: list[dict[str, Any]]):
    if not results: return
    name_len = max(len("RUN"), max((len(r["name"]) for r in results), default=0))
    header = f"{'RUN':<{name_len}} | {'START':>8} | {'FLOOR':>8} | {'END':>8} | {'IMP %':>7} | {'VERDICT vs BASE'}"
    print(header)
    print("-" * len(header))
    for r in results:
        imp_str = f"{r['improvement']*100:.1f}%" if r.get('improvement') is not None else "N/A"
        floor_str = f"{r['floor']:.3f}" if r.get('floor') is not None else "N/A"
        start_str = f"{r['start']:.3f}" if r.get('start') is not None else "N/A"
        end_str = f"{r['end']:.3f}" if r.get('end') is not None else "N/A"
        verdict = r.get("verdict", "BASELINE")
        row = f"{r['name']:<{name_len}} | {start_str:>8} | {floor_str:>8} | {end_str:>8} | {imp_str:>7} | {verdict}"
        print(row)
    print()

