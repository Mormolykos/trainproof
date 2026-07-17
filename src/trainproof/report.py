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
        evidence = finding.get("evidence", "")
        print(f"  [{level}] {msg}")
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
        evidence = H.escape(str(finding.get("evidence", "")))
        doc += f'<div class="card"><div class="level-{level}">[{level}] {msg}</div><div class="evidence">Evidence: {evidence}</div></div>\n'
        
    doc += "</div></body></html>"
    Path(path).write_text(doc, encoding="utf-8")
