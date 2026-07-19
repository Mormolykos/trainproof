import argparse
import sys
from pathlib import Path
from .speech.data import check_data
from .speech.tokenizer import check_tokenizer
from .epoch import check_epoch
from .compare import check_compare
from .watch import watch_loop
from .report import print_verdict_console, write_html_report, print_doctor_autopsy, print_doctor_footer, print_compare_table
from .adapters import parse_log_with_format

def main():
    parser = argparse.ArgumentParser(description="Trainproof: A deterministic linter for ML training runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data Command
    data_parser = subparsers.add_parser("data", help="Dataset preflight for speech/TTS corpora.")
    data_parser.add_argument("input", type=str, help="Directory or manifest.jsonl")

    # Tokenizer Command
    tok_parser = subparsers.add_parser("tokenizer", help="Tokenizer preflight.")
    tok_parser.add_argument("model", type=str, help="Path to tokenizer model (e.g., SentencePiece .model)")
    tok_parser.add_argument("transcripts", type=str, help="Path to transcripts text file or JSONL")

    # Epoch Command
    epoch_parser = subparsers.add_parser("epoch", help="First-epoch verdict from training logs.")
    epoch_parser.add_argument("logfile", type=str, help="Path to JSONL or CSV log file")
    epoch_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv"], default="auto", help="Log format override")

    # Doctor Command
    doctor_parser = subparsers.add_parser("doctor", aliases=["diagnose"], help="Flagship zero-config autopsy of training logs.")
    doctor_parser.add_argument("path", type=str, nargs="?", default=".", help="Path to file or directory")
    doctor_parser.add_argument("--baseline", type=str, help="Additionally run compare against this baseline")
    doctor_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv"], default="auto", help="Log format override (file mode only)")

    # Compare Command
    compare_parser = subparsers.add_parser("compare", help="Compare a run log against a baseline log.")
    compare_parser.add_argument("baseline", type=str, help="Path to baseline log file")
    compare_parser.add_argument("runs", type=str, nargs="+", help="Path to one or more run log files")
    compare_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv"], default="auto", help="Log format override")

    # Watch Command
    watch_parser = subparsers.add_parser("watch", help="Live guardian: poll a training log.")
    watch_parser.add_argument("logfile", type=str, help="Path to run log file")
    watch_parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds")
    watch_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv"], default="auto", help="Log format override")
    watch_parser.add_argument("--until-fail", action="store_true", help="Exit with code 1 as soon as verdict becomes FAIL")

    # Preflight Command
    preflight_parser = subparsers.add_parser("preflight", help="Pre-flight linter for LLM fine-tuning datasets.")
    preflight_parser.add_argument("dataset", type=str, help="Path to JSONL dataset")
    preflight_parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path")
    preflight_parser.add_argument("--max-len", type=int, help="Max context length")
    preflight_parser.add_argument("--text-field", type=str, help="Text field to extract")

    args = parser.parse_args()
    
    report_dict = {}

    if args.command == "data":
        report_dict = check_data(args.input)
    elif args.command == "tokenizer":
        report_dict = check_tokenizer(args.model, args.transcripts)
    elif args.command == "epoch":
        try:
            report_dict = check_epoch(args.logfile, fmt=args.format)
        except Exception as e:
            print(f"Error checking epoch: {e}")
            sys.exit(1)
    def _detect_format(path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if path.name == "trainer_state.json": return "hf"
        if path.suffix == ".json" and '"log_history"' in text: return "hf"
        if "-- GLOBAL_STEP:" in text: return "coqui"
        if path.suffix == ".csv" or (text and text[0] != '{'): return "csv"
        return "jsonl"

    if getattr(args, "command", None) in ("doctor", "diagnose"):
        path = Path(args.path)
        candidates = []
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            for p in path.rglob("*"):
                if not p.is_file(): continue
                if p.name == "trainer_state.json" or p.suffix in (".jsonl", ".csv", ".log", ".txt"):
                    try:
                        records = parse_log_with_format(p, fmt="auto")
                        if len(records) >= 3:
                            candidates.append(p)
                    except Exception:
                        pass
        else:
            print(f"Path not found: {path}")
            sys.exit(2)
            
        if not candidates:
            print("No valid logs found.")
            sys.exit(2)
            
        if path.is_dir():
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if len(candidates) > 20:
                print(f"Note: Found {len(candidates)} logs, capping to 20 most recently modified. Skipped {len(candidates)-20} logs.")
                candidates = candidates[:20]
                
        # Run checks
        results = []
        from .epoch import check_records
        for p in candidates:
            fmt = args.format if path.is_file() else "auto"
            try:
                records = parse_log_with_format(p, fmt)
            except Exception:
                records = []
            if not records: continue
            
            fmt_str = args.format if args.format != "auto" else _detect_format(p)
            step_range = f"{records[0].get('step', 0)}..{records[-1].get('step', len(records)-1)}"
            
            report = check_records(records)
            report["file"] = str(p)
            report["fmt"] = fmt_str
            report["num_records"] = len(records)
            report["step_range"] = step_range
            
            if args.baseline:
                try:
                    cmp = check_compare(p, args.baseline, fmt=fmt)
                    report["compare_findings"] = cmp["findings"]
                except Exception:
                    pass
            results.append(report)
            
        # triage order: failures first, then warnings, then passes
        _order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        results.sort(key=lambda r: _order.get(r["verdict"], 3))

        if len(results) > 1:
            print("SUMMARY")
            print("-" * 60)
            for r in results:
                print(f"{r['verdict']:<4} | {r['file']}")
            print()
            
        worst_verdict = "PASS"
        for r in results:
            if r["verdict"] == "FAIL": worst_verdict = "FAIL"
            elif r["verdict"] == "WARN" and worst_verdict == "PASS": worst_verdict = "WARN"
            
            print_doctor_autopsy(r["file"], r["fmt"], r["num_records"], r["step_range"], r["verdict"], r["findings"])
            if "compare_findings" in r:
                print("--- VS BASELINE ---")
                for f in r["compare_findings"]:
                    level = f.get("level", "INFO")
                    print(f"[{level}] {f.get('message', '')}")
                    if f.get('evidence'): print(f"       Evidence: {f.get('evidence')}")
                print()
                
        print_doctor_footer()
        sys.exit(1 if worst_verdict == "FAIL" else 0)

    elif args.command == "compare":
        try:
            from .compare import extract_metrics
            baseline_records = parse_log_with_format(args.baseline, args.format)
            base_metrics = extract_metrics(baseline_records) if baseline_records else None
            
            table_results = []
            if base_metrics:
                table_results.append({
                    "name": str(Path(args.baseline).name) + " (BASE)",
                    "start": base_metrics["start_med"],
                    "floor": base_metrics["floor"],
                    "end": base_metrics["end_med"],
                    "improvement": base_metrics["improvement"],
                    "verdict": "BASELINE"
                })
            
            all_reports = []
            for run_path in args.runs:
                report_dict = check_compare(run_path, args.baseline, fmt=args.format)
                all_reports.append(report_dict)
                
                run_records = parse_log_with_format(run_path, args.format)
                run_metrics = extract_metrics(run_records) if run_records else None
                if run_metrics:
                    table_results.append({
                        "name": str(Path(run_path).name),
                        "start": run_metrics["start_med"],
                        "floor": run_metrics["floor"],
                        "end": run_metrics["end_med"],
                        "improvement": run_metrics["improvement"],
                        "verdict": report_dict["verdict"]
                    })
            
            print_compare_table(table_results)
            
            # detailed output if exactly 2 paths (1 baseline + 1 run)
            if len(args.runs) == 1:
                print_verdict_console(all_reports[0]["verdict"], all_reports[0]["findings"])
                sys.exit(1 if all_reports[0]["verdict"] == "FAIL" else 0)
            
            sys.exit(1 if any(r["verdict"] == "FAIL" for r in all_reports) else 0)
                
        except Exception as e:
            print(f"Error checking compare: {e}")
            sys.exit(1)
    elif args.command == "watch":
        watch_loop(args.logfile, interval=args.interval, fmt=args.format, until_fail=args.until_fail)
        sys.exit(0)
    elif args.command == "preflight":
        from .preflight import preflight
        tokenizer = None
        if args.tokenizer:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            except ImportError:
                report_dict = {
                    "verdict": "FAIL",
                    "findings": [{"id": "transformers_missing", "level": "FAIL", "message": "pip install transformers is required to load tokenizer", "evidence": ""}]
                }
                print_verdict_console(report_dict["verdict"], report_dict["findings"])
                sys.exit(1)
        
        result = preflight(args.dataset, tokenizer=tokenizer, max_len=args.max_len, text_field=args.text_field)
        report_dict = {"verdict": result.verdict, "findings": result.findings}

    print_verdict_console(report_dict.get("verdict", "FAIL"), report_dict.get("findings", []))
    
    # Write self-contained HTML report
    html_out = Path("trainproof_report.html")
    write_html_report(report_dict, html_out)
    print(f"\nSaved detailed HTML report to {html_out.absolute()}")
    
    if report_dict.get("verdict") == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
