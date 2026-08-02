import argparse
import json
import sys
from pathlib import Path

from .adapters import parse_log_with_format, parse_log_with_format_info
from .compare import check_compare
from .envcheck import check_env
from .epoch import check_epoch
from .report import (
    print_compare_table,
    print_doctor_autopsy,
    print_doctor_footer,
    print_verdict_console,
    write_html_report,
)
from .speech.data import check_data
from .speech.tokenizer import check_tokenizer
from .watch import watch_loop


def parse_map(map_list):
    overrides = {}
    if map_list:
        for m in map_list:
            if '=' in m:
                canon, col = m.split('=', 1)
                overrides[canon.strip()] = col.strip()
    return overrides

SCHEMA_VERSION = 3

_JSON_MODE = False
_SARIF_PATH = None

_VERDICT_SEVERITY = {"FAIL": 0, "WARN": 1, "NOT-CHECKED": 2, "PASS": 3}

def _normalize_verdict(v):
    """An unrecognised verdict means trainproof could not judge this report.
    Map it to NOT-CHECKED so the enum stays closed and the exit code is 2."""
    return v if v in _VERDICT_SEVERITY else "NOT-CHECKED"

def _envelope(reports, worst_verdict, error=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "trainproof_version": __import__('trainproof').__version__,
        "reports": reports,
        "worst_verdict": worst_verdict,
        "error": error,
    }

def tag_source(findings, source):
    # every finding carries where it came from, so one flat `findings` array
    # can hold single-run and baseline-comparison results without ambiguity
    return [dict(f, source=source) for f in findings]

def emit_sarif(reports):
    # independent of --json: CI wants the file without JSON on stdout
    if not _SARIF_PATH:
        return
    from .sarif import to_sarif
    doc = to_sarif(reports, __import__('trainproof').__version__)
    Path(_SARIF_PATH).write_text(json.dumps(doc, indent=2), encoding="utf-8")

def _get_worst_verdict(reports):
    worst = "PASS"
    for r in reports:
        v = _normalize_verdict(r.get("verdict"))
        if _VERDICT_SEVERITY.get(v, _VERDICT_SEVERITY["NOT-CHECKED"]) < _VERDICT_SEVERITY.get(worst, 3):
            worst = v
    return worst

def _get_exit_code(reports):
    has_fail = False
    has_unjudged = False
    for r in reports:
        v = _normalize_verdict(r.get("verdict"))
        if v == "FAIL":
            has_fail = True
        elif v == "NOT-CHECKED":
            has_unjudged = True
        for f in r.get("findings", []):
            if f.get("id") in ("TP-NO-RECORDS", "TP-CMP-ERROR"):
                has_unjudged = True
    if has_fail:
        return 1
    elif has_unjudged:
        return 2
    return 0

def output_json(reports, worst_verdict):
    emit_sarif(reports)
    print(json.dumps(_envelope(reports, worst_verdict), indent=2))
    sys.exit(_get_exit_code(reports))

def fail_to_run(message):
    # trainproof could not judge: exit 2 and never synthesise a FAIL verdict,
    # or CI cannot tell a doomed run from an unreadable file.
    if _JSON_MODE:
        print(json.dumps(_envelope([], None, error=message), indent=2))
    else:
        print(f"trainproof: {message}", file=sys.stderr)
    sys.exit(2)

def main():
    try:
        _run()
    except Exception as e:
        fail_to_run(f"internal error: {e}")

def _run():
    parser = argparse.ArgumentParser(prog="trainproof", description="Trainproof: A deterministic linter for ML training runs.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__import__('trainproof').__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data Command
    data_parser = subparsers.add_parser("data", help="Dataset preflight for speech/TTS corpora.")
    data_parser.add_argument("input", type=str, help="Directory or manifest.jsonl")
    data_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")

    # Tokenizer Command
    tok_parser = subparsers.add_parser("tokenizer", help="Tokenizer preflight.")
    tok_parser.add_argument("model", type=str, help="Path to tokenizer model (e.g., SentencePiece .model)")
    tok_parser.add_argument("transcripts", type=str, help="Path to transcripts text file or JSONL")
    tok_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")

    # Epoch Command
    epoch_parser = subparsers.add_parser("epoch", help="First-epoch verdict from training logs.")
    epoch_parser.add_argument("logfile", type=str, help="Path to JSONL or CSV log file")
    epoch_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv", "tfevents"], default="auto", help="Log format override")
    epoch_parser.add_argument("--map", action="append", help="Override log column mapping (e.g. loss=my_loss)")
    epoch_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    epoch_parser.add_argument("--sarif", type=str, metavar="PATH", help="Also write a SARIF 2.1.0 file for GitHub code scanning")
    epoch_parser.add_argument("--html", type=str, metavar="PATH", nargs="?", const="trainproof_report.html", help="Also write a self-contained HTML report (bare flag uses trainproof_report.html)")

    # Doctor Command
    doctor_parser = subparsers.add_parser("doctor", aliases=["diagnose"], help="Flagship zero-config autopsy of training logs.")
    doctor_parser.add_argument("path", type=str, nargs="?", default=".", help="Path to file or directory")
    doctor_parser.add_argument("--baseline", type=str, help="Additionally run compare against this baseline")
    doctor_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv", "tfevents"], default="auto", help="Log format override (file mode only)")
    doctor_parser.add_argument("--map", action="append", help="Override log column mapping (e.g. loss=my_loss)")
    doctor_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    doctor_parser.add_argument("--sarif", type=str, metavar="PATH", help="Also write a SARIF 2.1.0 file for GitHub code scanning")

    # Compare Command
    compare_parser = subparsers.add_parser("compare", help="Compare a run log against a baseline log.")
    compare_parser.add_argument("baseline", type=str, help="Path to baseline log file")
    compare_parser.add_argument("runs", type=str, nargs="+", help="Path to one or more run log files")
    compare_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv", "tfevents"], default="auto", help="Log format override")
    compare_parser.add_argument("--map", action="append", help="Override log column mapping")
    compare_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    compare_parser.add_argument("--sarif", type=str, metavar="PATH", help="Also write a SARIF 2.1.0 file for GitHub code scanning")

    # Watch Command
    watch_parser = subparsers.add_parser("watch", help="Live guardian: poll a training log.")
    watch_parser.add_argument("logfile", type=str, help="Path to run log file")
    watch_parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds")
    watch_parser.add_argument("--format", choices=["auto", "hf", "coqui", "jsonl", "csv", "tfevents"], default="auto", help="Log format override")
    watch_parser.add_argument("--map", action="append", help="Override log column mapping")
    watch_parser.add_argument("--until-fail", action="store_true", help="Exit with code 1 as soon as verdict becomes FAIL")
    watch_parser.add_argument("--stall-timeout", type=int, default=300, help="Seconds of no log growth before warning of a stall")

    # Preflight Command
    preflight_parser = subparsers.add_parser("preflight", help="Pre-flight linter for LLM fine-tuning datasets.")
    preflight_parser.add_argument("dataset", type=str, help="Path to JSONL dataset")
    preflight_parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path")
    preflight_parser.add_argument("--max-len", type=int, help="Max context length")
    preflight_parser.add_argument("--text-field", type=str, help="Text field to extract")
    preflight_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    preflight_parser.add_argument("--sarif", type=str, metavar="PATH", help="Also write a SARIF 2.1.0 file for GitHub code scanning")

    # Env Command
    env_parser = subparsers.add_parser("env", help="Environment preflight: can this machine start this run?")
    env_parser.add_argument("--module", type=str, help="Python module to test-import in a subprocess")
    env_parser.add_argument("--python", type=str, help="Interpreter to run the import probe with")
    env_parser.add_argument("--cwd", type=str, help="Working directory for the import probe")
    env_parser.add_argument("--checkpoint", type=str, help="Path to a .pt/.pth/.ckpt to inspect (never unpickled)")
    env_parser.add_argument("--required-gb", type=float, help="System RAM this run needs (GB)")
    env_parser.add_argument("--output-dir", type=str, help="Directory where checkpoints will be saved")
    env_parser.add_argument("--checkpoint-gb", type=float, help="Size of one checkpoint (GB)")
    env_parser.add_argument("--keep", type=int, default=1, help="Number of checkpoints kept on disk (default: 1)")
    env_parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")

    args = parser.parse_args()
    
    mapping_overrides = parse_map(getattr(args, "map", None))
    is_json = getattr(args, "json", False)
    global _JSON_MODE, _SARIF_PATH
    _JSON_MODE = is_json
    _SARIF_PATH = getattr(args, "sarif", None)

    report_dict = {}

    if args.command == "data":
        report_dict = check_data(args.input)
    elif args.command == "tokenizer":
        report_dict = check_tokenizer(args.model, args.transcripts)
    elif args.command == "epoch":
        try:
            report_dict = check_epoch(args.logfile, fmt=args.format, mapping_overrides=mapping_overrides)
        except Exception as e:
            fail_to_run(f"could not read log {args.logfile}: {e}")

        # no records parsed means nothing was judged -- that is a tool outcome,
        # not a FAIL verdict on the training run
        if any(f.get("id") == "TP-NO-RECORDS" for f in report_dict.get("findings", [])):
            fail_to_run(f"no readable log records in {args.logfile}")


        report_dict["findings"] = tag_source(report_dict.get("findings", []), "single_run")
        if is_json:
            output_json([report_dict], report_dict.get("verdict", "UNKNOWN"))

    if getattr(args, "command", None) in ("doctor", "diagnose"):
        path = Path(args.path)
        candidates = []
        undiscoverable = []
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            for p in path.rglob("*"):
                if not p.is_file(): continue
                if (p.name == "trainer_state.json" or "tfevents" in p.name
                        or p.suffix in (".jsonl", ".csv", ".log", ".txt")):
                    try:
                        records, *_ = parse_log_with_format_info(p, fmt="auto", mapping_overrides=mapping_overrides)
                        if len(records) >= 3:
                            candidates.append(p)
                    except Exception:
                        # A file that raises here used to vanish: discovery
                        # dropped it before it could become a candidate, so it
                        # never reached the "could not be parsed" note either.
                        # A log the user can see on disk but cannot find in the
                        # report is indistinguishable from one that passed -
                        # the same failure NOT-CHECKED exists to prevent.
                        undiscoverable.append(p)
        else:
            fail_to_run(f"path not found: {path}")

        if not candidates:
            fail_to_run(f"no readable training logs found in {path}")
            
        if path.is_dir():
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if len(candidates) > 20:
                if not is_json:
                    print(f"Note: Found {len(candidates)} logs, capping to 20 most recently modified. Skipped {len(candidates)-20} logs.")
                candidates = candidates[:20]
                
        results = []
        unreadable = list(undiscoverable)
        from .epoch import check_records
        for p in candidates:
            fmt = args.format if path.is_file() else "auto"
            try:
                records, fmt_str, used_mapping, _meta = parse_log_with_format_info(p, fmt, mapping_overrides)
            except Exception:
                records = []
                fmt_str = "unknown"
                used_mapping = {}
            if not records:
                # dropped in silence before 0.11.1: a log that disappears from
                # the report is indistinguishable from a log that passed
                unreadable.append(p)
                continue
            
            step_range = f"{records[0].get('step', 0)}..{records[-1].get('step', len(records)-1)}"
            
            report = check_records(records)
            report["file"] = str(p)
            report["fmt"] = fmt_str
            report["num_records"] = len(records)
            report["step_range"] = step_range
            report["used_mapping"] = used_mapping
            report["findings"] = tag_source(report["findings"], "single_run")

            if args.baseline:
                try:
                    cmp = check_compare(p, args.baseline, fmt=fmt, mapping_overrides=mapping_overrides)
                    report["findings"] += tag_source(cmp["findings"], "compare")
                    # a failed baseline comparison is a failed run: without this
                    # doctor prints [FAIL] findings and still exits 0
                    if cmp["verdict"] == "FAIL":
                        report["verdict"] = "FAIL"
                    elif cmp["verdict"] == "WARN" and report["verdict"] == "PASS":
                        report["verdict"] = "WARN"
                except Exception as e:
                    # never swallowed: the report would otherwise show a
                    # single-run verdict with no sign that the comparison the
                    # user explicitly asked for never happened
                    report["findings"] += tag_source([{
                        "id": "TP-CMP-ERROR",
                        "level": "WARN",
                        "message": "The baseline comparison could not be run - this report judges the run on its own only.",
                        "evidence": f"{args.baseline}: {e}",
                    }], "compare")
                    if report["verdict"] == "PASS":
                        report["verdict"] = "WARN"
            results.append(report)
            
        if unreadable and not is_json:
            print(
                f"Note: {len(unreadable)} log(s) could not be parsed and were NOT judged: "
                + ", ".join(p.name for p in unreadable)
            )
            print()

        results.sort(key=lambda r: _VERDICT_SEVERITY.get(_normalize_verdict(r.get("verdict")), _VERDICT_SEVERITY["NOT-CHECKED"]))

        worst_verdict = _get_worst_verdict(results)

        if is_json:
            output_json(results, worst_verdict)

        if len(results) > 1:
            print("SUMMARY")
            print("-" * 60)
            for r in results:
                print(f"{r['verdict']:<4} | {r['file']}")
            print()
            
        for r in results:
            single = [f for f in r["findings"] if f.get("source") != "compare"]
            vs_baseline = [f for f in r["findings"] if f.get("source") == "compare"]

            print_doctor_autopsy(r["file"], r["fmt"], r["num_records"], r["step_range"], r["verdict"], single)
            if r.get("used_mapping") and r["fmt"] in ("csv", "jsonl"):
                print("COLUMNS: " + ", ".join(f"{canon}<-'{col}'" for canon, col in r["used_mapping"].items()))
                print()

            if vs_baseline:
                print("--- VS BASELINE ---")
                for f in vs_baseline:
                    level = f.get("level", "INFO")
                    rid = f.get("id", "")
                    prefix = f"[{level}] {rid}: " if rid else f"[{level}] "
                    print(f"{prefix}{f.get('message', '')}")
                    if f.get('evidence'): print(f"       Evidence: {f.get('evidence')}")
                print()
                
        print_doctor_footer()
        emit_sarif(results)
        sys.exit(_get_exit_code(results))

    elif args.command == "compare":
        try:
            from .compare import extract_metrics
            baseline_records = parse_log_with_format(args.baseline, args.format, mapping_overrides)
            base_metrics = extract_metrics(baseline_records) if baseline_records else None
            
            # Every gallery log is named trainer_state.json, so bare filenames leave
            # the baseline and the runs indistinguishable -- and reversing the two
            # arguments then produces a plausible, inverted verdict with nothing to
            # flag it. Show the shortest path suffix that tells them apart.
            paths = [args.baseline, *args.runs]
            parts = [Path(p).resolve().parts for p in paths]
            distinct = len(set(parts))
            labels = [Path(p).name for p in paths]
            for depth in range(1, max(len(q) for q in parts) + 1):
                candidate = ["/".join(q[-depth:]) for q in parts]
                if len(set(candidate)) == distinct:
                    labels = candidate
                    break
            label_of = dict(zip(paths, labels, strict=False))

            table_results = []
            if base_metrics:
                table_results.append({
                    "name": label_of[args.baseline] + " (BASE)",
                    "start": base_metrics["start_med"],
                    "floor": base_metrics["floor"],
                    "end": base_metrics["end_med"],
                    "improvement": base_metrics["improvement"],
                    "verdict": "BASELINE"
                })
            
            all_reports = []
            for run_path in args.runs:
                report_dict = check_compare(run_path, args.baseline, fmt=args.format, mapping_overrides=mapping_overrides)
                report_dict["findings"] = tag_source(report_dict["findings"], "compare")
                all_reports.append(report_dict)
                
                run_records = parse_log_with_format(run_path, args.format, mapping_overrides)
                run_metrics = extract_metrics(run_records) if run_records else None
                if run_metrics:
                    table_results.append({
                        "name": label_of[run_path],
                        "start": run_metrics["start_med"],
                        "floor": run_metrics["floor"],
                        "end": run_metrics["end_med"],
                        "improvement": run_metrics["improvement"],
                        "verdict": report_dict["verdict"]
                    })
            
            if is_json:
                worst_verdict = _get_worst_verdict(all_reports)
                output_json(all_reports, worst_verdict)
                
            print_compare_table(table_results)
            emit_sarif(all_reports)

            if len(args.runs) == 1:
                print_verdict_console(all_reports[0]["verdict"], all_reports[0]["findings"])
                sys.exit(_get_exit_code(all_reports))
            
            sys.exit(_get_exit_code(all_reports))
                
        except Exception as e:
            fail_to_run(f"could not compare against {args.baseline}: {e}")
    elif args.command == "watch":
        watch_loop(args.logfile, interval=args.interval, fmt=args.format, until_fail=args.until_fail, stall_timeout=args.stall_timeout)
        sys.exit(0)
    elif args.command == "preflight":
        from .preflight import preflight
        tokenizer = None
        if args.tokenizer:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            except ImportError:
                # a missing dependency is trainproof's problem, not a verdict
                # on the user's dataset
                fail_to_run("pip install transformers is required to load a tokenizer")
        
        result = preflight(args.dataset, tokenizer=tokenizer, max_len=args.max_len, text_field=args.text_field)
        report_dict = {"verdict": result.verdict, "findings": result.findings}

    elif args.command == "env":
        report_dict = check_env(
            module=args.module,
            checkpoint=args.checkpoint,
            required_gb=getattr(args, "required_gb", None),
            output_dir=getattr(args, "output_dir", None),
            checkpoint_gb=getattr(args, "checkpoint_gb", None),
            keep=args.keep,
            python=args.python,
            cwd=getattr(args, "cwd", None),
        )

    report_dict["findings"] = tag_source(report_dict.get("findings", []), "single_run")
    if is_json:
        # exits here: the HTML-report notice below would corrupt the JSON stream
        output_json([report_dict], report_dict.get("verdict", "UNKNOWN"))

    print_verdict_console(report_dict.get("verdict", "FAIL"), report_dict.get("findings", []))
    emit_sarif([report_dict])

    # Opt-in since 0.11.0. This used to write into the caller's working directory
    # on every single run, unasked -- it littered this project's own test suite,
    # which is how the smell was finally noticed. CONTRACTS.md explicitly excludes
    # the HTML report from the stability contract, so this breaks no promise.
    if getattr(args, "html", None):
        html_out = Path(args.html)
        write_html_report(report_dict, html_out)
        print(f"\nSaved detailed HTML report to {html_out.absolute()}")
    
    sys.exit(_get_exit_code([report_dict]))

if __name__ == "__main__":
    main()
