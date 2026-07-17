import argparse
import sys
from pathlib import Path
from .speech.data import check_data
from .speech.tokenizer import check_tokenizer
from .epoch import check_epoch
from .compare import check_compare
from .watch import watch_loop
from .report import print_verdict_console, write_html_report

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

    # Compare Command
    compare_parser = subparsers.add_parser("compare", help="Compare a run log against a baseline log.")
    compare_parser.add_argument("run", type=str, help="Path to run log file")
    compare_parser.add_argument("baseline", type=str, help="Path to baseline log file")
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
    elif args.command == "compare":
        try:
            report_dict = check_compare(args.run, args.baseline, fmt=args.format)
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
