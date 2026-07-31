import math
from pathlib import Path
from typing import Any
from . import rules
from .adapters import parse_log_with_format
from .epoch import check_epoch

def extract_metrics(records):
    losses = []
    grad_norms = []

    for r in records:
        loss = r.get("loss")
        gn = r.get("grad_norm")

        if loss is not None and not math.isnan(loss) and not math.isinf(loss):
            losses.append(loss)
        if gn is not None and not math.isnan(gn) and not math.isinf(gn):
            grad_norms.append(gn)

    if len(losses) < 10:
        return None

    floor = min(losses)

    w = rules.LOSS_IMPROVEMENT_WINDOW
    start_med = sorted(losses[:w])[w // 2]
    end_med = sorted(losses[-w:])[w // 2]

    # A run whose losses are all exactly zero has no scale to compare on. Its
    # floor (0.0) beats every possible baseline, so TP-FLOOR-RATIO and
    # TP-END-RATIO cannot fire; and substituting 0.0 for an undefined
    # improvement -- which this function used to do -- kept TP-NEG-IMPROVE
    # quiet too. Against a baseline that had not itself improved, nothing fired
    # at all and the report read TP-CMP-PASS, "compares favorably", for a run
    # that learned nothing. Mark it instead and let check_compare refuse.
    degenerate = None
    if all(l == 0.0 for l in losses):
        degenerate = "every logged loss is exactly 0.0"
    elif start_med <= 0:
        degenerate = "starting loss is not positive - relative improvement is undefined"

    improvement = None if degenerate else 1.0 - (end_med / start_med)

    gn_median = None
    if len(grad_norms) > 5:
        sorted_gns = sorted(grad_norms)
        gn_median = sorted_gns[len(sorted_gns) // 2]

    return {
        "floor": floor,
        "start_med": start_med,
        "end_med": end_med,
        "improvement": improvement,
        "degenerate": degenerate,
        "gn_median": gn_median,
        "losses_len": len(losses)
    }

def check_compare(run_path: str | Path, base_path: str | Path, fmt: str = "auto", mapping_overrides: dict[str, str] = None) -> dict[str, Any]:
    findings = []
    verdict = "PASS"

    # Baseline sanity — a suspect baseline downgrades the whole comparison
    base_epoch_res = check_epoch(base_path, fmt=fmt, mapping_overrides=mapping_overrides)
    if base_epoch_res["verdict"] == "FAIL":
        verdict = "WARN"
        findings.append({
            "id": "TP-BAD-BASELINE",
            "level": "WARN",
            "message": "baseline itself fails single-run checks - comparison may be meaningless",
            "evidence": str(base_path)
        })

    run_records = parse_log_with_format(run_path, fmt, mapping_overrides=mapping_overrides)
    base_records = parse_log_with_format(base_path, fmt, mapping_overrides=mapping_overrides)

    run_metrics = extract_metrics(run_records) if run_records else None
    base_metrics = extract_metrics(base_records) if base_records else None

    if not run_metrics:
        return {"verdict": "FAIL", "findings": [{"id": "TP-NO-LOSS", "level": "FAIL", "message": "Run log has fewer than 10 valid loss points.", "evidence": str(run_path)}]}
    if not base_metrics:
        return {"verdict": "FAIL", "findings": [{"id": "TP-NO-LOSS", "level": "FAIL", "message": "Baseline log has fewer than 10 valid loss points.", "evidence": str(base_path)}]}

    # Refuse to compare when either side has no usable loss scale, and refuse
    # BEFORE the ratio rules below, all of which a zero floor passes. The
    # principle is the one stated for exit code 2 in CONTRACTS.md: never
    # synthesise a verdict trainproof cannot support. A meaningless comparison
    # reported as a favorable one is worse than no comparison at all.
    uncomparable = []
    for side, metrics, path in (
        ("run", run_metrics, run_path),
        ("baseline", base_metrics, base_path),
    ):
        if metrics["degenerate"]:
            uncomparable.append({
                "id": "TP-CMP-UNCOMPARABLE",
                "level": "FAIL",
                "message": f"Cannot compare: the {side} has no usable loss scale.",
                "evidence": (
                    f"{side} {path}: {metrics['degenerate']}. Its loss floor of "
                    f"{metrics['floor']:.3f} would beat any baseline, so none of the ratio "
                    "rules can judge it."
                ),
            })
    if uncomparable:
        return {"verdict": "FAIL", "findings": findings + uncomparable}

    # Floor ratio
    if run_metrics["floor"] > base_metrics["floor"] * rules.MAX_FLOOR_RATIO:
        verdict = "FAIL"
        findings.append({
            "id": "TP-FLOOR-RATIO",
            "level": "FAIL",
            "message": "loss floor ratio exceeded limit",
            "evidence": f"Run floor {run_metrics['floor']:.3f} vs Baseline floor {base_metrics['floor']:.3f} (ratio {(run_metrics['floor'] / base_metrics['floor']):.1f}x > {rules.MAX_FLOOR_RATIO})"
        })

    # End-loss ratio: where the run landed vs where the baseline landed.
    # Robust to corrupted starts (see rules.MAX_END_RATIO comment).
    if base_metrics["end_med"] > 0 and run_metrics["end_med"] > base_metrics["end_med"] * rules.MAX_END_RATIO:
        verdict = "FAIL"
        findings.append({
            "id": "TP-END-RATIO",
            "level": "FAIL",
            "message": "end loss ratio exceeded limit",
            "evidence": f"Run end {run_metrics['end_med']:.3f} vs Baseline end {base_metrics['end_med']:.3f} (ratio {(run_metrics['end_med'] / base_metrics['end_med']):.1f}x > {rules.MAX_END_RATIO})"
        })

    # Improvement deficit. Both improvements are guaranteed non-None here: a
    # side with an undefined improvement is degenerate and returned above.
    run_imp = run_metrics["improvement"]
    base_imp = base_metrics["improvement"]

    if run_imp < 0:
        verdict = "FAIL"
        findings.append({
            "id": "TP-NEG-IMPROVE",
            "level": "FAIL",
            "message": "negative improvement",
            "evidence": f"Run improvement is {run_imp * 100:.1f}% vs Baseline {base_imp * 100:.1f}%"
        })
    elif base_imp > 0 and run_imp < base_imp * rules.MIN_IMPROVEMENT_FRACTION:
        verdict = "FAIL"
        findings.append({
            "id": "TP-IMPROVE-DEFICIT",
            "level": "FAIL",
            "message": "improvement deficit",
            "evidence": f"Run improvement {run_imp * 100:.1f}% vs Baseline {base_imp * 100:.1f}% (ratio {(run_imp / base_imp):.2f}x < {rules.MIN_IMPROVEMENT_FRACTION})"
        })

    # Gradnorm median ratio
    if run_metrics["gn_median"] is not None and base_metrics["gn_median"] is not None and base_metrics["gn_median"] > 0:
        if run_metrics["gn_median"] > base_metrics["gn_median"] * rules.MAX_GRADNORM_MEDIAN_RATIO:
            if verdict == "PASS": verdict = "WARN"
            findings.append({
                "id": "TP-GRADNORM-RATIO",
                "level": "WARN",
                "message": "gradient norm median significantly higher than baseline",
                "evidence": f"Run gn median {run_metrics['gn_median']:.2f} vs Baseline gn median {base_metrics['gn_median']:.2f} (ratio {(run_metrics['gn_median'] / base_metrics['gn_median']):.1f}x > {rules.MAX_GRADNORM_MEDIAN_RATIO})"
            })

    if verdict == "PASS" and not any(f["level"] == "WARN" for f in findings):
        findings.append({
            "id": "TP-CMP-PASS",
            "level": "PASS",
            "message": "Run compares favorably to baseline.",
            "evidence": f"Floor ratio {(run_metrics['floor'] / base_metrics['floor']):.2f}x, Improvement {run_imp*100:.1f}% vs {base_imp*100:.1f}%"
        })

    return {"verdict": verdict, "findings": findings}
