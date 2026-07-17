import math
from pathlib import Path
from typing import Any
from . import rules
from .adapters import parse_log_with_format

def check_epoch(log_path: str | Path, fmt: str = "auto") -> dict[str, Any]:
    records = parse_log_with_format(log_path, fmt)
    findings = []
    verdict = "PASS"
    
    if not records:
        return {"verdict": "FAIL", "findings": [{"level": "FAIL", "message": "No valid log records found.", "evidence": str(log_path)}]}

    # Find relevant keys
    def get_val(row, aliases):
        for a in aliases:
            for k in row:
                if a in k: return row[k]
        return None

    losses = []
    loss_steps = []  # kept aligned with losses; records may lack a loss field
    lrs = []
    grad_norms = []
    times = []
    time_steps = []

    for i, r in enumerate(records):
        step = get_val(r, ["step", "iter"])
        step = i if step is None else step
        loss = get_val(r, ["loss", "train_loss"])
        lr = get_val(r, ["lr", "learning_rate"])
        gn = get_val(r, ["grad_norm", "gnorm", "grad"])
        t = get_val(r, ["elapsed", "time", "timestamp"])

        if loss is not None:
            losses.append(loss)
            loss_steps.append(step)
        if lr is not None: lrs.append(lr)
        if gn is not None: grad_norms.append(gn)
        if t is not None:
            times.append(t)
            time_steps.append(step)

    if not losses:
        return {"verdict": "FAIL", "findings": [{"level": "FAIL", "message": "Could not find loss metric in logs.", "evidence": ""}]}
        
    # Check NaN / Inf in loss
    nan_steps = [s for s, l in zip(loss_steps, losses) if math.isnan(l) or math.isinf(l)]
    if nan_steps:
        findings.append({"level": "FAIL", "message": "NaN or Inf detected in loss.", "evidence": f"Steps: {nan_steps[:5]}..."})
        verdict = "FAIL"
        
    # Check flat loss
    valid_losses = [l for l in losses if not math.isnan(l) and not math.isinf(l)]
    if valid_losses:
        mean_loss = sum(valid_losses) / len(valid_losses)
        std_loss = math.sqrt(sum((l - mean_loss)**2 for l in valid_losses) / len(valid_losses))
        if mean_loss > 0 and (std_loss / mean_loss) < rules.MIN_LOSS_VARIATION:
            findings.append({"level": "FAIL", "message": "Loss curve is completely flat (dead run).", "evidence": f"Variation {std_loss/mean_loss:.5f} < {rules.MIN_LOSS_VARIATION}"})
            verdict = "FAIL"
            
        # Check divergence
        min_loss = min(valid_losses)
        if min_loss > 0 and valid_losses[-1] > min_loss * rules.MAX_LOSS_DIVERGENCE_RATIO:
            findings.append({"level": "FAIL", "message": "Loss curve is diverging.", "evidence": f"End loss {valid_losses[-1]:.3f} vs Min loss {min_loss:.3f}"})
            verdict = "FAIL"

    # Check Grad Norm
    valid_gns = [g for g in grad_norms if not math.isnan(g) and not math.isinf(g)]
    if valid_gns and len(valid_gns) > 5:
        sorted_gns = sorted(valid_gns)
        median_gn = sorted_gns[len(sorted_gns)//2]
        if median_gn > 0:
            spikes = [g for g in valid_gns if g > median_gn * rules.MAX_GRAD_NORM_SPIKE_RATIO]
            if spikes:
                findings.append({"level": "WARN", "message": "Gradient norm spikes detected.", "evidence": f"Max gn {max(spikes):.2f} > {rules.MAX_GRAD_NORM_SPIKE_RATIO}x median ({median_gn:.2f})"})
                if verdict == "PASS": verdict = "WARN"

    # Check LR
    if lrs:
        zeros = sum(1 for lr in lrs if lr <= 0)
        zero_frac = zeros / len(lrs)
        if zero_frac > rules.MAX_ZERO_LR_FRACTION:
            findings.append({"level": "WARN", "message": "Learning rate is zero for a large fraction of the run.", "evidence": f"{zero_frac*100:.1f}% of steps have lr=0"})
            if verdict == "PASS": verdict = "WARN"

    # Throughput — only when the log carries a time column; no guessing
    if len(times) >= 2 and times[-1] > times[0]:
        span = times[-1] - times[0]
        steps_covered = time_steps[-1] - time_steps[0]
        if steps_covered > 0:
            rate = steps_covered / span
            findings.append({"level": "INFO", "message": "Throughput measured from log timestamps.",
                             "evidence": f"{rate:.2f} steps/sec over {span:.0f}s observed."})

    if verdict == "PASS":
        findings.append({"level": "PASS", "message": "Loss curve shows healthy shape, grad norms are stable.", "evidence": f"{len(valid_losses)} steps analyzed."})

    return {"verdict": verdict, "findings": findings}
