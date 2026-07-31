import math
from pathlib import Path
from typing import Any
from . import rules
from .adapters import parse_log_with_format

# Every JUDGING check trainproof can run on a single log. TP-PASS reports which
# of these actually executed, and why each of the rest did not.
#
# INFO-only observations (TP-THROUGHPUT, TP-GPU-UTIL) are deliberately absent:
# they report context rather than judge the run, so a missing one cannot give a
# reader a false all-clear. Only checks that could have found something are
# accounted for here.
CHECK_GROUPS = (
    "zero-loss",
    "zero-grad",
    "flat-loss",
    "divergence",
    "dead-run",
    "grad-spike",
    "lr",
    "step-time",
    "loader",
    "overfit",
)


def _nothing_ran(reason: str) -> dict[str, Any]:
    return {"ran": [], "skipped": {g: reason for g in CHECK_GROUPS}}


def check_records(records: list[dict]) -> dict[str, Any]:
    findings = []
    verdict = "PASS"

    # Every group in CHECK_GROUPS lands in exactly one of these, and it is
    # decided at the point the check actually runs — never from the mere
    # presence of a column. Until v0.11.1 this was a hardcoded list plus
    # availability tests, so an all-zero loss curve skipped every shape guard
    # below and TP-PASS still reported those same checks as having run.
    ran: list[str] = []
    skipped: dict[str, str] = {}

    def ok(group: str) -> None:
        ran.append(group)

    def no(group: str, reason: str) -> None:
        skipped[group] = reason

    if not records:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-RECORDS", "level": "FAIL", "message": "No valid log records found.", "evidence": ""}],
            "checks": _nothing_ran("no records parsed"),
        }

    losses = []
    loss_steps = []  # kept aligned with losses; records may lack a loss field
    lrs = []
    grad_norms = []
    times = []
    time_steps = []

    for i, r in enumerate(records):
        step = r.get("step")
        step = i if step is None else step
        loss = r.get("loss")
        lr = r.get("lr")
        gn = r.get("grad_norm")
        t = r.get("time")

        if loss is not None:
            losses.append(loss)
            loss_steps.append(step)
        if lr is not None: lrs.append(lr)
        if gn is not None: grad_norms.append(gn)
        if t is not None:
            times.append(t)
            time_steps.append(step)

    if not losses:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-LOSS", "level": "FAIL", "message": "Could not find loss metric in logs.", "evidence": ""}],
            "checks": _nothing_ran("no loss column in the log"),
        }

    # Check NaN / Inf in loss
    nan_steps = [s for s, l in zip(loss_steps, losses) if math.isnan(l) or math.isinf(l)]
    if nan_steps:
        findings.append({"id": "TP-NAN", "level": "FAIL", "message": "NaN or Inf detected in loss.", "evidence": f"Steps: {nan_steps[:5]}..."})
        verdict = "FAIL"

    valid_losses = [l for l in losses if not math.isnan(l) and not math.isinf(l)]

    # Degenerate loss series. Runs BEFORE the shape checks below, each of which
    # bails on `> 0`: an all-zero curve used to skip all three and still reach
    # TP-PASS. See rules.MIN_POINTS_FOR_DEGENERATE_CHECK for why this is exact
    # equality and not a threshold.
    if len(valid_losses) >= rules.MIN_POINTS_FOR_DEGENERATE_CHECK:
        ok("zero-loss")
        if all(l == 0.0 for l in valid_losses):
            findings.append({
                "id": "TP-ZERO-LOSS",
                "level": "FAIL",
                "message": "Loss is exactly zero on every logged step - the run learned nothing.",
                "evidence": (
                    f"all {len(valid_losses)} finite losses are exactly 0.0. Cross-entropy "
                    "returns 0.0 when every target label is masked to -100, so check the "
                    "collator's prompt masking and whether the response was truncated out "
                    "of the context window."
                ),
            })
            verdict = "FAIL"
    else:
        no("zero-loss", f"fewer than {rules.MIN_POINTS_FOR_DEGENERATE_CHECK} finite loss points")

    if valid_losses:
        mean_loss = sum(valid_losses) / len(valid_losses)
        std_loss = math.sqrt(sum((l - mean_loss)**2 for l in valid_losses) / len(valid_losses))
        if mean_loss > 0:
            ok("flat-loss")
            if (std_loss / mean_loss) < rules.MIN_LOSS_VARIATION:
                findings.append({"id": "TP-FLAT", "level": "FAIL", "message": "Loss curve is completely flat (dead run).", "evidence": f"Variation {std_loss/mean_loss:.5f} < {rules.MIN_LOSS_VARIATION}"})
                verdict = "FAIL"
        else:
            no("flat-loss", "mean loss is not positive - relative variation is undefined")

        # Divergence. The floor is taken over NONZERO losses: a single 0.0
        # anywhere in the series used to drive min_loss to zero and disable this
        # check for the entire run.
        nonzero_losses = [l for l in valid_losses if l != 0.0]
        min_loss = min(nonzero_losses) if nonzero_losses else 0.0
        if min_loss > 0:
            ok("divergence")
            if valid_losses[-1] > min_loss * rules.MAX_LOSS_DIVERGENCE_RATIO:
                findings.append({"id": "TP-DIVERGE", "level": "FAIL", "message": "Loss curve is diverging.", "evidence": f"End loss {valid_losses[-1]:.3f} vs Min loss {min_loss:.3f}"})
                verdict = "FAIL"
        else:
            no("divergence", "no positive loss to measure a floor against")

        # Check no-improvement (dead run): robust start-vs-end median comparison
        if len(valid_losses) >= rules.MIN_POINTS_FOR_IMPROVEMENT_CHECK:
            w = rules.LOSS_IMPROVEMENT_WINDOW
            start_med = sorted(valid_losses[:w])[w // 2]
            end_med = sorted(valid_losses[-w:])[w // 2]
            if start_med > 0:
                ok("dead-run")
                if end_med >= start_med * (1 - rules.MIN_LOSS_IMPROVEMENT):
                    findings.append({"id": "TP-DEAD-RUN", "level": "FAIL", "message": "Loss never improved over the run (dead run).",
                                     "evidence": f"median of first {w} losses {start_med:.3f} vs last {w} {end_med:.3f} (needs >={rules.MIN_LOSS_IMPROVEMENT*100:.0f}% improvement)"})
                    verdict = "FAIL"
            else:
                no("dead-run", "starting loss is not positive - relative improvement is undefined")
        else:
            no("dead-run", f"fewer than {rules.MIN_POINTS_FOR_IMPROVEMENT_CHECK} finite loss points")
    else:
        for group in ("flat-loss", "divergence", "dead-run"):
            no(group, "every logged loss is NaN or Inf")

    # Check Grad Norm
    valid_gns = [g for g in grad_norms if not math.isnan(g) and not math.isinf(g)]

    # Degenerate gradient series — same shape of bug as zero-loss: the spike
    # test below is guarded by `median_gn > 0`, so an identically-zero gradient
    # norm skipped it in silence. An all-zero grad norm is the log signature of
    # a severed backward graph, which is a finding, not an absence of one.
    if len(valid_gns) >= rules.MIN_POINTS_FOR_DEGENERATE_CHECK:
        ok("zero-grad")
        if all(g == 0.0 for g in valid_gns):
            findings.append({
                "id": "TP-ZERO-GRAD",
                "level": "FAIL",
                "message": "Gradient norm is exactly zero on every logged step - no gradient reached the weights.",
                "evidence": (
                    f"all {len(valid_gns)} finite gradient norms are exactly 0.0. The backward "
                    "graph is severed or every parameter is frozen. With PEFT this is usually "
                    "reentrant gradient checkpointing (use_reentrant=True) over frozen input "
                    "embeddings, which detaches the graph before it reaches the adapters - call "
                    "enable_input_require_grads() or pass use_reentrant=False."
                ),
            })
            verdict = "FAIL"
    elif not valid_gns:
        no("zero-grad", "no finite gradient norms in the log")
    else:
        no("zero-grad", f"fewer than {rules.MIN_POINTS_FOR_DEGENERATE_CHECK} finite gradient norms")

    if len(valid_gns) > 5:
        sorted_gns = sorted(valid_gns)
        median_gn = sorted_gns[len(sorted_gns)//2]
        if median_gn > 0:
            ok("grad-spike")
            spikes = [g for g in valid_gns if g > median_gn * rules.MAX_GRAD_NORM_SPIKE_RATIO]
            if spikes:
                findings.append({"id": "TP-GRAD-SPIKE", "level": "WARN", "message": "Gradient norm spikes detected.", "evidence": f"Max gn {max(spikes):.2f} > {rules.MAX_GRAD_NORM_SPIKE_RATIO}x median ({median_gn:.2f})"})
                if verdict == "PASS": verdict = "WARN"
        else:
            no("grad-spike", "median gradient norm is zero - no scale to measure a spike against")
    elif not valid_gns:
        no("grad-spike", "no finite gradient norms in the log")
    else:
        no("grad-spike", "fewer than 6 finite gradient norms")

    # Check LR
    if lrs:
        ok("lr")
        zeros = sum(1 for lr in lrs if lr <= 0)
        zero_frac = zeros / len(lrs)
        if zero_frac >= rules.ZERO_LR_FAIL_FRACTION:
            findings.append({"id": "TP-ZERO-LR", "level": "FAIL", "message": "Learning rate is zero for the entire run - the optimizer never steps.", "evidence": f"{zero_frac*100:.1f}% of steps have lr=0"})
            verdict = "FAIL"
        elif zero_frac > rules.MAX_ZERO_LR_FRACTION:
            findings.append({"id": "TP-ZERO-LR-PARTIAL", "level": "WARN", "message": "Learning rate is zero for a large fraction of the run.", "evidence": f"{zero_frac*100:.1f}% of steps have lr=0"})
            if verdict == "PASS": verdict = "WARN"
    else:
        no("lr", "no learning-rate column in the log")

    # Throughput — only when the log carries a time column; no guessing.
    # INFO context, not a judging check, so it is not in CHECK_GROUPS.
    if len(times) >= 2 and times[-1] > times[0]:
        span = times[-1] - times[0]
        steps_covered = time_steps[-1] - time_steps[0]
        if steps_covered > 0:
            rate = steps_covered / span
            findings.append({"id": "TP-THROUGHPUT", "level": "INFO", "message": "Throughput measured from log timestamps.",
                             "evidence": f"{rate:.2f} steps/sec over {span:.0f}s observed."})

    # Overfit Rule
    eval_losses = []
    eval_loss_steps = []
    for i, r in enumerate(records):
        el = r.get("eval_loss")
        if el is not None and not math.isnan(el) and not math.isinf(el):
            eval_losses.append(el)
            eval_loss_steps.append(r.get("step", i))

    if len(eval_losses) >= rules.OVERFIT_MIN_EVALS:
        ok("overfit")
        eval_min = min(eval_losses)
        i_min = eval_losses.index(eval_min)
        if len(eval_losses) - i_min - 1 >= 3:
            last_3 = sorted(eval_losses[-3:])
            med_last_3 = last_3[len(last_3)//2]
            if med_last_3 > eval_min * rules.OVERFIT_RATIO:
                min_step = eval_loss_steps[i_min]

                tl_at_min = None
                for i, step in enumerate(loss_steps):
                    if step >= min_step:
                        tl_at_min = losses[i]
                        break

                if tl_at_min is None and losses:
                    tl_at_min = losses[-1]

                if tl_at_min is not None and valid_losses and valid_losses[-1] < tl_at_min:
                    ratio = med_last_3 / eval_min if eval_min > 0 else float('inf')
                    findings.append({
                        "id": "TP-OVERFIT",
                        "level": "WARN",
                        "message": "Overfitting detected: eval loss has significantly degraded while train loss continued falling.",
                        "evidence": f"eval_loss min {eval_min:.2f} @step{int(min_step)} rose to {med_last_3:.2f} ({ratio:.1f}x > {rules.OVERFIT_RATIO}) over the last 3 evals while train_loss fell to {valid_losses[-1]:.2f} - best checkpoint was near step {int(min_step)}."
                    })
                    if verdict == "PASS": verdict = "WARN"
    elif not eval_losses:
        no("overfit", "no eval_loss in the log - this run has no generalisation signal at all")
    else:
        no("overfit", f"fewer than {rules.OVERFIT_MIN_EVALS} eval points")

    # Telemetry Rules
    step_times = []
    for r in records:
        st = r.get("step_time")
        if st is not None and not math.isnan(st) and not math.isinf(st):
            step_times.append(st)

    if len(step_times) >= 10:
        first_50_idx = max(1, len(step_times) // 2)
        last_20_idx = len(step_times) - max(1, int(len(step_times) * 0.2))

        first_50 = sorted(step_times[:first_50_idx])
        last_20 = sorted(step_times[last_20_idx:])

        med_first_50 = first_50[len(first_50)//2] if first_50 else 0
        med_last_20 = last_20[len(last_20)//2] if last_20 else 0

        if med_first_50 > 0:
            ok("step-time")
            if med_last_20 > rules.STEP_TIME_CLIFF_RATIO * med_first_50:
                findings.append({"id": "TP-STEP-CLIFF", "level": "WARN", "message": "Step time cliff detected: recent steps are significantly slower.",
                                 "evidence": f"median recent step_time {med_last_20:.2f}s > {rules.STEP_TIME_CLIFF_RATIO}x median early step_time ({med_first_50:.2f}s)"})
                if verdict == "PASS": verdict = "WARN"
        else:
            no("step-time", "median early step_time is zero - no baseline to compare against")
    elif not step_times:
        no("step-time", "no step_time column in the log")
    else:
        no("step-time", "fewer than 10 step_time points")

    valid_loader_fractions = []
    gpu_utils = []
    for r in records:
        st = r.get("step_time")
        lt = r.get("loader_time")
        gu = r.get("gpu_util")
        if st is not None and lt is not None and st > 0:
            valid_loader_fractions.append(lt / st)
        if gu is not None and not math.isnan(gu) and not math.isinf(gu):
            gpu_utils.append(gu)

    if valid_loader_fractions:
        ok("loader")
        med_frac = sorted(valid_loader_fractions)[len(valid_loader_fractions)//2]
        if med_frac > rules.LOADER_FRACTION_MAX:
            findings.append({"id": "TP-LOADER-BOUND", "level": "WARN", "message": "Dataloader stall detected: spending too much time loading data.",
                             "evidence": f"median loader_time/step_time {med_frac*100:.1f}% > {rules.LOADER_FRACTION_MAX*100:.1f}%"})
            if verdict == "PASS": verdict = "WARN"
    else:
        no("loader", "no loader_time/step_time pair in the log")

    # INFO context, not a judging check.
    if gpu_utils:
        med_gpu = sorted(gpu_utils)[len(gpu_utils)//2]
        findings.append({"id": "TP-GPU-UTIL", "level": "INFO", "message": "GPU utilization context.",
                         "evidence": f"median gpu_util {med_gpu:.1f}% observed."})

    if verdict == "PASS":
        msg = "No mechanical failures detected."
        if ran:
            msg += f" Ran: {', '.join(sorted(ran))}."
        if skipped:
            msg += " Skipped: " + "; ".join(f"{g} ({r})" for g, r in sorted(skipped.items())) + "."
        findings.append({"id": "TP-PASS", "level": "PASS", "message": msg, "evidence": f"{len(valid_losses)} steps analyzed."})

    return {
        "verdict": verdict,
        "findings": findings,
        "checks": {"ran": sorted(ran), "skipped": skipped},
    }

def check_epoch(log_path: str | Path, fmt: str = "auto", mapping_overrides: dict[str, str] = None) -> dict[str, Any]:
    records = parse_log_with_format(log_path, fmt, mapping_overrides)
    if not records:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-RECORDS", "level": "FAIL", "message": "No valid log records found.", "evidence": str(log_path)}],
            "checks": _nothing_ran("no records parsed"),
        }
    return check_records(records)
