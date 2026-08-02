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


class CheckContext:
    def __init__(self, records: list[dict]):
        self.records = records
        self.ran = []
        self.skipped = {}
        self.losses = []
        self.loss_steps = []
        self.lrs = []
        self.grad_norms = []
        self.times = []
        self.time_steps = []

        for i, r in enumerate(records):
            step = r.get("step")
            step = i if step is None else step
            loss = r.get("loss")
            lr = r.get("lr")
            gn = r.get("grad_norm")
            t = r.get("time")

            if loss is not None:
                self.losses.append(loss)
                self.loss_steps.append(step)
            if lr is not None: self.lrs.append(lr)
            if gn is not None: self.grad_norms.append(gn)
            if t is not None:
                self.times.append(t)
                self.time_steps.append(step)

        self.valid_losses = [v for v in self.losses if not math.isnan(v) and not math.isinf(v)]
        self.valid_gns = [g for g in self.grad_norms if not math.isnan(g) and not math.isinf(g)]
        
        self.eval_losses = []
        self.eval_loss_steps = []
        for i, r in enumerate(records):
            el = r.get("eval_loss")
            if el is not None and not math.isnan(el) and not math.isinf(el):
                self.eval_losses.append(el)
                self.eval_loss_steps.append(r.get("step", i))
                
        self.step_times = []
        self.valid_loader_fractions = []
        self.gpu_utils = []
        for r in records:
            st = r.get("step_time")
            lt = r.get("loader_time")
            gu = r.get("gpu_util")
            
            if st is not None and not math.isnan(st) and not math.isinf(st):
                self.step_times.append(st)
                
            if st is not None and lt is not None and st > 0:
                self.valid_loader_fractions.append(lt / st)
                
            if gu is not None and not math.isnan(gu) and not math.isinf(gu):
                self.gpu_utils.append(gu)

    def ok(self, group: str) -> None:
        self.ran.append(group)

    def no(self, group: str, reason: str) -> None:
        self.skipped[group] = reason


def check_nan(ctx: CheckContext) -> list[dict]:
    nan_steps = [s for s, v in zip(ctx.loss_steps, ctx.losses, strict=False) if math.isnan(v) or math.isinf(v)]
    if nan_steps:
        return [{"id": "TP-NAN", "level": "FAIL", "message": "NaN or Inf detected in loss.", "evidence": f"Steps: {nan_steps[:5]}..."}]
    return []

def check_zero_loss(ctx: CheckContext) -> list[dict]:
    if len(ctx.valid_losses) >= rules.MIN_POINTS_FOR_DEGENERATE_CHECK:
        ctx.ok("zero-loss")
        if all(v == 0.0 for v in ctx.valid_losses):
            return [{
                "id": "TP-ZERO-LOSS",
                "level": "FAIL",
                "message": "Loss is exactly zero on every logged step - the run learned nothing.",
                "evidence": (
                    f"all {len(ctx.valid_losses)} finite losses are exactly 0.0. Cross-entropy "
                    "returns 0.0 when every target label is masked to -100, so check the "
                    "collator's prompt masking and whether the response was truncated out "
                    "of the context window."
                ),
            }]
    else:
        ctx.no("zero-loss", f"fewer than {rules.MIN_POINTS_FOR_DEGENERATE_CHECK} finite loss points")
    return []

def check_flat_loss(ctx: CheckContext) -> list[dict]:
    if not ctx.valid_losses:
        ctx.no("flat-loss", "every logged loss is NaN or Inf")
        return []
    mean_loss = sum(ctx.valid_losses) / len(ctx.valid_losses)
    std_loss = math.sqrt(sum((v - mean_loss)**2 for v in ctx.valid_losses) / len(ctx.valid_losses))
    if mean_loss > 0:
        ctx.ok("flat-loss")
        if (std_loss / mean_loss) < rules.MIN_LOSS_VARIATION:
            return [{"id": "TP-FLAT", "level": "FAIL", "message": "Loss curve is completely flat (dead run).", "evidence": f"Variation {std_loss/mean_loss:.5f} < {rules.MIN_LOSS_VARIATION}"}]
    else:
        ctx.no("flat-loss", "mean loss is not positive - relative variation is undefined")
    return []

def check_divergence(ctx: CheckContext) -> list[dict]:
    if not ctx.valid_losses:
        ctx.no("divergence", "every logged loss is NaN or Inf")
        return []
    nonzero_losses = [v for v in ctx.valid_losses if v != 0.0]
    min_loss = min(nonzero_losses) if nonzero_losses else 0.0
    if min_loss > 0:
        ctx.ok("divergence")
        if ctx.valid_losses[-1] > min_loss * rules.MAX_LOSS_DIVERGENCE_RATIO:
            return [{"id": "TP-DIVERGE", "level": "FAIL", "message": "Loss curve is diverging.", "evidence": f"End loss {ctx.valid_losses[-1]:.3f} vs Min loss {min_loss:.3f}"}]
    else:
        ctx.no("divergence", "no positive loss to measure a floor against")
    return []

def check_dead_run(ctx: CheckContext) -> list[dict]:
    if not ctx.valid_losses:
        ctx.no("dead-run", "every logged loss is NaN or Inf")
        return []
    if len(ctx.valid_losses) >= rules.MIN_POINTS_FOR_IMPROVEMENT_CHECK:
        w = rules.LOSS_IMPROVEMENT_WINDOW
        start_med = sorted(ctx.valid_losses[:w])[w // 2]
        end_med = sorted(ctx.valid_losses[-w:])[w // 2]
        if start_med > 0:
            ctx.ok("dead-run")
            if end_med >= start_med * (1 - rules.MIN_LOSS_IMPROVEMENT):
                return [{"id": "TP-DEAD-RUN", "level": "FAIL", "message": "Loss never improved over the run (dead run).",
                                 "evidence": f"median of first {w} losses {start_med:.3f} vs last {w} {end_med:.3f} (needs >={rules.MIN_LOSS_IMPROVEMENT*100:.0f}% improvement)"}]
        else:
            ctx.no("dead-run", "starting loss is not positive - relative improvement is undefined")
    else:
        ctx.no("dead-run", f"fewer than {rules.MIN_POINTS_FOR_IMPROVEMENT_CHECK} finite loss points")
    return []

def check_zero_grad(ctx: CheckContext) -> list[dict]:
    _nz_losses = [v for v in ctx.valid_losses if v != 0.0]
    _loss_improved = bool(_nz_losses) and min(_nz_losses) < _nz_losses[0] * (
        1 - rules.MIN_LOSS_IMPROVEMENT
    )
    if len(ctx.valid_gns) >= rules.MIN_POINTS_FOR_DEGENERATE_CHECK:
        if all(g == 0.0 for g in ctx.valid_gns) and _loss_improved:
            ctx.no("zero-grad",
               f"all {len(ctx.valid_gns)} gradient norms are 0.0 but the loss improved from "
               f"{_nz_losses[0]:.4f} to {min(_nz_losses):.4f} - the log is reporting an "
               "aggregate, not the true gradient norm")
        elif all(g == 0.0 for g in ctx.valid_gns):
            ctx.ok("zero-grad")
            return [{
                "id": "TP-ZERO-GRAD",
                "level": "FAIL",
                "message": "Gradient norm is exactly zero on every logged step - no gradient reached the weights.",
                "evidence": (
                    f"all {len(ctx.valid_gns)} finite gradient norms are exactly 0.0. The backward "
                    "graph is severed or every parameter is frozen. With PEFT this is usually "
                    "reentrant gradient checkpointing (use_reentrant=True) over frozen input "
                    "embeddings, which detaches the graph before it reaches the adapters - call "
                    "enable_input_require_grads() or pass use_reentrant=False."
                ),
            }]
        else:
            ctx.ok("zero-grad")
    elif not ctx.valid_gns:
        ctx.no("zero-grad", "no finite gradient norms in the log")
    else:
        ctx.no("zero-grad", f"fewer than {rules.MIN_POINTS_FOR_DEGENERATE_CHECK} finite gradient norms")
    return []

def check_grad_spike(ctx: CheckContext) -> list[dict]:
    if len(ctx.valid_gns) > 5:
        sorted_gns = sorted(ctx.valid_gns)
        median_gn = sorted_gns[len(sorted_gns)//2]
        if median_gn > 0:
            ctx.ok("grad-spike")
            spikes = [g for g in ctx.valid_gns if g > median_gn * rules.MAX_GRAD_NORM_SPIKE_RATIO]
            if spikes:
                return [{"id": "TP-GRAD-SPIKE", "level": "WARN", "message": "Gradient norm spikes detected.", "evidence": f"Max gn {max(spikes):.2f} > {rules.MAX_GRAD_NORM_SPIKE_RATIO}x median ({median_gn:.2f})"}]
        else:
            ctx.no("grad-spike", "median gradient norm is zero - no scale to measure a spike against")
    elif not ctx.valid_gns:
        ctx.no("grad-spike", "no finite gradient norms in the log")
    else:
        ctx.no("grad-spike", "fewer than 6 finite gradient norms")
    return []

def check_lr(ctx: CheckContext) -> list[dict]:
    if ctx.lrs:
        ctx.ok("lr")
        zeros = sum(1 for lr in ctx.lrs if lr <= 0)
        zero_frac = zeros / len(ctx.lrs)
        if zero_frac >= rules.ZERO_LR_FAIL_FRACTION:
            return [{"id": "TP-ZERO-LR", "level": "FAIL", "message": "Learning rate is zero for the entire run - the optimizer never steps.", "evidence": f"{zero_frac*100:.1f}% of steps have lr=0"}]
        elif zero_frac > rules.MAX_ZERO_LR_FRACTION:
            return [{"id": "TP-ZERO-LR-PARTIAL", "level": "WARN", "message": "Learning rate is zero for a large fraction of the run.", "evidence": f"{zero_frac*100:.1f}% of steps have lr=0"}]
    else:
        ctx.no("lr", "no learning-rate column in the log")
    return []

def check_throughput(ctx: CheckContext) -> list[dict]:
    if len(ctx.times) >= 2 and ctx.times[-1] > ctx.times[0]:
        span = ctx.times[-1] - ctx.times[0]
        steps_covered = ctx.time_steps[-1] - ctx.time_steps[0]
        if steps_covered > 0:
            rate = steps_covered / span
            return [{"id": "TP-THROUGHPUT", "level": "INFO", "message": "Throughput measured from log timestamps.",
                             "evidence": f"{rate:.2f} steps/sec over {span:.0f}s observed."}]
    return []

def check_overfit(ctx: CheckContext) -> list[dict]:
    if len(ctx.eval_losses) >= rules.OVERFIT_MIN_EVALS:
        ctx.ok("overfit")
        eval_min = min(ctx.eval_losses)
        i_min = ctx.eval_losses.index(eval_min)
        if len(ctx.eval_losses) - i_min - 1 >= 3:
            last_3 = sorted(ctx.eval_losses[-3:])
            med_last_3 = last_3[len(last_3)//2]
            if med_last_3 > eval_min * rules.OVERFIT_RATIO:
                min_step = ctx.eval_loss_steps[i_min]

                tl_at_min = None
                for i, step in enumerate(ctx.loss_steps):
                    if step >= min_step:
                        tl_at_min = ctx.losses[i]
                        break

                if tl_at_min is None and ctx.losses:
                    tl_at_min = ctx.losses[-1]

                if tl_at_min is not None and ctx.valid_losses and ctx.valid_losses[-1] < tl_at_min:
                    ratio = med_last_3 / eval_min if eval_min > 0 else float('inf')
                    return [{
                        "id": "TP-OVERFIT",
                        "level": "WARN",
                        "message": "Overfitting detected: eval loss has significantly degraded while train loss continued falling.",
                        "evidence": f"eval_loss min {eval_min:.2f} @step{int(min_step)} rose to {med_last_3:.2f} ({ratio:.1f}x > {rules.OVERFIT_RATIO}) over the last 3 evals while train_loss fell to {ctx.valid_losses[-1]:.2f} - best checkpoint was near step {int(min_step)}."
                    }]
    elif not ctx.eval_losses:
        ctx.no("overfit", "no eval_loss in the log - this run has no generalisation signal at all")
    else:
        ctx.no("overfit", f"fewer than {rules.OVERFIT_MIN_EVALS} eval points")
    return []

def check_step_time(ctx: CheckContext) -> list[dict]:
    if len(ctx.step_times) >= 10:
        first_50_idx = max(1, len(ctx.step_times) // 2)
        last_20_idx = len(ctx.step_times) - max(1, int(len(ctx.step_times) * 0.2))

        first_50 = sorted(ctx.step_times[:first_50_idx])
        last_20 = sorted(ctx.step_times[last_20_idx:])

        med_first_50 = first_50[len(first_50)//2] if first_50 else 0
        med_last_20 = last_20[len(last_20)//2] if last_20 else 0

        if med_first_50 > 0:
            ctx.ok("step-time")
            if med_last_20 > rules.STEP_TIME_CLIFF_RATIO * med_first_50:
                return [{"id": "TP-STEP-CLIFF", "level": "WARN", "message": "Step time cliff detected: recent steps are significantly slower.",
                                 "evidence": f"median recent step_time {med_last_20:.2f}s > {rules.STEP_TIME_CLIFF_RATIO}x median early step_time ({med_first_50:.2f}s)"}]
        else:
            ctx.no("step-time", "median early step_time is zero - no baseline to compare against")
    elif not ctx.step_times:
        ctx.no("step-time", "no step_time column in the log")
    else:
        ctx.no("step-time", "fewer than 10 step_time points")
    return []

def check_loader(ctx: CheckContext) -> list[dict]:
    if ctx.valid_loader_fractions:
        ctx.ok("loader")
        med_frac = sorted(ctx.valid_loader_fractions)[len(ctx.valid_loader_fractions)//2]
        if med_frac > rules.LOADER_FRACTION_MAX:
            return [{"id": "TP-LOADER-BOUND", "level": "WARN", "message": "Dataloader stall detected: spending too much time loading data.",
                             "evidence": f"median loader_time/step_time {med_frac*100:.1f}% > {rules.LOADER_FRACTION_MAX*100:.1f}%"}]
    else:
        ctx.no("loader", "no loader_time/step_time pair in the log")
    return []

def check_gpu_util(ctx: CheckContext) -> list[dict]:
    if ctx.gpu_utils:
        med_gpu = sorted(ctx.gpu_utils)[len(ctx.gpu_utils)//2]
        return [{"id": "TP-GPU-UTIL", "level": "INFO", "message": "GPU utilization context.",
                         "evidence": f"median gpu_util {med_gpu:.1f}% observed."}]
    return []


CHECK_REGISTRY = [
    check_nan,
    check_zero_loss,
    check_flat_loss,
    check_divergence,
    check_dead_run,
    check_zero_grad,
    check_grad_spike,
    check_lr,
    check_throughput,
    check_overfit,
    check_step_time,
    check_loader,
    check_gpu_util,
]


def check_records(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-RECORDS", "level": "FAIL", "message": "No valid log records found.", "evidence": ""}],
            "checks": _nothing_ran("no records parsed"),
        }

    ctx = CheckContext(records)

    if not ctx.losses:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-LOSS", "level": "FAIL", "message": "Could not find loss metric in logs.", "evidence": ""}],
            "checks": _nothing_ran("no loss column in the log"),
        }
        
    findings = []
    
    for check in CHECK_REGISTRY:
        findings.extend(check(ctx))
        
    verdict = "PASS"
    for f in findings:
        level = f["level"]
        if level == "FAIL":
            verdict = "FAIL"
        elif level == "WARN" and verdict == "PASS":
            verdict = "WARN"

    if verdict == "PASS":
        if not ctx.ran:
            verdict = "NOT-CHECKED"
            msg = f"{len(CHECK_GROUPS)} check groups considered, 0 executed."
            skipped_list = "; ".join(f"{g} ({r})" for g, r in sorted(ctx.skipped.items()))
            findings.append({
                "id": "TP-NOT-CHECKED", 
                "level": "NOT-CHECKED", 
                "message": msg, 
                "evidence": f"Skipped: {skipped_list}"
            })
        else:
            msg = "No mechanical failures detected."
            msg += f" Ran: {', '.join(sorted(ctx.ran))}."
            if ctx.skipped:
                msg += " Skipped: " + "; ".join(f"{g} ({r})" for g, r in sorted(ctx.skipped.items())) + "."
            findings.append({"id": "TP-PASS", "level": "PASS", "message": msg, "evidence": f"{len(ctx.valid_losses)} steps analyzed."})

    return {
        "verdict": verdict,
        "findings": findings,
        "checks": {"ran": sorted(ctx.ran), "skipped": ctx.skipped},
    }

def check_epoch(log_path: str | Path, fmt: str = "auto", mapping_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    records = parse_log_with_format(log_path, fmt, mapping_overrides)
    if not records:
        return {
            "verdict": "FAIL",
            "findings": [{"id": "TP-NO-RECORDS", "level": "FAIL", "message": "No valid log records found.", "evidence": str(log_path)}],
            "checks": _nothing_ran("no records parsed"),
        }
    return check_records(records)
