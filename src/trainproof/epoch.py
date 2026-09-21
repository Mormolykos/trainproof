import math
from pathlib import Path
from typing import Any

from . import coverage, rules
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
    "grad-finite",
    "flat-loss",
    "divergence",
    "dead-run",
    "grad-spike",
    "lr",
    "step-time",
    "loader",
    "overfit",
)


def _checks(ran: list[str], skipped: dict[str, str]) -> dict[str, Any]:
    """The `checks` block, with typed coverage alongside the sentences.

    Every report is assembled through here rather than by four separate dict
    literals, because a coverage field present on three paths out of four is
    worse than none: a reader who sees it once assumes it is always there, and
    the path that omits it becomes indistinguishable from full coverage.
    """
    return {
        "ran": sorted(ran),
        "skipped": skipped,
        "coverage": coverage.summarise(sorted(ran), skipped),
    }


def _nothing_ran(reason: str) -> dict[str, Any]:
    return _checks([], {g: reason for g in CHECK_GROUPS})


def _num(value: Any) -> float | None:
    """The domain of every metric column: a real number, or absent.

    R19: code that accepts or rejects evidence declares the predicate a value
    must satisfy, and a value failing it is treated exactly as ABSENT. Until
    v0.21 the only predicate here was `is not None`, so a metric logged as a
    string, a list or a dict reached `math.isnan()` and raised TypeError out of
    the constructor -- an uncaught crash where CONTRACTS.md promises a verdict
    or a documented "cannot judge" exit. Found by a property probe over
    non-numeric column values, not by a case anyone had thought of.

    `bool` is excluded deliberately: it is a subclass of `int`, so `True` would
    otherwise become a loss of 1.0 and be judged as if someone had measured it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class CheckContext:
    def __init__(self, records: list[dict]):
        self.records = records
        self.ran: list[str] = []
        self.skipped: dict[str, str] = {}
        self.losses = []
        self.loss_steps = []
        self.lrs = []
        self.grad_norms = []
        self.grad_steps = []
        self.times = []
        self.time_steps = []

        for i, r in enumerate(records):
            # `step` is arithmetic downstream (throughput spans, evidence
            # strings), so it carries the same domain as any metric. A
            # non-numeric step falls back to the record index rather than
            # propagating into a subtraction -- the fuzz in
            # tests/test_v021_nan_grad.py reached `check_throughput` with two
            # string steps and crashed there, in code older than this release.
            step = _num(r.get("step"))
            step = i if step is None else step
            loss = _num(r.get("loss"))
            lr = _num(r.get("lr"))
            gn = _num(r.get("grad_norm"))
            t = _num(r.get("time"))

            if loss is not None:
                self.losses.append(loss)
                self.loss_steps.append(step)
            if lr is not None: self.lrs.append(lr)
            if gn is not None:
                self.grad_norms.append(gn)
                self.grad_steps.append(step)
            if t is not None:
                self.times.append(t)
                self.time_steps.append(step)

        self.valid_losses = [v for v in self.losses if not math.isnan(v) and not math.isinf(v)]
        self.valid_gns = [g for g in self.grad_norms if not math.isnan(g) and not math.isinf(g)]
        
        self.eval_losses = []
        self.eval_loss_steps = []
        for i, r in enumerate(records):
            el = _num(r.get("eval_loss"))
            if el is not None and not math.isnan(el) and not math.isinf(el):
                self.eval_losses.append(el)
                # `r.get("step", i)` returned None whenever the key existed and
                # was null -- the default only covers a MISSING key, not a
                # present-and-unusable value. That None reached a `>=` against
                # an int in check_overfit. Same domain, same index fallback as
                # the loop above.
                eval_step = _num(r.get("step"))
                self.eval_loss_steps.append(i if eval_step is None else eval_step)
                
        self.step_times = []
        self.valid_loader_fractions = []
        self.gpu_utils = []
        for r in records:
            st = _num(r.get("step_time"))
            lt = _num(r.get("loader_time"))
            gu = _num(r.get("gpu_util"))
            
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

def check_nan_grad(ctx: CheckContext) -> list[dict]:
    """Judge the gradient-norm column against its domain.

    A gradient norm is a finite, non-negative real. A NaN or Inf is not a
    missing value: it is a positive statement that the backward pass produced a
    non-finite gradient, which the optimizer then applied to the weights.

    Every other gradient check in this file reads `valid_gns`, which filters
    non-finite values out. When they are ALL non-finite that list is empty, and
    those checks skip themselves with "no finite gradient norms in the log" -
    so the worse the corruption, the quieter trainproof became. coverage.py
    already drew this exact distinction for the skip *state* (see its note on
    the word "finite"); this check is the missing finding to go with it.
    """
    if not ctx.grad_norms:
        ctx.no("grad-finite", "no grad_norm column in the log")
        return []

    ctx.ok("grad-finite")
    bad = [
        s for s, g in zip(ctx.grad_steps, ctx.grad_norms, strict=False)
        if math.isnan(g) or math.isinf(g)
    ]
    if not bad:
        return []

    total = len(ctx.grad_norms)
    every = len(bad) == total
    return [{
        "id": "TP-NAN-GRAD",
        "level": "FAIL",
        "message": "Gradient norm is NaN or Inf - non-finite gradients reached the optimizer.",
        "evidence": (
            f"{len(bad)} of {total} logged gradient norms are non-finite"
            f"{' (every one)' if every else ''}; first at step {bad[0]:g}. "
            "Weights updated with a non-finite gradient are non-finite from that "
            "step on. A common cause is fp16 weights trained without a loss "
            "scaler (torch_dtype=float16 with fp16=False in TrainingArguments). "
            "Note that the loss column can still look clean: "
            "TrainingArguments.logging_nan_inf_filter defaults to True and "
            "substitutes an average for a NaN loss before it is ever logged."
        ),
    }]


def _zero_tail_onset(values: list[float]) -> int | None:
    """Index at which an exactly-zero tail begins, or None.

    Returns None when the series is entirely zero - TP-ZERO-LOSS owns that
    case - and None when nothing before the tail was positive, because then
    there is no live-then-dead transition to report.
    """
    if not values:
        return None
    i = len(values)
    while i > 0 and values[i - 1] == 0.0:
        i -= 1
    if i == 0 or i == len(values):
        return None
    if any(v > 0.0 for v in values[:i]):
        return i
    return None


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

        onset = _zero_tail_onset(ctx.valid_losses)
        if onset is not None and len(ctx.valid_losses) - onset >= rules.MIN_ZERO_TAIL_FOR_ONSET:
            tail = len(ctx.valid_losses) - onset
            return [{
                "id": "TP-ZERO-LOSS-ONSET",
                "level": "FAIL",
                "message": "Loss was positive and then became exactly zero for the rest of the run - the run died partway through.",
                "evidence": (
                    f"last positive loss {ctx.valid_losses[onset - 1]:.4f}, then {tail} "
                    "consecutive losses of exactly 0.0. Exact zero is a structural "
                    "signature, not convergence - a converging loss approaches zero "
                    "without reaching it. Every ratio-shaped rule reads this collapse "
                    "as a large improvement, which is why it is caught by equality "
                    "instead. Check the gradient norms: if they are non-finite, the "
                    "zeros are TrainingArguments.logging_nan_inf_filter (default True) "
                    "substituting an average for a NaN loss."
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
        # S-3 (2026-09-21): the predicate is `lr <= 0`, so negative learning
        # rates are counted here too. The evidence string used to read
        # "100.0% of steps have lr=0" for a series where every value was
        # -1e-4 -- a sentence that is false about the artifact. The predicate
        # is UNCHANGED (narrowing it would alter which runs FAIL, which needs
        # calibration); only the description of what was measured is fixed.
        nonpositive = [lr for lr in ctx.lrs if lr <= 0]
        zeros = len(nonpositive)
        zero_frac = zeros / len(ctx.lrs)
        n_negative = sum(1 for lr in nonpositive if lr < 0)
        _neg = f" ({n_negative} of them negative, not zero)" if n_negative else ""
        if zero_frac >= rules.ZERO_LR_FAIL_FRACTION:
            return [{"id": "TP-ZERO-LR", "level": "FAIL", "message": "Learning rate is zero for the entire run - the optimizer never steps.", "evidence": f"{zero_frac*100:.1f}% of logged steps have lr <= 0{_neg}"}]
        elif zero_frac > rules.MAX_ZERO_LR_FRACTION:
            return [{"id": "TP-ZERO-LR-PARTIAL", "level": "WARN", "message": "Learning rate is zero for a large fraction of the run.", "evidence": f"{zero_frac*100:.1f}% of logged steps have lr <= 0{_neg}"}]
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
    check_nan_grad,
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
        "checks": _checks(ctx.ran, ctx.skipped),
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
