import time

from trainproof.epoch import check_records
from trainproof.objective import TargetCoverage, check_ignore_index


def _infer_num_classes(model):
    """Output-layer width, preferred from config, else the last Linear's out_features."""
    cfg = getattr(model, "config", None)
    for attr in ("vocab_size", "num_labels", "n_classes"):
        v = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            return v
    out = None
    for module in model.modules():
        if module.__class__.__name__ == "Linear":
            out = getattr(module, "out_features", out)
    return out


def _infer_ignore_index(model):
    """-100 is the PyTorch/HF default; a model that overrides it is the interesting case."""
    cfg = getattr(model, "config", None)
    v = getattr(cfg, "ignore_index", None) if cfg is not None else None
    return v if isinstance(v, int) else -100


def _labels_from_batch(batch):
    if isinstance(batch, dict):
        for key in ("labels", "label", "target", "targets"):
            if key in batch:
                return batch[key]
        return None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[-1]
    return None


def _convert_state_to_records(state) -> list[dict]:
    history = getattr(state, "log_history", [])
    records = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if "train_runtime" in entry:  # end-of-training summary, not a step
            continue
        # HF logs eval results as SEPARATE entries that carry eval_loss and no
        # loss. Requiring "loss" here dropped every one of them, and eval_loss
        # was not copied even when it shared an entry with loss -- so
        # TP-OVERFIT could never reach its 4-eval minimum and was structurally
        # unreachable from the callback, while the file path saw it fine. This
        # now mirrors adapters.parse_hf_trainer_state.
        if "loss" not in entry and "eval_loss" not in entry:
            continue
        record = {}
        for k in ["loss", "eval_loss", "grad_norm", "step"]:
            if k in entry and entry[k] is not None:
                try:
                    record[k] = float(entry[k])
                except (ValueError, TypeError):
                    pass
        if "learning_rate" in entry and entry["learning_rate"] is not None:
            try:
                record["lr"] = float(entry["learning_rate"])
            except (ValueError, TypeError):
                pass
        records.append(record)
    return records

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass
    _HAS_TRANSFORMERS = False
else:
    _HAS_TRANSFORMERS = True

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False



class TrainproofCallback(TrainerCallback):
    """Judge a live HuggingFace training run with trainproof's deterministic rules.

    policy:
      "warn" (DEFAULT): observe and report only. On a FAIL verdict it prints the
        findings and lets training continue. It NEVER interrupts your run — safe
        to leave on by default, including for experiments you expect to fail.
      "stop_on_fail": opt-in. Additionally sets control.should_training_stop on a
        FAIL verdict, aborting the run to save GPU time. This is the only mode
        that takes an irreversible action, and you must ask for it explicitly.

    check_every: minimum steps between checks. min_points: minimum logged loss
    points before any verdict is issued (avoids judging warm-up noise).
    A FAIL is announced once, not re-announced every subsequent check.
    """

    def __init__(self, policy="warn", check_every=25, min_points=10,
                 objective_check=True, objective_batches=32,
                 num_classes=None, ignore_index=None):
        if not _HAS_TRANSFORMERS:
            raise ImportError("pip install transformers is required to use TrainproofCallback")
        self.policy = policy
        self.check_every = check_every
        self.min_points = min_points
        self.last_checked_step = 0
        self.last_verdict = None
        self.last_log_time = time.monotonic()
        self.last_log_step = 0
        self.telemetry = {}
        # Objective checks run ONCE, before training. They read the output-layer
        # width, the ignore sentinel and the first batches of labels. Nothing about
        # the loss curve can substitute for them: a collision between the sentinel
        # and a real class leaves the curve identical to a healthy run.
        self.objective_check = objective_check
        self.objective_batches = objective_batches
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.objective_check:
            return

        model = kwargs.get("model")
        loader = kwargs.get("train_dataloader")
        if model is None:
            return

        n = self.num_classes if self.num_classes is not None else _infer_num_classes(model)
        ig = self.ignore_index if self.ignore_index is not None else _infer_ignore_index(model)
        if not n:
            print("\nTRAINPROOF - objective check skipped: could not determine output-layer width")
            return

        findings = list(check_ignore_index(n, ig))

        if loader is not None:
            cov = TargetCoverage(n, ig)
            seen = 0
            try:
                # A FRESH iterator: the trainer's own epoch must not lose batches.
                for batch in loader:
                    labels = _labels_from_batch(batch)
                    if labels is not None:
                        cov.observe(labels)
                    seen += 1
                    if seen >= self.objective_batches:
                        break
            except Exception:  # noqa: S110
                # A dataloader we cannot sample is a check we cannot run, not a
                # finding. Reported as skipped rather than swallowed silently.
                print("\nTRAINPROOF - objective coverage skipped: train_dataloader not sampleable")
            else:
                if cov.n_targets:
                    findings.extend(cov.result())

        problems = [f for f in findings if f.get("level") == "FAIL"]
        if problems:
            if self.policy == "stop_on_fail":
                control.should_training_stop = True
                print("\nTRAINPROOF ABORT - the objective is broken before step 1:")
            else:
                print("\nTRAINPROOF WARNING - the objective is broken before step 1:")
            for f in problems:
                print(f"  [{f.get('level')}] {f.get('message', '')}")
                if f.get("evidence"):
                    print(f"         Evidence: {f['evidence']}")
            print("  No loss curve can reveal this. See RULES.md, Objective Rules.")

    def on_log(self, args, state, control, **kwargs):
        now = time.monotonic()
        step = getattr(state, "global_step", 0)
        
        steps_covered = step - self.last_log_step
        if steps_covered > 0:
            elapsed = now - self.last_log_time
            sec_per_step = elapsed / steps_covered
            
            tel = {"step_time": sec_per_step}
            if _HAS_PYNVML:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    tel["gpu_util"] = float(util.gpu)
                except Exception:  # noqa: S110
                    # GPU utilisation is optional CONTEXT, never a judgement:
                    # TP-GPU-UTIL is INFO-only and is excluded from CHECK_GROUPS
                    # for exactly this reason. If nvml cannot answer, the column
                    # is simply absent and the rules that need it report
                    # themselves as skipped. Swallowing here cannot hide a
                    # finding, because there is no finding to hide.
                    pass
            self.telemetry[step] = tel
            
        self.last_log_time = now
        self.last_log_step = step

        if step < self.last_checked_step + self.check_every:
            return

        records = _convert_state_to_records(state)
        for r in records:
            s = r.get("step")
            if s in self.telemetry:
                r.update(self.telemetry[s])

        if len(records) < self.min_points:
            return

        self.last_checked_step = step
        
        report = check_records(records)
        verdict = report.get("verdict", "UNKNOWN")
        findings = report.get("findings", [])

        if verdict == "FAIL" and self.last_verdict != "FAIL":
            if self.policy == "stop_on_fail":
                control.should_training_stop = True
                print(f"\nTRAINPROOF ABORT - stopping training at step {step}. Findings:")
            else:
                print("\nTRAINPROOF WARNING - this run looks doomed:")
                
            for f in findings:
                level = f.get("level", "INFO")
                msg = f.get("message", "")
                ev = f.get("evidence", "")
                print(f"  [{level}] {msg}")
                if ev:
                    print(f"         Evidence: {ev}")

        self.last_verdict = verdict
