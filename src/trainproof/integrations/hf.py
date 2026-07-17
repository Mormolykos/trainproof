from trainproof.epoch import check_records

def _convert_state_to_records(state) -> list[dict]:
    history = getattr(state, "log_history", [])
    records = []
    for entry in history:
        if not isinstance(entry, dict) or "loss" not in entry:
            continue
        record = {}
        for k in ["loss", "grad_norm", "step"]:
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

    def __init__(self, policy="warn", check_every=25, min_points=10):
        if not _HAS_TRANSFORMERS:
            raise ImportError("pip install transformers is required to use TrainproofCallback")
        self.policy = policy
        self.check_every = check_every
        self.min_points = min_points
        self.last_checked_step = 0
        self.last_verdict = None

    def on_log(self, args, state, control, **kwargs):
        step = getattr(state, "global_step", 0)
        if step < self.last_checked_step + self.check_every:
            return

        records = _convert_state_to_records(state)
        if len(records) < self.min_points:
            return

        self.last_checked_step = step
        
        report = check_records(records)
        verdict = report.get("verdict", "UNKNOWN")
        findings = report.get("findings", [])

        if verdict == "FAIL" and self.last_verdict != "FAIL":
            if self.policy == "stop_on_fail":
                setattr(control, "should_training_stop", True)
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
