import time

from trainproof.epoch import check_records
from trainproof.objective import TargetCoverage, check_ignore_index, check_label_alignment


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


def _input_ids_from_batch(batch):
    """The model inputs the labels are meant to be aligned with.

    Only the explicit key counts. A positional batch is guessed at for labels
    (batch[-1]) because that convention is near-universal, but there is no equally
    safe guess for the inputs, and a wrong guess here would produce a FAIL against
    a correct run. Absent means the alignment check does not run.
    """
    if isinstance(batch, dict):
        return batch.get("input_ids")
    return None


def _unwrap(model):
    """Peel the wrappers that sit between a Trainer and the real module.

    PEFT, DDP/FSDP (`.module`) and `torch.compile` (`._orig_mod`) all present a class
    whose name says nothing about the loss. Without this the confirmation below reports
    "unknown" for most real training setups -- safe, but it would disable the check for
    almost everyone.
    """
    seen = 0
    while seen < 5:
        nxt = None
        if hasattr(model, "get_base_model"):
            try:
                nxt = model.get_base_model()
            except Exception:  # noqa: BLE001 - a wrapper that refuses is just the end
                nxt = None
        if nxt is None:
            # Explicit None tests, not `or`: an nn.ModuleDict/ModuleList defines
            # __len__, so an empty one is FALSY while being perfectly present, and
            # `a or b` would silently skip it and halt the unwrap chain here.
            nxt = getattr(model, "_orig_mod", None)
            if nxt is None:
                nxt = getattr(model, "module", None)
        if nxt is None or nxt is model:
            return model
        model = nxt
        seen += 1
    return model


def _loss_shifts_labels(model):
    """Does THIS model's forward shift labels before the loss? True, False or None.

    Returns True only for a class that **transformers itself** defines and whose name
    ends in `ForCausalLM`, because those are the forwards that perform the shift. The
    module-origin half is the load-bearing half: a user subclass called
    `MyForCausalLM` that overrides `forward` with a non-shifting loss would otherwise
    be confirmed as shifting, and a correct run would be failed. A subclass defined in
    user code has its own `__module__`, so it reports unknown, which never fails.

    Never returns False. There is no way to confirm a negative about someone else's
    loss function, so `False` is only ever supplied explicitly by a caller.

    ⚠️ **This confirms the model, which is not the same as confirming the loss.** A
    `Trainer` subclass that overrides `compute_loss` can bypass `model(..., labels=...)`
    entirely and compute a loss with different conventions. That is not visible from the
    model object, and the callback is never handed the trainer. A caller in that position
    should pass `loss_shifts=` to `TrainproofCallback` explicitly; both adversarial
    reviewers raised this and it is recorded in RULES.md rather than papered over.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "is_encoder_decoder", False):
        # Encoder-decoder derives decoder_input_ids from labels internally, and
        # input_ids is the ENCODER's source sequence. The convention does not apply.
        return None
    # The UNWRAPPED module only. Testing the outer wrapper as well would widen the set
    # of objects that can answer True, and every widening here is a step toward failing
    # a correct run.
    cls = type(_unwrap(model))
    origin = getattr(cls, "__module__", "") or ""
    if origin.startswith("transformers.") and cls.__name__.endswith("ForCausalLM"):
        return True
    return None


def _is_streaming(loader):
    """Would sampling this dataloader consume the user's training data?

    A map-style dataset can be iterated again from the start at no cost. A streaming
    one may not be restartable, so taking a fresh iterator here can permanently drop
    batches from the real epoch. Detected without importing torch: by class name
    anywhere in the MRO -- `torch.utils.data.IterableDataset` and
    `datasets.IterableDataset` are different classes that share a name, and both are
    streaming, so the collision is harmless here -- and by the absence of `__len__`,
    which is the property that actually makes a dataset non-indexable.

    Name-based detection can miss an exotic wrapper. It is deliberately biased toward
    reporting "streaming" and skipping the sample: a missed check is recoverable, and
    silently eating a user's training data is not.
    """
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return False
    if any(c.__name__ == "IterableDataset" for c in type(ds).__mro__):
        return True
    # No __len__ AND no __getitem__. Requiring both narrows this away from map-style
    # datasets: a map-style dataset without __len__ cannot drive DataLoader's default
    # sampler anyway, so demanding the absence of __getitem__ too costs nothing and
    # stops the guard from disabling the check on a legitimate indexable dataset.
    return (
        hasattr(ds, "__iter__")
        and not hasattr(ds, "__len__")
        and not hasattr(ds, "__getitem__")
    )


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
    # The stand-in deliberately shadows the real class so the callback can be
    # defined without transformers installed. mypy sees two definitions of one
    # name and cannot know only one is ever live.
    class TrainerCallback:  # type: ignore[no-redef]
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
                 num_classes=None, ignore_index=None, loss_shifts=None):
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
        # Whether the LOSS shifts labels. None asks the model, which can only ever
        # answer True or "unknown". Pass it explicitly if you override
        # `Trainer.compute_loss`: that bypasses the model's own loss, so the model
        # object no longer describes the convention actually in force.
        self.loss_shifts = loss_shifts

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

        if loader is not None and _is_streaming(loader):
            # Sampling a stream is not free: a fresh iterator over a non-restartable
            # dataset takes batches the training epoch will never see. Refusing to
            # measure is the correct outcome; eating the user's data to produce a
            # finding is not.
            print("\nTRAINPROOF - objective sampling skipped: streaming dataloader, "
                  "reading it here would consume batches from the training epoch")
            loader = None

        if loader is not None:
            cov = TargetCoverage(n, ig)
            seen = 0
            # The alignment check needs input_ids and labels from the SAME batch, and
            # one batch carries far more than the positions it needs. Coverage wants
            # many batches; alignment wants one. Keep the first usable pair.
            first_pair = None
            try:
                # A FRESH iterator: the trainer's own epoch must not lose batches.
                for batch in loader:
                    labels = _labels_from_batch(batch)
                    if labels is not None:
                        cov.observe(labels)
                        if first_pair is None:
                            input_ids = _input_ids_from_batch(batch)
                            if input_ids is not None:
                                first_pair = (input_ids, labels)
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
                if first_pair is not None:
                    shifts = (
                        self.loss_shifts if self.loss_shifts is not None
                        else _loss_shifts_labels(model)
                    )
                    findings.extend(check_label_alignment(
                        first_pair[0], first_pair[1], ig, loss_shifts=shifts,
                    ))

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
