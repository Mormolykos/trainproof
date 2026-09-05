"""The objective check must fire from the callback, with no user action.

A check that requires the user to already suspect the bug detects nothing in
practice. These tests pin the automatic path: on_train_begin reads the output-layer
width, the ignore sentinel and the first batches of labels, and reports before step 1.
"""

import trainproof.integrations.hf as hf_mod
from trainproof.integrations.hf import (
    TrainproofCallback,
    _infer_ignore_index,
    _infer_num_classes,
    _labels_from_batch,
)

hf_mod._HAS_TRANSFORMERS = True


class MockControl:
    def __init__(self):
        self.should_training_stop = False


class MockConfig:
    def __init__(self, vocab_size=None, ignore_index=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if ignore_index is not None:
            self.ignore_index = ignore_index


class MockModel:
    def __init__(self, vocab_size=None, ignore_index=None):
        self.config = MockConfig(vocab_size, ignore_index)

    def modules(self):
        return []


def run_begin(model, loader=None, policy="warn", **kw):
    cb = TrainproofCallback(policy=policy, **kw)
    control = MockControl()
    cb.on_train_begin(None, None, control, model=model, train_dataloader=loader)
    return control


# --------------------------------------------------------------------------

def test_infer_num_classes_from_config():
    assert _infer_num_classes(MockModel(vocab_size=1025)) == 1025


def test_infer_ignore_index_defaults_to_minus_100():
    assert _infer_ignore_index(MockModel(vocab_size=10)) == -100


def test_infer_ignore_index_reads_config_override():
    assert _infer_ignore_index(MockModel(vocab_size=10, ignore_index=1024)) == 1024


def test_labels_from_dict_and_tuple_batches():
    assert _labels_from_batch({"input_ids": [1], "labels": [2]}) == [2]
    assert _labels_from_batch(([1], [2])) == [2]
    assert _labels_from_batch({"input_ids": [1]}) is None


def test_collision_is_reported_before_step_one(capsys):
    # The real failure: 1025 classes, sentinel 1024.
    run_begin(MockModel(vocab_size=1025, ignore_index=1024))
    out = capsys.readouterr().out
    assert "objective is broken before step 1" in out
    assert "ignore_index is a valid class" in out
    assert "No loss curve can reveal this" in out


def test_stop_on_fail_aborts_before_training():
    control = run_begin(MockModel(vocab_size=1025, ignore_index=1024), policy="stop_on_fail")
    assert control.should_training_stop is True


def test_warn_policy_never_stops_training():
    control = run_begin(MockModel(vocab_size=1025, ignore_index=1024), policy="warn")
    assert control.should_training_stop is False


def test_healthy_model_says_nothing(capsys):
    run_begin(MockModel(vocab_size=1025))          # sentinel -100, no labels
    assert capsys.readouterr().out == ""


def test_dead_class_found_from_the_dataloader(capsys):
    # 64 classes, class 17 never appears as a target anywhere in the data.
    loader = [{"labels": [c for c in range(64) if c != 17]} for _ in range(3)]
    run_begin(MockModel(vocab_size=64), loader=loader)
    out = capsys.readouterr().out
    assert "never appears as a training target" in out
    assert "[17]" in out


def test_objective_check_can_be_disabled(capsys):
    run_begin(MockModel(vocab_size=1025, ignore_index=1024), objective_check=False)
    assert capsys.readouterr().out == ""


def test_explicit_arguments_override_inference(capsys):
    # Model looks clean; the user knows the real numbers and passes them.
    run_begin(MockModel(vocab_size=10), num_classes=1025, ignore_index=1024)
    assert "ignore_index is a valid class" in capsys.readouterr().out


def test_unsampleable_dataloader_is_reported_not_swallowed(capsys):
    class Boom:
        def __iter__(self):
            raise RuntimeError("no")

    run_begin(MockModel(vocab_size=64), loader=Boom())
    assert "objective coverage skipped" in capsys.readouterr().out


def test_missing_output_width_is_reported_as_skipped(capsys):
    run_begin(MockModel())
    assert "could not determine output-layer width" in capsys.readouterr().out

# --------------------------------------------------------------------------
# label alignment from the callback
#
# The callback may only reach FAIL when it can CONFIRM the loss shifts. That
# confirmation requires a class transformers itself defines: a user subclass
# named `MyForCausalLM` that overrides forward with a non-shifting loss must
# report unknown, or a correct run gets aborted before step 1.
# --------------------------------------------------------------------------

from trainproof.integrations.hf import (  # noqa: E402
    _input_ids_from_batch,
    _is_streaming,
    _loss_shifts_labels,
)


class HFCausalModel(MockModel):
    """Stands in for a transformers-defined ...ForCausalLM."""


HFCausalModel.__name__ = "LlamaForCausalLM"
HFCausalModel.__module__ = "transformers.models.llama.modeling_llama"


class UserCausalModel(MockModel):
    """A user subclass whose forward may not shift. Must NOT be confirmed."""


UserCausalModel.__name__ = "MyForCausalLM"
UserCausalModel.__module__ = "my_project.models"


def _sft_batch(preshift, k=4, n=64, prompt=12):
    ids = [list(range(10 + r * 1000, 10 + r * 1000 + n)) for r in range(k)]
    labels = []
    for row in ids:
        lab = (row[1:] + [row[-1]]) if preshift else row[:]
        labels.append([-100] * prompt + lab[prompt:])
    return {"input_ids": ids, "labels": labels}


# --- confirmation -----------------------------------------------------------

def test_transformers_causal_lm_is_confirmed():
    assert _loss_shifts_labels(HFCausalModel(vocab_size=32000)) is True


def test_user_subclass_is_not_confirmed():
    """The load-bearing half: same name, different module, so unknown -> never FAIL."""
    assert _loss_shifts_labels(UserCausalModel(vocab_size=32000)) is None


def test_encoder_decoder_is_not_confirmed():
    m = HFCausalModel(vocab_size=32000)
    m.config.is_encoder_decoder = True
    assert _loss_shifts_labels(m) is None


def test_confirmation_unwraps_a_peft_style_wrapper():
    inner = HFCausalModel(vocab_size=32000)

    class Wrapper:
        config = inner.config
        def get_base_model(self):
            return inner
        def modules(self):
            return []

    assert _loss_shifts_labels(Wrapper()) is True


def test_confirmation_unwraps_ddp_and_compile_wrappers():
    inner = HFCausalModel(vocab_size=32000)

    class DDP:
        config = inner.config
        module = inner
        def modules(self):
            return []

    class Compiled:
        config = inner.config
        _orig_mod = inner
        def modules(self):
            return []

    assert _loss_shifts_labels(DDP()) is True
    assert _loss_shifts_labels(Compiled()) is True


def test_confirmation_never_returns_false():
    """False is caller-asserted only; it must never be inferred from a model."""
    for m in (HFCausalModel(vocab_size=10), UserCausalModel(vocab_size=10),
              MockModel(vocab_size=10)):
        assert _loss_shifts_labels(m) is not False


# --- the automatic path -----------------------------------------------------

def test_callback_catches_preshifted_labels_on_a_confirmed_model(capsys):
    control = run_begin(HFCausalModel(vocab_size=32000),
                        loader=[_sft_batch(preshift=True)], policy="stop_on_fail")
    assert control.should_training_stop is True
    out = capsys.readouterr().out
    assert "TRAINPROOF ABORT" in out
    assert "two tokens ahead" in out


def test_callback_does_not_abort_an_unconfirmable_model(capsys):
    """The regression that matters: a correct custom loop must not be aborted."""
    control = run_begin(UserCausalModel(vocab_size=32000),
                        loader=[_sft_batch(preshift=True)], policy="stop_on_fail")
    assert control.should_training_stop is False
    assert "TRAINPROOF ABORT" not in capsys.readouterr().out


def test_callback_accepts_aligned_labels(capsys):
    control = run_begin(HFCausalModel(vocab_size=32000),
                        loader=[_sft_batch(preshift=False)], policy="stop_on_fail")
    assert control.should_training_stop is False
    assert "TRAINPROOF ABORT" not in capsys.readouterr().out


def test_callback_skips_alignment_without_input_ids(capsys):
    batch = _sft_batch(preshift=True)
    control = run_begin(HFCausalModel(vocab_size=32000),
                        loader=[(batch["labels"],)], policy="stop_on_fail")
    assert control.should_training_stop is False
    assert "two tokens ahead" not in capsys.readouterr().out


def test_input_ids_helper_only_accepts_explicit_key():
    assert _input_ids_from_batch({"input_ids": [[1, 2]]}) == [[1, 2]]
    assert _input_ids_from_batch({"labels": [[1, 2]]}) is None
    assert _input_ids_from_batch(([[1, 2]], [[3, 4]])) is None


# --- streaming dataloaders --------------------------------------------------

class _MapDataset:
    def __len__(self):
        return 4
    def __getitem__(self, i):
        return i


class IterableDataset:            # noqa: D101 - name is the detection signal
    def __iter__(self):
        return iter(())


class _Loader:
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self._batches = batches
    def __iter__(self):
        return iter(self._batches)


def test_streaming_is_detected_by_class_name_and_by_missing_len():
    assert _is_streaming(_Loader(IterableDataset(), [])) is True

    class NoLen:
        def __iter__(self):
            return iter(())

    assert _is_streaming(_Loader(NoLen(), [])) is True
    assert _is_streaming(_Loader(_MapDataset(), [])) is False
    assert _is_streaming(_Loader(None, [])) is False


def test_streaming_loader_is_not_sampled(capsys):
    """Reading a stream here would take batches the training epoch never sees."""
    loader = _Loader(IterableDataset(), [_sft_batch(preshift=True)])
    control = run_begin(HFCausalModel(vocab_size=32000), loader=loader,
                        policy="stop_on_fail")
    out = capsys.readouterr().out
    assert "streaming dataloader" in out
    assert control.should_training_stop is False


def test_map_style_loader_is_still_sampled(capsys):
    loader = _Loader(_MapDataset(), [_sft_batch(preshift=True)])
    control = run_begin(HFCausalModel(vocab_size=32000), loader=loader,
                        policy="stop_on_fail")
    assert control.should_training_stop is True
    assert "TRAINPROOF ABORT" in capsys.readouterr().out


# --- round-2 adversarial findings -------------------------------------------

def test_unwrap_survives_a_falsy_module():
    """`a or b` skipped a present-but-falsy module and halted the chain."""
    inner = HFCausalModel(vocab_size=32000)

    class EmptyContainer:
        """Falsy: defines __len__ returning 0, like an empty nn.ModuleDict."""
        config = inner.config
        module = inner
        def __len__(self):
            return 0
        def modules(self):
            return []

    assert _loss_shifts_labels(EmptyContainer()) is True


def test_outer_wrapper_alone_is_not_enough():
    """Only the UNWRAPPED module may answer True; every widening risks a false FAIL."""
    inner = UserCausalModel(vocab_size=32000)

    class Wrapper:
        config = inner.config
        def get_base_model(self):
            return inner
        def modules(self):
            return []

    Wrapper.__name__ = "LlamaForCausalLM"
    Wrapper.__module__ = "transformers.models.llama.modeling_llama"
    assert _loss_shifts_labels(Wrapper()) is None


def test_explicit_loss_shifts_overrides_model_detection(capsys):
    """The escape hatch for a custom Trainer.compute_loss."""
    control = run_begin(HFCausalModel(vocab_size=32000),
                        loader=[_sft_batch(preshift=True)],
                        policy="stop_on_fail", loss_shifts=False)
    assert control.should_training_stop is False
    assert "TRAINPROOF ABORT" not in capsys.readouterr().out


def test_explicit_loss_shifts_can_judge_an_unconfirmable_model(capsys):
    control = run_begin(UserCausalModel(vocab_size=32000),
                        loader=[_sft_batch(preshift=True)],
                        policy="stop_on_fail", loss_shifts=True)
    assert control.should_training_stop is True
    assert "two tokens ahead" in capsys.readouterr().out


def test_map_style_dataset_without_len_is_not_called_streaming():
    """Narrowed in round 2: missing __len__ alone must not disable the check."""
    class Indexable:
        def __iter__(self):
            return iter(())
        def __getitem__(self, i):
            return i

    assert _is_streaming(_Loader(Indexable(), [])) is False


def test_three_dimensional_input_degrades_to_not_checked():
    """A 3-D batch must not crash or produce a verdict."""
    from trainproof.objective import check_label_alignment as cla
    cube = [[[1, 2], [3, 4]] for _ in range(4)]
    f = cla(cube, cube, loss_shifts=True)
    assert [x["id"] for x in f] == ["TP-OBJ-LABEL-SHIFT-NOT-MEASURED"]
