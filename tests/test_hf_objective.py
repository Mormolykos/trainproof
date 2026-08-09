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
