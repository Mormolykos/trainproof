import trainproof.integrations.hf as hf_mod
from trainproof.integrations.hf import _convert_state_to_records, TrainproofCallback

# Patch so tests can run without transformers installed
hf_mod._HAS_TRANSFORMERS = True
class MockState:
    def __init__(self):
        self.global_step = 0
        self.log_history = []

class MockControl:
    def __init__(self):
        self.should_training_stop = False

def test_convert_state_to_records():
    state = MockState()
    state.log_history = [
        {"loss": 1.0, "step": 10, "learning_rate": 0.01, "eval_loss": 2.0},
        {"eval_loss": 1.5, "step": 10}, 
        {"loss": 0.5, "step": 20, "grad_norm": 0.1}
    ]
    records = _convert_state_to_records(state)
    assert len(records) == 2
    assert records[0] == {"loss": 1.0, "step": 10.0, "lr": 0.01}
    assert records[1] == {"loss": 0.5, "step": 20.0, "grad_norm": 0.1}

def test_hf_callback_warn_policy():
    callback = TrainproofCallback(policy="warn", check_every=10, min_points=10)
    state = MockState()
    control = MockControl()
    
    # Add < 10 points -> should do nothing
    for i in range(5):
        state.log_history.append({"loss": 10.0, "step": i})
    state.global_step = 10
    callback.on_log(None, state, control)
    assert callback.last_verdict is None
    
    # Add up to 15 points (healthy) -> PASS
    for i in range(5, 15):
        state.log_history.append({"loss": 1.0, "step": i, "learning_rate": 1e-4})
    state.global_step = 20
    callback.on_log(None, state, control)
    assert callback.last_verdict == "PASS"
    assert not control.should_training_stop
    
    # Diverge -> FAIL but don't stop because policy=warn
    for i in range(15, 25):
        state.log_history.append({"loss": 50.0, "step": i, "learning_rate": 1e-4})
    state.global_step = 30
    callback.on_log(None, state, control)
    assert callback.last_verdict == "FAIL"
    assert not control.should_training_stop

def test_hf_callback_stop_on_fail():
    callback = TrainproofCallback(policy="stop_on_fail", check_every=10, min_points=10)
    state = MockState()
    control = MockControl()
    
    for i in range(20):
        state.log_history.append({"loss": 50.0, "step": i, "learning_rate": 1e-4}) 
    state.global_step = 20
    
    callback.on_log(None, state, control)
    assert callback.last_verdict == "FAIL"
    assert control.should_training_stop
