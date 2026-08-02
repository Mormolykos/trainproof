"""Generate torch checkpoint fixtures for testing."""

import os

import torch

# Seeded so the committed fixtures and oracle.json stay in step. Without this,
# regenerating produces different values and every previously recorded NaN count
# silently refers to a file that no longer exists.
torch.manual_seed(0)


def get_base_state():
    return {
        "model": {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.zeros(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.ones(5),
        },
        "optimizer": {
            "state": {},
            "param_groups": [{"lr": 0.001}]
        }
    }

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. clean.pt
    state = get_base_state()
    torch.save(state, os.path.join(out_dir, "clean.pt"))
    
    # 2. nan_weights.pt
    state = get_base_state()
    state["model"]["layer1.weight"][0, 0] = float('nan')
    state["model"]["layer1.weight"][1, 1] = float('nan')
    torch.save(state, os.path.join(out_dir, "nan_weights.pt"))
    
    # 3. inf_weights.pt
    state = get_base_state()
    state["model"]["layer2.weight"][0, 0] = float('inf')
    state["model"]["layer2.weight"][1, 1] = float('-inf')
    state["model"]["layer2.weight"][2, 2] = float('inf')
    torch.save(state, os.path.join(out_dir, "inf_weights.pt"))
    
    # 4. dead_layer.pt
    state = get_base_state()
    state["model"]["layer2.weight"] = torch.zeros(5, 10)
    torch.save(state, os.path.join(out_dir, "dead_layer.pt"))
    
    # 5. no_optimizer.pt
    state = get_base_state()
    del state["optimizer"]
    torch.save(state, os.path.join(out_dir, "no_optimizer.pt"))
    
    # 6. shape_drift.pt
    state = get_base_state()
    state["model"]["layer1.weight"] = torch.randn(10, 12)
    torch.save(state, os.path.join(out_dir, "shape_drift.pt"))
    
    # 7. mixed_dtype.pt
    state = get_base_state()
    state["model"]["layer1.weight"] = state["model"]["layer1.weight"].to(torch.float16)
    state["model"]["layer2.weight"] = state["model"]["layer2.weight"].to(torch.bfloat16)
    torch.save(state, os.path.join(out_dir, "mixed_dtype.pt"))

if __name__ == "__main__":
    main()
