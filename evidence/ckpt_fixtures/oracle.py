"""Read fixtures with torch and output oracle.json ground truth."""

import json
import os
import torch

def get_tensor_stats(t):
    if t.is_floating_point():
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
    else:
        nan_count = 0
        inf_count = 0
        
    is_all_zero = bool(torch.all(t == 0).item()) if t.numel() > 0 else True
    
    return {
        "dtype": str(t.dtype).split('.')[-1],
        "shape": list(t.shape),
        "numel": t.numel(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "is_all_zero": is_all_zero
    }

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures = [
        "clean.pt",
        "nan_weights.pt",
        "inf_weights.pt",
        "dead_layer.pt",
        "no_optimizer.pt",
        "shape_drift.pt",
        "mixed_dtype.pt"
    ]
    
    oracle_data = {}
    
    for f in fixtures:
        path = os.path.join(out_dir, f)
        if not os.path.exists(path):
            continue
            
        state = torch.load(path, map_location="cpu", weights_only=False)
        has_optim = "optimizer" in state
        model_state = state.get("model", state)
            
        file_data = {
            "has_optimizer": has_optim,
            "tensors": {}
        }
        
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor):
                file_data["tensors"][k] = get_tensor_stats(v)
                
        oracle_data[f] = file_data
        
    with open(os.path.join(out_dir, "oracle.json"), "w") as f:
        json.dump(oracle_data, f, indent=2)

if __name__ == "__main__":
    main()
