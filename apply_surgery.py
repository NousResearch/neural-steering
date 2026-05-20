import json
import os
import glob
from safetensors.torch import load_file, save_file

def apply_weight_surgery(model_dir, output_dir, ablation_map):
    os.makedirs(output_dir, exist_ok=True)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    
    ablation_map = {int(k): v for k, v in ablation_map.items()}
    total_ablated = 0

    for file_path in safetensor_files:
        print(f"Processing {os.path.basename(file_path)}...")
        state_dict = load_file(file_path)
        modified = False
        
        for layer_idx, neurons in ablation_map.items():
            gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            # Process each projection independently since they may be in different shards
            for key in [gate_key, up_key, down_key]:
                if key in state_dict:
                    for neuron_idx in neurons:
                        if "down_proj" in key:
                            state_dict[key][:, neuron_idx] = 0.0
                        else:
                            state_dict[key][neuron_idx, :] = 0.0
                    modified = True
                    if key == gate_key:  # Count once per layer per shard
                        total_ablated += len(neurons)
                        print(f"  Ablated {len(neurons)} neurons in layer {layer_idx} (gate+up+down where present)")
                
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        save_file(state_dict, save_path)
        
    print(f"\nSurgery complete. {total_ablated} total neurons hard-ablated.")
    print(f"Surgical checkpoint saved to {output_dir}")

with open("canonical_indices.json", "r") as f:
    indices = json.load(f)

apply_weight_surgery("/root/llama-3.1-8b-instruct-base", "/root/llama-ablated", indices)
