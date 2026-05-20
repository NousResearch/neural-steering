import argparse
import json
import os
import glob
import gc
from typing import Dict, List
from safetensors.torch import load_file, save_file

def apply_weight_surgery(
    model_dir: str, 
    output_dir: str, 
    ablation_map: Dict[int, List[int]],
    run_metadata: Dict[str, str]
):
    os.makedirs(output_dir, exist_ok=True)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
        
    ablation_map = {int(k): v for k, v in ablation_map.items()}
    total_ablated = 0

    # Serialize our ablation configuration and run info to embed in safetensors metadata
    serialized_ablation = json.dumps(ablation_map)
    custom_metadata = {
        **run_metadata,
        "ablation_circuit_map": serialized_ablation,
        "surgical_provenance": "apply_surgery.py v1.0 - offline weight ablation"
    }

    for file_path in safetensor_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Load shard (using zero-copy memory mapping under the hood)
        state_dict = load_file(file_path)
        
        for layer_idx, neurons in ablation_map.items():
            gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            for key in [gate_key, up_key, down_key]:
                if key in state_dict:
                    # In-place modification to avoid extra tensor memory allocation
                    for neuron_idx in neurons:
                        if "down_proj" in key:
                            state_dict[key][:, neuron_idx] = 0.0
                        else:
                            state_dict[key][neuron_idx, :] = 0.0
                    if key == gate_key:
                        total_ablated += len(neurons)
                        print(f"  Ablated {len(neurons)} neurons in layer {layer_idx} (gate+up+down where present)")
                
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        
        # Save file, embedding the metadata into the safetensors header!
        save_file(state_dict, save_path, metadata=custom_metadata)
        
        # Force immediate memory reclamation to prevent host RAM OOM
        del state_dict
        gc.collect()
        
    print(f"\nSurgery complete. {total_ablated} total neurons hard-ablated.")
    print(f"Auditable checkpoint saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditable Offline Safetensors Weight Ablation Surgery")
    parser.add_argument("--model_dir", type=str, default="/root/llama-3.1-8b-instruct-base")
    parser.add_argument("--output_dir", type=str, default="/root/llama-ablated")
    parser.add_argument("--indices_path", type=str, default="canonical_indices.json")
    parser.add_argument("--experiment_id", type=str, default="exp_refusal_ablation_001")
    args = parser.parse_args()

    # Load canonical indices
    if not os.path.exists(args.indices_path):
        raise FileNotFoundError(f"Missing circuit mapping index: {args.indices_path}")
        
    with open(args.indices_path, "r") as f:
        indices = json.load(f)

    meta = {
        "experiment_id": args.experiment_id,
        "source_model": os.path.abspath(args.model_dir),
    }

    apply_weight_surgery(args.model_dir, args.output_dir, indices, meta)
