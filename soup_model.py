import torch
import os
import glob
import shutil
from safetensors.torch import load_file, save_file

# --- CONFIGURATION ---
base_dir = "/working/runs/rescuenet_segformer_optimzed"
output_dir = "/working/runs/rescuenet_segformer_optimzed/checkpoint-soup"
N = 2 

def load_model_weights(path):
    """Smart loader that checks for bin or safetensors"""
    bin_path = os.path.join(path, "pytorch_model.bin")
    safe_path = os.path.join(path, "model.safetensors")
    
    if os.path.exists(safe_path):
        print(f"  -> Loading safetensors from {os.path.basename(path)}")
        return load_file(safe_path), "safetensors"
    elif os.path.exists(bin_path):
        print(f"  -> Loading bin from {os.path.basename(path)}")
        return torch.load(bin_path, map_location='cpu'), "bin"
    else:
        raise FileNotFoundError(f"No weights found in {path}")

def main():
    # 1. Find all checkpoints
    all_folders = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    
    # FILTER: Exclude the output directory itself to prevent crashes
    checkpoints = [c for c in all_folders if "checkpoint-soup" not in os.path.basename(c)]
    
    # Sort by time
    checkpoints = sorted(checkpoints, key=os.path.getmtime)
    
    if len(checkpoints) < N:
        print(f"Not enough checkpoints! Found {len(checkpoints)}")
        return

    to_average = checkpoints[-N:]
    print(f"🥣 Souping these {N} checkpoints:")
    for c in to_average:
        print(f"  - {os.path.basename(c)}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Load first model
    soup, fmt = load_model_weights(to_average[0])

    # Add others
    for ckpt in to_average[1:]:
        state_dict, _ = load_model_weights(ckpt)
        for key in soup:
            # Safetensors/PyTorch compatibility fix
            if key in state_dict:
                soup[key] += state_dict[key]

    # Average
    print("Averaging...")
    for key in soup:
        soup[key] = soup[key] / float(len(to_average))

    # Save
    print(f"Saving to {output_dir}...")
    if fmt == "safetensors":
        save_file(soup, os.path.join(output_dir, "model.safetensors"))
    else:
        torch.save(soup, os.path.join(output_dir, "pytorch_model.bin"))

    # Copy configs
    print("Copying config files...")
    last_ckpt = to_average[-1]
    for file in ["config.json", "preprocessor_config.json"]:
        src = os.path.join(last_ckpt, file)
        if os.path.exists(src):
            shutil.copy(src, output_dir)
    
    print("Model Soup Complete!")

if __name__ == "__main__":
    main()