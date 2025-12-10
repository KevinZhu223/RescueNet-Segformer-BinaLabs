import json
import os
import glob

# --- CONFIGURATION ---
BASE_DIR = '/media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/runs/rescuenet_final_b4_ohem_cosine_V2'
# ---------------------

def get_full_metrics(folder_path):
    json_path = os.path.join(folder_path, 'trainer_state.json')
    
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        history = data.get('log_history', [])
        eval_logs = [x for x in history if 'eval_mIoU' in x]
        
        if not eval_logs:
            return None
            
        latest = eval_logs[-1]
        
        # Build the dictionary
        metrics = {
            'Folder': os.path.basename(folder_path),
            'Epoch': latest.get('epoch', 0),
            'mIoU': latest.get('eval_mIoU', 0) * 100,
            'mAcc': latest.get('eval_mAcc', 0) * 100,
        }
        
        # Get all 11 classes
        for i in range(11):
            key = f'eval_IoU_{i}'
            metrics[f'C{i}'] = latest.get(key, 0) * 100
            
        return metrics
    except:
        return None

# 1. Find all folders
search_paths = [
    os.path.join(BASE_DIR, "checkpoint-*"),
    os.path.join(BASE_DIR, "BEST_MODELS_ARCHIVE", "checkpoint-*"),
    os.path.join(BASE_DIR, "BEST_ACCURACY_ARCHIVE", "checkpoint-*")
]

all_models = []
seen_folders = set()

print(f" Scanning all checkpoints in {BASE_DIR}...")

for pattern in search_paths:
    for folder in glob.glob(pattern):
        if os.path.isdir(folder):
            fname = os.path.basename(folder)
            if fname in seen_folders:
                continue
            
            metrics = get_full_metrics(folder)
            if metrics:
                all_models.append(metrics)
                seen_folders.add(fname)

# 2. Sort by Epoch
all_models.sort(key=lambda x: x['Epoch'])

# 3. Print the Massive Table
# Column Headers
headers = [
    "EPOCH", "mIoU", "mAcc", 
    "C0(Bg)", "C1(H2O)", "C2(No)", "C3(Min)", "C4(Maj)", 
    "C5(Rd)", "C6(Veh)", "C7(Tre)", "C8(Deb)", "C9(Pol)", "C10"
]

# Define column widths
widths = [6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 7, 6, 5]
header_row = "".join([f"{h:<{w}} | " for h, w in zip(headers, widths)])

print("\n" + "=" * len(header_row))
print(header_row)
print("-" * len(header_row))

best_miou = 0

for m in all_models:
    # Check for new record
    is_best = False
    if m['mIoU'] > best_miou:
        best_miou = m['mIoU']
        is_best = True
    
    # Format the row values
    row_values = [
        m['Epoch'], m['mIoU'], m['mAcc'],
        m['C0'], m['C1'], m['C2'], m['C3'], m['C4'],
        m['C5'], m['C6'], m['C7'], m['C8'], m['C9'], m['C10']
    ]
    
    # Build the string
    row_str = ""
    for val, w in zip(row_values, widths):
        row_str += f"{val:<{w}.1f} | "
    
    # Add marker for best run
    if is_best:
        row_str += " ⭐ NEW BEST"
    
    print(row_str)

print("=" * len(header_row))
print("LEGEND: C3=Minor Damage | C4=Major Damage | C8=Debris\n")