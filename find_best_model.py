import json
import os

# --- CONFIGURATION ---
# Update this path to your actual run folder
run_folder = '/media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/runs/rescuenet_final_b4_ohem_cosine_V2'
# ---------------------

# 1. Try to find the log file in the main folder
log_path = os.path.join(run_folder, 'trainer_state.json')

# 2. If not there, check inside the LAST checkpoint
if not os.path.exists(log_path):
    print(f"Not found in root, checking inside checkpoint-134700...")
    log_path = os.path.join(run_folder, 'checkpoint-134700', 'trainer_state.json')

if not os.path.exists(log_path):
    print("ERROR: Could not find 'trainer_state.json'.")
    exit()

print(f"Reading logs from: {log_path}")

with open(log_path, 'r') as f:
    data = json.load(f)

# Extract history
history = data.get('log_history', [])

# Filter for steps that have validation metrics
eval_steps = [x for x in history if 'eval_mIoU' in x]

if not eval_steps:
    print('No evaluation metrics found in logs.')
else:
    # Find the max mIoU
    best_step = max(eval_steps, key=lambda x: x['eval_mIoU'])
    
    print('\n' + '='*40)
    print('       🏆 BEST TRAINING RUN 🏆')
    print('='*40)
    print(f"Epoch:          {best_step['epoch']}")
    print(f"Step:           {best_step['step']}")
    print(f"Overall mIoU:   {best_step['eval_mIoU']:.5f}")
    print(f"Overall mAcc:   {best_step['eval_mAcc']:.5f}")
    print('-'*40)
    print('       📊 PER-CLASS SCORES')
    print('-'*40)
    
    # Extract and sort the class keys (eval_IoU_0, eval_IoU_1, etc.)
    iou_keys = [k for k in best_step.keys() if k.startswith('eval_IoU_')]
    
    # Sort them numerically (so Class 2 comes before Class 10)
    iou_keys.sort(key=lambda x: int(x.split('_')[-1]))
    
    for k in iou_keys:
        class_id = k.split('_')[-1]
        score = best_step[k]
        # Adding a visual bar for fun/clarity
        bar_len = int(score * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"Class {class_id:>2}: {bar} {score:.5f}")

    print('='*40 + '\n')