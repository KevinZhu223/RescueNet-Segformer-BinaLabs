import torch
import numpy as np
import os
import cv2
import math
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "/working/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0"
IMG_FOLDER = "/data/RescueNet/val-org-img"
GT_FOLDER = "/data/RescueNet/val-label-img"
OUT_DIR = "/working/runs/viz_paper_SMOOTH"

# --- CUSTOM PALETTE (BGR for OpenCV) ---
PALETTE_BGR = {
    0:  (0, 0, 0),       # Background
    1:  (255, 0, 0),     # Water (Blue)
    2:  (20, 255, 20),   # No Damage (Green)
    3:  (0, 215, 255),   # Minor Damage (Gold)
    4:  (0, 0, 255),     # Major Damage (Red)
    5:  (0, 0, 139),     # Total Destruction (Dark Red)
    6:  (128, 0, 128),   # Vehicle (Purple)
    7:  (128, 128, 128), # Road-Clear (Grey)
    8:  (64, 64, 64),    # Road-Blocked (Dark Grey)
    9:  (0, 100, 0),     # Tree (Dark Green)
    10: (255, 128, 0)    # Pool (Light Blue)
}

def colorize_mask(mask):
    """Paints the index mask into RGB"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for index, color in PALETTE_BGR.items():
        color_mask[mask == index] = color
    return color_mask

def predict_smooth_window(model, image_np, device, crop_size=1024, stride=768):
    """
    Predicts using overlapping windows to remove grid lines.
    stride < crop_size ensures overlap.
    """
    h, w, _ = image_np.shape
    num_classes = 11
    
    # Accumulators for logits and counts (to average overlap)
    full_probs = torch.zeros((num_classes, h, w), device=device)
    count_map = torch.zeros((1, h, w), device=device)
    
    processor = SegformerImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)

    n_rows = math.ceil((h - crop_size) / stride) + 1
    n_cols = math.ceil((w - crop_size) / stride) + 1
    
    # If image is smaller than crop, just pad and predict once
    if h <= crop_size or w <= crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img_padded = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        inputs = processor(images=img_padded, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs.pixel_values.to(device))
            # Resize logits to 1024x1024
            logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1) # (1, C, H, W)
            
        probs = probs[0, :, :h, :w] # Crop back
        return torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    # Sliding Window Loop
    for r in range(n_rows):
        for c in range(n_cols):
            y1 = int(r * stride)
            x1 = int(c * stride)
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)
            
            # Adjust start points if we hit the boundary to keep size fixed
            if y2 - y1 < crop_size: y1 = h - crop_size
            if x2 - x1 < crop_size: x1 = w - crop_size
            y2, x2 = y1 + crop_size, x1 + crop_size

            tile = image_np[y1:y2, x1:x2]
            
            inputs = processor(images=tile, return_tensors="pt")
            with torch.no_grad():
                outputs = model(inputs.pixel_values.to(device))
                logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                probs = F.softmax(logits, dim=1) # (1, C, H, W)
            
            # Accumulate
            full_probs[:, y1:y2, x1:x2] += probs[0]
            count_map[:, y1:y2, x1:x2] += 1.0

    # Average and Argmax
    full_probs /= count_map
    final_pred = torch.argmax(full_probs, dim=0).cpu().numpy().astype(np.uint8)
    return final_pred

def find_ground_truth(fname):
    """Robustly finds the GT file."""
    base_name = os.path.splitext(fname)[0]
    # Try typical name variations
    candidates = [
        f"{base_name}_lab.png",
        f"{base_name}.png",
        fname.replace(".jpg", "_lab.png").replace(".png", "_lab.png")
    ]
    
    for c in candidates:
        path = os.path.join(GT_FOLDER, c)
        if os.path.exists(path):
            return path
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    img_files = sorted([f for f in os.listdir(IMG_FOLDER) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Process ALL images
    print(f"Processing ALL {len(img_files)} images (Smooth Stitching)...")

    for fname in tqdm(img_files):
        # 1. LOAD IMAGE
        img_path = os.path.join(IMG_FOLDER, fname)
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # 2. PREDICT (SMOOTH SLIDING WINDOW)
        # stride=768 means 25% overlap to smooth out edges
        preds = predict_smooth_window(model, image_np, device, crop_size=1024, stride=768)

        # 3. LOAD GT (WITH DEBUGGING)
        gt_path = find_ground_truth(fname)
        
        if gt_path:
            gt_mask = np.array(Image.open(gt_path))
            # DEBUG: Print what values are actually in the GT
            unique_vals = np.unique(gt_mask)
            if len(unique_vals) == 1 and unique_vals[0] == 0:
                print(f"GT found for {fname}, but it is purely index 0 (Background)!")
            
            gt_color = colorize_mask(gt_mask)
        else:
            print(f"No GT file found for {fname}")
            gt_color = np.zeros(image_np.shape, dtype=np.uint8)

        # 4. VISUALIZE
        viz_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # BGR for OpenCV
        pred_color = colorize_mask(preds)
        
        # Overlay
        overlay = cv2.addWeighted(viz_img, 0.6, pred_color, 0.4, 0)

        # Combine
        combined = np.hstack([viz_img, gt_color, pred_color, overlay])
        cv2.imwrite(os.path.join(OUT_DIR, fname), combined)

    print(f"Smooth Visuals Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()