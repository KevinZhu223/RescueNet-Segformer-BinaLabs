import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from nh_datasets.loader import build_dataset_from_py
from tqdm import tqdm
import numpy as np

# --- CONFIGURATION ---
# Based on your screenshots, these paths are correct for inside the Docker container
CHECKPOINT_PATH = "/working/runs/rescuenet_final_b4_ohem_cosine_V2/checkpoint-36818"
CONFIG_FILE = "/working/nh_datasets/configs/segformer_rescuenet.py"
NUM_CLASSES = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tta_inference(model, batch):
    """Predicts: Original + Horizontal Flip"""
    pixel_values = batch["pixel_values"].to(DEVICE)
    
    # 1. Forward Pass (Original)
    with torch.no_grad():
        out1 = model(pixel_values).logits
        # Interpolate to 1024x1024
        out1 = F.interpolate(out1, size=(1024, 1024), mode="bilinear", align_corners=False)
        probs = F.softmax(out1, dim=1)

    # 2. Forward Pass (Horizontal Flip)
    # Flip image
    pixel_values_h = torch.flip(pixel_values, [3]) 
    with torch.no_grad():
        out2 = model(pixel_values_h).logits
        out2 = F.interpolate(out2, size=(1024, 1024), mode="bilinear", align_corners=False)
        # Flip output back
        out2 = torch.flip(out2, [3]) 
        probs += F.softmax(out2, dim=1)
        
    return probs / 2.0  # Average

class IoUEvaluator:
    """Helper to calculate mIoU on the fly"""
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def add_batch(self, preds, labels):
        preds = preds.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        
        # Filter out ignore_index (255)
        keep = (labels != self.ignore_index)
        preds = preds[keep]
        labels = labels[keep]

        # Update matrix using fast histogram
        self.confusion_matrix += np.bincount(
            self.num_classes * labels.astype(int) + preds.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        # Calculate IoU per class
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Avoid division by zero
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        macc = np.nanmean(intersection / (ground_truth_set + 1e-10))
        
        return miou, macc, iou

def main():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    model.eval()
    
    print("Loading Validation Dataset...")
    # Use standard processor
    image_processor = SegformerImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)
    # Load dataset
    val_ds = build_dataset_from_py(CONFIG_FILE, split="val", augment=False, image_processor=image_processor)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    evaluator = IoUEvaluator(num_classes=NUM_CLASSES)
    
    print("Starting TTA Evaluation Loop...")
    for batch in tqdm(loader):
        # Move labels to CPU for metric calc later
        labels = batch["labels"]
        
        # Get TTA Predictions
        probs = tta_inference(model, batch)
        preds = torch.argmax(probs, dim=1)

        # Update metrics
        evaluator.add_batch(preds, labels)

    # Final Calculation
    miou, macc, class_iou = evaluator.get_metrics()
    
    print("\n" + "="*40)
    print("🚀 TTA EVALUATION RESULTS")
    print("="*40)
    print(f"Overall mIoU:   {miou:.5f}")
    print(f"Pixel Accuracy: {macc:.5f}")
    print("-" * 40)
    print("Per-Class IoU:")
    for i, score in enumerate(class_iou):
        print(f"Class {i:>2}: {score:.5f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()