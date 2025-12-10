import torch
import numpy as np
import argparse
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from nh_datasets.rescuenet import RescueNetSegDataset 
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIGURATION ---
NUM_CLASSES = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/data/RescueNet")
    return parser.parse_args()

class IoUEvaluator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def add_batch(self, preds, labels):
        preds = preds.flatten()
        labels = labels.flatten()
        keep = (labels != self.ignore_index)
        preds = preds[keep]
        labels = labels[keep]
        self.confusion_matrix += np.bincount(
            self.num_classes * labels.astype(int) + preds.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        iou = intersection / (union + 1e-10)
        return np.nanmean(iou), np.nanmean(intersection / (ground_truth_set + 1e-10)), iou

def main():
    args = parse_args()
    print(f"Loading model from: {args.model_path}")
    
    # Load Model
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path).to(DEVICE)
    model.eval()

    # --- THE FIX: RESIZE TO 1024x1024 (Standard View) ---
    image_processor = SegformerImageProcessor(
        do_resize=True, 
        size={"height": 1024, "width": 1024}, # Force standard size
        do_normalize=True,
        reduce_labels=False
    )
    
    print("Initializing Dataset (Resizing to 1024x1024)...")
    val_ds = RescueNetSegDataset(
        root=args.data_root,
        split="val",
        image_processor=image_processor,
        augment=False
    )
    
    evaluator = IoUEvaluator(num_classes=NUM_CLASSES)
    
    print(f"Starting Evaluation on {len(val_ds)} images...")
    
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            sample = val_ds[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(DEVICE)
            labels = sample["labels"].numpy()
            
            # Predict
            outputs = model(pixel_values)
            logits = outputs.logits
            
            # Upsample logits to match label size (1024x1024)
            # Note: The dataset returns resized labels now too
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            evaluator.add_batch(preds, labels)

    miou, macc, class_iou = evaluator.get_metrics()
    
    print("\n" + "="*40)
    print(f"Mean IoU:      {miou:.5f}")
    print(f"Mean Accuracy: {macc:.5f}")
    print("="*40)
    print("Per Class IoU:")
    for idx, iou in enumerate(class_iou):
        print(f"Class {idx:<2}: {iou:.5f}")

if __name__ == "__main__":
    main()