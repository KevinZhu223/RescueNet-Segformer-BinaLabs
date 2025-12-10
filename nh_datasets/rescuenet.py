import os
import random
import numpy as np
import torch
import albumentations as A
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .registry import register_dataset

@register_dataset("rescuenet_segformer")
class RescueNetSegDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    
    CLASSES = [
        "Background", "Water", "Building_No_Damage", "Building_Minor_Damage",
        "Building_Major_Damage", "Building_Total_Destruction", "Vehicle",
        "Road-Clear", "Road-Blocked", "Tree", "Pool",
    ]
    label2id = {name: i for i, name in enumerate(CLASSES)}
    id2label = {i: name for i, name in enumerate(CLASSES)}

    def __init__(
        self, root: str, split: str, image_processor,
        image_size: int = 1024, augment: bool = False, ignore_index: int = 255,
        num_classes: int = 11
    ):
        self.root = Path(root)
        self.split = split
        self.image_processor = image_processor
        self.crop_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index
        
        self.img_dir = self.root / f"{split}-org-img"
        self.lbl_dir = self.root / f"{split}-label-img"
        
        if not self.img_dir.is_dir() or not self.lbl_dir.is_dir():
            raise FileNotFoundError(f"Missing directories: {self.img_dir} or {self.lbl_dir}")

        self.samples = []
        for fname in os.listdir(self.img_dir):
            stem, ext = os.path.splitext(fname)
            if ext not in self.IMG_EXTS: continue
            lbl_p = self.lbl_dir / f"{stem}_lab.png"
            if lbl_p.exists():
                self.samples.append((self.img_dir / fname, lbl_p))

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.9),
                ], p=0.8),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, fill_value=0, mask_fill_value=None, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=self.crop_size, min_width=self.crop_size),
            ])

    def __len__(self): return len(self.samples)

    def get_class_aware_crop(self, img, mask):
        # Explicit target classes
        rare_classes = [3, 4, 5, 6, 8]
        h, w = mask.shape
        crop_h, crop_w = self.crop_size, self.crop_size
        
        if h <= crop_h or w <= crop_w:
            return 0, 0, h, w

        # 50% chance to force a crop on a damage/rare class
        if random.random() < 0.5:
            rare_pixels = np.isin(mask, rare_classes)
            y_indices, x_indices = np.where(rare_pixels)
            
            if len(y_indices) > 0:
                idx = random.randint(0, len(y_indices) - 1)
                center_y, center_x = y_indices[idx], x_indices[idx]
                
                # Center the crop on the pixel, clamping to boundaries
                # This fixes the bias towards top-left logic
                top = max(0, min(center_y - crop_h // 2, h - crop_h))
                left = max(0, min(center_x - crop_w // 2, w - crop_w))
                return top, left, top + crop_h, left + crop_w

        # Fallback: Random crop
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        return top, left, top + crop_h, left + crop_w

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(lbl_path))

        # --- FIX: Ensure Mask is uint8 (Safety) ---
        mask = mask.astype(np.uint8) 

        if self.augment:
            top, left, bottom, right = self.get_class_aware_crop(image, mask)
            image = image[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
            
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Validation logic
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            # Logic: If crop_size is 1024 (Training validation), center crop.
            # If crop_size is 5000 (TTA validation), this block is skipped (Full Image).
            h, w, _ = image.shape
            if h > self.crop_size and w > self.crop_size:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
                image = image[top:top+self.crop_size, left:left+self.crop_size]
                mask = mask[top:top+self.crop_size, left:left+self.crop_size]

        mask = mask.astype(np.int64) # Convert back to Long for PyTorch
        
        # --- FIX: Explicit return_masks=False (Safety) ---
        encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": torch.from_numpy(mask), "id": img_path.stem}