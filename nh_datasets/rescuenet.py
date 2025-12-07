import os, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
# Use PIL for augmentation to avoid torchvision crashes
from transformers import SegformerImageProcessor
from .registry import register_dataset

@register_dataset("rescuenet_segformer")
class RescueNetSegDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    
    # BACK TO 11 CLASSES (Original Context)
    CLASSES = [
        "Background", "Water", "Building_No_Damage", "Building_Minor_Damage",
        "Building_Major_Damage", "Building_Total_Destruction", "Vehicle",
        "Road-Clear", "Road-Blocked", "Tree", "Pool",
    ]
    label2id = {name: i for i, name in enumerate(CLASSES)}
    id2label = {i: name for i, name in enumerate(CLASSES)}

    def __init__(
        self, root: str, split: str, image_processor: SegformerImageProcessor,
        image_size: int = 512, augment: bool = False, ignore_index: int = 255,
        num_classes: int = 11
    ):
        self.root = Path(root)
        self.split = split
        self.image_processor = image_processor
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.img_dir = self.root / f"{split}-org-img"
        self.lbl_dir = self.root / f"{split}-label-img"

        if not self.img_dir.is_dir() or not self.lbl_dir.is_dir():
            raise FileNotFoundError(f"Missing directories: {self.img_dir} or {self.lbl_dir}")

        self.samples = []
        for fname in os.listdir(self.img_dir):
            stem, ext = os.path.splitext(fname)
            if ext not in self.IMG_EXTS: continue
            img_p = self.img_dir / fname
            for ext2 in self.IMG_EXTS:
                lbl_p = self.lbl_dir / f"{stem}_lab.png"
                if lbl_p.exists():
                    self.samples.append((img_p, lbl_p))
                    break
        
        if len(self.samples) == 0: raise RuntimeError(f"No pairs found")
        self.rng = random.Random(1337)

    def __len__(self): return len(self.samples)

    def _load_pair(self, img_path, lbl_path):
        return Image.open(img_path).convert("RGB"), Image.open(lbl_path)

    # SAFE AUGMENTATION (PIL Only - No Crash)
    def _augment(self, img, lab):
        if not self.augment: return img, lab
        # Horizontal Flip
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        # Vertical Flip
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            lab = lab.transpose(Image.FLIP_TOP_BOTTOM)
        # 90-Deg Rotations
        rot_k = self.rng.randint(0, 3)
        if rot_k == 1:
            img = img.transpose(Image.ROTATE_90); lab = lab.transpose(Image.ROTATE_90)
        elif rot_k == 2:
            img = img.transpose(Image.ROTATE_180); lab = lab.transpose(Image.ROTATE_180)
        elif rot_k == 3:
            img = img.transpose(Image.ROTATE_270); lab = lab.transpose(Image.ROTATE_270)
        return img, lab

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img, lab = self._load_pair(img_path, lbl_path)
        img, lab = self._augment(img, lab) # Apply Augmentation

        # Resize to High Res
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        lab = lab.resize((self.image_size, self.image_size), Image.NEAREST)

        lab_np = np.array(lab, dtype=np.int64)
        # NO CLASS MERGING - Keep all 11 classes
        
        encoded = self.image_processor(images=img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.from_numpy(lab_np), "id": img_path.stem}