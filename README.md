# RescueNet Semantic Segmentation (Phase 2)

This repository contains a high-performance **SegFormer-B4** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying granular features such as "Building-Total-Destruction," "Road-Blocked," and "Flood Water."

This release represents **Phase 2** of our research, where we achieved a significant performance leap by optimizing the input resolution strategy and loss landscape.

##  Phase 1 vs. Phase 2 Comparison

We significantly outperformed our initial baselines by optimizing the input strategy and loss landscape. Phase 2 introduces **OHEM (Online Hard Example Mining)** and a **1024x1024 Center Crop** strategy to preserve high-frequency details.

### Global Metrics
| Experiment | Strategy | mIoU | Accuracy |
| :--- | :--- | :--- | :--- |
| **Original** | Standard Resize (512x512) | 70.30% | 81.60% |
| **Phase 1** | Resize + Test-Time Augmentation (TTA) | 72.29% | 83.10% |
| **Phase 2** | **1024x1024 Center Crop + OHEM** | **74.67%** | **85.92%** |

*Key Insight: While TTA (Phase 1) provided a ~2% boost, shifting to a high-resolution cropping strategy (Phase 2) unlocked a further ~2.4% gain and significantly improved texture recognition without needing slow inference-time ensembles.*
---

### Detailed Per-Class Performance
Comparing our **Original Baseline** against our final **Phase 2** model reveals massive improvements in the most difficult classes (Major/Minor Damage), proving that the cropping strategy allows the model to see fine-grained destruction details.

| ID | Class Name | Original IoU | Phase 2 IoU | Improvement |
|:---|:---|:---|:---|:---|
| 0 | Background | 86.50% | 84.70% | -1.80% |
| 1 | Water | 89.60% | **89.60%** | 0.00% |
| 2 | Building No Damage | 69.80% | 70.30% | +0.50% |
| 3 | Building Minor Damage | 58.10% | **72.00%** | **+13.90%** |
| 4 | Building Major Damage | 59.60% | **72.10%** | **+12.50%** |
| 5 | Total Destruction | 59.00% | **60.70%** | +1.70% |
| 6 | Vehicle | 69.60% | **76.10%** | +6.50% |
| 7 | Road-Clear | 77.80% | **83.50%** | **+5.70%** |
| 8 | Road-Blocked | 40.90% | 41.40% | +0.50% |
| 9 | Tree | 84.30% | 81.40% | -2.90% |
| 10 | Pool | 76.70% | **88.20%** | **+11.50%** |

**Analysis:**
* **Damage Sensitivity:** The "Squash/Resize" method (Original) destroyed the texture of damaged roofs, capping accuracy at ~59%. The "Crop" method (Phase 2) preserved these textures, driving **Minor Damage** and **Major Damage** scores up by over **13%**.
* **Small Objects:** Classes like **Pool** and **Vehicle** saw massive gains because they weren't being interpolated out of existence.

---

## The Visualization Challenge & Solution
A major engineering challenge in this project was generating accurate visualizations for high-resolution satellite imagery (3000x4000px).

### The Problem
During initial inference, predictions appeared as **"chaotic blobs"** or low-confidence noise.
* **Root Cause:** The model was trained on **1024x1024 zoomed-in crops**. Standard inference scripts attempted to **squash** the massive full-resolution image into a 1024x1024 square. This destroyed the aspect ratio and scale, presenting the model with distorted features it had never seen during training.

### The Solution: "Training-Aligned Inference"
We engineered a custom visualization pipeline (`viz_smooth_stitch.py`) that strictly mirrors the training logic:
1.  **No Squashing:** We perform **Smooth Sliding Window Inference** with overlap to handle the full native resolution.
2.  **Palette Alignment:** We mapped the custom 11-class training indices to the correct visual palette, fixing discrepancies where roads appeared as "Debris."
3.  **Result:** Sharp, pixel-perfect segmentation maps that accurately reflect the model's high mIoU score.

![Prediction Example](demo_figures/10781.jpg)
*(Left: Original, Middle: Ground Truth, Right: Model Prediction, Far Right: Overlay)*

---

## Methodology

This implementation builds upon the Hugging Face Transformers library and utilizes:

* **Architecture:** `nvidia/segformer-b4-finetuned-ade-512-512` (MiT-B4 Encoder).
* **Input Strategy (The Key Differentiator):**
    * *Phase 1:* Resizing images (Destroys small debris details).
    * *Phase 2:* **1024x1024 Center Cropping**. This forces the model to learn high-resolution textures, crucial for distinguishing "Rubble" from "Road."
* **Loss Function:** **Compound Loss** (Dice Loss + OHEM Cross Entropy) to penalize the model heavily for missing rare classes.
* **Optimization:** AdamW optimizer with a **Cosine Annealing** scheduler (Warmup ratio 0.1).

---

## Usage

### 1. Environment Setup
The project runs inside a Docker container for full reproducibility.

```bash
# Pull the docker image
docker pull letatanu/semseg_2d:latest

# Start the container
docker run --rm -ti --gpus all -v $(pwd):/working letatanu/semseg_2d:latest bash
```

### 2. Training
To reproduce the Phase 2 training run (OHEM + Cosine):

```bash
./run_train_rescuenet_phase2.sh
```

### 3. Evaluation
To benchmark the model using the validated Center Crop strategy:

```bash
./run_eval_simple.sh
```

### 4. Visualization
To generate the smooth, stitched visualizations seen above:

```bash
./run_smooth_stitch.sh
```

### Attribution & Research Team
This project was developed as part of research work at Bina Labs at Lehigh University.

Principal Investigator: Dr. Maryam Rahnemoonfar

Primary Author: Nhut Le, PhD Candidate

Research Lead: Kevin Zhu

Modifications by Kevin Zhu (Phase 2):

Implementation of OHEM + Compound Loss for class imbalance.

Development of 1024x1024 Training Strategy to resolve scale artifacts.

Engineering of Robust Visualization Pipeline to solve scale/index mismatches.

### License
This code is released for academic and educational use. Please cite the original RescueNet Paper if you use this in your research.
