# RescueNet Semantic Segmentation (Phase 2)

This repository contains a high-performance **SegFormer-B4** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying granular features such as "Building-Total-Destruction," "Road-Blocked," and "Flood Water."

This release represents **Phase 2** of our research, where we achieved a significant performance leap by optimizing the input resolution strategy and loss landscape.

##  Phase 1 vs. Phase 2 Comparison

We significantly outperformed our initial baselines by optimizing the input strategy and loss landscape. Phase 2 introduces **OHEM (Online Hard Example Mining)** and a **1024x1024 Center Crop** strategy to preserve high-frequency details.

### Global Metrics
| Metric | Phase 1 | Phase 2 | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean IoU (mIoU)** | 72.29% | **74.67%** | **+2.38%** |
| **Pixel Accuracy (mAcc)** | 83.10% | **85.92%** | **+2.82%** |

*Note: Phase 1 utilized Test-Time Augmentation (TTA). Phase 2 achieves higher scores **without** TTA, relying purely on better feature learning.*

---

### Detailed Per-Class Performance
The Phase 2 strategy specifically targeted difficult damage classes. Note the massive gains in distinguishing damage levels (Major vs. Minor vs. Total Destruction), which was a key weakness in Phase 1.

| ID | Class Name | Phase 1 IoU | Phase 2 IoU | Delta |
|:---|:---|:---|:---|:---|
| 0 | Background | 86.87% | 84.70% | -2.17% |
| 1 | Water | 89.17% | **89.60%** | +0.43% |
| 2 | Building No Damage | 74.70% | 70.30% | -4.40% |
| 3 | Building Minor Damage | 63.01% | **72.00%** | **+8.99%** |
| 4 | Building Major Damage | 62.51% | **72.10%** | **+9.59%** |
| 5 | Total Destruction | 59.01% | **60.70%** | +1.69% |
| 6 | Vehicle | 71.75% | **76.10%** | +4.35% |
| 7 | Road-Clear | 77.78% | **83.50%** | **+5.72%** |
| 8 | Road-Blocked | 43.71% | 41.40% | -2.31% |
| 9 | Tree | 84.60% | 81.40% | -3.20% |
| 10 | Pool | 80.44% | **88.20%** | **+7.76%** |

**Analysis:**
* **Damage Assessment:** The most critical improvements were in **Major Damage (+9.6%)** and **Minor Damage (+9.0%)**. This confirms that the high-resolution crop strategy allowed the model to see the subtle texture differences (e.g., missing shingles vs. collapsed roofs) that the resized Phase 1 model missed.
* **Environmental Features:** Classes like "Road-Clear" and "Pool" saw significant boosts due to sharper object boundaries.

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
