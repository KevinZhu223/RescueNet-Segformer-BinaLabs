# RescueNet Semantic Segmentation with SegFormer

This repository contains a fine-tuned **SegFormer-b4** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying features such as "Building-No-Damage," "Building-Major-Damage," and "Road-Blocked."

##  Key Results
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **mIoU (TTA)** | **72.29%** | Best Model (Epoch 82) with Test-Time Augmentation |
| **mIoU (Base)** | **72.14%** | Standard Validation Score |
| **Pixel Accuracy** | **83.10%** | Validation Set |
| **Background IoU** | **89.2%** | Highly accurate on environmental features |

*Training was optimized using OHEM (Online Hard Example Mining) and Cosine Decay scheduling to handle the severe class imbalance in disaster imagery.*

### Detailed Per-Class Performance
The model achieves strong performance on environmental features but highlights challenges with small debris objects (Class 8), which is a focus for future optimization.

| Class ID | Class Name (Inferred) | IoU Score | 
| :--- | :--- | :--- | :--- |
| **0** | Background | **86.87%** |
| **1** | Water | **89.17%** |
| **2** | Building No Damage | **74.70%** |
| **3** | Building Minor Damage | **63.01%** |
| **4** | Building Major Damage | **62.51%** |
| **5** | Building Total Destruction | **59.01%** |
| **6** | Vehicle | **71.75%** |
| **7** | Road-Clear | **77.78%** |
| **8** | Road-Blocked / Debris | **43.71%** |
| **9** | Tree | **84.60%** |
| **10** | Pool | **80.44%** |

##  Methodology
This implementation builds upon the Hugging Face Transformers library and utilizes:
* **Architecture:** SegFormer-b4 (mit-b4 encoder).
* **Loss Function:** Compound Loss (Dice Loss + OHEM Cross Entropy) to focus on hard-to-classify damage levels.
* **Optimization:** AdamW optimizer with Cosine Annealing scheduler.
* **Augmentation:** Geometric (Flips/Rotations) and Photometric distortions (Brightness/Contrast) via Albumentations.

##  Usage

### 1. Environment Setup
The project is designed to run inside a Docker container to ensure reproducibility.

```bash
# Pull the docker image
docker pull letatanu/semseg_2d:latest

# Start the container
docker run --rm -ti --gpus all -v $(pwd):/working letatanu/semseg_2d:latest bash
```

### 2. Training
To reproduce the training run:
```bash
python train_segformer.py \
    --config_file nh_datasets/configs/segformer_rescuenet.py \
    --output_dir ./runs/my_rescuenet_model
```

### 3. Evaluation with TTA
To run inference with Test-Time Augmentation (Horizontal Flip averaging):

python eval_tta.py

### Attribution & Research Team
This project was developed as part of research work at Bina Labs at Lehigh University.

Principal Investigator: Dr. Maryam Rahnemoonfar

Primary Author: Nhut Le (letatanu), PhD Candidate

Research Lead: Kevin Zhu

Modifications by Kevin Zhu include:

Implementation of advanced augmentation pipelines using Albumentations.

Integration of Test-Time Augmentation (TTA) for validation.

Customization of training schedules (Cosine Decay) and loss functions.

### License

This code is released for academic and educational use. Please cite the original RescueNet Paper if you use this in your research.
