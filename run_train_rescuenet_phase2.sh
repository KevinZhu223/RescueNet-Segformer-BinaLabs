#!/usr/bin/env bash
set -e

# --- CONFIGURATION ---
DEVICES="0,1,2,3"
DOCKER_IMAGE="letatanu/semseg_2d:latest"
# Calculates number of GPUs automatically based on the list above
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))

# Output directory for the new Phase 2 run
OUTPUT_DIR="/working/runs/rescuenet_b4_CROP_AUG_phase2"

# --- DOCKER COMMAND ---
docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        
        # 1. Activate Conda Environment
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            . /opt/conda/etc/profile.d/conda.sh
        else
            echo 'conda source not found'
        fi
        conda activate semseg
        
        # 2. Install Phase 2 Dependencies (Albumentations)
        echo 'Installing dependencies...'
        pip install albumentations
        
        # 3. Start Training
        echo 'Starting Phase 2 Training: Random Crop + Albumentations...'
        echo 'Output Directory: ${OUTPUT_DIR}'
        
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        /working/train_segformer.py \
        --config_file /working/nh_datasets/configs/segformer_rescuenet.py \
        --output_dir ${OUTPUT_DIR}
        "