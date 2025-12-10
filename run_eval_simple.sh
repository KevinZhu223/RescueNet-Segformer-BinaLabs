#!/usr/bin/env bash
set -e

# --- CONFIGURATION ---
DOCKER_IMAGE="letatanu/semseg_2d:latest"
SCRIPT_NAME="eval_simple.py"
# Pointing to your BEST model (Epoch 255)
MODEL_PATH="/working/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0"
DATA_ROOT="/data/RescueNet"

echo "🚀 Starting Standard (Resized) Evaluation for Epoch 255..."

# --- DOCKER COMMAND ---
docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -c "
    echo '✅ Inside Docker. GPU Mode Active.'
    export CUDA_VISIBLE_DEVICES='0'
    
    # --- AUTO-FIX: Install missing libraries ---
    echo '🔧 Installing albumentations...'
    /opt/conda/envs/semseg/bin/pip install albumentations opencv-python-headless
    
    # Run the evaluation
    echo '▶️ Running Simple Evaluation...'
    /opt/conda/envs/semseg/bin/python /working/${SCRIPT_NAME} \
      --model_path ${MODEL_PATH} \
      --data_root ${DATA_ROOT}
  "

echo "🎉 Evaluation Complete!"