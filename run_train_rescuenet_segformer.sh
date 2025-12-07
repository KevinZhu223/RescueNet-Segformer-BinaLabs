#!/usr/bin/env bash
set -e
DEVICES="0,1,2,3"
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            . /opt/conda/etc/profile.d/conda.sh
        else
            echo 'conda source not found'
        fi
        conda activate semseg
        
        # --- MODIFIED LINE BELOW ---
        # I added the resume flag. IMPORTANT: Replace 'rescuenet_final_b4_oh...' 
        # with the ACTUAL full folder name from your VSCode explorer.
        
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        /working/train_segformer.py \
        --config_file /working/nh_datasets/configs/segformer_rescuenet.py \
        --resume /working/runs/rescuenet_final_b4_ohem_cosine_V2/checkpoint-11225
        "