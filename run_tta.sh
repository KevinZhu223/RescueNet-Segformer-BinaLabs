#!/usr/bin/env bash
set -e

DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -c "
    # DIRECT EXECUTION: Use the python binary from the semseg environment
    # This avoids the 'conda not found' and 'module not found' errors.
    
    /opt/conda/envs/semseg/bin/python /working/eval_tta.py \
    --model_path /working/runs/rescuenet_segformer_optimzed/checkpoint-soup \
    --data_root /data/RescueNet
  "