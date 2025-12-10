#!/usr/bin/env bash
set -e
DOCKER_IMAGE="letatanu/semseg_2d:latest"
SCRIPT_NAME="viz_smooth_stitch.py"

# Wipe the folder
sudo rm -rf /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/runs/viz_paper_SMOOTH

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -c "
    /opt/conda/envs/semseg/bin/pip install opencv-python-headless
    /opt/conda/envs/semseg/bin/python /working/${SCRIPT_NAME}
  "