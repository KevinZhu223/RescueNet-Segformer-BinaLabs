#!/usr/bin/env bash
set -e

## --------------------------------------------------------- ##
DATASET="rescuenet"
DOCKER_IMAGE="letatanu/semseg_2d:latest"

# MOUNTING EXPLANATION:
# 1. Mount your code folder to /working
# 2. Mount your main volume to /data (so we can reach RescueNet)

docker run --rm -it \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        export OMP_NUM_THREADS=16
        
        # 1. Activate Environment
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            . /opt/conda/etc/profile.d/conda.sh
        elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
            . \$HOME/miniconda3/etc/profile.d/conda.sh
        else
            echo 'conda.sh not found in image' >&2; exit 1
        fi
        conda activate semseg

        # 2. Run Visualization
        # We point directly to the test folders we verified earlier.
        
        # RUN VISUALIZATION
        python /working/viz_segformer.py \
        --model /working/runs/rescuenet_segformer_optimzed/checkpoint-127350 \
        --folder /data/RescueNet/test-org-img \
        --gt_folder /data/RescueNet/test-label-img \
        --gt_suffix _lab.png \
        --outdir /working/runs/viz_rescuenet_specialist \
        --no_show
        "
