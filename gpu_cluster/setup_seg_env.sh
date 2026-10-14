#!/bin/bash
# One-time setup of the conda environment for the labelling jobs (yoloe_job.sbatch), run on the cluster's
# master node. Until 2026-10-09 it also held SAM 2, which pinned numpy<2.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q '^landerpi-seg ' || conda create -y -q -n landerpi-seg python=3.11
conda activate landerpi-seg
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# same ultralytics as the laptop's landerpi env; YOLOE's text encoder wants ultralytics' CLIP package (without
# it the first job installed it by itself, 2026-10-09)
pip install -q ultralytics==8.4.174 "git+https://github.com/ultralytics/CLIP.git"
# the weights and the text encoder (mobileclip_blt.ts) are copied once from the laptop's data/models to
# the storage's models/yoloe folder (CLUSTER_MODELS_ON_MASTER in .env, yoloe_frames.py --models)
python -c "import torch, numpy, cv2, ultralytics; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__, 'ultralytics', ultralytics.__version__)"
