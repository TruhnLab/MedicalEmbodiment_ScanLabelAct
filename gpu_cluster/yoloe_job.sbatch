#!/bin/bash
# YOLOE masks for a vocabulary in every keyframe of one LanderPi scene (yoloe_frames.py) on a GPU node.
# Submitted by projects/scene_understanding/detect_frames.py:
#   sbatch --job-name=yoloe_<session> --output=<scene>/seg/slurm_yoloe.log yoloe_job.sbatch <scene> --models <dir>
# yoloe-11l: 1361 keyframes in ~30 s on an L40S (2026-10-09).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
SCENE="$1"
shift
source ~/miniconda3/etc/profile.d/conda.sh
conda activate landerpi-seg
echo "node $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader), scene $SCENE"
python -u ~/code/LanderPi/gpu_cluster/yoloe_frames.py --scene "$SCENE" "$@"
