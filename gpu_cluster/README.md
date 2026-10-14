# GPU jobs on the Slurm cluster

The cluster is optional. One job runs there: the scene labelling's open-vocabulary detector (YOLOE masks
in every keyframe, `yoloe_frames.py`). Without a cluster, the labelling runs the detector on the laptop's
CPU with a smaller model (`projects/scene_understanding/README.md`).

Everything heavy runs inside a Slurm job; the laptop only uploads, submits and fetches.

| What | Where |
|---|---|
| Code (rsync of this repo, no data) | cluster master (`CLUSTER_MASTER`), `~/code/LanderPi` |
| Scenes (input + results) | `<scenes>/<session>/`: `CLUSTER_SCENES_ON_MASTER` on the cluster = `CLUSTER_SCENES_ON_STORAGE` on the storage server (`CLUSTER_STORAGE`) |
| YOLOE weights | `CLUSTER_MODELS_ON_MASTER` |
| Conda env | `landerpi-seg` in `~/miniconda3` (`setup_seg_env.sh`) |

## Files

- `yoloe_job.sbatch`: the Slurm job; submitted by `projects/scene_understanding/detect_frames.py`
  (the console's **Label** button), which also uploads the keyframes and fetches the masks.
- `yoloe_frames.py`: the detector over all keyframes of one scene.
- `setup_seg_env.sh`: creates the conda env (once).
- `tools/gpu_cluster.py`: hosts, paths, ssh/rsync and `sync-code`, shared by the jobs' callers.

## Setup (once)

1. Hosts and paths are settings in the repo's `.env` (names in `.env.example`); `CLUSTER_PARTITION` is
   passed to `sbatch` (empty: the cluster's default).
2. WSL on the laptop needs `rsync` and ssh keys for both hosts (`ssh <CLUSTER_MASTER>` and
   `ssh <CLUSTER_STORAGE>` work without a password).
3. Copy the code and create the env:
   ```powershell
   python tools/gpu_cluster.py sync-code
   ```
   ```bash
   bash ~/code/LanderPi/gpu_cluster/setup_seg_env.sh    # on the master
   ```
4. Copy the YOLOE weights and the text encoder from the laptop's `data/models/` to
   `CLUSTER_MODELS_ON_MASTER` (the comment in `setup_seg_env.sh` lists them).

## Earlier: Gaussian splatting and pose refinement (removed 2026-10-14)

Until 2026-10-06 the cluster also ran Gaussian splatting of recorded sessions and a pose refinement
(feature matching + bundle adjustment). Neither beat the laptop's TSDF reconstruction with our recordings:
low camera (26 cm), mostly forward motion, 640 x 400 images, people moving through the room. Both were
removed from the repository.
