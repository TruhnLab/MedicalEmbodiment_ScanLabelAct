"""Access to the Slurm GPU cluster: hosts, paths, ssh/rsync, code sync. Used by the scene labelling's YOLOE
job (projects/scene_understanding/detect_frames.py, gpu_cluster/yoloe_job.sbatch).

    python tools/gpu_cluster.py sync-code            # this repo's code -> ~/code/LanderPi on the cluster

All remote access goes through WSL's ssh/rsync (keys are set up there). Data goes directly to the
storage server, jobs are submitted on the cluster's master; both see the same storage.

Until 2026-10-14 this also ran Gaussian splatting and a pose refinement (bundle adjustment) of recorded
sessions on the cluster. Neither beat the laptop's reconstruction with our recordings (low camera,
forward motion, 640x400, people moving), and neither was used after 2026-10-06, so both were removed.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from landerpi.settings import setting  # noqa: E402

# Hosts and storage paths of the cluster are site-specific: settings in the repo's .env (.env.example)
MASTER = setting("CLUSTER_MASTER", "user@slurm-master")      # jobs are submitted here
OCEAN = setting("CLUSTER_STORAGE", "user@storage-server")    # data goes directly to the storage server
SCENES_ON_MASTER = setting("CLUSTER_SCENES_ON_MASTER", "/path/on/master/LanderPiScenes")
SCENES_ON_OCEAN = setting("CLUSTER_SCENES_ON_STORAGE", "/path/on/storage/LanderPiScenes")  # the same folder
MODELS_ON_MASTER = setting("CLUSTER_MODELS_ON_MASTER", "/path/on/master/models/yoloe")  # yoloe_frames.py --models
# the jobs' partition comes from the command line, not the .sbatch files (empty: the cluster's default)
PARTITION = setting("CLUSTER_PARTITION", "")
SBATCH = "sbatch" + (f" --partition={PARTITION}" if PARTITION else "")
REMOTE_CODE = "~/code/LanderPi"


def wsl_path(p):
    p = Path(p).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def sh(cmd, timeout=600):
    """Runs a bash command in WSL; returns stdout, raises with stderr on failure."""
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-500:])
    return r.stdout


SSH_OPTS = "-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"


def ssh(host, cmd, timeout=120):
    quoted = cmd.replace("'", "'\"'\"'")
    return sh(f"ssh {SSH_OPTS} {host} '{quoted}'", timeout)


def rsync(args, timeout=1800, attempts=4):
    """rsync over ssh; the laptop's link to the cluster drops now and then ("Broken pipe"), so
    partial files are kept and the transfer is resumed a few times before giving up."""
    for attempt in range(1, attempts + 1):
        try:
            return sh(f"rsync -a --partial -e 'ssh {SSH_OPTS}' {args}", timeout)
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def sync_code():
    rsync(f"--delete --exclude .git --exclude .env --exclude data --exclude logs --exclude __pycache__ "
          f"--exclude '*.egg-info' {wsl_path(REPO)}/ {MASTER}:{REMOTE_CODE}/", timeout=300)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["sync-code"])
    p.parse_args()
    sync_code()
    print("code synced")


if __name__ == "__main__":
    main()
