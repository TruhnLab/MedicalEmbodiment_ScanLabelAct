"""Calibrates the depth/colour camera's tilt and roll in the survey pose (key P), from a corner.

    python projects/mapping/camera_calibration.py [data/sessions/<name>] [--save]

Record ~20 s standing still, the robot 1-1.5 m in front of a room corner, arm in the survey pose:
    python projects/mapping/record.py --all-frames --max-minutes 0.35 --live-port 0

Why: the robot model computes the camera pose from the COMMANDED servo positions; under the arm's
weight the servos sit a few steps lower, and bracket/servo tolerances add up, so the real camera
looks a few degrees up and is slightly rolled (measured 2026-10-02: ~4 deg / ~2.5 deg). Every depth
frame then places the floor and walls tilted, which blurs the merged map.

How: in the robot's base frame the floor must be level and the walls vertical. Per frame, the floor
and the two walls are fitted as planes; the small rotation (about the base's x and y axes) that
makes all of them right at once is solved by least squares over all frames. The heading (pan) is
calibrated in the servo itself (tools/arm_calibration.py, checked with camera_pan_check.py).

--save writes config/camera_correction.json; reconstruct.py applies it to keyframes taken in the
survey pose (others are left as they are, with a note).
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from reconstruct import DEPTH_FRAME, REPO, TransformTree, depth_points

CONFIG = REPO / "config" / "camera_correction.json"


def planes(points, max_planes=4):
    """Largest planes: [(kind, normal (unit, floor up / walls facing the robot), n_points)]."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    out = []
    for _ in range(max_planes):
        if len(pc.points) < 3000:
            break
        (a, b, c, _), inl = pc.segment_plane(0.01, 3, 1000)
        n = np.array([a, b, c])
        if abs(n[2]) > 0.8:
            out.append(("floor", n * np.sign(n[2]), len(inl)))
        elif abs(n[2]) < 0.3 and len(inl) > 5000:
            centre = np.asarray(pc.points)[inl].mean(0)
            out.append(("wall", n if n @ centre < 0 else -n, len(inl)))
        pc = pc.select_by_index(inl, invert=True)
    return out


def solve(observations):
    """Small rotation w = (wx, wy) about the base's x and y axes with R n ~ n + w x n such that floor
    normals become (0, 0, 1) and wall normals horizontal. Least squares, weighted by plane size."""
    A, b, w = [], [], []
    for kind, n, size in observations:
        weight = np.sqrt(size)
        if kind == "floor":
            # x: n_x + wy n_z = 0 ; y: n_y - wx n_z = 0
            A += [[0.0, n[2]], [-n[2], 0.0]]
            b += [-n[0], -n[1]]
            w += [weight, weight]
        else:
            # z: n_z + wx n_y - wy n_x = 0
            A.append([n[1], -n[0]])
            b.append(-n[2])
            w.append(weight)
    A, b, w = np.array(A), np.array(b), np.array(w)
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return sol, A @ sol - b


def rotation(wx, wy):
    """Exact rotation for the small-angle solution (roll wx about x, then pitch wy about y)."""
    cx, sx, cy, sy = np.cos(wx), np.sin(wx), np.cos(wy), np.sin(wy)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, nargs="?")
    p.add_argument("--save", action="store_true", help="write config/camera_correction.json")
    p.add_argument("--frames", type=int, default=25)
    a = p.parse_args()
    session = a.session or max((REPO / "data" / "sessions").iterdir(), key=lambda d: d.name)
    tree = TransformTree(session)
    depth_info = json.loads((session / "calib" / "depth_camera_info.json").read_text())
    with open(session / "keyframes.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    obs = []
    for k in rows[:: max(1, len(rows) // a.frames)]:
        t = float(k["depth_stamp"])
        depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        T_depth, _ = tree.map_to(DEPTH_FRAME, t)
        T_base, _ = tree.map_to("base_footprint", t)
        B = np.linalg.inv(T_base) @ T_depth
        B_ref = B
        pts, _, _ = depth_points(depth, depth_info, 0.3, 2.5)
        obs += planes(pts @ B[:3, :3].T + B[:3, 3])
    kinds = [o[0] for o in obs]
    if kinds.count("floor") < 3 or kinds.count("wall") < 3:
        raise SystemExit(f"not enough planes ({kinds.count('floor')} floor, {kinds.count('wall')} wall): "
                         "stand 1-1.5 m in front of a corner with floor in view")
    (wx, wy), residual = solve(obs)
    print(f"{session.name}: {kinds.count('floor')} floor and {kinds.count('wall')} wall planes")
    print(f"  correction: roll {np.degrees(wx):+.2f} deg (about the base x axis), "
          f"pitch {np.degrees(wy):+.2f} deg (about the base y axis)")
    print(f"  remaining misfit: median {np.degrees(np.median(np.abs(residual))):.2f} deg, "
          f"max {np.degrees(np.max(np.abs(residual))):.2f} deg")
    if a.save:
        CONFIG.parent.mkdir(exist_ok=True)
        CONFIG.write_text(json.dumps({
            "description": "rotation applied in the robot base frame to the camera pose in the survey pose "
                           "(projects/mapping/camera_calibration.py)",
            "session": session.name, "roll_deg": round(float(np.degrees(wx)), 3),
            "pitch_deg": round(float(np.degrees(wy)), 3),
            # applies only to keyframes whose camera sits like this in the base frame (the survey pose)
            "base_to_depth_rotation": B_ref[:3, :3].round(6).tolist(),
            # servo 1 had no zero offset until 2026-10-02 (now +38, tools/arm_calibration.py): older
            # recordings' camera looked 7.15 deg further right than modelled (camera_pan_check.py)
            "pan_deg_before": {"session": "20261002_093700", "pan_deg": -7.15}}, indent=1))
        print(f"  saved to {CONFIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
