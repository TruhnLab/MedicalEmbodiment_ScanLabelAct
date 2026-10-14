"""Measures how far the depth camera's heading deviates from the robot model, using the lidar.

    python projects/mapping/camera_pan_check.py [data/sessions/<name>]     # newest session by default

Record ~20 s standing still in the pose to check, e.g.
    python projects/mapping/record.py --all-frames --max-minutes 0.35 --live-port 0

The lidar is fixed to the chassis, the camera sits on the arm. Both see the same walls: seen from
above, the depth points on walls and furniture sides must lie on the lidar scan. Placed with the modelled camera pose, they
are rotated by the camera's heading error; a 2D alignment (rotation about the arm's base axis +
small shift) measures it. Negative rotation = the camera really looks further right than modelled.
Needs a wall (best: a corner) 1-1.5 m in front of the camera, not open space or table legs.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from reconstruct import DEPTH_FRAME, REPO, TransformTree, depth_points, stamp

LIDAR_FRAME = "lidar_frame"
ARM_BASE_FRAME = "link1"  # rotates with servo 1 (arm base); its axis is what a pan error turns


def scan_points(msg):
    r = np.array([np.nan if x is None else x for x in msg["ranges"]], dtype=float)
    a = msg["angle_min"] + np.arange(len(r)) * msg["angle_increment"]
    ok = np.isfinite(r) & (r > msg["range_min"]) & (r < min(msg["range_max"], 4.0))
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum())], 1)


def align_yaw(src, dst, centre, iterations=30):
    """2D ICP: rotation (about `centre`) + shift that moves src onto dst. Returns degrees, metres, residual."""
    tree = cKDTree(dst)
    yaw, t = 0.0, np.zeros(2)
    for _ in range(iterations):
        c, s = np.cos(yaw), np.sin(yaw)
        moved = (src - centre) @ np.array([[c, -s], [s, c]]).T + centre + t
        d, idx = tree.query(moved)
        keep = d < max(0.05, np.percentile(d, 70))
        A, B = src[keep] - centre, dst[idx[keep]] - centre
        mu_a, mu_b = A.mean(0), B.mean(0)
        H = (A - mu_a).T @ (B - mu_b)
        yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        c, s = np.cos(yaw), np.sin(yaw)
        t = mu_b - mu_a @ np.array([[c, -s], [s, c]]).T
    c, s = np.cos(yaw), np.sin(yaw)
    d, _ = tree.query((src - centre) @ np.array([[c, -s], [s, c]]).T + centre + t)
    return np.degrees(yaw), t, float(np.median(d))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, nargs="?")
    p.add_argument("--band-low", type=float, default=0.05, help="use depth points from this height above the floor ...")
    p.add_argument("--band-high", type=float, default=0.5, help="... up to this height (m)")
    p.add_argument("--frames", type=int, default=40, help="depth frames to use")
    a = p.parse_args()
    session = a.session or max((REPO / "data" / "sessions").iterdir(), key=lambda d: d.name)
    tree = TransformTree(session)
    depth_info = json.loads((session / "calib" / "depth_camera_info.json").read_text())
    with open(session / "keyframes.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    scans = [json.loads(line)["msg"] for line in open(session / "streams" / "scan.jsonl", encoding="utf-8")]
    scan_t = np.array([stamp(m["header"]) for m in scans])

    results = []
    for k in rows[:: max(1, len(rows) // a.frames)]:
        t = float(k["depth_stamp"])
        depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        T_depth, _ = tree.map_to(DEPTH_FRAME, t)
        T_lidar, _ = tree.map_to(LIDAR_FRAME, t)
        T_arm, _ = tree.map_to(ARM_BASE_FRAME, t)
        pts, _, _ = depth_points(depth, depth_info, 0.3, 3.0)
        cam = pts @ T_depth[:3, :3].T + T_depth[:3, 3]
        # everything in the lidar frame: the lidar plane is z = 0 there
        L = np.linalg.inv(T_lidar)
        cam_l = cam @ L[:3, :3].T + L[:3, 3]
        # the lidar is only ~9 cm above the floor: points "at lidar height" would mostly be floor.
        # Use vertical structures instead (walls, legs, furniture sides) from 5 to 50 cm above the
        # floor as the camera sees it; their footprint from above must match the scan
        hist, edges = np.histogram(cam_l[:, 2], bins=np.arange(-0.4, 0.4, 0.01))
        floor = edges[np.argmax(hist)] + 0.005
        band = cam_l[(cam_l[:, 2] > floor + a.band_low) & (cam_l[:, 2] < floor + a.band_high)][:, :2]
        if len(band) < 300:
            continue
        scan = scans[int(np.argmin(np.abs(scan_t - t)))]
        sp = scan_points(scan)[:, :2]
        # only the part of the scan the camera can see (its horizontal field of view)
        ang_band = np.arctan2(band[:, 1], band[:, 0])
        lo, hi = np.percentile(ang_band, 1) - 0.15, np.percentile(ang_band, 99) + 0.15
        ang_scan = np.arctan2(sp[:, 1], sp[:, 0])
        sp = sp[(ang_scan > lo) & (ang_scan < hi)]
        if len(sp) < 30:
            continue
        centre = (L @ T_arm)[:2, 3]  # arm base axis in the lidar frame
        band = band[np.random.default_rng(0).choice(len(band), min(len(band), 4000), replace=False)]
        yaw, shift, resid = align_yaw(band, sp, centre)
        results.append((yaw, *shift, resid, len(band), len(sp)))
    if not results:
        raise SystemExit("no usable frames: the camera needs walls/furniture 0.5-3 m away at lidar height")
    r = np.array(results)
    yaw = r[:, 0]
    print(f"{session.name}: {len(r)} frames, depth points at lidar height vs lidar scan")
    print(f"  camera heading error: {np.median(yaw):+.2f} deg (spread {np.percentile(yaw, 10):+.2f}..{np.percentile(yaw, 90):+.2f}),"
          f" {'camera looks further RIGHT than modelled' if np.median(yaw) < 0 else 'camera looks further LEFT than modelled'}")
    print(f"  remaining shift {100 * np.median(r[:, 1]):+.1f} / {100 * np.median(r[:, 2]):+.1f} cm, "
          f"median distance after alignment {100 * np.median(r[:, 3]):.1f} cm "
          f"({int(np.median(r[:, 4]))} depth / {int(np.median(r[:, 5]))} scan points per frame)")


if __name__ == "__main__":
    main()
