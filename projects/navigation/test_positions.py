"""Offline check of 3D positions from detections: detection in the colour image + depth -> where the object is
in the room (detector.py, perception.py), on a room's own recording. (No robot, nothing moves.)

    python projects/navigation/test_positions.py 20261007_101108

For every object's best views (scene.json best_views): the detector with the object's label as prompt, the
detection overlapping the labelling's box, its mask's depth points in the map (keyframe poses from
reconstruction/cameras.json), and the distance of their median to the object's box (0 inside it). The scan's
boxes are the reference here, with their own 10-20 cm errors (and cabinet boxes reaching 20-30 cm beyond
their fronts: navigation README), so this checks for gross errors (wrong frames, timing, flipped axes)
rather than centimetres.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import Detector  # noqa: E402
from landerpi.room import Room  # noqa: E402
from landerpi.session import Session  # noqa: E402
from perception import inner, locate, mask_points, nearest_group  # noqa: E402
from scan_cells import ScanCells  # noqa: E402
from test_detector import iou  # noqa: E402


def box_distance(o, xy):
    yaw = math.radians(o["yaw_deg"])
    dx, dy = xy[0] - o["center"][0], xy[1] - o["center"][1]
    lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.hypot(max(abs(lx) - o["size"][0] / 2, 0), max(abs(ly) - o["size"][1] / 2, 0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room")
    args = p.parse_args()
    room = Room(args.room)
    s = Session(room.session_path)
    frames = {f.index: f for f in s.frames}
    objs = json.loads((room.session_path / "scene" / "scene.json").read_text())["objects"]
    Kd, Kr = s.K["depth"][2:], s.K["rgb"][2:]
    d = Detector()
    cells = ScanCells.of(room)
    rows, ms = [], []
    for o in objs:
        d.set_prompts([o["label"]])
        for v in o.get("best_views", []):
            f = frames.get(v["frame"])
            if f is None:
                continue
            rgb = cv2.imread(str(f.rgb_file))
            depth = cv2.imread(str(f.depth_file), cv2.IMREAD_UNCHANGED)
            t0 = time.perf_counter()
            dets = [x for x in d.detect(rgb) if iou(x["box"], v["box"]) >= 0.3 and x["mask"] is not None]
            if not dets:
                rows.append((o, None, None))
                continue
            m = dets[0]["mask"]
            loc = locate(mask_points(m, depth, Kd, f.T_map_depth, Kr, f.T_map_rgb))
            ms.append((time.perf_counter() - t0) * 1000)
            # the live method (2026-10-09): the mask without its edge; the nearest solid group of its points only
            # if that position floats (for stools it took the one in front: 138 cm off)
            pts = mask_points(inner(m), depth, Kd, f.T_map_depth, Kr, f.T_map_rgb)
            new = locate(pts)
            if new is not None and cells is not None and not cells.supported(new["position"], new["size"]):
                new = locate(nearest_group(pts, f.T_map_rgb[:3, 3]))
            rows.append((o, None if loc is None else (box_distance(o, loc["position"][:2]), loc),
                         None if new is None else (box_distance(o, new["position"][:2]), new)))
    print(f"{room}: {len(rows)} views; time per view {np.median(ms):.0f} ms (detection + 3D position)")
    for name, k in (("median of the mask (before)", 1), ("inner mask, nearest group if floating (live)", 2)):
        got = [(r[k], o) for r in rows if r[k] is not None for o in [r[0]]]
        dist = np.array([g[0][0] for g in got])
        sup = np.mean([cells.supported(g[0][1]["position"], g[0][1]["size"]) for g in got]) if cells else float("nan")
        print(f"{name}: located {len(got)}; distance to the object's box: median {np.median(dist) * 100:.0f} cm, "
              f"{(dist < 0.10).mean() * 100:.0f} % within 10 cm, {(dist < 0.25).mean() * 100:.0f} % within 25 cm, "
              f"worst {dist.max() * 100:.0f} cm; on a scanned surface {sup * 100:.0f} %")
        worst = sorted([(g[0][0], g[1]) for g in got], key=lambda x: -x[0])[:6]
        print("  largest: " + ", ".join(f"#{o['id']} {o['label']} {dd * 100:.0f} cm" for dd, o in worst))


if __name__ == "__main__":
    main()
