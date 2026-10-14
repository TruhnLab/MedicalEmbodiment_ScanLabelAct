"""Step 3 of scene understanding: small things on the floor that the VLM did not list, found in 3D.

    python projects/scene_understanding/floor_items.py 20261005_124116

Run after build_objects.py, before label_objects.py. Adds candidates to scene/objects.json (labelled
"floor item?", source "geometry"); label_objects.py names them or drops them if the VLM sees nothing.

Why geometry: things below the lidar's scan plane are what the robot can bump into or pick up, and a
VLM easily overlooks a small item in a wide view; depth sees it from several keyframes.
A candidate: measured depth points 3-30 cm above the floor, seen in 3+ keyframes and at least 6 cm high, not part of
something taller (walls, furniture: skirting boards and chair legs would qualify otherwise) and not
inside a known object's box.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landerpi.session import Session  # noqa: E402
from build_objects import inside  # noqa: E402

CELL = 0.03            # m, floor grid
LOW = (0.03, 0.30)     # floor noise is 1-2 cm after levelling; above 30 cm it is furniture, not an item
TALL = 0.35            # cells with points above this are walls / furniture
TALL_MARGIN = 0.12     # m around tall cells (skirting boards, legs, wall bases)
MIN_VIEWS = 3
# blobs only 3-4 cm high, seen in 2 views, were floor noise (8 of them, 20261005_124116)
MIN_HEIGHT = 0.06
SIZE = (0.04, 0.6)     # m, footprint of a candidate
MAX_DIST = 2.0         # m, only close measurements: floor items are small


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    sd = s.path / "scene"
    d = json.loads((sd / "objects.json").read_text())
    d["objects"] = [o for o in d["objects"] if o.get("source") != "geometry"]  # rerun-safe
    area = np.array(d["area_polygon"])
    pts_all = dict(np.load(sd / "objects_points.npz"))

    allp = []
    for f in s.frames:
        p = s.depth_points(f, max_d=MAX_DIST)[0]
        allp.append(np.c_[p, np.full(len(p), f.index)])
    P = np.concatenate(allp)
    lo = P[:, :2].min(0) - 1
    ij = np.floor((P[:, :2] - lo) / CELL).astype(int)
    shape = tuple(ij.max(0) + 1)
    tall = np.zeros(shape, bool)
    tall[tuple(ij[P[:, 2] > TALL].T)] = True
    tall = ndimage.binary_dilation(tall, iterations=int(round(TALL_MARGIN / CELL)))
    low_sel = (P[:, 2] > LOW[0]) & (P[:, 2] < LOW[1])
    # views per cell: how many keyframes measured something low there
    key = ij[low_sel, 0] * shape[1] + ij[low_sel, 1]
    pairs = np.unique(np.c_[key, P[low_sel, 3].astype(int)], axis=0)
    views = np.bincount(pairs[:, 0], minlength=shape[0] * shape[1]).reshape(shape)
    cand = (views >= MIN_VIEWS) & ~tall
    # inside a known object's box (dilated): already covered
    known = np.zeros(shape, np.uint8)
    for o in d["objects"]:
        (cx, cy, _), (w, l, _) = o["center"], o["size"]
        box = cv2.boxPoints(((cx, cy), (w + 0.2, l + 0.2), o["yaw_deg"]))
        cv2.fillPoly(known, [np.round((box - lo) / CELL).astype(np.int32)[:, ::-1]], 1)
    cand &= known == 0
    lab, n = ndimage.label(ndimage.binary_closing(cand), structure=np.ones((3, 3)))
    next_id = max([o["id"] for o in d["objects"]], default=-1) + 1
    by_index = {f.index: f for f in s.frames}
    added = 0
    # points of each blob, grouped once instead of scanning all ~30 M points for every blob
    plab = np.where(low_sel, lab[ij[:, 0], ij[:, 1]], 0)
    order = np.argsort(plab, kind="stable")
    bounds = np.searchsorted(plab[order], np.arange(n + 2))
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        ext = np.array([sl[0].stop - sl[0].start, sl[1].stop - sl[1].start]) * CELL
        if ext.max() < SIZE[0] or ext.max() > SIZE[1]:
            continue
        sel = order[bounds[k]:bounds[k + 1]]
        pts = P[sel, :3]
        frames = np.unique(P[sel, 3]).astype(int)
        c = pts.mean(0)
        if len(pts) < 30 or np.percentile(pts[:, 2], 99) < MIN_HEIGHT or not inside(area, c[:2]):
            continue
        # best views: close and the item well inside the image
        cands = []
        for fi in frames:
            f = by_index[int(fi)]
            uv, ok = s.project(f, pts, "rgb")
            if ok.mean() < 0.9:
                continue
            x1, y1 = uv[ok].min(0)
            x2, y2 = uv[ok].max(0)
            dist = float(np.linalg.norm(c - f.T_map_rgb[:3, 3]))
            cands.append((abs(dist - 0.9), {"frame": int(fi), "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                                            "distance": round(dist, 2)}))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        (cx, cy), (w, l), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
        z0, z1 = np.percentile(pts[:, 2], [1, 99])
        st = [float(f) for f in frames]
        d["objects"].append({
            "id": next_id, "source": "geometry", "label": "floor item?", "labels": {"floor item?": 1},
            "category": "floor_item", "movable": "movable", "state": None, "state_seen": [],
            "center": [round(float(cx), 3), round(float(cy), 3), round(float((z0 + z1) / 2), 3)],
            "size": [round(float(w), 3), round(float(l), 3), round(float(z1), 3)], "yaw_deg": round(float(ang), 1),
            "z_range": [0.0, round(float(z1), 3)], "n_views": len(frames),
            "first_seen": min(by_index[i].stamp for i in frames), "last_seen": max(by_index[i].stamp for i in frames),
            "best_views": [c_[1] for c_ in cands[:3]], "views": sorted(int(i) for i in frames),
            "measured_fraction": 1.0})
        pts_all[str(next_id)] = pts.astype(np.float32)
        next_id += 1
        added += 1
    (sd / "objects.json").write_text(json.dumps(d, indent=1))
    np.savez_compressed(sd / "objects_points.npz", **pts_all)
    print(f"{n} low blobs, {added} floor item candidates added")


if __name__ == "__main__":
    main()
