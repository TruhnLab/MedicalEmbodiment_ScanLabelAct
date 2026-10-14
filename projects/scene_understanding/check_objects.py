"""Quick look at build_objects.py's result: a top view of the room with each object's box and label,
and a gallery of each object's best view (box from the VLM, points of the object projected in).

    python projects/scene_understanding/check_objects.py 20261005_124116

Writes <session>/scene/check_topview.png and check_gallery.jpg.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402


def colour(i):
    return plt.cm.tab20(i % 20)


def topview(s, objs, area, out):
    cloud = o3d.io.read_point_cloud(str(s.path / "reconstruction" / "cloud.ply")).voxel_down_sample(0.03)
    P, C = np.asarray(cloud.points), np.asarray(cloud.colors)
    keep = P[:, 2] < 1.6
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(P[keep, 0], P[keep, 1], c=C[keep], s=1, alpha=0.35, linewidths=0)
    ax.plot(*np.vstack([area, area[:1]]).T, "k--", lw=1)
    path = np.array([f.T_map_rgb[:2, 3] for f in s.frames])
    ax.plot(path[:, 0], path[:, 1], color="0.5", lw=0.8)
    for o in objs:
        (cx, cy, _), (w, l, _) = o["center"], o["size"]
        box = cv2.boxPoints(((cx, cy), (w, l), o["yaw_deg"]))
        ax.fill(box[:, 0], box[:, 1], color=colour(o["id"]), alpha=0.35)
        ax.plot(*np.vstack([box, box[:1]]).T, color=colour(o["id"]), lw=1.5)
        ax.text(cx, cy, f"{o['id']} {o['label']}", fontsize=7, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_title(f"{s.name}: {len(objs)} objects (dashed = area of interest, grey = driven path)")
    fig.savefig(out, dpi=110, bbox_inches="tight")


def gallery(s, objs, points, out, tile=(240, 150)):
    by_index = {f.index: f for f in s.frames}
    tiles = []
    for o in objs:
        if str(o["id"]) not in points.files:
            continue
        v = o["best_views"][0]
        f = by_index[v["frame"]]
        img = f.rgb()
        uv, ok = s.project(f, points[str(o["id"])], "rgb")
        for x, y in uv[ok][::3].astype(int):
            img[y, x] = (0, 0, 255)
        x1, y1, x2, y2 = (int(c) for c in v["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # crop around the box with some context
        pad = 40
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        scale = min(tile[0] / crop.shape[1], tile[1] / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        t = np.full((tile[1] + 22, tile[0], 3), 255, np.uint8)
        t[22:22 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(t, f"#{o['id']} {o['label']} ({len(o['views'])}v)"[:34], (3, 16), 0, 0.45, (0, 0, 0), 1)
        tiles.append(t)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    # the labelled scene if there is one, else build_objects.py's raw result
    f = s.path / "scene" / "scene.json"
    d = json.loads((f if f.exists() else s.path / "scene" / "objects.json").read_text())
    points = np.load(s.path / "scene" / "objects_points.npz")
    topview(s, d["objects"], np.array(d["area_polygon"]), s.path / "scene" / "check_topview.png")
    gallery(s, d["objects"], points, s.path / "scene" / "check_gallery.jpg")
    print("wrote", s.path / "scene" / "check_topview.png", "and check_gallery.jpg")


if __name__ == "__main__":
    main()
