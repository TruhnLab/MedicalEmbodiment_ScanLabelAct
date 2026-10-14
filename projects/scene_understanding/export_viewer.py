"""Data for the console's Labels view (3D view, console/labels3d.js): which object each mesh vertex
belongs to, and each object's best keyframes with its mask drawn in.

    python projects/scene_understanding/export_viewer.py 20261005_124116

Writes to <session>/scene/: vertex_objects.bin (int16 per vertex of reconstruction/mesh_view.ply, in
file order, -1 = no object) and views/<id>_<n>.jpg.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402

# a vertex belongs to the object with the nearest point within this distance: the object points come
# from single depth images (noise ~1-2 cm at 1 m), the mesh is the fused surface
VERTEX_RADIUS = 0.04


def palette(i):
    """Same colours as console/labels3d.js (golden-angle hues), BGR for OpenCV."""
    h = (i * 137.508) % 360
    hsv = np.uint8([[[h / 2, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def vertex_objects(s, objects, points):
    V = np.asarray(o3d.io.read_triangle_mesh(str(s.mesh_file(view=True))).vertices)
    ids = np.full(len(V), -1, np.int16)
    objs = [o for o in objects if str(o["id"]) in points.files and len(points[str(o["id"])])]
    if not objs:
        return ids
    # all objects' points in one tree: each vertex takes the object of its nearest point (one query
    # instead of one per object)
    P = np.concatenate([points[str(o["id"])] for o in objs])
    owner = np.concatenate([[o["id"]] * len(points[str(o["id"])]) for o in objs])
    d, k = cKDTree(P).query(V, distance_upper_bound=VERTEX_RADIUS)
    hit = np.isfinite(d)
    ids[hit] = owner[k[hit]]
    return ids


def mask_views(s, objects, sd):
    """Keyframe with the object's own mask filled in, the other masks of that frame outlined."""
    out = sd / "views"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    by_index = {f.index: f for f in s.frames}
    frames_vlm = {r["index"]: r for r in map(json.loads, (sd / "frames_vlm.jsonl").read_text().splitlines()) if r}
    for o in objects:
        files = []
        for n, v in enumerate(o["best_views"][:3]):
            mf = sd / "masks" / f"{v['frame']}.npz"
            img = by_index[v["frame"]].rgb()
            mine = None
            if mf.exists() and o.get("source", "vlm") == "vlm":
                z = np.load(mf)
                masks = np.unpackbits(z["masks"], axis=-1, count=int(z["width"])).astype(bool)
                boxes = [frames_vlm[v["frame"]]["objects"][int(k)]["box"] for k in z["objects"]]
                # this object's mask: the one whose VLM box is the view's box
                k = int(np.argmin([np.abs(np.subtract(b, v["box"])).sum() for b in boxes]))
                for j, m in enumerate(masks):
                    if j != k:
                        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, cs, -1, (200, 200, 200), 1)
                mine = masks[k]
            col = palette(o["id"])
            if mine is not None:
                img[mine] = (0.45 * img[mine] + 0.55 * np.array(col)).astype(np.uint8)
                cs, _ = cv2.findContours(mine.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 2)
            else:
                x1, y1, x2, y2 = (int(c) for c in v["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            name = f"{o['id']}_{n}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files.append(f"views/{name}")
        o["mask_views"] = files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    sd = s.path / "scene"
    scene = json.loads((sd / "scene.json").read_text())
    points = np.load(sd / "objects_points.npz")
    ids = vertex_objects(s, scene["objects"], points)
    ids.tofile(sd / "vertex_objects.bin")
    mask_views(s, scene["objects"], sd)
    (sd / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"{(ids >= 0).sum():,} of {len(ids):,} mesh vertices belong to {len(np.unique(ids[ids >= 0]))} objects; "
          f"mask views for {len(scene['objects'])} objects")


if __name__ == "__main__":
    main()
