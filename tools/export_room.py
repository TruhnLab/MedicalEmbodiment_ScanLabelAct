"""The labelled room for the page's 3D viewer (assets/js/room3d.js): a decimated copy of a session's mesh
with the photo colours and, per vertex, the object it belongs to.

    conda run -n landerpi python tools/export_room.py [--session 20261007_101108] [--faces 160000]

Reads ../LanderPi/data/sessions/<session>/: reconstruction/mesh_view.ply, scene/scene.json and
scene/vertex_objects.bin (int16 per vertex, -1 = no object). Writes assets/data/room.bin and room.json.

room.bin, little endian, in this order:
  uint32  indices[3 * faces]
  uint16  positions[3 * vertices]   quantized between room.json's box_min and box_max
  uint8   colours[3 * vertices]     sRGB
  uint8   objects[vertices]         index into room.json's objects, 255 = none
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PAGE = Path(__file__).resolve().parents[1]
CODE = PAGE.parent / "LanderPi"

# stray bits far from the room (reflections, the view through a door) only distract in the viewer
MARGIN_M = 0.6
MIN_CLUSTER_TRIANGLES = 400
# not published: in 20261007_101108 the "shoe" is worn by a person who sat in the kitchen during the scan
SKIP_LABELS = {"shoe"}
# descriptions that talk about the labelling's image crops instead of the object (e.g. the kitchen's table)
META_WORDS = ("crop", "highlight", "bounding box", "green box")


def first_sentence(text):
    text = (text or "").strip()
    i = text.find(". ")
    text = text if i < 0 else text[: i + 1]
    if any(w in text.lower() for w in META_WORDS):
        return ""
    if text.startswith("The object is "):
        text = text[len("The object is "):]
        text = text[:1].upper() + text[1:]
    return text


def inside(poly, xy, margin):
    """Points within `margin` of the polygon (even-odd test on the polygon, plus distance to its edges)."""
    poly = np.asarray(poly, float)
    x, y = xy[:, 0], xy[:, 1]
    inn = np.zeros(len(xy), bool)
    dist = np.full(len(xy), np.inf)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inn ^= cross
        d = np.array([x2 - x1, y2 - y1])
        t = np.clip(((x - x1) * d[0] + (y - y1) * d[1]) / max(d @ d, 1e-12), 0, 1)
        dist = np.minimum(dist, np.hypot(x - (x1 + t * d[0]), y - (y1 + t * d[1])))
    return inn | (dist < margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20261007_101108")
    ap.add_argument("--faces", type=int, default=160000)
    args = ap.parse_args()

    s = CODE / "data" / "sessions" / args.session
    scene = json.loads((s / "scene" / "scene.json").read_text(encoding="utf-8"))
    mesh = o3d.io.read_triangle_mesh(str(s / "reconstruction" / "mesh_view.ply"))
    # copies: np.asarray only views Open3D's buffers, which the cropping below rewrites
    V0 = np.array(mesh.vertices)
    C0 = np.array(mesh.vertex_colors)
    ids0 = np.fromfile(s / "scene" / "vertex_objects.bin", np.int16)
    assert len(ids0) == len(V0), "vertex_objects.bin belongs to another reconstruction"
    print(f"mesh: {len(V0)} vertices, {len(mesh.triangles)} faces")

    # crop to the room's area and drop small floating pieces
    keep = inside(scene["area_polygon"], V0[:, :2], MARGIN_M)
    mesh.remove_vertices_by_mask(~keep)
    tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
    small = np.asarray(cluster_n)[np.asarray(tri_cluster)] < MIN_CLUSTER_TRIANGLES
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    print(f"cropped: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.faces)
    mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles).astype(np.uint32)
    print(f"decimated: {len(V)} vertices, {len(F)} faces")

    # colours and objects from the nearest vertex of the full mesh (decimation moves vertices)
    _, nn = cKDTree(V0).query(V)
    colours = np.clip(np.round(C0[nn] * 255), 0, 255).astype(np.uint8)
    ids = ids0[nn]

    objects, index = [], {}
    for o in scene["objects"]:
        if o["label"] in SKIP_LABELS:
            continue
        index[o["id"]] = len(objects)
        objects.append({
            "id": o["id"],
            "label": o["label"],
            "category": o.get("category"),
            "description": first_sentence(o.get("description")),
            "center": [round(v, 3) for v in o["center"]],
            "size": [round(v, 3) for v in o["size"]],
            "yaw_deg": o["yaw_deg"],
            "z_range": [round(v, 3) for v in o["z_range"]],
        })
    obj = np.full(len(V), 255, np.uint8)
    for oid, k in index.items():
        obj[ids == oid] = k
    print(f"{len(objects)} objects, {np.count_nonzero(obj != 255)} vertices on objects")

    lo, hi = V.min(0), V.max(0)
    q = np.round((V - lo) / (hi - lo) * 65535).astype(np.uint16)

    out = PAGE / "assets" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "room.bin", "wb") as f:
        f.write(F.astype("<u4").tobytes())
        f.write(q.astype("<u2").tobytes())
        f.write(colours.tobytes())
        f.write(obj.tobytes())
    meta = {
        "session": args.session,
        "vertices": int(len(V)),
        "faces": int(len(F)),
        "box_min": [round(float(v), 4) for v in lo],
        "box_max": [round(float(v), 4) for v in hi],
        "area_polygon": scene["area_polygon"],
        "objects": objects,
    }
    (out / "room.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"room.bin {(out / 'room.bin').stat().st_size / 1e6:.2f} MB, room.json {(out / 'room.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
