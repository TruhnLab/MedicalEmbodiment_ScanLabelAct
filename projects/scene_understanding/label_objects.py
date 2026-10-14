"""Step 5 of scene understanding: the VLM names each merged object from its best views, and the result
is written as the scene description scene.json.

    python projects/scene_understanding/label_objects.py 20261005_124116

Needs scene/objects.json (build_objects.py). Writes scene/crops/<id>_<n>.jpg (the views sent to the
VLM, kept for a later agent that wants to look at an object) and scene/scene.json:
objects with label, description, category, movable/articulated, state (with the time it was seen),
upright box in the map frame, clearance below, graspable flag, keyframes, crops.

States (door open, chair position) are what the scan saw at its time, not facts: a later run can
compare them with what the robot sees now.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landerpi.session import Session  # noqa: E402
import progress  # noqa: E402
import vlm  # noqa: E402
from schemas import ObjectAnswer  # noqa: E402

PROMPT = """These {n} crops come from a small ground robot's camera (0.3 m above the floor) at different
moments. The green box marks the same object in each crop. Its measured size is about
{w:.2f} x {l:.2f} m footprint and {h:.2f} m tall, from {z0:.2f} to {z1:.2f} m above the floor.
Name the object in the green box with 1-3 words (e.g. "cabinet", "bar stool"); details go into the
description. Name the whole object the box outlines, not a part that stands out in a close-up (a door, not
its handle, 2026-10-09), unless the box itself is only that part. If the boxes clearly show different
objects, set same_object false and describe the one in the first crop. If the box shows only a part of the
building (floor, wall, a fixed wall panel, a skirting board, a door threshold), a shadow or a reflection,
set real_object false (floor-level candidates found in 3D were skirting boards three times, 2026-10-06;
YOLOE's objects included "wall", "threshold" and skirting boards named as real objects, 2026-10-09)."""

# the arm's gripper opens ~6 cm; things up to ~15 cm that lie on the floor are candidates to pick up
GRASP_MAX = 0.15
# below this the robot (0.3 m with the camera on the arm) cannot drive under something
ROBOT_HEIGHT = 0.30


def crop(session, frame, box, pad=0.35, size=448, colour=(0, 255, 0)):
    img = frame.rgb()
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
    w, h = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * w - 20)), int(max(0, y1 - pad * h - 20))
    X2, Y2 = int(min(img.shape[1], x2 + pad * w + 20)), int(min(img.shape[0], y2 + pad * h + 20))
    c = img[Y1:Y2, X1:X2]
    s = size / max(c.shape[:2])
    if s < 1:
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return c


def geometry(o, pts, category):
    """Clearance below (table tops, sofa seats: what the robot can drive under) and floor contact."""
    (cx, cy, _), (w, l, _) = o["center"], o["size"]
    z0, z1 = o["z_range"]
    # inner part of the footprint: the legs at the edges don't count
    yaw = np.radians(o["yaw_deg"])
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local = (pts[:, :2] - [cx, cy]) @ R.T
    inner = (np.abs(local[:, 0]) < 0.3 * w) & (np.abs(local[:, 1]) < 0.3 * l)
    under = None
    # furniture only: small things on a counter got "drive under" from the counter's points below them
    if category == "furniture" and min(w, l) > 0.35 and inner.sum() > 30 and z1 > 0.4:
        lowest = float(np.percentile(pts[inner, 2], 5))
        under = round(lowest, 2) if lowest > 0.1 else 0.0
    on_floor = z0 < 0.05
    graspable = on_floor and max(w, l, z1 - max(z0, 0)) <= GRASP_MAX
    return {"clearance_below_m": under, "robot_fits_below": None if under is None else under > ROBOT_HEIGHT + 0.05,
            "on_floor": bool(on_floor), "graspable_size": bool(graspable)}


def label_one(session, by_index, o, crops_dir):
    views = o["best_views"][:3]
    parts, files = [], []
    for n, v in enumerate(views):
        c = crop(session, by_index[v["frame"]], v["box"])
        f = crops_dir / f"{o['id']}_{n}.jpg"
        cv2.imwrite(str(f), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
        files.append(f"crops/{f.name}")
        parts.append(vlm.image_part(f.read_bytes()))
    w, l, h = o["size"]
    # without the per-image labels: Qwen followed YOLOE's confusions (a stool named "chair", a radiator "door",
    # 2026-10-09); the name comes from the crops alone
    text = PROMPT.format(n=len(views), w=w, l=l, h=h, z0=o["z_range"][0], z1=o["z_range"][1])
    content = parts + [{"type": "text", "text": text}]
    try:
        ans = ObjectAnswer.model_validate_json(vlm.chat(content, max_tokens=600, schema=ObjectAnswer))
    except ValueError:
        # with the schema enforced the model sometimes runs on with whitespace until max_tokens
        # (1 of 62 objects, 2026-10-05): once more without it, then fall back to the per-frame labels
        try:
            ans = ObjectAnswer.model_validate(vlm.parse_json(vlm.chat(content, max_tokens=600)))
        except ValueError:
            ans = ObjectAnswer(real_object=True, same_object=False, label=o["label"], description="",
                               category=o["category"] or "other", movable=o["movable"] or "movable",
                               state=o["state"], confidence="low")
    return ans, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    s = Session(a.session)
    sd = s.path / "scene"
    d = json.loads((sd / "objects.json").read_text())
    points = np.load(sd / "objects_points.npz")
    by_index = {f.index: f for f in s.frames}
    crops_dir = sd / "crops"
    crops_dir.mkdir(exist_ok=True)
    for f in crops_dir.glob("*.jpg"):
        f.unlink()
    t0 = time.time()
    finished, lock = [], threading.Lock()

    def one(o):
        r = label_one(s, by_index, o, crops_dir)
        with lock:  # one writer at a time for progress.json
            finished.append(o["id"])
            progress.update(sd, done=len(finished), total=len(d["objects"]))
        return r

    with ThreadPoolExecutor(a.workers) as pool:
        answers = list(pool.map(one, d["objects"]))
    objects = []
    for o, (ans, files) in zip(d["objects"], answers):
        if ans.category == "person" or not ans.real_object:
            continue
        g = geometry(o, points[str(o["id"])], ans.category)
        # a state only for things that open; the VLM also called stools "open" (2026-10-05)
        articulated = ans.movable == "articulated" or ans.category == "door"
        # a label is a name, not a description: if the model wrote a sentence anyway (once, 2026-10-06:
        # "cabinet with a side panel and drawers, viewed from ..."), use the most common per-frame label
        label = ans.label.strip().lower()
        if len(label.split()) > 4:
            label = max(o["labels"], key=o["labels"].get)
        objects.append({
            "id": o["id"], "label": label, "description": ans.description, "category": ans.category,
            "movable": ans.movable,
            # articulated / movable things: what the scan saw, with the time, not a fixed fact
            "state": ans.state if articulated else None, "state_seen": o["state_seen"] if articulated else [],
            "center": o["center"], "size": o["size"], "yaw_deg": o["yaw_deg"], "z_range": o["z_range"], **g,
            "confidence": ans.confidence if ans.same_object else "low",
            "views": o["views"], "best_views": o["best_views"], "crops": files,
            "first_seen": o["first_seen"], "last_seen": o["last_seen"],
            "frame_labels": o["labels"], "source": o.get("source", "vlm"),
            # an operable part (drawer, cabinet door) of the object with this id (judge_pairs.py)
            "part_of": o.get("part_of"),
        })
    scene = {"session": s.name, "created": time.strftime("%Y-%m-%d %H:%M"), "model": vlm.MODEL,
             "frame": "map: metres, floor at z = 0, same frame as the SLAM map and reconstruction/mesh.ply",
             "box": "center = middle of the upright box, size = [along yaw, across, height], yaw in degrees",
             "camera_limits": "camera 0.3 m above the floor: table tops and things above ~1.5 m are not seen",
             "area_polygon": d["area_polygon"], "objects": objects}
    (sd / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"{len(objects)} objects labelled in {time.time() - t0:.0f} s -> {sd / 'scene.json'}")
    for o in objects:
        print(f"  #{o['id']:3d} {o['label']:<24} {o['category']:<10} {o['confidence']:<6} {o['description'][:70]}")


if __name__ == "__main__":
    main()
