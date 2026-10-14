"""How well small objects are detected with distance, and what helps: detector input size, a zoom, the model's
size, the score threshold, synonyms as prompts.

    python projects/navigation/test_far.py 20261007_101108                    # input sizes and zoom (2026-10-08)
    python projects/navigation/test_far.py 20261007_101108 --models yoloe-11s-seg.pt yoloe-11m-seg.pt yoloe-11l-seg.pt
    python projects/navigation/test_far.py 20261007_101108 --conf 0.05 0.1 --synonyms

Small labelled objects of the room (largest side < MAX_SIZE m: bottles, a shoe, a power outlet) in every
keyframe where their box centre is in the colour image (final poses, reconstruction/cameras.json), by the
distance from the camera. Each setting gets the object's label as prompt; it counts as detected if a box of
that label contains the projected centre. Settings: the detector's input at 640 / 960 / 1280 px, and "zoom":
640 px plus a second pass on the image region around the object enlarged 2x (what a search can do with its
area). Reported: share detected per distance band, and time per image.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import Detector  # noqa: E402
from landerpi.room import Room  # noqa: E402
from landerpi.session import Session  # noqa: E402

MAX_SIZE = 0.4
BANDS = [(0.3, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5)]
PER_BAND = 6           # keyframes per object and band (spread over the run); 12 took over an hour with 3 models
ZOOM_HALF = 160        # px around the projected centre, enlarged 2x for the zoom pass


# a few words for the same thing: the detector reacts to the exact word (the chat model will give such lists)
SYNONYMS = {"bottle": ["bottle", "water bottle", "plastic bottle"], "shoe": ["shoe", "sneaker", "boot"],
            "power outlet": ["power outlet", "wall socket", "electrical outlet"], "fixture": ["fixture", "wall lamp", "switch"],
            "bar stool": ["bar stool", "stool", "chair"], "cabinet": ["cabinet", "cupboard", "drawer unit"],
            "table": ["table", "desk", "counter"], "cup": ["cup", "mug", "coffee cup"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room")
    p.add_argument("--models", nargs="*", help="compare models (at 640 px) instead of input sizes")
    p.add_argument("--conf", type=float, nargs="*", default=[0.1], help="score thresholds to compare")
    p.add_argument("--synonyms", action="store_true", help="also with SYNONYMS as prompts (any of them counts)")
    args = p.parse_args()
    room = Room(args.room)
    s = Session(room.session_path)
    objs = [o for o in json.loads((room.session_path / "scene" / "scene.json").read_text())["objects"]
            if max(o["size"]) < MAX_SIZE]
    print(f"{room}: small objects: " + ", ".join(f"#{o['id']} {o['label']}" for o in objs))
    cases = []  # (object, frame, distance, u, v)
    for o in objs:
        c = np.array([o["center"][0], o["center"][1], (o["z_range"][0] + o["z_range"][1]) / 2])
        per = defaultdict(list)
        for f in s.frames:
            uv, ok = s.project(f, c[None], "rgb")
            if not ok[0]:
                continue
            d = float(np.linalg.norm(c - f.T_map_rgb[:3, 3]))
            band = next((b for b in BANDS if b[0] <= d < b[1]), None)
            if band:
                per[band].append((o, f, d, *uv[0]))
        for band, lst in per.items():
            cases += [lst[i] for i in np.linspace(0, len(lst) - 1, min(PER_BAND, len(lst))).astype(int)]
    print(f"{len(cases)} object views")

    def hit(dets, labels, u, v, dx=0, dy=0, scale=1.0, conf=0.0):
        return any(d["label"] in labels and d["score"] >= conf and d["box"][0] / scale + dx <= u <= d["box"][2] / scale + dx
                   and d["box"][1] / scale + dy <= v <= d["box"][3] / scale + dy for d in dets)

    if args.models:
        settings = [(f"{m} conf {c}" + (" + syn" if syn else ""), m, 640, False, c, syn)
                    for m in args.models for syn in ([False, True] if args.synonyms else [False]) for c in args.conf]
    else:
        settings = [(n, None, sz, z, args.conf[0], False) for n, sz, z in
                    (("640", 640, False), ("960", 960, False), ("1280", 1280, False), ("640 + zoom", 640, True))]
    detectors = {}
    for name, model, imgsz, zoom, conf, syn in settings:
        key = (model, imgsz)
        if key not in detectors:
            detectors[key] = Detector(model, imgsz=imgsz, conf=min(args.conf)) if model else Detector(imgsz=imgsz)
            img0 = cv2.imread(str(cases[0][1].rgb_file))
            detectors[key].set_prompts(["object"])
            for _ in range(2):
                detectors[key].detect(img0)  # warm-up
        d = detectors[key]
        res, ms = defaultdict(list), []
        for o, f, dist, u, v in cases:
            labels = SYNONYMS.get(o["label"], [o["label"]]) if syn else [o["label"]]
            d.set_prompts(labels)
            img = cv2.imread(str(f.rgb_file))
            t0 = time.perf_counter()
            ok = hit(d.detect(img), labels, u, v, conf=conf)
            if zoom and not ok:
                h, w = img.shape[:2]
                x0, y0 = int(max(0, u - ZOOM_HALF)), int(max(0, v - ZOOM_HALF * 0.75))
                x1, y1 = int(min(w, u + ZOOM_HALF)), int(min(h, v + ZOOM_HALF * 0.75))
                crop = cv2.resize(img[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ok = hit(d.detect(crop), labels, u, v, x0, y0, 2.0, conf=conf)
            ms.append((time.perf_counter() - t0) * 1000)
            band = next(b for b in BANDS if b[0] <= dist < b[1])
            res[band].append(ok)
        print(f"{name:>36}: " + "  ".join(f"{b[0]:.1f}-{b[1]:.1f} m {np.mean(res[b]) * 100:3.0f}% ({len(res[b])})"
                                          for b in BANDS if res[b]) + f"   {np.median(ms):.0f} ms/image", flush=True)


if __name__ == "__main__":
    main()
