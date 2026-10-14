"""How fast and how well open-vocabulary detectors find a room's objects by name, on the laptop's CPU.

    python projects/navigation/test_detector.py 20261007_101108                       # default models
    python projects/navigation/test_detector.py 20261007_101108 --models yoloe-11s-seg.pt yolov8s-worldv2.pt

Test set: every object's best views from the labelling (scene.json best_views: keyframe + the VLM's box,
taken as the reference). Each model gets the room's labels as prompts and each view; an object counts as found
in a view if a detection with its label overlaps the reference box (IoU >= IOU) with at least the score
threshold. Reported per model: time per image (after a warm-up), the share of views found at two score
thresholds, false hits (a label detected in a view of another object, at the higher threshold), per label.
"""

import argparse
import csv
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

IOU = 0.3
THRESHOLDS = (0.10, 0.25)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room")
    p.add_argument("--models", nargs="*", default=["yoloe-11s-seg.pt", "yolov8s-worldv2.pt"])
    args = p.parse_args()
    room = Room(args.room)
    s = room.session_path
    objs = json.loads((s / "scene" / "scene.json").read_text())["objects"]
    with open(s / "keyframes.csv") as f:
        rgb = {int(r["index"]): s / r["rgb"] for r in csv.DictReader(f)}
    views = [(o, v["frame"], v["box"]) for o in objs for v in o.get("best_views", []) if v["frame"] in rgb]
    labels = sorted({o["label"] for o in objs})
    print(f"{room}: {len(views)} views of {len(objs)} objects, prompts: {labels}")
    images = {}
    for _, f, _ in views:
        if f not in images:
            images[f] = cv2.imread(str(rgb[f]))

    for name in args.models:
        t0 = time.time()
        d = Detector(name)
        d.set_prompts(labels)
        d.detect(next(iter(images.values())))  # warm-up (first call builds the graph)
        print(f"\n{name}: loaded in {time.time() - t0:.1f} s")
        found = {t: defaultdict(list) for t in THRESHOLDS}
        false_hits, ms, cache = 0, [], {}
        for o, f, ref in views:
            if f not in cache:
                cache[f] = d.detect(images[f])
                ms.append(d.last_ms)
            dets = cache[f]
            for t in THRESHOLDS:
                found[t][o["label"]].append(any(x["label"] == o["label"] and x["score"] >= t and iou(x["box"], ref) >= IOU
                                                for x in dets))
        # false hits: confident detections overlapping no reference box of their label in that frame
        refs = defaultdict(list)
        for o, f, ref in views:
            refs[(f, o["label"])].append(ref)
        for f, dets in cache.items():
            for x in dets:
                if x["score"] >= THRESHOLDS[1] and not any(iou(x["box"], r) >= IOU for r in refs.get((f, x["label"]), [])):
                    false_hits += 1
        print(f"  {np.median(ms):.0f} ms per image (median, 640 px), {1000 / np.median(ms):.1f} images/s")
        for t in THRESHOLDS:
            allv = [v for l in found[t].values() for v in l]
            print(f"  score >= {t:.2f}: found in {np.mean(allv) * 100:.0f} % of the views")
        print(f"  detections >= {THRESHOLDS[1]:.2f} with no matching reference: {false_hits} in {len(cache)} images "
              "(partly real objects the labelling's box didn't cover)")
        print("  per label (>= 0.10): " + ", ".join(f"{l} {np.mean(v) * 100:.0f}% ({len(v)})"
                                                    for l, v in sorted(found[THRESHOLDS[0]].items())))


if __name__ == "__main__":
    main()
