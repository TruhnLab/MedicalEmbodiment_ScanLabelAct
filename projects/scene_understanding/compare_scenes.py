"""Compares two labellings of the same room (scene.json + objects_points.npz), e.g. Qwen boxes + SAM against
detect_frames.py's vocabulary + YOLOE (experiment 2026-10-09).

    python projects/scene_understanding/compare_scenes.py 20261007_101108 20261007_101108_yoloe

Objects are matched one-to-one by their points: overlap = geometric mean of the shares of each object's
points within TOUCH of the other's. Lists the matched pairs (labels side by side), the objects only one labelling has,
counts per category and the seating sizes (two stools as one, leg fragments).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import SESSIONS  # noqa: E402

TOUCH = 0.05      # m
MIN_OVERLAP = 0.3
SEATING = ("stool", "chair", "seat")


def load(name):
    sd = SESSIONS / name / "scene"
    objs = json.loads((sd / "scene.json").read_text())["objects"]
    pts = np.load(sd / "objects_points.npz")
    return objs, {o["id"]: pts[str(o["id"])] for o in objs if str(o["id"]) in pts.files}


def share(p, q):
    """Share of p's points within TOUCH of q's."""
    p = p[:: max(1, len(p) // 2000)]
    d, _ = cKDTree(q).query(p, distance_upper_bound=TOUCH)
    return float(np.isfinite(d).mean())


def overlap(p, q):
    """Both ways (geometric mean): a drawer inside a cabinet shares all its points with it, but not the other way."""
    return float(np.sqrt(share(p, q) * share(q, p)))


def describe(o):
    c, s = o["center"], o["size"]
    return f"{o['label'][:24]:<24} ({c[0]:5.2f}, {c[1]:5.2f})  {max(s[:2]):.2f} m"


def seating(objs):
    seats = [o for o in objs if any(w in o["label"] for w in SEATING)]
    big = sum(max(o["size"][:2]) > 0.7 for o in seats)
    small = sum(max(o["size"][:2]) < 0.25 for o in seats)
    return f"{len(seats)} seating objects, {big} over 70 cm, {small} under 25 cm"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A, pa = load(args.a)
    B, pb = load(args.b)
    M = np.zeros((len(A), len(B)))
    for i, o in enumerate(A):
        for j, q in enumerate(B):
            if o["id"] in pa and q["id"] in pb and np.linalg.norm(np.subtract(o["center"][:2], q["center"][:2])) < 3:
                M[i, j] = overlap(pa[o["id"]], pb[q["id"]])
    rows, cols = linear_sum_assignment(-M)
    pairs = [(i, j) for i, j in zip(rows, cols) if M[i, j] >= MIN_OVERLAP]
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    print(f"{args.a}: {len(A)} objects   {args.b}: {len(B)} objects   matched: {len(pairs)}\n")
    print(f"matched ({args.a} | {args.b} | overlap):")
    for i, j in sorted(pairs, key=lambda p: A[p[0]]["label"]):
        print(f"  {describe(A[i])} | {describe(B[j])} | {M[i, j]:.2f}")
    for name, objs, matched in ((args.a, A, ma), (args.b, B, mb)):
        rest = [o for k, o in enumerate(objs) if k not in matched]
        print(f"\nonly in {name} ({len(rest)}):")
        for o in sorted(rest, key=lambda o: o["label"]):
            print(f"  {describe(o)}  views {len(o.get('views', []))}")
    print("\ncategories:")
    ca, cb = Counter(o["category"] for o in A), Counter(o["category"] for o in B)
    for c in sorted(set(ca) | set(cb)):
        print(f"  {c:<11} {ca[c]:3d} {cb[c]:3d}")
    print(f"\nseating {args.a}: {seating(A)}\nseating {args.b}: {seating(B)}")
    print(f"parts (part_of): {sum(o.get('part_of') is not None for o in A)} / {sum(o.get('part_of') is not None for o in B)}")


if __name__ == "__main__":
    main()
