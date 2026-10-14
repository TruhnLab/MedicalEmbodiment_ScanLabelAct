"""Step 4 of scene understanding: Qwen judges close pairs of objects (same object / one part of the
other / different) and merges them. Between build_objects.py / floor_items.py and label_objects.py.

    python projects/scene_understanding/judge_pairs.py 20261007_101108
    python projects/scene_understanding/judge_pairs.py 20261007_101108 --out objects_judged   # compare

Why: the geometric merge rules stop where only the meaning decides. In 20261007_101108 a cabinet and its
door (labelled "door") stayed two objects, and one stool split in two; merging regardless of the label
made bottles "parts" of a counter and a radiator a "part" of a whiteboard.

- Candidates: objects whose 3D points come within CLOSE of each other, or whose centres are less than
  NEAR apart (floor-level candidates from floor_items.py excluded).
- Qwen sees a keyframe with both in it (A green, B red) if there is one, the best view of each, and
  the facts (sizes, heights, distance, how often they were two masks in one keyframe).
- Merged on "same" / "part of" with medium or high confidence. Without a keyframe showing both, "same"
  counts only within SAME_WITHOUT_SHARED: identical stools can't be told apart from separate views.
Writes scene/objects.json + objects_points.npz (or --out) and the log scene/pairs.json.
"""

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landerpi.session import Session  # noqa: E402
import build_objects as bo  # noqa: E402
import progress  # noqa: E402
import vlm  # noqa: E402
from label_objects import crop  # noqa: E402
from schemas import PairAnswer  # noqa: E402

CLOSE = 0.05               # m between the two objects' points: touching
NEAR = 0.4                 # m between centres
SAME_WITHOUT_SHARED = 0.3  # m: "same" without a keyframe showing both only this close
# one object (or a part inside its whole) doesn't grow much when merged; two neighbours side by side do:
# Qwen sometimes called two tightly packed identical stools "the same" (a 1.09 m "stool",
# 20261007_101108). Merged footprint at most GROW x the larger one + GROW_ABS (drift between visits)
GROW, GROW_ABS = 1.3, 0.15
PART_SHARE = 0.5   # an operable part has at most this share of its whole's points (drawer vs counter: ~5 %)
GREEN, RED = (0, 255, 0), (0, 0, 255)
# Qwen's answer to the same pair varied between runs (a cabinet door "part of" its cabinet in one run,
# "different" in the next; 20261007_101108): ask VOTES times, merge only on a majority
VOTES = 3
VOTE_TEMPERATURE = 0.3   # independent answers; at 0 repeated calls tend to repeat each other

PROMPT = """A small ground robot (camera 0.3 m above the floor) mapped a room and detected objects in its
camera images. The object list is for a robot that will later move to and handle things, so every part
that can be handled on its own must stay a separate entry.

The labels are a detector's guesses per image and may be wrong; trust the images.
Object A (green): "{a_label}", per-image labels {a_labels}; {a_w:.2f} x {a_l:.2f} m, from {a_z0:.2f} to {a_z1:.2f} m high.
Object B (red): "{b_label}", per-image labels {b_labels}; {b_w:.2f} x {b_l:.2f} m, from {b_z0:.2f} to {b_z1:.2f} m high.
Their centres are {dist:.2f} m apart; their surfaces come within {gap:.2f} m of each other. The detector saw
them as two separate objects in {together} of the {both} images that show both.
{images}
Answer with one of:
- "same": one physical object detected twice (e.g. the same stool seen from two sides);
- "fragment": the smaller one is only a piece of the other that cannot be used on its own (a leg, a
  seat, a backrest, an armrest, a side panel, a strip or edge of the same surface, a partial detection of
  it); fixed parts of one piece of furniture are fragments, not different objects (a chair's seat and
  backrest were judged "different", 2026-10-09);
- "operable_part": the smaller one belongs to the other but can be used on its own: a drawer, a cabinet
  door, a lid, a handle, a knob, a switch, a tap;
- "different": two physical objects, also if they look alike (e.g. two stools side by side) or one
  stands on, in or next to the other (a bottle on a counter, a bin on a cabinet).
If unsure, answer "different"."""


def box_of(session, frame, pts):
    """The object's box in a keyframe's colour image (its points projected), or None if not in view."""
    uv, ok = session.project(frame, pts[:: max(1, len(pts) // 2000)], "rgb")
    if ok.mean() < 0.8:
        return None
    x1, y1 = np.percentile(uv[ok], 2, axis=0)
    x2, y2 = np.percentile(uv[ok], 98, axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def both_image(session, by_index, a, b, pa, pb):
    """The keyframe showing both best (both boxes in view, as large as possible), with A green and B red."""
    best = None
    frames = sorted(set(a["views"]) & set(b["views"])) or sorted(set(a["views"]) | set(b["views"]))
    for fi in frames[:: max(1, len(frames) // 40)]:
        f = by_index.get(fi)
        if f is None:
            continue
        ba, bb = box_of(session, f, pa), box_of(session, f, pb)
        if ba is None or bb is None:
            continue
        area = min((ba[2] - ba[0]) * (ba[3] - ba[1]), (bb[2] - bb[0]) * (bb[3] - bb[1]))
        if best is None or area > best[0]:
            best = (area, f, ba, bb)
    if best is None:
        return None
    _, f, ba, bb = best
    img = f.rgb()
    for box, col in ((ba, GREEN), (bb, RED)):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 3)
    return img


def candidates(objects, points):
    objs = [o for o in objects if o.get("source", "vlm") != "geometry"]
    trees = {o["id"]: cKDTree(points[str(o["id"])][:, :2][:: max(1, len(points[str(o["id"])]) // 3000)]) for o in objs}
    out = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            dist = float(np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1]))
            if dist > NEAR + max(a["size"][:2]) + max(b["size"][:2]):
                continue
            pb = points[str(b["id"])][:, :2][:: max(1, len(points[str(b["id"])]) // 3000)]
            gap = float(trees[a["id"]].query(pb, k=1)[0].min())
            if gap < CLOSE or dist < NEAR:
                out.append((a, b, dist, gap))
    return out


# The views decide, not one image pair: two objects the detector found as separate masks in many of the images
# that show both are two objects, whatever Qwen says about one image (a stool merged as the "support leg" of
# a whiteboard it was detected apart from in 37 images; neighbouring stools 46 cm apart merged as
# "fragments", 20261007_101108, 2026-10-09). A real fragment (a leg found on its own) is in view in nearly
# every image of its whole but found apart only now and then. Same share as build_objects.py's rule.
VIEW_SAMPLE = 150   # images checked for "both in view" (sampled evenly when more)


def in_view_both(session, by_index, a, b, pa, pb):
    """In how many keyframes (of those where either was detected) both objects are in the image."""
    frames = sorted(set(a["views"]) | set(b["views"]))
    step = max(1, len(frames) // VIEW_SAMPLE)
    hits = sum(box_of(session, by_index[fi], pa) is not None and box_of(session, by_index[fi], pb) is not None
               for fi in frames[::step] if fi in by_index)
    return int(round(hits * step))


def judge(session, by_index, points, a, b, dist, gap):
    pa, pb = points[str(a["id"])], points[str(b["id"])]
    parts, notes = [], []
    img = both_image(session, by_index, a, b, pa, pb)
    if img is not None:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        parts.append(vlm.image_part(enc.tobytes()))
        notes.append(f"Image {len(parts)}: one camera image with A outlined green and B red.")
    for o, col, name in ((a, GREEN, "A"), (b, RED, "B")):
        v = o["best_views"][0]
        c = crop(session, by_index[v["frame"]], v["box"], colour=col)
        ok, enc = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
        parts.append(vlm.image_part(enc.tobytes()))
        notes.append(f"Image {len(parts)}: {name} alone ({'green' if name == 'A' else 'red'} box).")
    together = len(set(a["views"]) & set(b["views"]))
    both = max(in_view_both(session, by_index, a, b, pa, pb), together)
    labels = lambda o: ", ".join(f"{k} x{n}" for k, n in list(o["labels"].items())[:4])
    text = PROMPT.format(
        a_label=a["label"], a_labels=labels(a), a_w=a["size"][0], a_l=a["size"][1], a_z0=a["z_range"][0],
        a_z1=a["z_range"][1], b_label=b["label"], b_labels=labels(b), b_w=b["size"][0], b_l=b["size"][1],
        b_z0=b["z_range"][0], b_z1=b["z_range"][1], dist=dist, gap=gap, together=together, both=both,
        images="\n".join(notes))
    answers = []
    for _ in range(VOTES):
        try:
            answers.append(PairAnswer.model_validate_json(vlm.chat(
                parts + [{"type": "text", "text": text}], max_tokens=300, schema=PairAnswer,
                temperature=VOTE_TEMPERATURE)))
        except ValueError:
            pass  # a cut-off answer counts as no vote
    if not answers:
        answers = [PairAnswer(verdict="different", confidence="low", reason="no valid answer")]
    # Majorities, not single answers (they varied between runs, 20261007_101108). "same" and "fragment"
    # merge; an "operable_part" (drawer, cabinet door) stays its own object, linked to its whole: the
    # robot may have to open it (user, 2026-10-07). Belonging together but split between fragment and
    # operable part: kept separate, the safer choice for handling. Which one is the part: the smaller.
    votes = [x for x in answers if x.confidence != "low"]
    n = lambda *kinds: sum(x.verdict in kinds for x in votes)
    if n("same", "fragment") * 2 > VOTES:
        verdict = "same" if n("same") >= n("fragment") else "fragment"
    elif n("same", "fragment", "operable_part") * 2 > VOTES:
        verdict = "operable_part"
    else:
        verdict = "different"
    merge = verdict in ("same", "fragment")
    ans = next((x for x in votes if x.verdict == verdict), answers[0])
    if verdict == "same" and img is None and dist > SAME_WITHOUT_SHARED:
        merge, verdict = False, "different"
    r = {"a": a["id"], "b": b["id"], "a_label": a["label"], "b_label": b["label"], "dist": round(dist, 2),
         "gap": round(gap, 3), "shared_image": img is not None, "together": together, "both_in_view": both,
         "verdict": verdict, "confidence": ans.confidence, "reason": ans.reason, "merge": merge,
         "votes": [x.verdict for x in answers]}
    if merge and together >= bo.SEEN_TOGETHER and together >= bo.SEEN_APART_SHARE * both:
        r["merge"] = False
        r["blocked"] = f"detected as two objects in {together} of {both} images showing both"
    return r


def merged_object(group, objects, points):
    """One object from a group: the whole (or the most seen) keeps its id, name and best views; the
    points are combined and the box recomputed from the main body."""
    os_ = [objects[i] for i in group]
    keep = max(os_, key=lambda o: (o.get("_whole", 0), len(o["views"])))
    P = bo.main_body(np.concatenate([points[str(o["id"])] for o in os_]))
    (cx, cy), (w, l), ang = cv2.minAreaRect(P[:, :2].astype(np.float32))
    z0, z1 = np.percentile(P[:, 2], [1, 99])
    labels = Counter()
    for o in os_:
        labels.update(o["labels"])
    views = sorted({v for o in os_ for v in o["views"]})
    out = dict(keep)
    out.update({
        "labels": dict(labels.most_common()),
        "state_seen": sorted((s for o in os_ for s in o["state_seen"]), key=lambda s: s["stamp"]),
        "center": [round(float(cx), 3), round(float(cy), 3), round(float((z0 + z1) / 2), 3)],
        "size": [round(float(w), 3), round(float(l), 3), round(float(z1 - max(z0, 0)), 3)],
        "yaw_deg": round(float(ang), 1), "z_range": [round(float(z0), 3), round(float(z1), 3)],
        "n_views": len(views), "views": views,
        "first_seen": min(o["first_seen"] for o in os_), "last_seen": max(o["last_seen"] for o in os_),
        "merged_from": sorted(o["id"] for o in os_),
    })
    out.pop("_whole", None)
    return out, P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--out", default="objects")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    s = Session(a.session, rgb_time_offset=bo.RGB_TIME_OFFSET)
    sd = s.path / "scene"
    d = json.loads((sd / "objects.json").read_text())
    points = dict(np.load(sd / "objects_points.npz"))
    by_index = {f.index: f for f in s.frames}
    pairs = candidates(d["objects"], points)
    print(f"{len(d['objects'])} objects, {len(pairs)} close pairs to judge", flush=True)
    results, lock, t0 = [], threading.Lock(), time.time()

    def one(p):
        r = judge(s, by_index, points, *p)
        with lock:
            results.append(r)
            progress.update(sd, done=len(results), total=len(pairs))
        return r
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, pairs))
    # groups of merged objects; "part of" marks the whole, which keeps its name
    by_id = {o["id"]: o for o in d["objects"]}
    parent = {o["id"]: o["id"] for o in d["objects"]}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # a smaller group never joins a bigger one if one of its members was judged "different" from the
    # bigger group's main object: the answers can contradict each other (a bin "part of" a cabinet door, but
    # "different" from the cabinet it stands on, 20261007_101108). Not between all members: two drawers are
    # "different" from each other but both parts of one counter. Best-supported merges first: "same"
    # before "fragment", then the objects seen most
    different = {frozenset((r["a"], r["b"])) for r in results
                 if not r["merge"] and sum(v == "different" for v in r["votes"]) * 2 > VOTES}
    members = {o["id"]: {o["id"]} for o in d["objects"]}
    order = sorted((r for r in results if r["merge"]),
                   key=lambda r: (r["verdict"] != "same", -(len(by_id[r["a"]]["views"]) + len(by_id[r["b"]]["views"]))))
    for r in order:
        ra, rb = root(r["a"]), root(r["b"])
        if ra == rb:
            continue
        small, big = (ra, rb) if len(members[ra]) < len(members[rb]) or (
            len(members[ra]) == len(members[rb]) and len(by_id[ra]["views"]) < len(by_id[rb]["views"])) else (rb, ra)
        main = max(members[big], key=lambda i: len(by_id[i]["views"]))
        if any(frozenset((x, main)) in different for x in members[small]):
            r["merge"] = False
            r["blocked"] = "judged different from a member of the other group"
            continue
        # the part is the smaller one; the whole keeps its id, name and views for the crops
        side = lambda P: float(max(cv2.minAreaRect(P[:, :2].astype(np.float32))[1]))
        before = max(side(bo.main_body(points[str(i)])) for i in members[small] | members[big])
        after = side(bo.main_body(np.concatenate([points[str(i)] for i in members[small] | members[big]])))
        if after > GROW * before + GROW_ABS:
            r["merge"] = False
            r["blocked"] = f"merged footprint {after:.2f} m vs {before:.2f} m: two objects side by side"
            continue
        whole = max((r["a"], r["b"]), key=lambda i: len(points[str(i)])) if r["verdict"] == "fragment" else None
        if whole is not None:
            by_id[whole]["_whole"] = by_id[whole].get("_whole", 0) + 1
        parent[small] = big
        members[big] |= members.pop(small)
    groups = {}
    for i in parent:
        groups.setdefault(root(i), []).append(i)
    objects, pts_out, new_id = [], {}, {}
    for g in groups.values():
        o, P = merged_object(g, by_id, points) if len(g) > 1 else (by_id[g[0]], points[str(g[0])])
        o.pop("_whole", None)
        objects.append(o)
        pts_out[str(o["id"])] = P.astype(np.float32)
        # the merged object keeps its whole's id, not necessarily the group's root (KeyError, 2026-10-09)
        new_id.update({i: o["id"] for i in g})
    # operable parts: linked to their whole (both after the merges above); the smaller one is the part
    for o in objects:
        o.pop("part_of", None)
    by_new = {o["id"]: o for o in objects}
    for r in results:
        # only to a whole it touches: a cabinet handle was linked to the stool in front of it (shown as
        # stool + handle, Qwen answered "a handle, part of the cabinet"; 20261007_101108)
        if r["verdict"] != "operable_part" or r["gap"] > CLOSE:
            continue
        pa, pb = new_id[r["a"]], new_id[r["b"]]
        if pa == pb:
            continue
        part, whole = sorted((pa, pb), key=lambda i: len(pts_out[str(i)]))
        # a part is clearly smaller than its whole: a stool was linked as "part" of its neighbour
        if len(pts_out[str(part)]) > PART_SHARE * len(pts_out[str(whole)]):
            continue
        old = by_new[part].get("part_of")
        if old is None or len(pts_out[str(whole)]) > len(pts_out[str(old)]):
            by_new[part]["part_of"] = whole
    # no chains through a part: a part's whole is never itself a part of the part
    for o in objects:
        if o.get("part_of") is not None and by_new[o["part_of"]].get("part_of") == o["id"]:
            del by_new[o["part_of"]]["part_of"]
    objects.sort(key=lambda o: o["id"])
    d["objects"] = objects
    (sd / f"{a.out}.json").write_text(json.dumps(d, indent=1))
    np.savez_compressed(sd / f"{a.out}_points.npz", **pts_out)
    (sd / "pairs.json").write_text(json.dumps(results, indent=1))
    n = sum(r["merge"] for r in results)
    parts = sum(o.get("part_of") is not None for o in objects)
    print(f"{len(results)} pairs judged in {time.time() - t0:.0f} s, {n} merged -> {len(objects)} objects, "
          f"{parts} operable parts linked to their whole")
    for r in sorted(results, key=lambda r: (not r["merge"], r["a"])):
        tag = 'MERGE' if r['merge'] else 'PART ' if r['verdict'] == 'operable_part' else '     '
        print(f"  {tag} #{r['a']} {r['a_label']} / #{r['b']} {r['b_label']}: "
              f"{r['verdict']} ({r['confidence']}, {'both in one image' if r['shared_image'] else 'separate views'}) "
              f"- {r['reason']}")


if __name__ == "__main__":
    main()
