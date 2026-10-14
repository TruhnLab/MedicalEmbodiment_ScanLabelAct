"""Finding things live: detections (detector.py) turned into 3D positions with the depth image, a watcher that decides
"found", and search areas from what the user says ("between the radiator and the chair"). ROS-free: the
navigator (navigator.py) feeds it, test_positions.py checks it on recordings.

    pts = mask_points(mask, depth_mm, K_depth, T_map_depth, K_rgb, T_map_rgb)   # the object's depth points (map)
    w = Watcher(); w.start(["pencil"], area=search_area(objs, [4, 12], "between"))
    w.add(sightings, now)        # [{"label", "score", "position", "size", "box"}] of one image
    w.found                      # {label: {...}} once followed in CONFIRM_VIEWS images and checked

Found = a box followed in CONFIRM_VIEWS images (see the watcher below), confirmed by the image check.

Sightings are observations, not facts (user, 2026-10-09): "sure" ones have a depth position that rests on
something the room's scan knows (scan_cells.py); they go on the map, can stop a drive and be approached.
"Vague" ones (no usable depth, or a place floating in the air) keep only where they were seen from, the
direction and an estimate along the line of sight: the chat gets them as text, to go and look. Sightings not
found again at a visit are marked; searches without a find are remembered too. All of it lasts as long as the
chat (forget()).
"""

import math
import time

import cv2
import numpy as np

MIN_DEPTH, MAX_DEPTH = 0.25, 3.0   # m: the gripper fills the image below 0.25 m; beyond 3 m depth is noisy
MIN_POINTS = 15                    # depth points inside a mask for a position
SCORE_MIN = 0.05                   # test_far.py (2026-10-08): small objects at 0.7-1 m found in 33 % of views at
                                   # 0.05 (with synonyms) vs 25 % at 0.10; prompts for absent things gave a box in
                                   # 1.4 % of frames at 0.05, and a find needs a second sighting at the same place
ERODE_SHARE = 0.15                 # the mask's edge (this share of its size) is left out: there depth pixels mix
                                   # the item with what is behind or in front of it
GROUP_M = 0.05                     # depth points in bins of 5 cm from the camera; the nearest bin with
GROUP_SHARE = 0.2                  # this share of the largest bin's points starts the item's group,
GROUP_DEPTH_M = 0.20               # which reaches this deep (a cup or a shoe; larger things are only placed)
CONFIRM_VIEWS = 2                  # images a box must be followed in before it counts (one stray false hit
                                   # does not stop the robot: the detector has false hits, test_detector.py)
RECHECK_GROWTH = 2.0               # a checked candidate (fits or not) is checked again if seen this much larger,
MAX_CHECKS = 3                     # checks per sighting in all (from far, closer, close)
ADOPT_MAX = 8                      # earlier checked candidates re-judged for a new search (one or two requests)
MIN_CHECK_PX = 18                  # a box smaller than this is not sent yet (nothing to judge: wait until larger)
AREA_MARGIN = 0.30                 # m: sightings this far outside the search area still count
STEP = 2                           # every 2nd depth pixel


def depth_to_map(depth_mm, K, T_map_cam):
    """Depth image -> Nx3 points in the map (valid range only)."""
    fx, fy, cx, cy = K
    z = depth_mm[::STEP, ::STEP].astype(np.float32) / 1000.0
    v, u = np.nonzero((z > MIN_DEPTH) & (z < MAX_DEPTH))
    zz = z[v, u]
    pts = np.stack([(u * STEP - cx) * zz / fx, (v * STEP - cy) * zz / fy, zz], 1)
    return pts @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]


def mask_points(mask, depth_mm, K_depth, T_map_depth, K_rgb, T_map_rgb):
    """The depth points that fall into a colour-image mask (HxW bool), in the map frame. Each camera with
    its own pose at its own time: the colour images are ~0.1 s late against depth
    (measured with scene_understanding/check_camera_timing.py), the poses of both come from their own stamps."""
    pts = depth_to_map(depth_mm, K_depth, T_map_depth)
    if not len(pts):
        return pts
    R, t = T_map_rgb[:3, :3], T_map_rgb[:3, 3]
    cam = (pts - t) @ R  # map -> colour camera (inverse of a rigid transform)
    fx, fy, cx, cy = K_rgb
    front = cam[:, 2] > 0.05
    u = np.full(len(cam), -1, int)
    v = np.full(len(cam), -1, int)
    u[front] = np.round(cam[front, 0] * fx / cam[front, 2] + cx).astype(int)
    v[front] = np.round(cam[front, 1] * fy / cam[front, 2] + cy).astype(int)
    h, w = mask.shape
    ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ok[ok] = mask[v[ok], u[ok]]
    return pts[ok]


def place_dist(a, b):
    """Distance of two positions (x, y, height above the floor) in 3D."""
    return float(np.linalg.norm(np.asarray(a[:3], float) - np.asarray(b[:3], float)))


def inner(mask):
    """The mask without its edge (ERODE_SHARE of its size), unless too little would remain."""
    n = int(mask.sum())
    k = int(round(ERODE_SHARE * math.sqrt(max(n, 1)) / 2))
    if k < 1:
        return mask
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    return m if m.sum() >= 0.25 * n else mask


def nearest_group(points, camera_xyz):
    """The item's own depth points: the nearest solid group seen from the camera, not the median of everything
    in the mask (with background in it, the median lay between the item and the wall behind: cups placed in
    mid-air, real run 2026-10-09)."""
    if len(points) < MIN_POINTS:
        return points
    d = np.linalg.norm(points - np.asarray(camera_xyz), axis=1)
    bins = np.floor((d - d.min()) / GROUP_M).astype(int)
    counts = np.bincount(bins)
    first = int(np.argmax(counts >= max(5, GROUP_SHARE * counts.max())))
    keep = (bins >= first - 1) & (bins * GROUP_M <= first * GROUP_M + GROUP_DEPTH_M)
    return points[keep] if keep.sum() >= MIN_POINTS else points


def locate(points, floor_z=0.0):
    """Position (median) and size (5-95 % extent) of an object's points; None if too few."""
    if len(points) < MIN_POINTS:
        return None
    lo, hi = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
    p = np.median(points, axis=0)
    return {"position": [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2] - floor_z), 3)],
            "size": [round(float(v), 3) for v in hi - lo], "points": int(len(points))}


# ---------- search areas from words ----------

def _box_corners(o):
    yaw = math.radians(o["yaw_deg"])
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    c = np.array(o["center"][:2])
    L, W = o["size"][0] / 2, o["size"][1] / 2
    return np.array([c + a * L * u + b * W * v for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def search_area(objects, anchor_ids, relation, robot_xy=None):
    """A polygon (Nx2, map) where to look. relation:
      "between" (2 anchors): the strip between them, BETWEEN_WIDTH wide;
      "near" (1+ anchors): around them, NEAR_M beyond their boxes;
      "in_front_of" (1 anchor): FRONT_M deep in front of the box's face nearest to the robot;
      "here": around the robot (robot_xy), HERE_M."""
    objs = {o["id"]: o for o in objects}
    anchors = [objs[i] for i in anchor_ids if i in objs]
    if relation == "here" or not anchors:
        if robot_xy is None:
            raise ValueError("no anchor objects and no robot position")
        return _circle(np.array(robot_xy), HERE_M)
    if relation == "between" and len(anchors) >= 2:
        a, b = np.array(anchors[0]["center"][:2]), np.array(anchors[1]["center"][:2])
        d = b - a
        n = np.linalg.norm(d)
        if n < 0.1:
            return _circle(a, NEAR_M)
        u, v = d / n, np.array([-d[1], d[0]]) / n
        hw = BETWEEN_WIDTH / 2
        return np.array([a + v * hw, b + v * hw, b - v * hw, a - v * hw])
    if relation == "in_front_of":
        o = anchors[0]
        corners = _box_corners(o)
        c = np.array(o["center"][:2])
        ref = np.array(robot_xy) if robot_xy is not None else c + 1.0
        # the face whose outward normal points most towards the robot
        faces = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        best = max(faces, key=lambda f: np.dot((f[0] + f[1]) / 2 - c, ref - c))
        n = (best[0] + best[1]) / 2 - c
        n /= np.linalg.norm(n) + 1e-9
        return np.array([best[0], best[1], best[1] + n * FRONT_M, best[0] + n * FRONT_M])
    # "near" (and anything else): the anchors' boxes grown by NEAR_M, as one hull
    pts = np.concatenate([_box_corners(o) for o in anchors])
    grown = np.concatenate([pts + NEAR_M * np.array([math.cos(a), math.sin(a)])
                            for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    return cv2.convexHull(grown.astype(np.float32)).reshape(-1, 2).astype(float)


BETWEEN_WIDTH = 1.5
NEAR_M = 1.0
FRONT_M = 1.2
HERE_M = 1.5


def _circle(c, r, n=16):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def in_area(xy, area, margin=AREA_MARGIN):
    if area is None:
        return True
    d = cv2.pointPolygonTest(np.asarray(area, np.float32).reshape(-1, 1, 2), (float(xy[0]), float(xy[1])), True)
    return d >= -margin


# ---------- the watcher ----------
#
# Each thing in the camera picture is followed by its box from image to image: the box overlapping it most in the
# next image (after moving the boxes by how far the robot turned meanwhile). With the camera still that is
# unambiguous, also for touching blocks. A box followed for CONFIRM_VIEWS images becomes a remembered sighting
# (numbered, with its 3D place), gets one image check (verifier.py), and keeps its label while it is followed.
# Lost and seen again: a new sighting with a new check. Nothing is merged or deleted by place (user, 2026-10-09:
# "keep it much simpler"; 3D places of small things wobble by 2-5 cm, things lie close together, scenes change),
# so the memory may hold one thing twice, which is fine for hints. It lasts as long as the chat (forget()).

TRACK_IOU = 0.3                    # a box overlapping a followed box this much is the same thing
RELINK_M = 0.10                    # a newly followed box is the remembered sighting within this, if only one is
                                   # (lost and found again while the robot drives: the list grew by dozens per
                                   # thing and the fire extinguisher was checked 15 times, real run 2026-10-12)
TRACK_LOST_S = 1.0                 # a followed box not seen again within this is lost (~5 images)
VIEWS_KEPT = 8                     # the latest views of a followed box: its place is their median


def iou(a, b):
    """Overlap of two image boxes [x0, y0, x1, y1] (intersection over union)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _verdict_text(e):
    v, what = e.get("verdict"), e.get("verdict_what") or ""
    return {"yes": f" (checked: {what})", "no": f", REJECTED by the image check: {what}",
            "pending": " (image check running)"}.get(v, "")


class Watcher:
    """What to look for, where, what is followed in the picture, and what was found."""

    def __init__(self):
        self.prompts, self.area, self.since = [], None, None
        self.tracks = []    # followed boxes: {"label", "box", "t", "frames", "views", "entry"}
        self.found = {}     # label -> its confirmed sighting inside the search area (stops the robot)
        # every confirmed sighting of something looked for, inside the area or not, numbered for the chat ("go back
        # to the cup you saw, #2: that was the one I meant"; user, 2026-10-08: a cup outside the area was dropped)
        self.seen = []      # {"id", "label", "position", "score", "views", "time", "in_area", "verdict", ...}
        self.next_id = 1
        self.new = []       # labels found since the last take_new()
        # what the last search was for (kept after it ended): the map shows only its sightings, the others stay
        # remembered for a later search for them (user, 2026-10-09: the map got busy with old items)
        self.objective = []
        self.searches = []  # {"item", "where", "result", "time"}
        # a second opinion (verifier.py) before a candidate counts: sightings get "verdict" pending / yes / no /
        # unchecked; set by live.py (off in the simulation: no images)
        self.verify = False
        self.to_check = []  # sightings waiting to be sent to the verifier
        self.recheck = []   # confirmed ones to be looked at once more, closer (they stay confirmed meanwhile)
        self.stood_still = None  # (stamp, stamp) -> whether the robot stayed put between them (live.py sets it)

    @property
    def active(self):
        return bool(self.prompts)

    def start(self, prompts, area=None):
        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.area = None if area is None else np.asarray(area, float)
        self.since = time.time()
        self.objective = list(self.prompts)
        for p in self.prompts:
            self.found.pop(p, None)  # a new search for it: its earlier sightings count against the new area,
        for e in self.seen:
            if e["label"] in self.prompts and e.get("position") is not None:
                e["in_area"] = bool(in_area(e["position"][:2], self.area))
        # ... and a confirmed one in it is the find at once (real run 2026-10-09 16:24: the red block was
        # confirmed just after a search failed; the chat searched again, which dropped it, twice)
        for e in sorted(self.seen, key=lambda e: e["time"]):
            if e["label"] in self.prompts and self.counts(e):
                self.found[e["label"]] = e

    def counts(self, e):
        """Whether a sighting is a find: sure place, in the area, not missed at a visit, and checked (or unchecked
        when there is no check: the simulation, or the gateway did not answer)."""
        ok = ("yes", "unchecked") if self.verify else ("yes", "unchecked", None)
        return e.get("quality") == "sure" and e.get("in_area") and not e.get("missing_since") \
            and e.get("verdict") in ok

    def current(self):
        """The sure sightings of the current objective that were not missed at a visit (what the map shows)."""
        return [e for e in self.seen if e["label"] in self.objective and e.get("quality") == "sure"
                and not e.get("missing_since") and e.get("verdict") != "no"]

    def mark_missing(self, entry, now=None):
        """Not seen at its place when the camera looked there: it may have been moved (live.seen_empty, the final
        look after an approach)."""
        entry["missing_since"] = now or time.time()
        if self.found.get(entry["label"]) is entry:
            self.found.pop(entry["label"])

    def searched(self, item, where, result, now=None):
        """A search's outcome, also without a find ("searched near [22]: nothing")."""
        self.searches.append({"item": item, "where": where, "result": result, "time": now or time.time()})
        del self.searches[:-20]

    def memory(self, now=None, labels=6, per_label=3):
        """The conversation's memory as short text for the chat (newest items first); "" if nothing. One thing
        seen twice (followed, lost, seen again) is listed once: the newest at about the same place and verdict."""
        now = now or time.time()

        def age(t):
            m = (now - t) / 60
            return "just now" if m < 1 else f"{m:.0f} min ago"
        latest = {}
        for e in self.seen:
            latest[e["label"]] = max(latest.get(e["label"], 0), e["time"])
        for s in self.searches:
            latest[s["item"]] = max(latest.get(s["item"], 0), s["time"])
        out = []
        for label in sorted(latest, key=lambda k: -latest[k])[:labels]:
            parts, listed = [], []
            es = sorted((e for e in self.seen if e["label"] == label), key=lambda e: -e["time"])
            for e in es:
                p = e.get("position")
                if p is not None and any(v == e.get("verdict") and place_dist(p, q) < 0.15 for q, v in listed):
                    continue
                if len(listed) >= per_label:
                    break
                if p is not None:
                    listed.append((p, e.get("verdict")))
                if e.get("quality") == "sure":
                    parts.append(f"#{e['id']} at ({p[0]:.1f}, {p[1]:.1f}) {age(e['time'])}" + _verdict_text(e)
                                 + (f", NOT there when checked {age(e['missing_since'])}" if e.get("missing_since")
                                    else ""))
                elif e.get("verdict") != "no" and e.get("from") is not None:
                    parts.append(f"#{e['id']} vague, only a direction: seen from ({e['from'][0]:.1f}, "
                                 f"{e['from'][1]:.1f}) towards {e['bearing_deg']:.0f} deg (map)"
                                 + (f", roughly at ({p[0]:.1f}, {p[1]:.1f})" if p is not None else "")
                                 + f" {age(e['time'])}" + _verdict_text(e))
            for s in [s for s in self.searches if s["item"] == label][-2:]:
                parts.append(f"searched {s['where']} {age(s['time'])}: {s['result']}")
            out.append(f"{label}: " + "; ".join(parts))
        return " | ".join(out)

    def stop(self):
        self.prompts, self.area = [], None

    # ---------- following boxes ----------

    def add(self, sightings, now, labels=None, shift=0.0):
        """One image's sightings ({"label", "score", "box", and "position" or the line of sight}). labels: the
        words followed (default: what is looked for); shift: px the picture moved sideways since the previous
        image (the robot turned). Sets s["_entry"] on the boxes that belong to a sighting. Returns labels
        newly found."""
        labels = set(self.prompts if labels is None else labels)
        self.tracks = [t for t in self.tracks if now - t["t"] <= TRACK_LOST_S]
        here = [s for s in sightings if s["label"] in labels and s["score"] >= SCORE_MIN and s.get("box") is not None]
        moved = lambda b: [b[0] + shift, b[1], b[2] + shift, b[3]]
        pairs = sorted(((iou(moved(t["box"]), s["box"]), ti, si) for ti, t in enumerate(self.tracks)
                        for si, s in enumerate(here) if t["label"] == s["label"]), key=lambda p: -p[0])
        used_t, used_s = set(), set()
        for o, ti, si in pairs:  # the best overlaps first, each box and each followed box once
            if o < TRACK_IOU:
                break
            if ti not in used_t and si not in used_s:
                used_t.add(ti)
                used_s.add(si)
                self.follow(self.tracks[ti], here[si], now)
        for si, s in enumerate(here):
            if si not in used_s:
                t = {"label": s["label"], "box": s["box"], "t": now, "frames": 0, "views": [], "entry": None}
                self.tracks.append(t)
                self.follow(t, s, now)
        new = []
        for s in here:
            e = s.get("_entry")
            if e is None or self.needs_check(e, s):
                continue
            if e["label"] in self.prompts and e["label"] not in self.found and self.counts(e):
                self.found[e["label"]] = e
                new.append(e["label"])
        self.new += new
        return new

    def follow(self, t, s, now):
        """One more view of a followed box; after CONFIRM_VIEWS views it is a remembered sighting."""
        t.update(box=s["box"], t=now, frames=t["frames"] + 1)
        t["views"] = (t["views"] + [s])[-VIEWS_KEPT:]
        if t["entry"] is None and t["frames"] >= CONFIRM_VIEWS:
            t["entry"] = self.relink(t, s)
            if t["entry"] is None:
                t["entry"] = {"id": self.next_id, "label": t["label"], "time": now, "views": 0}
                self.next_id += 1
                self.seen.append(t["entry"])
        e = t["entry"]
        if e is None:
            return
        s["_entry"] = e
        views = t["views"]
        best = max(views, key=lambda v: v["score"])  # its crop goes to the image check
        sure = [v for v in views if v.get("quality", "sure") == "sure" and v.get("position") is not None]
        e.update(time=now, views=e.get("views", 0) + 1, score=round(max(e.get("score", 0), best["score"]), 3),
                 word=best.get("word"), crop=best.get("crop") or e.get("crop"), stamp=s.get("stamp"))
        e.pop("missing_since", None)  # seen (again)
        if sure:
            pos = [round(float(v), 3) for v in np.median([v["position"] for v in sure], axis=0)]
            e.update(position=pos, size=sure[-1].get("size"), quality="sure", placed=sure[-1].get("placed"),
                     in_area=bool(in_area(pos[:2], self.area)))
        elif e.get("quality") != "sure":  # no usable depth so far: where it was seen from, the direction
            last = views[-1]
            about = last.get("about")
            e.update(quality="vague", position=about, **{k: last.get(k) for k in ("from", "bearing_deg", "about")},
                     in_area=bool(about is not None and in_area(about[:2], self.area)))

    def relink(self, t, s):
        """The remembered sighting a newly followed box continues: the only one of its label within RELINK_M that no
        other followed box holds (touching blocks: several, so none). It keeps its check result if the robot moved
        while the box was lost (the motion lost it); if the robot stood still, something in the scene happened (a
        hand, a swap): it is checked again."""
        p = s.get("position") if s.get("quality", "sure") == "sure" else None
        if p is None:
            return None
        held = {id(x["entry"]) for x in self.tracks if x is not t and x["entry"] is not None}
        near = [e for e in self.seen if e["label"] == t["label"] and e.get("position") is not None
                and e.get("quality") == "sure" and id(e) not in held and place_dist(e["position"], p) < RELINK_M]
        if len(near) != 1:
            return None
        e = near[0]
        if e.get("verdict") not in (None, "pending") and self.stood_still is not None \
                and self.stood_still(e.get("stamp"), s.get("stamp")):
            e.update(verdict=None, checks=0)  # checked again (needs_check)
        e.pop("missing_since", None)
        return e

    def needs_check(self, entry, s):
        """Queue the sighting for the verifier if it was never checked, or is now seen RECHECK_GROWTH larger than
        at its last check (at most MAX_CHECKS in all). Returns True while it waits for a verdict (no find yet)."""
        if not self.verify or not entry.get("crop"):
            return False
        h = (s["box"][3] - s["box"][1]) if s.get("box") else 0
        w = (s["box"][2] - s["box"][0]) if s.get("box") else 0
        v, n, h0 = entry.get("verdict"), entry.get("checks", 0), entry.get("checked_h", h + 1)
        if v is None and s.get("box") and min(w, h) < MIN_CHECK_PX:
            return True  # too small to judge yet: no find and no label; sent once it is MIN_CHECK_PX
        if v is None or (v in ("no", "yes") and n < MAX_CHECKS and h >= RECHECK_GROWTH * h0):
            # also a "fits" once more when seen twice as large (approaching it: a judgement from far may be wrong;
            # user, 2026-10-09); a still picture never grows, so it is never checked again
            if v == "yes":  # it stays the find while the closer look runs
                entry.update(checks=n + 1, checked_h=h)
                self.recheck.append(entry)
                return False
            entry.update(verdict="pending", checks=n + 1, checked_h=h)
            self.to_check.append(entry)
            return True
        return v == "pending"

    def set_verdict(self, entry, fits, what, now=None):
        """The verifier's answer (fits None: not checked). Returns True if the sighting is now a find."""
        entry["verdict"] = "unchecked" if fits is None else ("yes" if fits else "no")
        entry["verdict_what"] = what
        if fits is False and self.found.get(entry["label"]) is entry:
            self.found.pop(entry["label"])
        if entry["label"] in self.prompts and entry["label"] not in self.found and self.counts(entry):
            self.found[entry["label"]] = entry
            self.new.append(entry["label"])
            return True
        return False

    def adopt_earlier(self, now=None):
        """A new search may be for something already seen: the green search's "rejected: a blue block" is the
        blue search's find (real run 2026-10-09). The newest ADOPT_MAX checked sightings of other searches (with
        their crops) become sightings of this one, to be judged against its description. Returns them (queued
        in to_check)."""
        if not self.verify:
            return []
        olds = sorted((e for e in self.seen if e["label"] not in self.prompts and e.get("crop")
                       and e.get("quality") == "sure" and e.get("verdict") in ("yes", "no")
                       and not e.get("missing_since")), key=lambda e: -e["time"])[:ADOPT_MAX]
        out = []
        for label in self.prompts:
            for e in olds:
                new = {k: v for k, v in e.items() if k not in ("verdict", "verdict_what", "checks", "checked_h", "id")}
                new.update(label=label, id=self.next_id, verdict="pending", checks=1, earlier=e["id"],
                           in_area=bool(in_area(e["position"][:2], self.area)))
                self.next_id += 1
                self.seen.append(new)
                out.append(new)
        self.to_check += out
        return out

    def take_to_check(self):
        out, self.to_check, self.recheck = self.to_check + self.recheck, [], []
        return out

    def lookup(self, item=None, seen_id=None):
        """A sighting by its number, or for an item its find in the search area (else its latest sighting)."""
        if seen_id is not None:
            return next((e for e in self.seen if e["id"] == int(seen_id)), None)
        if item:
            item = item.strip().lower()
            return self.found.get(item) or max((e for e in self.seen if e["label"] == item
                                                and e.get("quality") == "sure" and not e.get("missing_since")
                                                and e.get("verdict") != "no"),
                                               key=lambda e: e["time"], default=None)
        return None

    def forget(self):
        """The chat was cleared or another room: nothing seen before applies."""
        self.found.clear()
        self.seen.clear()
        self.tracks = []
        self.searches = []
        self.to_check = []
        self.recheck = []
        self.objective = []

    def take_new(self):
        new, self.new = self.new, []
        return new
