"""Step 2 of scene understanding: lifts the masks to 3D and merges the views of each object.

    python projects/scene_understanding/build_objects.py 20261005_124116

Needs scene/frames_vlm.jsonl and scene/masks/ (detect_frames.py). Writes
scene/objects.json (one entry per object: VLM labels of all its views, upright box, observations,
best views) and scene/objects_points.npz (points per object, map frame).

- 3D points of a mask: only that keyframe's own depth. Filling gaps from the room mesh (until
  2026-10-06) gave unmeasured objects whatever lay along the ray: a wall TV 3-6 m away, never measured
  (glossy black, far), took the depth of bar stools in front of it (20261005_142856).
- Area of interest: a polygon in scene/area.json ({"polygon": [[x, y], ...]}, map frame) if there is
  one; else the floor reachable from the driven path without crossing walls or no-go lines (runs from
  2026-10-05 on record them); for older runs the driven path's hull plus a margin.
- Merging: each keyframe's masks are matched one-to-one to the objects so far (object projected into
  the frame, 2D IoU and 3D overlap); masks of one frame are never one object. Afterwards, objects with
  a shared label word that were never seen together and lie on each other are joined. Labels differ
  between views ("stool" / "bar stool" / "chair"), so they only decide close calls.
- People and anything seen in fewer than 3 keyframes or from fewer than 2 viewpoints are dropped (people
  walk through, single views are often reflections in glass or VLM mistakes; keyframes taken while the
  robot turns on the spot are one look, not independent confirmations).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402

AREA_MARGIN = 1.2      # m around the driven path's hull: walls are ~1-1.5 m from the path (20261005_124116)
AREA_CELL = 0.05       # m, floor grid of the area of interest
# the reachable floor grown by this much: objects standing at walls or on a no-go line belong to the room
AREA_GROW = 0.3
FLOOR_Z = 0.015        # points below are floor that leaked into a mask (floor at z = 0 after levelling)
VOXEL = 0.04           # component filter of one view
MATCH_VOXEL = 0.08     # an object's final points: cells confirmed by 2+ views (tolerant of few-cm pose errors)
MIN_POINTS = 40
# a view counts only if enough of its mask has measured depth: objects seen through a doorway or glass,
# or beyond the depth camera's ~2.5 m (90 % of its readings, 20261005_124116), cannot be placed from it.
# 0.3, not more: thin stool legs lose their edges to the depth filter; with 0.6 most stools, cabinets and
# doors were lost (19 instead of 45 objects in 20261005_124116, 28 with 0.3)
MIN_COVERAGE = 0.3
# farther views put points on whatever is in front (a far sofa got a stool leg's points); the depth
# camera measures up to ~2.5 m, closer views of the same object exist on a loop through the room
MAX_VIEW_DISTANCE = 4.0
MAX_POINTS_PER_VIEW = 3000
# the colour images are ~0.1 s late against depth and poses (check_camera_timing.py: -0.09 to -0.12 s in
# 20261005_142856 and _160227): while turning, masks landed next to thin stool legs and one wall outlet
# became two objects 19 cm apart. The total correction wanted: reconstructions since 2026-10-06 contain it
# already (cameras.json rgb_time_offset_s), Session only adds what is missing (older reconstructions)
RGB_TIME_OFFSET = -0.1
# independent views of an object: camera positions this far apart, or looking at it from directions this
# different. The TV's 4 views in 20261005_142856 came from 4-9 cm apart (the robot turning on the spot).
VIEWPOINT_DIST = 0.4   # m
VIEWPOINT_ANGLE = 15   # degrees
MIN_VIEWPOINTS = 2
# Seating (bar stools, chairs): one seat = one object. Seen from 0.3 m, rows of identical stools touch in
# the image from almost every direction; matching masks left 7 leg fragments and 5 pairs of stools merged
# among 20 stool objects in 20261006_142432, although the merged ones had the most views (median 20, from
# 184 deg). The seat (its underside) is what the camera measures best: seat centres are the peaks of the
# seating points' density at seat height, and every view goes to the seat its points lie around.
SEATING = {"stool", "chair", "seat"}
SEAT_Z = (0.4, 0.85)      # m: chair seats from ~0.45, bar stool seats ~0.65-0.8
SEAT_CELL = 0.02          # m, top-view grid
SEAT_SMOOTH = 0.06        # m, density smoothing (a seat is ~35-40 cm across)
SEAT_SPACING = 0.3        # m, closest seat centres (neighbouring bar stools ~45 cm apart)
SEAT_PEAK = 0.1           # weakest seat peak, relative to the strongest
SEAT_RADIUS = 0.3         # m: a stool's splayed legs reach ~25 cm from its seat centre
MIN_SEAT_FOOTPRINT = 0.25 # m: smaller "stools" / "chairs" were leg fragments


# ---------- area of interest ----------

def area_polygon(session):
    f = session.path / "scene" / "area.json"
    if f.exists():
        return np.array(json.loads(f.read_text())["polygon"], dtype=float)
    lines = session.nogo_lines()
    if lines is not None:
        return reachable_area(session, lines)
    xy = np.array([fr.T_map_rgb[:2, 3] for fr in session.frames])
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1) * AREA_MARGIN
    pts = (xy[:, None, :] + ring[None]).reshape(-1, 2)  # Minkowski sum: hull of the buffered path
    return pts[ConvexHull(pts).vertices]


def reachable_area(session, lines):
    """Floor reachable from the driven path, walls and no-go lines as barriers, from the mesh: cells with
    surface above 10 cm block, cells with only floor are free. Furniture inside is filled in afterwards."""
    V = np.asarray(o3d.io.read_triangle_mesh(str(session.mesh_file(view=True))).vertices)
    path = np.array([fr.T_map_rgb[:2, 3] for fr in session.frames])
    lo = np.minimum(V[:, :2].min(0), path.min(0)) - 1.0
    shape = tuple(np.ceil((np.maximum(V[:, :2].max(0), path.max(0)) + 1.0 - lo) / AREA_CELL).astype(int))

    def cells(xy):
        return np.clip(np.floor((xy - lo) / AREA_CELL).astype(int), 0, np.array(shape) - 1)

    floor = np.zeros(shape, bool)
    wall = np.zeros(shape, np.uint8)
    floor[tuple(cells(V[V[:, 2] < 0.05, :2]).T)] = True
    wall[tuple(cells(V[V[:, 2] > 0.10, :2]).T)] = 1
    for x1, y1, x2, y2 in lines:
        (i1, j1), (i2, j2) = cells(np.array([[x1, y1], [x2, y2]]))
        cv2.line(wall, (int(j1), int(i1)), (int(j2), int(i2)), 1, thickness=3)
    # where the robot drove is floor even if the mesh has a hole there
    floor[tuple(cells(path).T)] = True
    free = ndimage.binary_closing(floor, iterations=2) & (wall == 0)
    lab, _ = ndimage.label(free)
    seeds = np.unique(lab[tuple(cells(path).T)])
    region = np.isin(lab, seeds[seeds > 0])
    region = ndimage.binary_fill_holes(region)
    region = ndimage.binary_dilation(region, iterations=int(round(AREA_GROW / AREA_CELL)))
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]  # within ~7 cm
    # contour points are (col, row) = (j, i) -> map x from i, y from j
    return np.stack([lo[0] + (c[:, 1] + 0.5) * AREA_CELL, lo[1] + (c[:, 0] + 0.5) * AREA_CELL], 1)


def inside(poly, xy):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(xy[0]), float(xy[1])), False) >= 0


# ---------- one view ----------

class Lifter:
    """Per keyframe: a map point for every colour pixel that the depth camera measured (NaN elsewhere)."""

    def __init__(self, session):
        self.s = session
        w, h = session.K["rgb"][:2]
        self.w, self.h = int(w), int(h)

    def pixel_points(self, frame):
        """HxWx3 map points (NaN = not measured) and an HxW mask of pixels with measured depth."""
        P = np.full((self.h, self.w, 3), np.nan)
        pts = self.s.depth_points(frame)[0]
        uv, ok = self.s.project(frame, pts, "rgb")
        u = np.round(uv[ok, 0]).astype(int)
        v = np.round(uv[ok, 1]).astype(int)
        # the depth camera has a different viewpoint, so a pixel can get several points: the nearest one
        # is what the colour camera sees
        d = np.linalg.norm(pts[ok] - frame.T_map_rgb[:3, 3], axis=1)
        order = np.argsort(-d)
        P[v[order], u[order]] = pts[ok][order]
        return P, np.isfinite(P[..., 0])


def viewpoints(obs, centre):
    """Number of independent viewpoints among an object's views (see VIEWPOINT_DIST / _ANGLE)."""
    reps = []
    for b in obs:
        cam = np.asarray(b["camera"])
        ray = centre - cam
        ray /= max(np.linalg.norm(ray), 1e-6)
        new = True
        for c, r in reps:
            angle = np.degrees(np.arccos(np.clip(ray @ r, -1, 1)))
            if np.linalg.norm(cam[:2] - c[:2]) < VIEWPOINT_DIST and angle < VIEWPOINT_ANGLE:
                new = False
                break
        if new:
            reps.append((cam, ray))
    return len(reps)


def largest_component(pts):
    """Points of the largest 26-connected blob (4 cm voxels): drops background seen through gaps and
    edge pixels that landed on the wall behind the object."""
    if len(pts) < MIN_POINTS:
        return pts
    key = np.floor((pts - pts.min(0)) / VOXEL).astype(int)
    grid = np.zeros(key.max(0) + 1, bool)
    grid[tuple(key.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return pts
    sizes = np.bincount(lab[tuple(key.T)])
    sizes[0] = 0
    return pts[lab[tuple(key.T)] == sizes.argmax()]


def lift_view(lifter, frame, fr, masks_file, area, P, measured):
    """The frame's masks as observations: 3D points, VLM labels, and the mask itself (kept only while
    this frame is associated)."""
    z = np.load(masks_file)
    masks = np.unpackbits(z["masks"], axis=-1, count=int(z["width"])).astype(bool)
    obs = []
    kernel = np.ones((5, 5), np.uint8)
    for k, (m, score) in enumerate(zip(masks, z["scores"])):
        o = fr["objects"][int(z["objects"][k])]
        if o.get("category") == "person" or "person" in o["label"]:
            continue
        # 2 px off the outline: the masks' edges and the colour/depth registration are not pixel exact
        core = cv2.erode(m.astype(np.uint8), kernel) > 0
        pts = P[core]
        if not len(pts) or np.isfinite(pts[:, 0]).mean() < MIN_COVERAGE:
            continue
        pts = pts[np.isfinite(pts[:, 0])]
        pts = pts[pts[:, 2] > FLOOR_Z]
        if len(pts) > MAX_POINTS_PER_VIEW:
            pts = pts[np.random.default_rng(0).choice(len(pts), MAX_POINTS_PER_VIEW, replace=False)]
        pts = largest_component(pts)
        if len(pts) < MIN_POINTS or not inside(area, np.median(pts[:, :2], axis=0)):
            continue
        x1, y1, x2, y2 = o["box"]
        cam = frame.T_map_rgb[:3, 3]
        if np.linalg.norm(np.median(pts, 0) - cam) > MAX_VIEW_DISTANCE:
            continue
        obs.append({"frame": frame.index, "stamp": frame.stamp, "label": o["label"], "category": o.get("category"),
                    "movable": o.get("movable"), "state": o.get("state"), "box": o["box"],
                    "score": float(score), "area_px": int(m.sum()), "measured": float(measured[core].mean()),
                    "distance": float(np.linalg.norm(np.median(pts, 0) - cam)), "camera": cam.tolist(),
                    "touches_border": bool(x1 < 3 or y1 < 3 or x2 > lifter.w - 3 or y2 > lifter.h - 3),
                    "points": pts.astype(np.float32), "mask": m})
    return obs


# ---------- association ----------
# Each keyframe's masks are matched one-to-one to the objects so far: an object is projected into the
# frame (only its parts not hidden behind the frame's surface) and compared with each mask by 2D IoU
# (objects hidden in the frame: share of the mask's 3D points on the object). Two masks of one frame are never the same object;
# that is remembered ("seen together") and blocks later merges. The previous version merged views by
# 3D surface overlap alone: 50 pairs of same-label objects < 40 cm apart remained in 20261005_124116,
# 23 of them never seen in the same keyframe (e.g. one wall outlet as two objects, 2 cm apart).

CLOUD_VOXEL = 0.02     # m, an object's accumulated points
NEAR_3D = 0.05         # m, a mask point "on" an object (pose and depth errors: mesh 90 % < 4.4 cm)
MATCH_SIMILAR = 0.25   # score to join an object with a similar label (shared word or category)
MATCH_OTHER = 0.5      # ... with a different label (the VLM names one thing differently across frames)
# separate masks in this many keyframes = different objects. One is not enough: the VLM sometimes boxes
# a stool's legs and the whole stool separately (83 close same-label pairs "seen together" once, 20261005_124116)
SEEN_TOGETHER = 2
# ... and in this share of the smaller object's keyframes: a fixed count stopped working in long runs
# (16.6 min, 1361 keyframes, 20261007_101108): one stool split in two was seen as two masks in 4 of its
# 22 keyframes (Qwen's occasional double box), real neighbours are two masks in most frames showing both
SEEN_APART_SHARE = 0.4
# Small things (outlets, switches, items; not furniture, see Obj.small_thing): an outline ~6 px wide matches the next keyframe's mask poorly after
# 1-2 cm of pose error (5 views of one outlet became 5 objects, 20261006_142432), so they also match by
# 3D distance. And the poses of two visits can differ by ~20 cm along a blank wall (that outlet: 21 cm
# between the start and the end of the run): same-label small objects never seen as two masks are one
# object up to SMALL_DRIFT apart. Two real outlets that close show up as two masks in the same keyframe.
SMALL = 0.25        # m, footprint of a "small" object
SMALL_MATCH = 0.08  # m between centres: the same small object in another keyframe
SMALL_DRIFT = 0.3   # m
# a fragment lying mostly on a bigger object with the same label word is part of it (stool legs / stool)
PART_OF = 0.6


def words(label):
    """The label's head noun (its last word), as a set: "bar stool" is a stool, "door handle" a handle,
    "cabinet door" a door. Any shared word made a door handle "named alike" to its door, and the handle's
    masks were merged into the door as a fragment (20261007_101108, YOLOE, 2026-10-09)."""
    w = label.replace("-", " ").split()
    return {w[-1]} if w else set()


class Obj:
    def __init__(self, uid):
        self.uid, self.obs, self.together = uid, [], Counter()  # uid -> keyframes seen as separate masks
        self.cloud = np.zeros((0, 3), np.float32)
        self.words, self.cats = set(), set()
        self._tree = None

    def add(self, ob):
        self.obs.append(ob)
        self.words |= words(ob["label"])
        self.cats.add(ob["category"])
        self.add_points(ob["points"])

    def add_points(self, pts):
        c = np.concatenate([self.cloud, pts])
        _, first = np.unique(np.floor(c / CLOUD_VOXEL).astype(np.int64), axis=0, return_index=True)
        self.cloud = c[np.sort(first)]
        self._tree = None

    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.cloud)
        return self._tree

    def centre(self):
        return np.median(self.cloud, axis=0)

    def extent(self):
        """Largest footprint side, m."""
        return float(np.ptp(self.cloud[:, :2], axis=0).max())

    def small_thing(self):
        """A small object that is not furniture (an outlet, a switch, an item): small pieces of furniture
        are fragments (leg pieces, a 4 cm "cabinet"), and the small-object rules made more of them."""
        category = Counter(b["category"] for b in self.obs).most_common(1)[0][0]
        return self.extent() < SMALL and category != "furniture"

    def frames(self):
        return {o["frame"] for o in self.obs}

    def main_words(self):
        """Words of the most common label. Not the union of all labels: merged objects collected the
        words of everything they absorbed ("cabinet" + "cabinet door" -> merged with doors) and snowballed
        into 24 objects for the whole room (20261005_124116)."""
        return words(Counter(o["label"] for o in self.obs).most_common(1)[0][0])


def similar(a_words, a_cats, b_words, b_cats):
    return bool(a_words & b_words) or bool(a_cats & b_cats)


def footprint(s, frame, obj, P):
    """The object's visible part in this frame as a pixel mask (None if it is not in view)."""
    uv, ok = s.project(frame, obj.cloud, "rgb")
    if ok.sum() < 10:
        return None
    u = np.round(uv[ok, 0]).astype(int)
    v = np.round(uv[ok, 1]).astype(int)
    cam = frame.T_map_rgb[:3, 3]
    d = np.linalg.norm(obj.cloud[ok] - cam, axis=1)
    surf = np.linalg.norm(P[v, u] - cam, axis=1)
    # hidden behind the surface this frame sees there (NaN = no surface: counts as visible)
    vis = ~(surf < d - (0.05 + 0.03 * d))
    if vis.sum() < 10:
        return None
    img = np.zeros(P.shape[:2], np.uint8)
    img[v[vis], u[vis]] = 1
    # a 2 cm cell covers ~fx * 0.02 / d pixels: close the gaps between projected points
    r = int(np.clip(s.K["rgb"][2] * CLOUD_VOXEL / np.median(d[vis]), 1, 10))
    return cv2.dilate(img, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def associate(s, frame, obs, objects, P, next_uid):
    cam = frame.T_map_rgb[:3, 3]
    cands = [o for o in objects if np.linalg.norm(o.centre() - cam) < MAX_VIEW_DISTANCE + 1.0]
    feet = [footprint(s, frame, o, P) for o in cands]
    cands = [o for o, f in zip(cands, feet) if f is not None]
    feet = [f for f in feet if f is not None]
    score = np.zeros((len(obs), len(cands)))
    for i, ob in enumerate(obs):
        m = ob["mask"]
        for j, (o, f) in enumerate(zip(cands, feet)):
            # 2D overlap only. The 3D share let objects snowball: a neighbouring stool's legs lie within 5 cm
            # of a grown object, and rows of stools became 1 x 2.5 m objects (20261005_124116); a grown
            # object's footprint has a low IoU with one stool's mask
            sc = np.count_nonzero(m & f) / max(1, np.count_nonzero(m | f))
            if (o.small_thing() and ob["category"] != "furniture"
                    and np.ptp(ob["points"][:, :2], axis=0).max() < SMALL):
                d = np.linalg.norm(np.median(ob["points"], axis=0) - o.centre())
                if d < SMALL_MATCH:
                    sc = max(sc, 0.9 * (1 - d / SMALL_MATCH) + 0.1)
            sim = similar(words(ob["label"]), {ob["category"]}, o.words, o.cats)
            score[i, j] = sc if sc >= (MATCH_SIMILAR if sim else MATCH_OTHER) else 0.0
    seen, taken = [], set()
    if score.size:
        rows, cols = linear_sum_assignment(-score)
        for i, j in zip(rows, cols):
            if score[i, j] > 0:
                cands[j].add(obs[i])
                seen.append(cands[j])
                taken.add(i)
    for i, ob in enumerate(obs):
        if i not in taken:
            o = Obj(next_uid)
            next_uid += 1
            o.add(ob)
            objects.append(o)
            seen.append(o)
    for o in seen:
        o.together.update(p.uid for p in seen if p is not o)
    for ob in obs:
        del ob["mask"]
    return next_uid


def is_seating(obs):
    return bool(words(Counter(b["label"] for b in obs).most_common(1)[0][0]) & SEATING)


def split_seating(groups):
    """groups: lists of views. Seating groups are regrouped by seat (see SEATING); other groups and views
    not around any seat stay as they were."""
    seat = [g for g in groups if is_seating(g)]
    if not seat:
        return groups
    rest = [g for g in groups if not is_seating(g)]
    P = np.concatenate([b["points"] for g in seat for b in g])
    band = P[(P[:, 2] > SEAT_Z[0]) & (P[:, 2] < SEAT_Z[1]), :2]
    if len(band) < MIN_POINTS:
        return groups
    lo = band.min(0) - 0.5
    shape = tuple(np.ceil((band.max(0) + 0.5 - lo) / SEAT_CELL).astype(int))
    grid = np.zeros(shape)
    np.add.at(grid, tuple(np.floor((band - lo) / SEAT_CELL).astype(int).T), 1)
    grid = ndimage.gaussian_filter(grid, SEAT_SMOOTH / SEAT_CELL)
    # a peak must be the highest within SEAT_SPACING all around (round window): the square window was
    # SEAT_SPACING wide, i.e. +-15 cm, and one seat seen on many passes kept two peaks 23-29 cm apart
    # (20261007_101108); a seat is ~35-40 cm across, so real seat centres are further apart than that
    r = int(round(SEAT_SPACING / SEAT_CELL))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    window = xx ** 2 + yy ** 2 <= r ** 2
    peaks = (grid == ndimage.maximum_filter(grid, footprint=window)) & (grid >= SEAT_PEAK * grid.max())
    centres = lo + (np.argwhere(peaks) + 0.5) * SEAT_CELL
    tree = cKDTree(centres)
    by_seat = defaultdict(list)
    leftover = defaultdict(list)
    for gi, g in enumerate(seat):
        for b in g:
            d, k = tree.query(b["points"][:, :2], distance_upper_bound=SEAT_RADIUS)
            near = np.isfinite(d)
            if near.mean() < 0.5:
                leftover[gi].append(b)       # not around a seat (e.g. only legs, seat out of view)
                continue
            best = np.bincount(k[near]).argmax()
            keep = near & (k == best)
            by_seat[best].append(dict(b, points=b["points"][keep]))
    print(f"  seating: {len(seat)} objects -> {len(centres)} seats", flush=True)
    return rest + list(by_seat.values()) + list(leftover.values())


def footprint_size(obs):
    P = np.concatenate([b["points"] for b in obs])
    return float(np.ptp(P[:, :2], axis=0).max())


def merge_pair(a, b):
    """(distance, a, b) if a and b should be one object, else None. The same object seen at different
    times and a part lying on a bigger object need a shared label word; a small thing named differently
    in different views ("power strip" / "device", 5 cm apart, never two masks, 20261007_101108) not."""
    dist = float(np.hypot(*(a.centre() - b.centre())[:2]))
    if dist > 2.0:
        return None
    small, big = (a, b) if len(a.cloud) < len(b.cloud) else (b, a)
    near = big.tree().query(small.cloud, distance_upper_bound=NEAR_3D)[0]
    frac = float(np.isfinite(near).mean())
    diag = min(np.ptp(a.cloud[:, :2], 0).max(), np.ptp(b.cloud[:, :2], 0).max())
    together = a.together[b.uid]
    apart = together >= SEEN_TOGETHER and together >= SEEN_APART_SHARE * min(len(a.frames()), len(b.frames()))
    tolerance = SMALL_DRIFT if a.small_thing() and b.small_thing() else max(0.15, 0.5 * diag)
    named_alike = bool(a.main_words() & b.main_words())
    same = not apart and named_alike and (frac >= 0.3 or dist < tolerance)
    # a part needs the same label word, like the same object seen twice: without it, bottles on a counter
    # and a radiator under a whiteboard (same category, same wall) became "parts" (20261007_101108). A
    # cabinet's door labelled only "door" stays a separate object, which is the lesser error
    part = named_alike and frac >= PART_OF and len(small.frames()) < len(big.frames())
    small_same = (not apart and a.small_thing() and b.small_thing() and dist < SMALL_MATCH
                  and Counter(x["category"] for x in a.obs).most_common(1)[0][0]
                  == Counter(x["category"] for x in b.obs).most_common(1)[0][0])
    return (dist, a, b) if same or part or small_same else None


def merge_close(objects):
    """Joins objects with a shared label word that are one thing seen at different times (never seen
    as separate masks in SEEN_TOGETHER keyframes, on each other in 3D or nearly the same centre), and
    fragments that lie mostly on a bigger such object (parts). Closest pairs first; after a merge only
    the pairs of the merged object are checked again (all pairs each time took minutes for ~350)."""
    objects = list(objects)
    pairs = {}
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            r = merge_pair(a, b)
            if r:
                pairs[(a.uid, b.uid)] = r
    alive = {o.uid: o for o in objects}
    while pairs:
        key = min(pairs, key=lambda k: pairs[k][0])
        _, a, b = pairs.pop(key)
        a.obs += b.obs
        a.words |= b.words
        a.cats |= b.cats
        a.add_points(b.cloud)
        a.together.update(b.together)
        del a.together[a.uid], a.together[b.uid]
        for o in alive.values():
            if b.uid in o.together:
                o.together[a.uid] += o.together.pop(b.uid)
        del alive[b.uid]
        pairs = {k: v for k, v in pairs.items() if a.uid not in k and b.uid not in k}
        for o in alive.values():
            if o is not a:
                r = merge_pair(a, o)
                if r:
                    pairs[(a.uid, o.uid)] = r
    return list(alive.values())


def vote(values, weights):
    c = Counter()
    for v, w in zip(values, weights):
        if v:
            c[v] += w
    return c.most_common(1)[0][0] if c else None


def main_body(pts):
    """The largest connected blob of an object's points (8 cm cells)."""
    if len(pts) < MIN_POINTS:
        return pts
    key = np.floor((pts - pts.min(0)) / MATCH_VOXEL).astype(int)
    grid = np.zeros(key.max(0) + 1, bool)
    grid[tuple(key.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return pts
    ids = lab[tuple(key.T)]
    return pts[ids == np.bincount(ids)[1:].argmax() + 1]


def summarize(i, o):
    obs = o["obs"]
    pts = np.concatenate([b["points"] for b in obs])
    # keep 8 cm cells confirmed by 2+ views when there are 3+ views: drops one-off mask leaks
    if len({b["frame"] for b in obs}) >= 3:
        votes = defaultdict(set)
        keys = [tuple(k) for k in np.floor(pts / MATCH_VOXEL).astype(int)]
        for k, b in zip(keys, np.concatenate([[n] * len(b["points"]) for n, b in enumerate(obs)])):
            votes[k].add(b)
        keep = np.array([len(votes[k]) >= 2 for k in keys])
        if keep.sum() > MIN_POINTS:
            pts = pts[keep]
    # the main body only: a small stray blob far away stretched a 0.8 m cabinet's box to 2.2 m (3 % of its
    # points, 20261007_101108)
    pts = main_body(pts)
    # upright box: furniture stands on the floor, a yaw-only box is what a planner / simulator wants
    (cx, cy), (w, l), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    z0, z1 = np.percentile(pts[:, 2], [1, 99])
    weights = [b["score"] * np.sqrt(b["area_px"]) for b in obs]
    label = vote([b["label"] for b in obs], weights)
    # best views: big, close, the whole object in the image if possible. Only views detected under the
    # object's own name: a door's "best view" was its handle (8 x 30 px, fully in the image, higher score),
    # and size counts more than "not touching the border": seen from 0.3 m, a door close up always touches
    # it (20261007_101108, YOLOE, 2026-10-09)
    own = [b for b in obs if words(b["label"]) == words(label)] or obs
    def view_quality(b):
        return min(b["area_px"] / 20000, 1) * 2 + (not b["touches_border"]) + b["score"] - abs(b["distance"] - 1.2) / 2
    best = sorted(own, key=view_quality, reverse=True)[:3]
    states = sorted({(b["stamp"], b["state"]) for b in obs if b["state"]})
    return {
        "id": i,
        "label": label,
        "labels": dict(Counter(b["label"] for b in obs).most_common()),
        "category": vote([b["category"] for b in obs], weights),
        "movable": vote([b["movable"] for b in obs], weights),
        "state": vote([b["state"] for b in obs], weights),
        "state_seen": [{"stamp": s, "state": v} for s, v in states],
        "center": [round(float(cx), 3), round(float(cy), 3), round(float((z0 + z1) / 2), 3)],
        "size": [round(float(w), 3), round(float(l), 3), round(float(z1 - max(z0, 0)), 3)],
        "yaw_deg": round(float(ang), 1),
        "z_range": [round(float(z0), 3), round(float(z1), 3)],
        "n_views": len({b["frame"] for b in obs}),
        "first_seen": min(b["stamp"] for b in obs),
        "last_seen": max(b["stamp"] for b in obs),
        "best_views": [{"frame": b["frame"], "box": b["box"], "distance": round(b["distance"], 2)} for b in best],
        "views": sorted({b["frame"] for b in obs}),
        "measured_fraction": round(float(np.mean([b["measured"] for b in obs])), 2),
    }, pts


_worker = {}


def lift_frame(args):
    """One keyframe's depth -> points and its masks -> views, in a worker process: independent of the
    other keyframes, so they run in parallel (they took ~2/3 of this step on one core, 2026-10-06)."""
    session, idx, fr, mask_file, area, min_coverage = args
    global MIN_COVERAGE
    MIN_COVERAGE = min_coverage
    if _worker.get("session") != session:
        s = Session(session, rgb_time_offset=RGB_TIME_OFFSET)
        _worker.update(session=session, s=s, lifter=Lifter(s), by_index={f.index: f for f in s.frames})
    frame = _worker["by_index"][idx]
    P, measured = _worker["lifter"].pixel_points(frame)
    obs = lift_view(_worker["lifter"], frame, fr, mask_file, area, P, measured)
    return P.astype(np.float32), obs


def main():
    global MIN_COVERAGE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    # objects seen in only 2 keyframes were mostly leg fragments and VLM one-offs (20261005_124116)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--out", default="objects", help="output name: <out>.json, <out>_points.npz")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE, help="measured share of a mask")
    # test 2026-10-09: is the seating-only rule still needed with YOLOE's instance masks and the pair check?
    ap.add_argument("--no-seat-rule", action="store_true", help="no seat split / seat footprint minimum")
    a = ap.parse_args()
    MIN_COVERAGE = a.min_coverage
    s = Session(a.session, rgb_time_offset=RGB_TIME_OFFSET)
    scene_dir = s.path / "scene"
    frames_vlm = {r["index"]: r for r in map(json.loads, (scene_dir / "frames_vlm.jsonl").read_text().splitlines()) if r}
    area = area_polygon(s)
    by_index = {f.index: f for f in s.frames}
    files = [mf for mf in sorted((scene_dir / "masks").glob("*.npz"), key=lambda p: int(p.stem))
             if int(mf.stem) in frames_vlm and int(mf.stem) in by_index]
    jobs = [(s.name, int(mf.stem), frames_vlm[int(mf.stem)], mf, area, MIN_COVERAGE) for mf in files]
    objects, uid, n_obs = [], 0, 0
    # lifting in parallel, matching in time order; in chunks, so not all keyframes sit in memory at once
    chunk = 32
    with ProcessPoolExecutor(max(1, min(8, (os.cpu_count() or 2) - 1))) as pool:
        for c in range(0, len(jobs), chunk):
            for job, (P, obs) in zip(jobs[c:c + chunk], pool.map(lift_frame, jobs[c:c + chunk])):
                n_obs += len(obs)
                uid = associate(s, by_index[job[1]], obs, objects, P, uid)
            print(f"  {min(c + chunk, len(jobs))} keyframes, {n_obs} views, {len(objects)} objects", flush=True)
    before = len(objects)
    objects = merge_close(objects)
    print(f"  merged {before - len(objects)} objects that were never seen together")
    groups = [o.obs for o in objects] if a.no_seat_rule else split_seating([o.obs for o in objects])

    def keep(obs):
        frames = {b["frame"] for b in obs}
        centre = np.median(np.concatenate([b["points"] for b in obs]), axis=0)
        if len(frames) < a.min_views or viewpoints(obs, centre) < MIN_VIEWPOINTS:
            return False
        return a.no_seat_rule or not (is_seating(obs) and footprint_size(obs) < MIN_SEAT_FOOTPRINT)
    objs = [{"obs": g} for g in groups if keep(g)]
    objs.sort(key=lambda o: -len(o["obs"]))
    observations = [b for o in objs for b in o["obs"]]
    out, points = [], {}
    for i, o in enumerate(objs):
        d, pts = summarize(i, o)
        out.append(d)
        points[str(i)] = pts.astype(np.float32)
    (scene_dir / f"{a.out}.json").write_text(json.dumps({
        "session": s.name, "frame": "map (metres, floor at z = 0)", "area_polygon": area.round(3).tolist(),
        "objects": out}, indent=1))
    np.savez_compressed(scene_dir / f"{a.out}_points.npz", **points)
    print(f"{len(observations)} views -> {len(out)} objects (seen in {a.min_views}+ keyframes, "
          f"{MIN_VIEWPOINTS}+ viewpoints)")
    for d in out[:60]:
        print(f"  #{d['id']:3d} {d['label']:<22} {d['category'] or '':<10} views {d['n_views']:3d}  "
              f"at ({d['center'][0]:.2f}, {d['center'][1]:.2f})  size {d['size'][0]:.2f} x {d['size'][1]:.2f} x {d['size'][2]:.2f}"
              f"  {d['state'] or ''}")


if __name__ == "__main__":
    main()
