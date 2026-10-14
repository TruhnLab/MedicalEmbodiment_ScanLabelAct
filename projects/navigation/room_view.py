"""The room for the chat's language model: walls with what is on them (text), a top-view picture (sent once per
conversation), and which wall is ahead of / behind / beside the robot (with every message). The user asked for
it (2026-10-09): with the object list alone the model had trouble with "the wall behind you", "the wall with
the two doors"; it picks a point and drives there with drive_to_point.

    walls = room_walls(room)           # [{"name": "W1", "a": (x, y), "b": (x, y), "wall": True, "objects": [...]}]
    png = room_picture(room)           # bytes, cached in data/rooms/<room>/chat_map.png
    around(room, pose)                 # {"ahead": "W2, 3.1 m", "behind": ..., "left": ..., "right": ...}

Walls are the sides of the scan's room outline (scene.json area_polygon, simplified); a side counts as a wall
if the lidar map has obstacles along most of it, otherwise it is an open side (the room continues there).
"""

import json
import math

import cv2
import numpy as np

OUTLINE_M = 0.3        # outline simplification: the room 20261007_101108's 31 corners -> its 6 sides
WALL_NEAR_M = 0.3      # a side point counts as wall if a map obstacle is this close
WALL_SHARE = 0.5       # ... for at least this share of the side's points
ON_WALL_M = 0.7        # an object whose centre is this close to a wall is "on" it (doors, radiators, outlets)
PICTURE_PX = 1000      # the picture's longer side
MARGIN_M = 0.8
STOOLS = {"bar stool", "stool", "chair"}  # many of them: only their [id] in the picture
CLEAR_M = 0.45         # a point to stand at: this far from the lidar map's obstacles (the robot's half width + margin)
# The model picked points as if the walls ran along x and y (room 20261007_101108 is turned ~28 deg in the map):
# "the wall in front of you" became a point behind the cabinets (test_chat_room.py, 2026-10-09). So the text gives
# ready points to stand at: in front of each wall's middle, and in front of the wall ahead / behind / beside.

_walls = {}
_clearance = {}


def _clear(room):
    """Distance (m) of each map cell to the nearest obstacle."""
    if room.name not in _clearance:
        free = (room.grid < 65).astype(np.uint8)
        _clearance[room.name] = cv2.distanceTransform(free, cv2.DIST_L2, 5) * room.resolution
    return _clearance[room.name]


def stand_point(room, start, toward, max_m=2.5):
    """From start (at a wall) step toward a direction until the robot fits there (CLEAR_M); None if not."""
    clear = _clear(room)
    d = np.asarray(toward, float) / max(np.linalg.norm(toward), 1e-9)
    for t in np.arange(0.2, max_m, 0.05):
        p = np.asarray(start, float) + t * d
        c, r = room.world_to_cell(p)
        if 0 <= r < clear.shape[0] and 0 <= c < clear.shape[1] and clear[r, c] >= CLEAR_M:
            return p
    return None


def _outline(room):
    scene = json.loads((room.session_path / "scene" / "scene.json").read_text())
    poly = np.asarray(scene.get("area_polygon") or [], np.float32)
    if len(poly) < 3:
        return None
    return cv2.approxPolyDP(poly.reshape(-1, 1, 2), OUTLINE_M, True).reshape(-1, 2).astype(float)


def _seg_dist(p, a, b):
    ab, ap = b - a, p - a
    t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab))), t


def room_walls(room):
    """The room outline's sides, walls numbered W1.. around the room, open sides named "open side"."""
    if room.name in _walls:
        return _walls[room.name]
    poly = _outline(room)
    out = []
    if poly is not None:
        occ = (room.grid >= 65).astype(np.uint8)
        k = 2 * int(WALL_NEAR_M / room.resolution) + 1
        near = cv2.dilate(occ, np.ones((k, k), np.uint8)) > 0
        n = 0
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            length = float(np.linalg.norm(b - a))
            ts = np.linspace(0, 1, max(2, int(length / 0.1)))
            c, r = room.world_to_cell(a + ts[:, None] * (b - a)).T
            inside = (r >= 0) & (r < near.shape[0]) & (c >= 0) & (c < near.shape[1])
            wall = inside.any() and near[r[inside], c[inside]].mean() >= WALL_SHARE
            if wall:
                n += 1
            out.append({"name": f"W{n}" if wall else "open side", "a": a, "b": b, "length": length, "wall": bool(wall),
                        "objects": []})
        centre = poly.mean(0)
        for w in out:  # where to stand in front of the side's middle (inward, clear of furniture)
            mid, e = (w["a"] + w["b"]) / 2, w["b"] - w["a"]
            nrm = np.array([-e[1], e[0]])
            if nrm @ (centre - mid) < 0:
                nrm = -nrm
            w["stand"] = stand_point(room, mid, nrm)
        for o in room.objects():
            p = np.array(o["center"][:2])
            best = min(((_seg_dist(p, w["a"], w["b"]), w) for w in out if w["wall"]), key=lambda x: x[0][0],
                       default=None)
            if best is not None and best[0][0] <= ON_WALL_M and 0.02 < best[0][1] < 0.98:
                best[1]["objects"].append(o)
    _walls[room.name] = out
    return out


def _nogo_along(room, w):
    """No-go lines that run along a side (both ends near it)."""
    return [i + 1 for i, (x0, y0, x1, y1) in enumerate(room.nogo_lines())
            if max(_seg_dist(np.array(p, float), w["a"], w["b"])[0] for p in ((x0, y0), (x1, y1))) < ON_WALL_M]


def walls_text(room):
    """For the system message. No-go lines are named as such, not as walls (user, 2026-10-09: the room may
    continue beyond one, the robot just must not cross it)."""
    lines = []
    for w in room_walls(room):
        things = ", ".join(f"{o['label']} [{o['id']}]" for o in w["objects"] if o["label"] not in STOOLS)
        nogo = _nogo_along(room, w)
        lines.append(f"- {w['name']}: from ({w['a'][0]:.1f}, {w['a'][1]:.1f}) to ({w['b'][0]:.1f}, {w['b'][1]:.1f}), "
                     f"{w['length']:.1f} m" + (f"; along it: {things}" if things else "")
                     + ("" if w["wall"] else " (no wall: the room continues there)")
                     + (f"; no-go line {', '.join(f'N{i}' for i in nogo)} along it" if nogo else "")
                     + (f"; to stand in front of its middle: ({w['stand'][0]:.1f}, {w['stand'][1]:.1f})"
                        if w.get("stand") is not None else ""))
    for i, (x0, y0, x1, y1) in enumerate(room.nogo_lines()):
        lines.append(f"- no-go line N{i + 1}: from ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f}), set by the user: the "
                     f"robot never crosses it (not a wall: the room may continue beyond it)")
    return "\n".join(lines)


def around(room, pose):
    """Which side of the room is ahead of, behind, left and right of the robot, and how far."""
    walls = room_walls(room)
    if pose is None or not walls:
        return None
    p = np.array(pose[:2], float)
    out = {}
    for name, turn in (("ahead", 0), ("left", 90), ("behind", 180), ("right", -90)):
        d = np.array([math.cos(pose[2] + math.radians(turn)), math.sin(pose[2] + math.radians(turn))])
        hits = []
        for w in walls:
            e = w["b"] - w["a"]
            m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
            if abs(np.linalg.det(m)) < 1e-9:
                continue
            t, s = np.linalg.solve(m, w["a"] - p)
            if t > 0 and 0 <= s <= 1:
                hits.append((t, w["name"]))
        if hits:
            t, nm = min(hits)
            stand = stand_point(room, p + t * d, -d, max_m=t)
            out[name] = f"{nm}, {t:.1f} m" + (f", stand in front of it at ({stand[0]:.1f}, {stand[1]:.1f})"
                                              if stand is not None else "")
    return out


def room_picture(room):
    """The top view as PNG bytes (cached next to the room; made again if the room's files are newer)."""
    path = room.path / "chat_map.png"
    sources = [room.session_path / "scene" / "scene.json", room.path / "map.pgm", room.path / "room.json"]
    if path.exists() and all(path.stat().st_mtime >= s.stat().st_mtime for s in sources if s.exists()):
        return path.read_bytes()
    walls = room_walls(room)
    pts = np.array([w["a"] for w in walls]) if walls else np.array([o["center"][:2] for o in room.objects()])
    lo, hi = pts.min(0) - MARGIN_M, pts.max(0) + MARGIN_M
    s = PICTURE_PX / float(max(hi - lo))
    W, H = int((hi[0] - lo[0]) * s), int((hi[1] - lo[1]) * s)

    def px(x, y):
        return int(round((x - lo[0]) * s)), int(round((hi[1] - y) * s))  # y up

    # the lidar map underneath: obstacles dark, unknown light grey
    img = np.full((H, W, 3), 255, np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    c, r = room.world_to_cell(np.stack([lo[0] + xs.ravel() / s, hi[1] - ys.ravel() / s], 1)).T
    ok = (r >= 0) & (r < room.grid.shape[0]) & (c >= 0) & (c < room.grid.shape[1])
    g = np.full(len(r), -1)
    g[ok] = room.grid[r[ok], c[ok]]
    g = g.reshape(H, W)
    img[g < 0] = (232, 232, 232)
    img[g >= 65] = (90, 90, 90)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 1 m grid with its coordinates, for picking points
    for x in range(math.ceil(lo[0]), math.floor(hi[0]) + 1):
        cv2.line(img, px(x, lo[1]), px(x, hi[1]), (215, 225, 240), 1)
        cv2.putText(img, f"x={x}", (px(x, 0)[0] + 2, H - 6), font, 0.4, (120, 120, 160), 1, cv2.LINE_AA)
    for y in range(math.ceil(lo[1]), math.floor(hi[1]) + 1):
        cv2.line(img, px(lo[0], y), px(hi[0], y), (215, 225, 240), 1)
        cv2.putText(img, f"y={y}", (3, px(0, y)[1] - 3), font, 0.4, (120, 120, 160), 1, cv2.LINE_AA)
    # walls and open sides, named as in the text
    centre = np.mean([w["a"] for w in walls], 0) if walls else None
    for w in walls:
        a, b = px(*w["a"]), px(*w["b"])
        if w["wall"]:
            cv2.line(img, a, b, (40, 110, 230), 4, cv2.LINE_AA)
        else:
            for t in np.arange(0, 1, 0.06):
                cv2.line(img, px(*(w["a"] + t * (w["b"] - w["a"]))), px(*(w["a"] + min(1, t + 0.03) * (w["b"] - w["a"]))),
                         (150, 150, 150), 2, cv2.LINE_AA)
        mid = (w["a"] + w["b"]) / 2
        mid = mid + 0.35 * (centre - mid) / max(1e-6, np.linalg.norm(centre - mid))  # the label inside the room
        _label(img, w["name"], px(*mid), (40, 110, 230) if w["wall"] else (120, 120, 120), 0.6, 2)
    # no-go lines (the robot never crosses them; not walls), named as in the text
    for i, (x0, y0, x1, y1) in enumerate(room.nogo_lines()):
        for t in np.arange(0, 1, 0.04):
            a, b = np.array([x0, y0]), np.array([x1, y1])
            cv2.line(img, px(*(a + t * (b - a))), px(*(a + min(1, t + 0.025) * (b - a))), (60, 60, 220), 3, cv2.LINE_AA)
        _label(img, f"N{i + 1} no-go", px((x0 + x1) / 2, (y0 + y1) / 2), (60, 60, 220), 0.45, 1)
    # objects: their boxes and [id] (with the label, except for the many stools)
    for o in room.objects():
        cx, cy = o["center"][:2]
        L, Wd = o["size"][0] / 2, o["size"][1] / 2
        yaw = math.radians(o.get("yaw_deg", 0.0))
        u, v = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        box = [px(*(np.array([cx, cy]) + i * L * u + j * Wd * v)) for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        stool = o["label"] in STOOLS
        col = (170, 170, 170) if stool else (30, 140, 60)
        cv2.polylines(img, [np.array(box, np.int32)], True, col, 1 if stool else 2, cv2.LINE_AA)
        text = f"[{o['id']}]" if stool else f"[{o['id']}] {o['label']}"
        _label(img, text, px(cx, cy), (110, 110, 110) if stool else (20, 100, 40), 0.38 if stool else 0.45, 1)
    _label(img, "top view, map frame in metres: x to the right, y up", (W // 2, 14), (60, 60, 60), 0.45, 1)
    ok, buf = cv2.imencode(".png", img)
    path.write_bytes(buf.tobytes())
    return buf.tobytes()


def _label(img, text, centre, color, scale, thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = int(centre[0] - tw / 2), int(centre[1] + th / 2)
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + 3), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
