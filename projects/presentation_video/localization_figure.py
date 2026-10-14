"""A slide of how the robot finds out where it is in a known room (navigation's localize.py), from real data: one turn
on the spot, the search through the room's map, the result. Three panels with a short label each (minimal text, the
user's wish for the talk; the slide says the rest), also saved one by one.

    python projects/presentation_video/localization_figure.py                     # kitchen, a run of 2026-10-06
    python projects/presentation_video/localization_figure.py --room <room> --run <session>

The turn is the start of a recorded run (its scans stitched with its odometry, as the navigator does after ⟳ Localize),
matched against a room made from another run: the result is a real one, not the room's own scan. Output:
data/demo/localization.png (1920 x 1080) and localization_1/2/3.png.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from style import ACCENT, BORDER, F_BOLD, F_SMALL, F_TEXT, F_TITLE, H, MAP_FREE, MAP_OCC, MUTED, SURFACE, TEXT, W, wrap

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "projects" / "navigation")]
from landerpi.room import Room  # noqa: E402
import localize as L  # noqa: E402  (navigation's localization, used as it is)
from test_localize import Track, lidar_in_base, read, stamp, yaw_of  # noqa: E402

OUT = REPO / "data" / "demo"
PANEL_W, PANEL_H, GAP = 586, 640, 41
PANEL_Y = (H - PANEL_H) // 2 + 30  # no title: centred, room for the labels above
ROBOT = (239, 68, 68)
HEAT = (255, 140, 0)
MARGIN_M = 0.6


def start_turn(run):
    """The run's scans of its first full turn, stitched with odometry (points in the frame of the turn's end)."""
    odom = read(run, "odom")
    odo = Track([stamp(m) for m in odom], [(m["pose"]["pose"]["position"]["x"], m["pose"]["pose"]["position"]["y"],
                                           yaw_of(m["pose"]["pose"]["orientation"])) for m in odom])
    lidar = lidar_in_base(run)
    scans = [(stamp(m), L.scan_points(m, lidar)) for m in read(run, "scan")]
    scans = [(t, p) for t, p in scans if odo.t[0] <= t <= odo.t[-1] and len(p) > 20]
    turned, i = 0.0, 0
    while i + 1 < len(scans) and turned < 2 * math.pi:
        i += 1
        turned += abs(L.wrap(odo.at(scans[i][0])[2] - odo.at(scans[i - 1][0])[2]))
    pts = L.stitch([(p, odo.at(t)) for t, p in scans[:i + 1]])
    return pts, scans[i][0] - scans[0][0], i + 1


def coarse_scores(loc, pts):
    """The global search's first stage (localize.Localizer.global_search): every position 20 cm apart and heading
    3 deg apart; per position the best heading's score."""
    coarse_pts = L.spread(L.thin(pts, L.COARSE_STEP / 2), L.MAX_COARSE_POINTS)
    rows, cols = np.nonzero(loc.free)
    xy = np.stack([cols, rows], 1) * loc.res + loc.origin + loc.res / 2
    xy = xy[np.unique(np.floor(xy / L.COARSE_STEP).astype(np.int64), axis=0, return_index=True)[1]]
    yaws = np.arange(-math.pi, math.pi, L.COARSE_YAW)
    poses = np.concatenate([np.column_stack([xy, np.full(len(xy), a)]) for a in yaws])
    sc = loc.search(loc.coarse, coarse_pts, poses).reshape(len(yaws), len(xy)).max(0)
    return xy, sc, len(yaws)


class MapView:
    """The room's map cropped to its area, scaled into a panel; world <-> panel pixels."""

    def __init__(self, room, area, scale=None):
        g, res, (ox, oy) = room.grid, room.resolution, room.origin
        poly = np.asarray(area, float)[:, :2]
        lo, hi = poly.min(0) - MARGIN_M, poly.max(0) + MARGIN_M
        self.scale = scale or min(PANEL_W / (hi[0] - lo[0]), PANEL_H / (hi[1] - lo[1]))  # px per m
        self.lo, self.hi = lo, hi
        self.size = (int((hi[0] - lo[0]) * self.scale), int((hi[1] - lo[1]) * self.scale))
        img = np.full((g.shape[0], g.shape[1], 3), SURFACE, np.uint8)
        img[g == 0] = MAP_FREE
        img[g >= 65] = MAP_OCC
        # world -> map cells for the crop, then flip (row 0 = largest y)
        c0, r0 = ((lo - (ox, oy)) / res).astype(int)
        c1, r1 = ((hi - (ox, oy)) / res).astype(int)
        crop = img[max(r0, 0):r1, max(c0, 0):c1][::-1]
        self.base = cv2.resize(crop, self.size, interpolation=cv2.INTER_NEAREST)

    def px(self, x, y):
        return (x - self.lo[0]) * self.scale, (self.hi[1] - y) * self.scale


def legend(d, x, y, text):
    """A small note on a white backing (over map or points)."""
    w = d.textlength(text, font=F_SMALL)
    d.rectangle([x - 6, y - 4, x + w + 6, y + 28], fill=SURFACE)
    d.text((x, y), text, font=F_SMALL, fill=MUTED)


def robot(d, px, yaw, scale, colour=ROBOT):
    """The chassis (0.31 x 0.25 m, at least 14 px) with its heading."""
    s = max(scale, 14 / 0.31)
    c, si = math.cos(yaw), math.sin(yaw)
    corners = [(px[0] + (c * dx - si * dy) * s, px[1] - (si * dx + c * dy) * s)
               for dx, dy in ((-0.155, -0.125), (0.155, -0.125), (0.155, 0.125), (-0.155, 0.125))]
    d.polygon(corners, fill=colour, outline=SURFACE)
    d.line([px, (px[0] + c * 0.3 * s, px[1] - si * 0.3 * s)], fill=SURFACE, width=3)


def panel_frame(img, d, k, x, title, caption, body):
    """Number, title above; the panel picture; the caption below."""
    d.ellipse([x, PANEL_Y - 62, x + 40, PANEL_Y - 22], fill=ACCENT)
    d.text((x + 20 - d.textlength(str(k), font=F_BOLD) / 2, PANEL_Y - 58), str(k), font=F_BOLD, fill=SURFACE)
    d.text((x + 54, PANEL_Y - 58), title, font=F_BOLD, fill=TEXT)
    ox = x + (PANEL_W - body.width) // 2
    img.paste(body, (ox, PANEL_Y))
    d.rectangle([ox - 1, PANEL_Y - 1, ox + body.width, PANEL_Y + body.height], outline=BORDER, width=2)
    y = PANEL_Y + PANEL_H + 24
    for line in wrap(d, caption, F_TEXT, PANEL_W) if caption else []:
        d.text((x, y), line, font=F_TEXT, fill=TEXT)
        y += 34


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default="20261007_101108", help="the kitchen")
    ap.add_argument("--run", default="20261006_164000", help="another run in it (its start turn)")
    a = ap.parse_args()
    room = Room(a.room)
    area = json.loads((room.session_path / "scene" / "scene.json").read_text()).get("area_polygon")
    loc = L.Localizer(room.grid, room.resolution, room.origin, area=area)
    pts, turn_s, n_scans = start_turn(a.run)
    pose, info = loc.global_search(pts)
    xy, sc, n_yaw = coarse_scores(loc, pts)
    view = MapView(room, area)
    print(f"turn {turn_s:.0f} s, {n_scans} scans, {len(pts)} points; localized at {np.round(pose, 2)}, score "
          f"{info['score']:.2f}, next best elsewhere {info['runner_up']:.2f}; {len(xy)} positions x {n_yaw} headings")

    # 1: the turn's points around the robot (same scale as the map panels)
    p1 = Image.new("RGB", view.base.shape[1::-1], SURFACE)  # the map panels' size and scale
    d1 = ImageDraw.Draw(p1)
    mid = (np.percentile(pts, 1, axis=0) + np.percentile(pts, 99, axis=0)) / 2  # the cloud's middle in the picture
    c = (p1.width / 2 - mid[0] * view.scale, p1.height / 2 + mid[1] * view.scale)  # where the robot is
    for m in range(-20, 21):  # 1 m grid
        d1.line([(c[0] + m * view.scale, 0), (c[0] + m * view.scale, p1.height)], fill=MAP_FREE)
        d1.line([(0, c[1] + m * view.scale), (p1.width, c[1] + m * view.scale)], fill=MAP_FREE)
    for x, y in pts:
        q = (c[0] + x * view.scale, c[1] - y * view.scale)
        d1.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=ACCENT)
    robot(d1, c, 0.0, view.scale)
    rr = 34  # the turn: a circular arrow around the robot (no words)
    d1.arc([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], 20, 320, fill=ROBOT, width=4)
    a_ = math.radians(320)
    tip = (c[0] + rr * math.cos(a_), c[1] + rr * math.sin(a_))
    d1.polygon([(tip[0] + 12, tip[1] + 2), (tip[0] - 4, tip[1] - 12), (tip[0] - 6, tip[1] + 10)], fill=ROBOT)

    # 2: the room's map and how well the turn fits at each position
    p2 = Image.fromarray(view.base).convert("RGBA")
    heat = Image.new("RGBA", p2.size, (0, 0, 0, 0))
    dh = ImageDraw.Draw(heat)
    r = L.COARSE_STEP * view.scale / 2
    lo_s, hi_s = np.percentile(sc, 40), sc.max()
    for (x, y), s in zip(xy, sc):
        v = float(np.clip((s - lo_s) / (hi_s - lo_s), 0, 1))
        if v > 0:
            q = view.px(x, y)
            dh.rectangle([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=HEAT + (int(25 + 230 * v ** 1.5),))
    p2 = Image.alpha_composite(p2, heat).convert("RGB")
    d2 = ImageDraw.Draw(p2)
    q = view.px(*pose[:2])
    d2.ellipse([q[0] - 14, q[1] - 14, q[0] + 14, q[1] + 14], outline=ROBOT, width=5)

    # 3: the turn's points placed at the result: they lie on the walls
    p3 = Image.fromarray(view.base).convert("RGB")
    d3 = ImageDraw.Draw(p3)
    T = L.pose_matrix(pose)
    for x, y in pts @ T[:2, :2].T + T[:2, 2]:
        q = view.px(x, y)
        d3.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=ACCENT)
    robot(d3, view.px(*pose[:2]), pose[2], view.scale)

    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    xs = [40 + k * (PANEL_W + GAP) for k in range(3)]
    panels = [(p1, "Turn on the spot", ""), (p2, "Search the map", ""), (p3, "Located", "")]
    for k, (body, title, caption) in enumerate(panels, 1):
        panel_frame(img, d, k, xs[k - 1], title, caption, body)
    for k in range(2):  # arrows between the panels
        x = xs[k] + PANEL_W + GAP / 2
        y = PANEL_Y + PANEL_H / 2
        d.polygon([(x - 12, y - 16), (x - 12, y + 16), (x + 12, y)], fill=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / "localization.png")
    for k, (body, _, _) in enumerate(panels, 1):
        body.save(OUT / f"localization_{k}.png")
    print(f"wrote {OUT / 'localization.png'} and localization_1/2/3.png")


if __name__ == "__main__":
    sys.exit(main())
