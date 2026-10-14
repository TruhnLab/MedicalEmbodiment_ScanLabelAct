"""Simulated exploration on a saved SLAM map (laptop, no ROS, no robot).

    python offboard_ros/test_explore.py [data/maps/<map>.yaml] [--start x y] [--minutes 15]

The saved map is the "real world" (unknown counts as solid); the explorer starts with an empty
map that a simulated lidar (same blind rear sector as the robot's) fills in as it drives. The robot
follows the commanded velocities exactly. Writes <map>_explore.png: path, coverage, final map.
Checks: never closer than ROBOT_RADIUS to an obstacle, never driving towards the blind sector.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import explore_core as ec  # noqa: E402

DT = 0.1
LIDAR_RANGE = 8.0


def load_world(yaml_path):
    meta = {}
    for line in Path(yaml_path).read_text().splitlines():  # map_saver's simple "key: value" format
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    meta["resolution"] = float(meta["resolution"])
    meta["origin"] = [float(v) for v in meta["origin"].strip("[]").split(",")]
    img = cv2.imread(str(Path(yaml_path).parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    img = np.flipud(img)  # image row 0 is the top (max y); grid row 0 is min y
    cells = np.full(img.shape, ec.UNKNOWN, np.uint8)
    cells[img >= 250] = ec.FREE
    cells[img <= 10] = ec.OCC
    return ec.Grid(cells, meta["resolution"], meta["origin"][0], meta["origin"][1])


def lidar(world, known, pose):
    """Casts rays in the true world; marks what it sees in `known` (the explorer's map) and
    returns the hit points in the robot frame (like /scan). All rays at once (numpy)."""
    x, y, yaw = pose
    lo, hi = ec.LIDAR_SECTOR
    ang = np.arange(lo, hi, math.radians(1.0))
    d = np.arange(world.res * 0.5, LIDAR_RANGE, world.res * 0.5)
    px = x + np.cos(yaw + ang)[:, None] * d[None, :]
    py = y + np.sin(yaw + ang)[:, None] * d[None, :]
    r = np.floor((py - world.origin_y) / world.res).astype(int)
    c = np.floor((px - world.origin_x) / world.res).astype(int)
    h, w = world.cells.shape
    inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    rc, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    solid = (world.cells[rc, cc] != ec.FREE) | ~inside
    first = np.where(solid.any(1), solid.argmax(1), len(d))  # index of the first hit per ray
    steps = np.arange(len(d))[None, :]
    free_mask = (steps < first[:, None]) & inside
    fr, fc = rc[free_mask], cc[free_mask]
    keep = known[fr, fc] != ec.OCC
    known[fr[keep], fc[keep]] = ec.FREE
    hit = (first < len(d))
    hi_idx = np.nonzero(hit)[0]
    hr, hc = rc[hi_idx, first[hit]], cc[hi_idx, first[hit]]
    ok = inside[hi_idx, first[hit]]
    known[hr[ok], hc[ok]] = ec.OCC
    dist = d[first[hit]][ok]
    a = ang[hit][ok]
    return np.stack([dist * np.cos(a), dist * np.sin(a)], 1) if len(dist) else np.zeros((0, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    maps = sorted(Path("data/maps").glob("*.yaml"), key=lambda f: f.with_suffix(".pgm").stat().st_size)
    p.add_argument("map", nargs="?", default=str(maps[-1]) if maps else None)
    p.add_argument("--start", type=float, nargs=2, help="start position (m); default: a free cell near the middle")
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--mode", default=ec.MODE, choices=list(ec.MODES))
    p.add_argument("--wall", type=float, nargs=4, action="append", default=[],
                   help="no-go line x1 y1 x2 y2 (m); the robot must never cross it")
    p.add_argument("--wall-at", type=float, default=0.0,
                   help="draw the no-go lines only after this many seconds (like drawing them mid-run)")
    p.add_argument("--out", default=None, help="picture file (default: <map>_explore.png)")
    p.add_argument("--slip", type=float, nargs=3, action="append", default=[], metavar=("X", "Y", "R"),
                   help="floor transition (circle, m): sideways motion slips there, forward only from 0.15 m/s")
    p.add_argument("--events", action="store_true", help="print the explorer's events (goals, give-ups)")
    a = p.parse_args()
    world = load_world(a.map)
    free = np.argwhere((world.cells == ec.FREE) & (ndi_clearance(world) > 0.4))
    if a.start:
        pose = [a.start[0], a.start[1], 0.0]
    else:
        r, c = free[len(free) // 2]
        pose = [*world.to_xy(r, c), 0.0]
    known = np.full(world.cells.shape, ec.UNKNOWN, np.uint8)
    ex = ec.Explorer()
    ex.command(f"start {a.mode}")
    walls_drawn = False
    trail, min_clear, rear_moves, blocked, plan_times, crossings = [], math.inf, 0, 0, [], 0
    clear_world = ndi_clearance(world)
    t, t_map = 0.0, -10.0
    while t < a.minutes * 60:
        scan = lidar(world, known, pose)
        if t - t_map >= 2.0:  # slam_toolbox publishes the map every 2 s
            ex.set_map(ec.Grid(known.copy(), world.res, world.origin_x, world.origin_y))
            t_map = t
        if a.wall and not walls_drawn and t >= a.wall_at:
            ex.command("walls " + json.dumps(a.wall))
            walls_drawn, drawn_at = True, (t, tuple(round(v, 2) for v in pose[:2]))
        t0 = time.perf_counter()
        vx, vy, wz = ex.step(t, tuple(pose), scan)
        plan_times.append(time.perf_counter() - t0)
        if ex.status.state != "exploring":
            break
        if vx or vy:
            rel = math.atan2(vy, vx)
            if not (ec.LIDAR_SECTOR[0] <= rel <= ec.LIDAR_SECTOR[1]):
                rear_moves += 1
        blocked += ex.status.blocked
        while ex.events:
            text = ex.events.pop(0)
            if a.events:
                print(f"  {t:6.1f} s  ({pose[0]:5.2f}, {pose[1]:5.2f})  {text}")
        if any(math.hypot(pose[0] - sx, pose[1] - sy) < sr for sx, sy, sr in a.slip):
            vy = 0.0                            # rollers slip at the edge
            vx = vx if vx >= 0.15 else 0.0      # only a forward push gets over it
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose[0] += (c * vx - s * vy) * DT
        pose[1] += (s * vx + c * vy) * DT
        pose[2] += wz * DT
        r, cc = world.to_cell(pose[0], pose[1])
        min_clear = min(min_clear, clear_world[r, cc])
        if trail and walls_drawn:
            n = sum(crosses(trail[-1], (pose[0], pose[1]), w) for w in a.wall)
            if n:
                print(f"  crossed a no-go line at {t:.0f} s ({ex.status.phase})")
            crossings += n
        trail.append((pose[0], pose[1]))
        t += DT
    dist = sum(math.hypot(b[0] - a_[0], b[1] - a_[1]) for a_, b in zip(trail, trail[1:]))
    explored = ((known == ec.FREE) & (world.cells == ec.FREE)).sum() / max(1, (world.cells == ec.FREE).sum())
    print(f"{Path(a.map).name}: {ex.status.state} after {t / 60:.1f} min ({ex.status.message})")
    print(f"  driven {dist:.1f} m, free area discovered {100 * explored:.0f} %, camera coverage {100 * ex.coverage():.0f} %")
    print(f"  re-driven (within 35 cm of its own path from > 30 s before): {100 * redriven(trail):.0f} % of the distance")
    print(f"  closest to an obstacle {100 * min_clear:.0f} cm (robot radius {100 * ec.ROBOT_RADIUS:.0f}), "
          f"steps towards the blind sector {rear_moves}, blocked steps {blocked}")
    if a.wall:
        print(f"  no-go lines drawn at {drawn_at[0]:.0f} s with the robot at {drawn_at[1]}; crossed {crossings} times "
              f"afterwards; start {tuple(round(v, 2) for v in trail[0])}, end {tuple(round(v, 2) for v in trail[-1])}")
    if ex.grid is not None:
        why_unseen(ex, tuple(pose))
    print(f"  control step: median {1000 * np.median(plan_times):.1f} ms, max {1000 * max(plan_times):.0f} ms")
    draw(world, known, ex, trail, Path(a.out) if a.out else Path(a.map).with_name(Path(a.map).stem + "_explore.png"))


def why_unseen(ex, pose):
    """Why the surfaces the camera has not seen are left: given up, too small, or no viewpoint."""
    from scipy import ndimage
    g = ex.grid
    unseen = ex.surfaces() & ~ex.seen
    given_up = unseen & (ex.attempts >= ex.p["views"])
    start = g.to_cell(pose[0], pose[1])
    dist, _ = ex.cost_field(start)
    views = ex.coverage_goals(dist, start)
    with_vp = np.zeros(unseen.shape, bool)
    blocked = np.zeros(unseen.shape, bool)
    for rc, n, (rr, cc) in views:
        x, y = g.to_xy(*rc)
        if all(math.hypot(x - bx, y - by) > 1.0 for bx, by in ex.blacklist):
            with_vp[rr, cc] = True
        else:
            blocked[rr, cc] = True
    labels, n = ndimage.label(unseen & ~given_up, structure=np.ones((3, 3)))
    small = np.isin(labels, [i for i in range(1, n + 1) if (labels == i).sum() < ec.MIN_SURFACE])
    tot = max(1, unseen.sum())
    print(f"  unseen surface cells {unseen.sum()}: given up {given_up.sum() / tot:.0%}, in small bits {small.sum() / tot:.0%}, "
          f"viewpoint blocked by a checked one {blocked.sum() / tot:.0%}, still with a viewpoint {with_vp.sum() / tot:.0%}, "
          f"no viewpoint at all {(unseen & ~given_up & ~small & ~blocked & ~with_vp).sum() / tot:.0%}")


def redriven(trail):
    """Share of the distance driven within 35 cm of where the robot was more than 30 s before."""
    from scipy.spatial import cKDTree
    pts = np.array(trail[::5])  # every 0.5 s
    if len(pts) < 3:
        return 0.0
    seg = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]
    lag = int(30 / (5 * DT))
    again = 0.0
    for i in range(lag + 1, len(pts)):
        d, _ = cKDTree(pts[:i - lag]).query(pts[i])
        if d < 0.35:
            again += seg[i]
    return again / max(seg.sum(), 1e-9)


def crosses(p, q, wall):
    """Does the segment p-q cross the line wall = (x1, y1, x2, y2)?"""
    a, b = (wall[0], wall[1]), (wall[2], wall[3])
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return side(a, b, p) * side(a, b, q) < 0 and side(p, q, a) * side(p, q, b) < 0


def ndi_clearance(world):
    from scipy import ndimage
    return ndimage.distance_transform_edt(world.cells == ec.FREE) * world.res


def draw(world, known, ex, trail, out):
    img = np.full((*world.cells.shape, 3), 60, np.uint8)
    img[known == ec.FREE] = (235, 235, 235)
    img[world.cells == ec.OCC] = (90, 90, 90)
    s = ex.surfaces()
    img[s & ex.seen] = (40, 160, 40)
    img[s & ~ex.seen] = (40, 40, 220)
    for (x0, y0), (x1, y1) in zip(trail, trail[1:]):
        cv2.line(img, world.to_cell(x0, y0)[::-1], world.to_cell(x1, y1)[::-1], (220, 120, 0), 1)
    img = cv2.resize(np.flipud(img), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out), img)
    print(f"  picture: {out} (green: surfaces the camera saw, red: not seen, orange: path)")


if __name__ == "__main__":
    main()
