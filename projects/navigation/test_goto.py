"""Simulated drives to the objects of a room (goto.py), without the robot.

    python projects/navigation/test_goto.py 20261007_101108                 # every object, 2 start spots
    python projects/navigation/test_goto.py 20261007_101108 --objects 5 11 --block --picture

The room's 2D map is the world: a simulated robot (perfect pose, commanded velocities applied as given)
with a simulated lidar (front ~195 deg, like /scan, 6 m). --block puts a 30 x 30 cm obstacle the planner's
map doesn't know onto the first path (a chair moved since the scan). Reported per drive: arrived or not,
time, driven length, the closest the robot's centre came to an obstacle, and at the end the gap between the
robot's front and the object's box and the heading error to facing it. No depth camera here: objects only
the depth camera would see (table tops) are not obstacles in the simulation.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import goto as gt  # noqa: E402
from goto import ec  # noqa: E402
from landerpi.room import Room  # noqa: E402

DT = 0.1
TIMEOUT_S = 120.0
LIDAR_X = 0.073
RAYS = np.arange(ec.LIDAR_SECTOR[0], ec.LIDAR_SECTOR[1], 2 * math.pi / 504)


def box_gap(obj, x, y):
    """Distance from a point to an object's upright box footprint."""
    yaw = math.radians(obj["yaw_deg"])
    dx, dy = x - obj["center"][0], y - obj["center"][1]
    lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.hypot(max(abs(lx) - obj["size"][0] / 2, 0), max(abs(ly) - obj["size"][1] / 2, 0))


class World:
    def __init__(self, room):
        self.occ = room.grid >= 65
        self.res, self.origin = room.resolution, np.array(room.origin)

    def cell(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)

    def add_box(self, x, y, half=0.15):
        c0, c1 = self.cell((x - half, y - half)), self.cell((x + half, y + half))
        self.occ[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1] = True

    def ranges(self, pose, max_range=6.0, step=0.025):
        """Ranges for all 504 rays of a full turn (lidar frame, angle 0 = ahead), None where the LD19's
        filtered rear sector or nothing within max_range: like /scan (fake_robot.py --sim)."""
        a = np.arange(504) * 2 * math.pi / 504
        rel = np.angle(np.exp(1j * a))
        front = (rel >= ec.LIDAR_SECTOR[0]) & (rel <= ec.LIDAR_SECTOR[1])
        x, y, yaw = pose
        lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
        d = np.arange(0.15, max_range, step)
        px = lx + np.cos(yaw + a)[:, None] * d[None, :]
        py = ly + np.sin(yaw + a)[:, None] * d[None, :]
        c = np.floor((px - self.origin[0]) / self.res).astype(int)
        r = np.floor((py - self.origin[1]) / self.res).astype(int)
        inside = (r >= 0) & (r < self.occ.shape[0]) & (c >= 0) & (c < self.occ.shape[1])
        hit = np.zeros(px.shape, bool)
        hit[inside] = self.occ[r[inside], c[inside]]
        out = [None] * 504
        for i in np.nonzero(front & hit.any(1))[0]:
            out[i] = float(d[hit[i].argmax()])
        return out

    def scan(self, pose, max_range=6.0, step=0.025):
        """Lidar points in the robot frame."""
        x, y, yaw = pose
        lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
        d = np.arange(0.15, max_range, step)
        a = yaw + RAYS
        px = lx + np.cos(a)[:, None] * d[None, :]
        py = ly + np.sin(a)[:, None] * d[None, :]
        c = np.floor((px - self.origin[0]) / self.res).astype(int)
        r = np.floor((py - self.origin[1]) / self.res).astype(int)
        inside = (r >= 0) & (r < self.occ.shape[0]) & (c >= 0) & (c < self.occ.shape[1])
        hit = np.zeros(px.shape, bool)
        hit[inside] = self.occ[r[inside], c[inside]]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        k = first >= 0
        rng = d[first[k]]
        ang = RAYS[k]
        return np.stack([LIDAR_X + rng * np.cos(ang), rng * np.sin(ang)], 1)


def start_spots(room, area, n, seed=1):
    """n free spots 0.45+ m from obstacles inside the room's area, spread out."""
    free = room.grid == 0
    clear = ndimage.distance_transform_edt(room.grid < 65) * room.resolution
    ok = free & (clear > 0.45)
    if area:
        mask = np.zeros(ok.shape, np.uint8)
        cv2.fillPoly(mask, [((np.asarray(area)[:, :2] - room.origin) / room.resolution).astype(np.int32)], 1)
        ok &= mask.astype(bool)
    rows, cols = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [((cols[i] + 0.5) * room.resolution + room.origin[0], (rows[i] + 0.5) * room.resolution + room.origin[1],
             float(rng.uniform(-math.pi, math.pi))) for i in pick]


def drive(room, obj, start, block=False, box_error=0.0, seed=0):
    world = World(room)
    truth = obj
    if box_error:  # the planner gets the box shifted (a scan's box can be 10-20 cm off); the world stays
        a = np.random.default_rng(seed).uniform(0, 2 * math.pi)
        obj = dict(obj, center=[obj['center'][0] + box_error * math.cos(a), obj['center'][1] + box_error * math.sin(a),
                                obj['center'][2]])
    g = gt.GoTo(room.grid, room.resolution, room.origin, walls=room.nogo_lines())
    pose = np.array(start, float)
    t = 0.0
    g.start(obj, t)
    g.plan(pose, t)
    if block and len(g.path) > 12:  # an obstacle the map doesn't know, ~1/3 along the first path
        bx, by = g.path[len(g.path) // 3]
        world.add_box(bx, by)
    clear_map = ndimage.distance_transform_edt(~world.occ) * room.resolution
    driven, closest, trace = 0.0, math.inf, [tuple(pose)]
    t_wall = time.time()
    while g.state in ("planning", "driving", "docking") and t < TIMEOUT_S:
        scan = world.scan(pose)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        if g.plan_due(t) or g.last_plan is None:
            g.set_obstacles(scan @ np.array([[c, s], [-s, c]]) + pose[:2])
            g.plan(pose, t)
        vx, vy, wz = g.step(t, pose, scan)
        dx, dy = (c * vx - s * vy) * DT, (s * vx + c * vy) * DT
        pose = pose + [dx, dy, wz * DT]
        pose[2] = gt.wrap(pose[2])
        driven += math.hypot(dx, dy)
        cell = world.cell(pose[:2])
        closest = min(closest, clear_map[cell[1], cell[0]])
        trace.append(tuple(pose))
        t += DT
    fx, fy = pose[0] + gt.HALF_LENGTH * math.cos(pose[2]), pose[1] + gt.HALF_LENGTH * math.sin(pose[2])
    obj = truth
    facing = math.atan2(obj["center"][1] - pose[1], obj["center"][0] - pose[0])
    herr = math.degrees(abs(gt.wrap(g.goal[2] - pose[2]))) if g.goal else math.nan
    return {"state": g.state if t < TIMEOUT_S else "timeout", "message": g.message, "t": t, "driven": driven,
            "closest": closest, "gap": box_gap(obj, fx, fy), "seen": g.front_gap(world.scan(pose)), "herr": herr,
            "facing_err": math.degrees(abs(gt.wrap(facing - pose[2]))), "trace": trace, "goal": g.goal,
            "wall_s": time.time() - t_wall, "blocked": block}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room")
    p.add_argument("--objects", type=int, nargs="*", help="object ids (default: all)")
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--block", action="store_true", help="an unknown obstacle on the first path")
    p.add_argument("--box-error", type=float, default=0.0, help="m: shift each object's box for the planner")
    p.add_argument("--picture", action="store_true", help="write data/rooms/<room>/goto_<id>.png")
    args = p.parse_args()

    room = Room(args.room)
    scene = json.loads((room.session_path / "scene" / "scene.json").read_text())
    objs = {o["id"]: o for o in scene["objects"]}
    ids = args.objects or sorted(objs)
    starts = start_spots(room, scene.get("area_polygon"), args.starts)
    print(f"{room}: {len(ids)} objects x {len(starts)} start spots{' (with an unknown obstacle)' if args.block else ''}")
    print(f"{'object':>26} {'start':>5} {'result':>8} {'time s':>6} {'m':>5} {'closest':>7} {'gap':>5} {'seen':>5} {'head':>5}")
    results = []
    for i in ids:
        o = objs[i]
        for k, st in enumerate(starts):
            r = drive(room, o, st, args.block, args.box_error, seed=i * 10 + k)
            results.append(r)
            print(f"{("#" + str(i) + " " + o["label"])[:26]:>26} {k:>5} {r['state']:>8} {r['t']:6.1f} "
                  f"{r['driven']:5.1f} {r['closest']:7.2f} {r['gap']:5.2f} {r['seen']:5.2f} {r['herr']:5.1f}"
                  + ("" if r["state"] == "arrived" else f"  {r['message']}"))
            if args.picture:
                picture(room, o, r, room.path / f"goto_{i}_{k}.png")
    ok = [r for r in results if r["state"] == "arrived"]
    print(f"\narrived {len(ok)}/{len(results)}; of those: time median {np.median([r['t'] for r in ok]):.1f} s, "
          f"gap median {np.median([r['gap'] for r in ok]):.2f} m (range {min(r['gap'] for r in ok):.2f}-"
          f"{max(r['gap'] for r in ok):.2f}), seen ahead within 0.8 m: {sum(r['seen'] < 0.8 for r in ok)} drives, "
          f"their gap {np.median([r['seen'] for r in ok if r['seen'] < 0.8]):.2f} m ("
          f"{min(r['seen'] for r in ok):.2f}-{max(r['seen'] for r in ok if r['seen'] < 0.8):.2f}), closest to anything {min(r['closest'] for r in results):.2f} m, "
          f"heading error max {max(r['herr'] for r in ok):.1f} deg; simulation {np.mean([r['wall_s'] for r in results]):.1f} s per drive")


def picture(room, obj, r, path):
    img = np.full(room.grid.shape + (3,), 128, np.uint8)
    img[room.grid == 0] = 255
    img[room.grid >= 65] = 0
    s = 3
    img = cv2.resize(img[::-1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    h = room.grid.shape[0]

    def px(x, y):
        return int((x - room.origin[0]) / room.resolution * s), int((h - (y - room.origin[1]) / room.resolution) * s)
    yaw = math.radians(obj["yaw_deg"])
    L, W = obj["size"][:2]
    corners = [(obj["center"][0] + a * L / 2 * math.cos(yaw) - b * W / 2 * math.sin(yaw),
                obj["center"][1] + a * L / 2 * math.sin(yaw) + b * W / 2 * math.cos(yaw)) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    cv2.polylines(img, [np.array([px(*c) for c in corners])], True, (0, 160, 0), 2)
    pts = np.array([px(x, y) for x, y, _ in r["trace"]])
    cv2.polylines(img, [pts], False, (255, 120, 0), 2)
    x, y, a = r["trace"][-1]
    cv2.arrowedLine(img, px(x, y), px(x + 0.3 * math.cos(a), y + 0.3 * math.sin(a)), (0, 0, 255), 2)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main()
