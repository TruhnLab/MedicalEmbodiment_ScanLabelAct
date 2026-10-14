"""Offline test of localize.py: a recorded run of the same room, localized in a room made from another run.

    python projects/navigation/test_localize.py 20261006_154055 20261006_161813    # room, run
    python projects/navigation/test_localize.py 20261006_154055 20261006_154055    # self-test: must be identity

The run has its own SLAM map frame, so there is no ground truth in the room's frame. But the transform
room <- run frame is one fixed transform: every correct localization gives the same one
(T = pose in room * (pose in run)^-1). The test
1. localizes from scratch at many moments of the run (global search on the scans of the seconds
   before, stitched with odometry; at the start the exploration's turn on the spot),
2. takes the transform most results agree on (within 20 cm / 3 deg) as the truth and reports each
   result's distance from it (success = within 30 cm / 5 deg),
3. tracks the whole run scan by scan from the first global result and reports the error against
   the run's SLAM pose taken into the room's frame.
Writes <room>/test_<run>.png: the room's map with the run's scans placed by tracking (they should lie on
the walls) and the global results.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landerpi.room import SESSIONS, Room  # noqa: E402
from localize import (Localizer, compose, matrix_pose, pose_matrix, relative, scan_points,  # noqa: E402
                      stitch, wrap)

AGREE = (0.20, math.radians(3))
SUCCESS = (0.30, math.radians(5))


def stamp(msg):
    return msg["header"]["stamp"]["sec"] + msg["header"]["stamp"]["nanosec"] * 1e-9


def yaw_of(q):
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def read(session, name):
    with open(SESSIONS / session / "streams" / f"{name}.jsonl") as f:
        return [json.loads(line)["msg"] for line in f]


class Track:
    """A pose stream (t, x, y, yaw), linearly interpolated."""

    def __init__(self, t, poses):
        o = np.argsort(t)
        self.t, p = np.asarray(t)[o], np.asarray(poses)[o]
        self.x, self.y, self.yaw = p[:, 0], p[:, 1], np.unwrap(p[:, 2])

    def at(self, t):
        return np.array([np.interp(t, self.t, self.x), np.interp(t, self.t, self.y),
                         wrap(np.interp(t, self.t, self.yaw))])


def lidar_in_base(session):
    """lidar_frame in base_footprint from the session's static TF (planar part)."""
    tfs = json.loads((SESSIONS / session / "calib" / "tf_static.json").read_text())["transforms"]
    parent = {t["child_frame_id"]: t for t in tfs}
    T, f = np.eye(3), "lidar_frame"
    while f != "base_footprint":
        t = parent[f]["transform"]
        T = pose_matrix((t["translation"]["x"], t["translation"]["y"], yaw_of(t["rotation"]))) @ T
        f = parent[f]["header"]["frame_id"]
    return matrix_pose(T)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room")
    p.add_argument("run")
    p.add_argument("--every", type=float, default=15.0, help="s between global tests")
    p.add_argument("--window", type=float, default=4.0, help="s of scans per global test (not at the start)")
    args = p.parse_args()

    room = Room(args.room)
    scene = room.session_path / "scene" / "scene.json"
    area = json.loads(scene.read_text()).get("area_polygon") if scene.exists() else None
    t0 = time.time()
    loc = Localizer(room.grid, room.resolution, room.origin, area=area)
    print(f"{room}: {loc.free.sum() * room.resolution ** 2:.0f} m2 where the robot can be "
          f"({'area polygon' if area else 'all free cells'}), field in {time.time() - t0:.1f} s")

    lidar = lidar_in_base(args.run)
    odom = read(args.run, "odom")
    odo = Track([stamp(m) for m in odom], [(m["pose"]["pose"]["position"]["x"], m["pose"]["pose"]["position"]["y"],
                                           yaw_of(m["pose"]["pose"]["orientation"])) for m in odom])
    mo = read(args.run, "slam_map_odom")
    map_odom = Track([stamp(m) for m in mo], [(m["transform"]["translation"]["x"], m["transform"]["translation"]["y"],
                                               yaw_of(m["transform"]["rotation"])) for m in mo])
    scans = [(stamp(m), scan_points(m, lidar)) for m in read(args.run, "scan")]
    scans = [(t, pts) for t, pts in scans if odo.t[0] <= t <= odo.t[-1] and len(pts) > 20]
    run_pose = lambda t: compose(map_odom.at(t), odo.at(t))  # noqa: E731  the run's SLAM pose

    # 1. global tests: the start (turn on the spot until 360 deg) and then every --every s
    tests = []
    yaw0, i = odo.at(scans[0][0])[2], 0
    turned = 0.0
    while i + 1 < len(scans) and turned < 2 * math.pi:
        i += 1
        turned += abs(wrap(odo.at(scans[i][0])[2] - odo.at(scans[i - 1][0])[2]))
    tests.append(("start turn", 0, i))
    t = scans[0][0] + max(args.every, scans[i][0] - scans[0][0])
    while t < scans[-1][0]:
        j = np.searchsorted([s[0] for s in scans], t)
        k = np.searchsorted([s[0] for s in scans], t - args.window)
        tests.append((f"t={t - scans[0][0]:.0f}s", k, j))
        t += args.every
    results = []
    for name, a, b in tests:
        ts = time.time()
        pose, info = loc.global_search(stitch([(pts, odo.at(t)) for t, pts in scans[a:b + 1]]))
        T = matrix_pose(pose_matrix(pose) @ np.linalg.inv(pose_matrix(run_pose(scans[b][0]))))
        results.append(dict(name=name, t=scans[b][0], pose=pose, T=T, sec=time.time() - ts, **info))

    def close(a, b, tol):
        return np.hypot(*(a[:2] - b[:2])) < tol[0] and abs(wrap(a[2] - b[2])) < tol[1]

    T_true = max((r["T"] for r in results), key=lambda T: sum(close(T, r["T"], AGREE) for r in results))
    print(f"\nroom <- run: x {T_true[0]:+.3f} m, y {T_true[1]:+.3f} m, yaw {math.degrees(T_true[2]):+.2f} deg "
          f"({sum(close(T_true, r['T'], AGREE) for r in results)}/{len(results)} results agree)")
    print(f"{'test':>12} {'points':>6} {'score':>6} {'2nd':>5} {'err cm':>7} {'err deg':>7} {'s':>5}")
    ok = 0
    for r in results:
        truth = compose(T_true, run_pose(r["t"]))
        e, ea = np.hypot(*(r["pose"][:2] - truth[:2])), abs(math.degrees(wrap(r["pose"][2] - truth[2])))
        good = e < SUCCESS[0] and math.radians(ea) < SUCCESS[1]
        ok += good
        print(f"{r['name']:>12} {r['points']:>6} {r['score']:6.3f} {r['runner_up']:5.3f} {e * 100:7.1f} {ea:7.2f} "
              f"{r['sec']:5.1f} {'' if good else 'FAIL'}{' ambiguous' if r['ambiguous'] else ''}")
    print(f"global: {ok}/{len(results)} within {SUCCESS[0] * 100:.0f} cm / {math.degrees(SUCCESS[1]):.0f} deg, "
          f"{sum(r['ambiguous'] for r in results)} flagged ambiguous")

    # 3. tracking from the start result, odometry prediction + scan match
    ts = time.time()
    pose, prev_odo = results[0]["pose"], odo.at(scans[tests[0][2]][0])
    errs, placed, fit = [], [], []
    for t, pts in scans[tests[0][2] + 1:]:
        o = odo.at(t)
        pose, info = loc.track(pts, compose(pose, relative(prev_odo, o)))
        prev_odo = o
        truth = compose(T_true, run_pose(t))
        errs.append((np.hypot(*(pose[:2] - truth[:2])), abs(wrap(pose[2] - truth[2]))))
        fit.append((info["score"], loc.score(loc.fine, pts, truth[None])[0]))
        placed.append(pts @ pose_matrix(pose)[:2, :2].T + pose[:2])
    errs = np.array(errs)
    print(f"tracking {len(errs)} scans ({(time.time() - ts) / len(errs) * 1000:.0f} ms each): error median "
          f"{np.median(errs[:, 0]) * 100:.1f} cm / {math.degrees(np.median(errs[:, 1])):.2f} deg, 95 % "
          f"{np.percentile(errs[:, 0], 95) * 100:.1f} cm / {math.degrees(np.percentile(errs[:, 1], 95)):.2f} deg, "
          f"max {errs[:, 0].max() * 100:.1f} cm")
    # which one is off, tracking or the reference? the one whose scans fit the room's map worse
    fit = np.array(fit)
    print(f"map fit (mean likelihood of the scan points): tracked {fit[:, 0].mean():.3f}, "
          f"run's SLAM pose {fit[:, 1].mean():.3f}; tracked fits better in {(fit[:, 0] > fit[:, 1]).mean() * 100:.0f} %")

    # picture: map grey, robot area tinted, tracked scans in blue, global results as dots
    img = np.full(room.grid.shape + (3,), 128, np.uint8)
    img[room.grid == 0] = 255
    img[loc.free] = (235, 245, 235)
    img[room.grid >= 65] = 0
    for pts in placed[::5]:
        c = room.world_to_cell(pts)
        k = (c[:, 0] >= 0) & (c[:, 0] < img.shape[1]) & (c[:, 1] >= 0) & (c[:, 1] < img.shape[0])
        img[c[k, 1], c[k, 0]] = (255, 120, 0)
    img = cv2.resize(img[::-1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for r in results:
        c = room.world_to_cell(r["pose"][:2])
        q = (int(c[0] * 3), int((img.shape[0] // 3 - 1 - c[1]) * 3))
        good = close(r["pose"], compose(T_true, run_pose(r["t"])), SUCCESS)
        cv2.circle(img, q, 5, (0, 170, 0) if good else (0, 0, 255), -1)
        cv2.line(img, q, (int(q[0] + 15 * math.cos(r["pose"][2])), int(q[1] - 15 * math.sin(r["pose"][2]))),
                 (0, 0, 0), 2)
    out = room.path / f"test_{args.run}.png"
    cv2.imwrite(str(out), img)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
