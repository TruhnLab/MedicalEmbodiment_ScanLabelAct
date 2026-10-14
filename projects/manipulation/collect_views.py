"""Pictures of the calibration sheet from many arm poses, for the camera's mounting correction (solve_camera.py).
The sheet lies still in front of the robot; each pose is approached from above, the servos' measured positions and
one colour picture are saved. Moves the arm (no base motion).

    python projects/manipulation/collect_views.py --plan       # only list the poses (no robot)
    python projects/manipulation/collect_views.py              # -> data/calibration/views_<time>/
    python projects/manipulation/collect_views.py --sweeps     # one joint at a time (the joints' degrees per pulse)

Poses: pan (servo 1), shoulder (2) and elbow (3) varied; the tilt (4) with the most markers in view; kept
if arm.collision() (with the chassis box) passes and at least 3 markers should be in the picture.
"""

import argparse
import asyncio
import glob
import json
import time
import urllib.request

import cv2
import numpy as np

import arm
import grid_sheet
from check_arm import REPORTS, Servos
from landerpi.robot import ROSBRIDGE_URL, SURVEY_POSE
from landerpi.rosbridge import RosbridgeClient
from look_at_sheet import SNAPSHOT, camera_info

APPROACH = 40          # pulses above the pose first (from above the servos stop 0-5 short, from below 5-13)
W, H = 640, 400
# rgb camera_info of the robot (sessions of 2026-10-07); the live one is used when the robot is connected
K0 = np.array([[417.41, 0, 315.41], [0, 418.09, 191.83], [0, 0, 1]])


def latest_sheet():
    path = sorted(glob.glob(str(REPORTS / "*_sheet.json")))[-1]
    return np.array(json.loads(open(path).read())["base_sheet"])


def markers_in_view(pose, S, K=K0):
    """How many marker centres the camera would see (fully, with a margin) in this pose."""
    C = arm.frame("rgb_camera_link", arm.pulses_to_angles(pose))
    n = 0
    for x, y in grid_sheet.MARKERS.values():
        p = np.linalg.inv(C) @ S @ np.array([x / 1000, y / 1000, 0, 1])
        if p[2] < 0.1:
            continue
        u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
        n += 40 < u < W - 40 and 40 < v < H - 40
    return n


def plan(S):
    """Arm poses with >= 3 markers in view: several tilts per pan / shoulder / elbow (different tilts separate the
    camera's mounting from the arm's angles)."""
    poses = []
    for p2 in (650, 730, 800):
        for p3 in (80, 130, 200, 260):
            for p1 in (380, 440, 500, 560, 620):
                base = {1: p1, 2: p2, 3: p3, 5: 500}
                tilts = [p4 for p4 in range(100, 500, 10) if markers_in_view({**base, 4: p4}, S) >= 3]
                for p4 in tilts[::4]:  # 40 pulses (~10 deg) apart
                    pose = {**base, 4: p4}
                    pre = {i: min(1000, p + APPROACH) for i, p in pose.items()}
                    q, qpre = arm.pulses_to_angles(pose), arm.pulses_to_angles(pre)
                    if arm.collision(q) or arm.collision(qpre) or arm.path_collision(qpre, q):
                        continue
                    poses.append((pose, markers_in_view(pose, S)))
    return poses


def sweeps(S):
    """One joint at a time through its range (20 pulses apart), the others still, sheet in view: the camera's
    rotation between the pictures is that joint's angle alone, whatever the camera's mounting (solve_gains.py)."""
    bases = {1: ({2: 730, 3: 130, 4: 110, 5: 500}, range(300, 701, 20)),
             2: ({1: 500, 3: 130, 4: 110, 5: 500}, range(600, 861, 20)),
             3: ({1: 500, 2: 650, 4: 140, 5: 500}, range(40, 301, 20)),
             4: ({1: 500, 2: 730, 3: 130, 5: 500}, range(100, 401, 20))}
    poses = []
    for joint, (base, values) in bases.items():
        for v in values:
            pose = {**base, joint: v}
            pre = {i: min(1000, p + APPROACH) for i, p in pose.items()}
            q, qpre = arm.pulses_to_angles(pose), arm.pulses_to_angles(pre)
            if arm.collision(q) or arm.collision(qpre) or arm.path_collision(qpre, q):
                continue
            if markers_in_view(pose, S) >= 2:  # 8 corners give the camera's rotation well enough
                poses.append((pose, joint))
    return poses


async def collect(poses):
    out = REPORTS / f"views_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        K, D = await camera_info(client)
        (out / "camera_info.json").write_text(json.dumps({"k": K.tolist(), "d": D.tolist()}))
        current = await s.read(arm.ARM_IDS)
        for n, (pose, _) in enumerate(poses):
            pre = {i: min(1000, p + APPROACH) for i, p in pose.items()}
            hit = arm.path_collision(arm.pulses_to_angles(current), arm.pulses_to_angles(pre))
            if hit:  # from the last pose to this one: go through the survey pose
                for via in ({i: min(1000, p + APPROACH) for i, p in SURVEY_POSE.items()},):
                    await s.move(via, 2.0)
                    await s.settle(arm.ARM_IDS, 2.0)
            await s.move(pre, 1.5)
            await s.settle(arm.ARM_IDS, 1.5)
            await s.move(pose, 1.0)
            current = await s.settle(arm.ARM_IDS, 1.0)
            await asyncio.sleep(0.7)  # a picture taken after the arm stopped
            jpg = urllib.request.urlopen(SNAPSHOT, timeout=5).read()
            (out / f"{n:02d}.jpg").write_bytes(jpg)
            (out / f"{n:02d}.json").write_text(json.dumps({"command": pose, "measured": current}))
            print(f"{n:2d}/{len(poses)} command {pose} measured {current}", flush=True)
        await s.move({i: min(1000, p + APPROACH) for i, p in SURVEY_POSE.items()}, 2.0)
        await s.settle(arm.ARM_IDS, 2.0)
        await s.move(SURVEY_POSE, 1.0)
        await s.settle(arm.ARM_IDS, 1.0)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--sweeps", action="store_true", help="one joint at a time instead (joint gains)")
    a = p.parse_args()
    poses = sweeps(latest_sheet()) if a.sweeps else plan(latest_sheet())
    for pose, n in poses:
        print(pose, f"{n} markers" if not a.sweeps else f"joint {n}")
    print(f"{len(poses)} poses")
    if not a.plan:
        asyncio.run(collect(poses))


if __name__ == "__main__":
    main()
