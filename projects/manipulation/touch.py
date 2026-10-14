"""The gripper tip touches the calibration sheet (grid_sheet.py) at its targets: where the floor really is for the arm.
For each target: guarded move to 7 cm above it, then down in small steps (the gripper's direction kept) until a joint
stops following (contact), record, lift. Moves the arm (no base motion).

    python projects/manipulation/touch.py --plan              # which targets are reachable, how (no robot)
    python projects/manipulation/touch.py 5 8 2               # touch T5, T8, T2 -> data/calibration/touch_<time>.json
    python projects/manipulation/touch.py --hold 0.01 5       # tip 1 cm above T5 by the model, and stay there
    python projects/manipulation/touch.py --lift              # from there up and back to the survey pose

--hold is for someone measuring where the jaw tips really are (contact by "a joint lags" was unreliable: pressing
on the floor went unnoticed at 3 of 5 targets, 2026-10-12).

The sheet's place: the latest look_at_sheet.py result (its x, y and turn; the floor taken as z 0). At contact the
tip's height by the arm's model (arm.py with the measured gains) says how far model and floor disagree; the picture
shows the tip on the grid.
"""

import argparse
import asyncio
import json
import math
import time
import urllib.request

import numpy as np

import arm
import grid_sheet
from check_arm import REPORTS, Blocked, Servos, guarded_move
from collect_views import latest_sheet
from landerpi.robot import ROSBRIDGE_URL, SURVEY_POSE
from landerpi.rosbridge import RosbridgeClient
from look_at_sheet import SNAPSHOT

HOVER = 0.09          # m above the floor before going down (the block fully in view above it, 2026-10-12)
LOWEST = -0.03        # m: the model's floor may be off; never further down than this
GRIPPER_CLOSED = 530  # jaws ~13 mm apart: one point to touch with (config/gripper.json)
CONTACT_LAG = 8       # pulses: the lag changed this much against the free descent = touching


def target_xy(n, S):
    x, y = grid_sheet.TARGETS[n]
    p = S @ np.array([x / 1000, y / 1000, 0, 1])
    return p[0], p[1]


def plan_target(n, S):
    """(down angle, hover pulses, [(z, pulses) going down]) or None."""
    x, y = target_xy(n, S)
    q, down = arm.reach(np.array([x, y, 0.02]), chassis=False)  # above arm.FLOOR_Z: the descent goes on below
    if q is None:
        return None
    path = []
    for z in [HOVER] + list(np.arange(0.05, 0.019, -0.005)) + list(np.arange(0.018, LOWEST - 1e-6, -0.002)):
        t = np.array([x, y, z])
        qz, err, derr = min((arm.ik(t, down, s0, iterations=60) for s0 in arm.planar_ik(t, down)[:1] + [q]),
                            key=lambda r: r[1])
        if err > 0.002 or (z > 0.012 and arm.collision(qz, chassis=False)):
            break
        path.append((round(float(z), 3), arm.angles_to_pulses(qz)))
        q = qz
    if not path or path[0][0] != HOVER:
        return None
    return down, path


async def touch(s, n, S, results):
    p = plan_target(n, S)
    if p is None:
        print(f"T{n}: not reachable")
        return
    down, path = p
    hover = {i: v for i, v in path[0][1].items() if i in arm.ARM_IDS[:4]}
    now = await s.read(arm.ARM_IDS)
    hit = arm.path_collision(arm.pulses_to_angles(now), arm.pulses_to_angles({**hover, 5: now[5]}), chassis=False)
    if hit:
        print(f"T{n}: the way to above it: {hit}; skipped")
        return
    print(f"T{n}: gripper {down} deg down; to 7 cm above it", flush=True)
    await guarded_move(s, hover)
    baseline = None
    contact = None
    for z, pulses in path[1:]:
        pose = {i: v for i, v in pulses.items() if i in arm.ARM_IDS[:4]}
        await s.move(pose, 0.25)
        got = await s.settle(list(pose), 0.25, timeout=1.0)
        lag = {i: pose[i] - got[i] for i in pose}
        if baseline is None and z <= 0.03:
            baseline = lag  # the free descent's own lag (gravity, deadband)
        if baseline is not None and any(abs(lag[i] - baseline[i]) > CONTACT_LAG for i in pose):
            contact = (z, pose, got, lag)
            break
    if contact is None:
        print(f"T{n}: no contact down to {path[-1][0]:+.3f} m (model)")
    else:
        z, pose, got, lag = contact
        q = arm.pulses_to_angles({**got, 5: 500})
        tip = arm.frame("jaw_tips", q)[:3, 3]
        await asyncio.sleep(0.3)
        jpg = urllib.request.urlopen(SNAPSHOT, timeout=5).read()
        shot = REPORTS / f"touch_{time.strftime('%H%M%S')}_T{n}.jpg"
        shot.write_bytes(jpg)
        print(f"T{n}: contact at commanded z {z:+.3f} m; tip by the model at ({tip[0]:.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) "
              f"m, target ({target_xy(n, S)[0]:.3f}, {target_xy(n, S)[1]:+.3f}); lags {lag}; picture {shot.name}")
        results.append({"target": n, "down_deg": down, "z_command": z, "measured": got, "lag": lag,
                        "tip_model": tip.round(4).tolist(), "target_xy": [round(v, 4) for v in target_xy(n, S)],
                        "picture": shot.name})
    await guarded_move(s, hover)


async def hold(n, z_hold):
    S = latest_sheet()
    p = plan_target(n, S)
    if p is None:
        raise SystemExit(f"T{n}: not reachable")
    down, path = p
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        await s.move({arm.GRIPPER_ID: GRIPPER_CLOSED}, 0.8)
        hover = {i: v for i, v in path[0][1].items() if i in arm.ARM_IDS[:4]}
        now = await s.read(arm.ARM_IDS)
        hit = arm.path_collision(arm.pulses_to_angles(now), arm.pulses_to_angles({**hover, 5: now[5]}), chassis=False)
        if hit:
            raise SystemExit(f"the way to above T{n}: {hit}")
        got = await guarded_move(s, hover)
        pose = hover  # z_hold at or above the hover height: stay there
        for z, pulses in path[1:]:
            if z < z_hold - 1e-6:
                break
            pose = {i: v for i, v in pulses.items() if i in arm.ARM_IDS[:4]}
            got = await guarded_move(s, pose, max_step=8)
        # stand exactly: correct the servos' shortfall once (deadband, gravity)
        fix = {i: pose[i] + (pose[i] - got[i]) for i in pose if abs(pose[i] - got[i]) > 2}
        if fix:
            await s.move({**pose, **fix}, 0.4)
            got = await s.settle(list(pose), 0.4)
        x, y = target_xy(n, S)
        for nominal in (False,):
            tip = arm.frame("jaw_tips", arm.pulses_to_angles({**got, 5: 500}, nominal))[:3, 3]
            print(f"  model: jaw tips at ({tip[0]:.3f}, {tip[1]:+.3f}, "
                  f"{tip[2]:+.3f}) m; target ({x:.3f}, {y:+.3f}) -> tip {1000 * (tip[0] - x):+.0f} mm ahead, "
                  f"{1000 * (tip[1] - y):+.0f} mm left, {1000 * tip[2]:+.0f} mm up")
        rec = {"target": n, "down_deg": down, "z_hold": z_hold, "command": pose, "measured": got,
               "target_xy": [round(x, 4), round(y, 4)], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(REPORTS / "holds.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"T{n}: holding, gripper {down} deg down, servos {got}. Measure the jaw tips: height above the paper, "
              f"mm ahead/behind and left/right of the cross.")


async def lift():
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        now = await s.read(arm.ARM_IDS)
        q = arm.pulses_to_angles(now)
        f = arm.frame("end_effector_link", q)
        up_target = f[:3, 3] + [0, 0, 0.06]
        d = f[:3, 2]
        down = math.degrees(math.asin(max(-1.0, min(1.0, -d[2]))))
        qu, err, _ = arm.ik(up_target, down, q, iterations=80)
        if err < 0.003:
            try:
                await guarded_move(s, {i: v for i, v in arm.angles_to_pulses(qu).items() if i in arm.ARM_IDS[:4]},
                                   max_step=8)
            except Blocked as e:  # leaving the floor the first step can lag (tips resting on the paper): go on up
                print(f"lift: {e}; going on to the survey pose")
        await s.move({i: min(1000, p + 40) for i, p in SURVEY_POSE.items()}, 2.0)
        await s.settle(arm.ARM_IDS, 2.0)
        await s.move(SURVEY_POSE, 1.0)
        print("survey pose:", await s.settle(arm.ARM_IDS, 1.0))


async def run(targets):
    S = latest_sheet()
    results = []
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        await s.move({arm.GRIPPER_ID: GRIPPER_CLOSED}, 0.8)
        await s.settle([arm.GRIPPER_ID], 0.8)
        try:
            for n in targets:
                try:
                    await touch(s, n, S, results)
                except Blocked as e:
                    print(f"T{n}: stopped, {e} (at {e.reached})")
                    up = {i: v + 40 for i, v in e.reached.items()}  # back off from above
                    await s.move(up, 1.0)
                    await s.settle(list(up), 1.0)
        finally:
            await s.move({i: min(1000, p + 40) for i, p in SURVEY_POSE.items()}, 2.0)
            await s.settle(arm.ARM_IDS, 2.0)
            await s.move(SURVEY_POSE, 1.0)
            await s.settle(arm.ARM_IDS, 1.0)
    out = REPORTS / f"touch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", type=int)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--hold", type=float, help="go down to this height (m, by the model) above one target and stay")
    ap.add_argument("--lift", action="store_true", help="up from where the arm is, back to the survey pose")
    a = ap.parse_args()
    if a.lift:
        return asyncio.run(lift())
    if a.hold is not None:
        if len(a.targets) != 1:
            raise SystemExit("--hold takes one target")
        return asyncio.run(hold(a.targets[0], a.hold))
    if a.plan:
        S = latest_sheet()
        for n in grid_sheet.TARGETS:
            p = plan_target(n, S)
            print(f"T{n:2d}: " + (f"{p[0]} deg down, hover {p[1][0][1]}, down to z {p[1][-1][0]:+.3f}" if p else "not reachable"))
        return
    asyncio.run(run(a.targets))


if __name__ == "__main__":
    main()
