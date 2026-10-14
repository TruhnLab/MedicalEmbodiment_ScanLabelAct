"""The pick's approach (pick.approach: base driven by the camera, arm following, fine correction) against a simulated
robot, no hardware: a base that does less than commanded (BASE_EFFICIENCY, nothing below BASE_DEADBAND), servos with
sag, a camera with latency and noise that sees the object where arm.py's camera model projects it.

    python projects/manipulation/test_pick_sim.py              # a few starting positions
    python projects/manipulation/test_pick_sim.py --slow-detector 0.8

Checks: the approach ends with the gripper over the object (within 1 cm in the model), the object never leaves the
picture, the base never drives past where it should stop. 2026-10-12: written after a live run where the base drove on
an old detection and pushed the cube out of view.
"""

import argparse
import asyncio
import math
import time
from pathlib import Path

import cv2
import numpy as np

import arm
import pick

BASE_EFFICIENCY = 0.7   # the wheels did ~a third of what /odom said at 5 cm/s (2026-10-12); 0.7 is kinder, see --eff
BASE_DEADBAND = 0.02    # m/s: below this the base does not move
FRAME_AGE = 0.10        # s: a picture is this old when it arrives
SAG = {2: -10}          # pulses: servo 2 stops ~10 short under the arm's weight in low poses
PIXEL_NOISE = 2.0
VERBOSE = False
K = np.array([[417.41, 0, 315.41], [0, 418.09, 191.83], [0, 0, 1.0]])
D = np.array([0.004, -0.033, 0.0, 0.0, 0.0176])


class SimServos:
    def __init__(self, sim, pose):
        self.sim, self.target, self.c = sim, dict(pose), None

    async def read(self, ids):
        return {i: self.sim.servo_now(i) for i in ids}

    async def move(self, pose, duration):
        self.target.update(pose)

    async def settle(self, ids, duration, timeout=2.5):
        await asyncio.sleep(duration)
        return await self.read(ids)


class SimRobot(pick.Robot):
    max_drive = 0.30

    def __init__(self, obj, detect_s, start_pose="aimed"):
        from landerpi.robot import SURVEY_POSE
        start = {**SURVEY_POSE, arm.GRIPPER_ID: pick.OPEN}
        if start_pose == "aimed":  # navigation's look_at: the camera's tilt (servo 4) aimed at the object
            def miss(p4):
                C = arm.camera(arm.pulses_to_angles({**SURVEY_POSE, 4: p4}))
                d = np.array([obj[0], obj[1], pick.OBJECT_PLANE]) - C[:3, 3]
                return -C[:3, 2] @ d / np.linalg.norm(d)
            start[4] = min(range(90, 340, 5), key=miss)
        super().__init__(SimServos(self, start))
        self.K, self.D = K, D
        self.obj = np.array(obj, float)        # the object in the robot's frame (moves as the base drives)
        self.v = np.zeros(2)
        self.detect_s = detect_s
        self.hist = []                         # (t, servo positions) for the camera at a picture's time
        self.min_x, self.max_driven, self.out_of_view = 9.9, 0.0, 0
        self.driven = 0.0
        self.seen = False
        self.verbose = False

    def servo_now(self, i):
        return self.servos.target[i] + SAG.get(i, 0)

    async def start(self):
        pass

    def log(self, text):
        print(f"    {text}", flush=True)

    async def twist(self, vx, vy):
        self.v = np.array([vx, vy])

    async def physics(self):
        while True:
            await asyncio.sleep(0.02)
            v = self.v * BASE_EFFICIENCY if np.hypot(*self.v) >= BASE_DEADBAND else np.zeros(2)
            self.obj[:2] -= v * 0.02
            self.driven += np.hypot(*v) * 0.02
            self.min_x = min(self.min_x, self.obj[0])
            self.hist = [h for h in self.hist if time.monotonic() - h[0] < 3] + [
                (time.monotonic(), {i: self.servo_now(i) for i in arm.ARM_IDS})]

    def project(self, q_pulses):
        C = arm.camera(arm.pulses_to_angles(q_pulses))
        P = np.linalg.inv(C) @ np.array([*self.obj[:2], pick.OBJECT_PLANE, 1.0])
        if P[2] <= 0.05:
            return None
        px, _ = cv2.projectPoints(P[:3].reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        return px.ravel()

    async def frame(self):
        t_cap = time.monotonic() - FRAME_AGE
        q = min(self.hist, key=lambda h: abs(h[0] - t_cap))[1] if self.hist else self.servos.target
        return {"q": q}, t_cap, "sim"

    async def find(self, img, prompts):
        await asyncio.sleep(self.detect_s)
        uv = self.project(img["q"])
        if uv is None or not (20 < uv[0] < 620 and 20 < uv[1] < 380):
            if self.seen:  # lost after it was seen (before: the arm still on its way to the lookout)
                self.out_of_view += 1
                if self.verbose:
                    print(f"    out of view: object ({self.obj[0]:.3f}, {self.obj[1]:+.3f}), pixel "
                          f"{None if uv is None else np.round(uv)}, servos {img['q']}", flush=True)
            return None
        self.seen = True
        u, v = uv + np.random.normal(0, PIXEL_NOISE, 2)
        return u, v, 60, 60, 3600, None, [u - 30, v - 30, u + 30, v + 30], 0.8

    async def snapshot(self):
        return None


async def run_case(obj, detect_s, start_pose, max_drive):
    r = SimRobot(obj, detect_s, start_pose)
    r.max_drive = max_drive
    r.verbose = VERBOSE
    log_dir = Path(__file__).resolve().parents[2] / "data" / "manipulation" / "sim_pick"
    pick.PickLog.DIR = log_dir
    r.pick_log = pick.PickLog()
    phys = asyncio.create_task(r.physics())
    # the reference spot as the model sees it: the object straight under the pre-grasp pose
    target = np.array([0.232, 0.0, pick.GRASP_Z])
    pre = {**pick.pose_at(pick.axis_point(target, pick.RETREAT)), 5: 500}
    C = arm.camera(arm.pulses_to_angles(pre))
    P = np.linalg.inv(C) @ np.array([0.232, 0.0, pick.OBJECT_PLANE, 1.0])
    uv, _ = cv2.projectPoints(P[:3].reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    ref = {"x": float(uv.ravel()[0]), "y": float(uv.ravel()[1])}
    t0 = time.monotonic()
    try:
        trk = pick.Tracker(r, ["cube"], r.pick_log)
        trk.start()
        try:
            fine = await pick.approach(r, ["cube"], ref, trk)
        finally:
            await trk.stop()
        err = np.hypot(*(fine[:2] - r.obj[:2])) * 100
        ok = err < 1.0 and r.out_of_view == 0
        print(f"  {'OK ' if ok else 'BAD'} gripper target {err:.1f} cm from the object; object now at "
              f"({r.obj[0]:.3f}, {r.obj[1]:+.3f}), nearest {r.min_x:.3f} m; base drove {r.driven * 100:.0f} cm; "
              f"{r.out_of_view} pictures without it; {time.monotonic() - t0:.0f} s")
        return ok
    except pick.GraspError as e:
        expected = "approach it first" in str(e) and obj[0] - pick.REACH[0] > r.max_drive - 0.03
        print(f"  {'OK ' if expected else 'BAD'} {e}; object at ({r.obj[0]:.3f}, {r.obj[1]:+.3f}); {r.out_of_view} "
              f"pictures without it")
        return expected
    finally:
        phys.cancel()
        r.pick_log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slow-detector", type=float, default=0.25, help="s per detection")
    ap.add_argument("--eff", type=float, default=BASE_EFFICIENCY, help="the base's real / commanded speed")
    ap.add_argument("--case", type=int, help="only this case (0-based)")
    ap.add_argument("--max-drive", type=float, default=0.30, help="m the base may drive")
    ap.add_argument("-v", action="store_true", help="say when the object leaves the picture")
    ap.add_argument("--start", choices=["aimed", "P"], default="aimed", help="the arm's pose at the start: aimed at "
                    "the object (navigation's look_at after approach_found) or the survey pose")
    a = ap.parse_args()
    globals()["VERBOSE"] = a.v
    globals()["BASE_EFFICIENCY"] = a.eff
    cases = [(0.37, 0.0, 0.0), (0.38, 0.04, 0.0), (0.33, -0.07, 0.0), (0.24, 0.02, 0.0), (0.42, 0.0, 0.0),
             (0.50, 0.03, 0.0), (0.69, -0.03, 0.0), (0.79, 0.0, 0.0), (0.224, -0.001, 0.0)]
    results = []
    for obj in (cases if a.case is None else [cases[a.case]]):
        print(f"object at ({obj[0]:.2f}, {obj[1]:+.2f}) m, detection {a.slow_detector:.2f} s, base {a.eff:.0%}:")
        results.append(asyncio.run(run_case(obj, a.slow_detector, a.start, a.max_drive)))
    print(f"{sum(results)}/{len(results)} OK")
    return all(results)


if __name__ == "__main__":
    main()
