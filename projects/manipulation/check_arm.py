"""Checks and calibration of the arm and gripper on the robot (run while you watch it; nothing else may move the
arm meanwhile: no task console drive or chat, no exploration).

    python projects/manipulation/check_arm.py health    # no motion: servo positions, measured and commanded
    python projects/manipulation/check_arm.py gripper   # gripper range + jaw opening by pulse (a ruler) -> config/gripper.json
    python projects/manipulation/check_arm.py grip      # close on an object held between the jaws: where it stops
    python projects/manipulation/check_arm.py joints    # commanded vs reached, from both sides, under load
    python projects/manipulation/check_arm.py gripper-to 400   # one gripper position (e.g. to measure its opening)

Every motion waits for Enter first (--yes: not; someone must watch the robot); Ctrl+C aborts (the servos keep their last position). Positions are read from
the servos themselves (ros_robot_controller's bus_servo/get_state): /controller_manager/servo_states only echoes the
last command (Hiwonder's ServoManager.set_position records what it sent; read on the robot 2026-10-09). Hiwonder's
read waits for the servo's answer without a timeout and under a lock: one lost answer blocks all later reads (and the
node's other callbacks) until ros_robot_controller restarts. Reports go to data/calibration/.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import arm
from landerpi.robot import ROSBRIDGE_URL, SURVEY_POSE, TOPICS
from landerpi.rosbridge import RosbridgeClient

REPO = Path(__file__).resolve().parents[2]
GRIPPER_FILE = REPO / "config" / "gripper.json"
REPORTS = REPO / "data" / "calibration"
GET = "/ros_robot_controller/bus_servo/get_state"
ALL_IDS = arm.ARM_IDS + [arm.GRIPPER_ID]

SETTLE_TOL = 2         # pulses: two reads this close count as standing still
STALL = 20             # pulses between command and reached: blocked (a limit, or an object in the gripper)
GRIP_STEP = 20         # pulses per gripper step while searching the limits
# closing on an object: contact = the servo lags this much behind its command. At 20 pulses (and 10-pulse steps)
# a 29.5 mm block was "touched" at 403 (26 mm by the table; touching is at ~376): the jaws had already bent by
# 3-4 mm, a firm grip (2026-10-12). The servos' deadband is ~5 pulses (joints check)
CONTACT_LAG = 8
CONTACT_STEP = 5
SQUEEZE = 15           # pulses past the contact while holding: the 29.5 mm block held firmly at a 14-pulse lag,
                       # "still okay" at 10 (2026-10-12); lifting adds its weight and the arm's jolts


class Servos:
    def __init__(self, client):
        self.c = client
        self.topic = {}          # id -> last commanded position (/controller_manager/servo_states)
        self.last = {}           # id -> last measured position

    async def start(self):
        await self.c.subscribe(*TOPICS["servo_state"], self._on_topic, throttle_ms=100)

    def _on_topic(self, msg, recv):
        for s in msg.get("servo_state", []):
            self.topic[s["id"]] = s["position"]

    async def read(self, ids=ALL_IDS):
        """{id: position} measured by the servos. Positions only: asking also for offset, voltage, temperature,
        limits and torque state of all six servos got no answer and hung ros_robot_controller (2026-10-12, a
        reboot freed it); positions alone answer in ~50 ms."""
        cmd = [{"id": i, "get_id": 0, "get_position": 1, "get_offset": 0, "get_voltage": 0, "get_temperature": 0,
                "get_position_limit": 0, "get_voltage_limit": 0, "get_max_temperature_limit": 0,
                "get_torque_state": 0} for i in ids]
        out = {}
        for attempt in range(3):
            try:
                r = await self.c.call_service(GET, {"cmd": cmd}, timeout=3.0)
            except (RuntimeError, asyncio.TimeoutError):
                raise SystemExit("the servos don't answer (ros_robot_controller hangs or is not running): reboot the "
                                 "robot")
            out.update({i: v for i, v in ((i, (s.get("position") or [None])[-1])
                                          for i, s in zip(ids, r.get("state", []))) if v is not None})
            if all(i in out for i in ids):
                break
        # a servo the board did not answer for (now and then one, 2026-10-12: a pick crashed on it): its last value
        for i in ids:
            if i not in out:
                out[i] = self.last.get(i)
        self.last.update({i: v for i, v in out.items() if v is not None})
        return out

    async def move(self, pose, duration):
        await self.c.publish(*TOPICS["servo_cmd"], {
            "duration": float(duration), "position_unit": "pulse",
            "position": [{"id": int(i), "position": int(p)} for i, p in pose.items()]})

    async def settle(self, ids, duration, timeout=2.5):
        """Waits for the move, then until the servos stand still; returns their positions."""
        await asyncio.sleep(duration)
        last, t0 = await self.read(ids), time.monotonic()
        while time.monotonic() - t0 < timeout:
            await asyncio.sleep(0.08)  # 0.15 made a pick's descent, closing and lift take ~85 s (2026-10-12)
            now = await self.read(ids)
            if all(now[i] is not None and last[i] is not None and abs(now[i] - last[i]) <= SETTLE_TOL for i in ids):
                return now
            last = now
        return last


YES = False  # --yes: no Enter before motions (someone watches the robot; conda run passes no input)


class Blocked(Exception):
    """A guarded move stopped: a joint no longer follows its command (contact or a mechanical stop)."""

    def __init__(self, joint, lag, reached):
        super().__init__(f"servo {joint} lags {lag} pulses behind its command")
        self.joint, self.lag, self.reached = joint, lag, reached


class Stopped(Exception):
    """STOP in the console (or disarming) during a motion: the servos were told to stay where they are."""


async def guarded_move(s, pose, max_step=15, lag_limit=20, baseline=None, stop=None):
    """Moves the arm servos to pose in steps of at most max_step pulses; after each step, if a servo lags more than
    lag_limit pulses behind (or, with baseline {id: lag}, its lag changed by more than lag_limit), it stops pushing
    (commands where the servos are) and raises Blocked. Gravity alone makes up to 13 pulses, the deadband ~5 (joints
    check, 2026-10-12). stop: a function returning True when STOP was pressed (then Stopped). Returns the reached
    positions."""
    ids = list(pose)
    now = await s.read(ids)
    steps = max(1, math.ceil(max(abs(pose[i] - now[i]) for i in ids) / max_step))
    start = dict(now)
    for k in range(1, steps + 1):
        if stop is not None and stop():
            await s.move(now, 0.2)
            raise Stopped()
        cmd = {i: round(start[i] + (pose[i] - start[i]) * k / steps) for i in ids}
        await s.move(cmd, 0.12)
        now = await s.settle(ids, 0.12, timeout=1.0)
        for i in ids:
            lag = cmd[i] - now[i]
            if abs(lag - (baseline or {}).get(i, 0)) > lag_limit:
                await s.move(now, 0.2)
                raise Blocked(i, lag, now)
    return now


async def ask(prompt):
    """input() without blocking the event loop (the rosbridge connection must keep answering pings)."""
    if YES:
        print(prompt + "(--yes)")
        return ""
    return (await asyncio.to_thread(input, prompt)).strip()


def save_report(kind, data):
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"report: {path.relative_to(REPO)}")


# ---------- health (no motion) ----------

async def health(s):
    d = await s.read(ALL_IDS)
    await asyncio.sleep(0.5)
    print(f"{'servo':>6} {'measured':>9} {'commanded':>10}")
    for i in ALL_IDS:
        print(f"{i:>6} {d.get(i)!s:>9} {s.topic.get(i)!s:>10}")
    f = arm.fk(arm.pulses_to_angles({i: d[i] for i in arm.ARM_IDS if d.get(i) is not None}))
    print("gripper tip (end_effector_link) in base_footprint:", np.round(f["end_effector_link"][:3, 3], 3),
          "| camera:", np.round(f["depth_camera_link"][:3, 3], 3))
    save_report("health", {"measured": d, "commanded": s.topic})


# ---------- gripper range and opening ----------

async def find_limit(s, start, direction):
    """Steps the gripper until it stops following (or the range ends); returns (last reached position, why)."""
    target = start
    while True:
        target = max(0, min(1000, target + direction * GRIP_STEP))
        await s.move({arm.GRIPPER_ID: target}, 0.2)
        got = (await s.settle([arm.GRIPPER_ID], 0.2))[arm.GRIPPER_ID]
        print(f"  command {target:4d} -> {got}", end="\r")
        if got is None:
            raise SystemExit("\nno gripper position")
        if abs(got - target) > STALL:
            await s.move({arm.GRIPPER_ID: got}, 0.2)  # stop pushing against the stop
            print(f"\n  blocked at {got} (command {target})")
            return got, "blocked"
        if target in (0, 1000):
            print(f"\n  end of the servo range at {got}")
            return got, "range end"


async def gripper(s):
    p0 = (await s.read([arm.GRIPPER_ID]))[arm.GRIPPER_ID]
    print(f"gripper (servo {arm.GRIPPER_ID}) at {p0}. It will step up to its limit, then down to the other one, "
          f"{GRIP_STEP} pulses at a time. Keep fingers out of the jaws.")
    await ask("Enter to start ")
    hi, hi_why = await find_limit(s, p0, +1)
    lo, lo_why = await find_limit(s, hi, -1)
    print(f"range {lo}..{hi}. Now the opening at a few positions: measure the inner gap between the jaw pads "
          f"(mm, Enter skips one).")
    widths = []
    for p in np.linspace(lo + 10, hi - 10, 6).round().astype(int):
        await s.move({arm.GRIPPER_ID: int(p)}, 0.8)
        got = (await s.settle([arm.GRIPPER_ID], 0.8))[arm.GRIPPER_ID]
        a = await ask(f"  at {got} (command {p}): gap in mm? ")
        if a:
            widths.append([int(got), float(a.replace(",", "."))])
    data = {"date": time.strftime("%Y-%m-%d %H:%M"), "servo": arm.GRIPPER_ID, "low": int(lo), "low_stop": lo_why,
            "high": int(hi), "high_stop": hi_why, "widths_mm": widths}
    if len(widths) >= 2:
        w = np.array(widths)
        data["closed"], data["open"] = (int(lo), int(hi)) if w[0, 1] < w[-1, 1] else (int(hi), int(lo))
        print(f"closed = {data['closed']}, open = {data['open']}, opening {w[:, 1].min():.0f}-{w[:, 1].max():.0f} mm "
              f"in the measured range")
    else:
        print("fewer than 2 widths: which end is closed stays unknown")
    GRIPPER_FILE.write_text(json.dumps(data, indent=1) + "\n")
    print(f"saved {GRIPPER_FILE.relative_to(REPO)}")


def width_at(g, pulse):
    w = np.array(sorted(g["widths_mm"]))
    return float(np.interp(pulse, w[:, 0], w[:, 1]))


async def grip(s):
    if not GRIPPER_FILE.exists() or "closed" not in json.loads(GRIPPER_FILE.read_text()):
        raise SystemExit("run the gripper check first (config/gripper.json with widths)")
    g = json.loads(GRIPPER_FILE.read_text())
    direction = 1 if g["closed"] > g["open"] else -1
    await ask("Enter opens the gripper ")
    await s.move({arm.GRIPPER_ID: g["open"]}, 1.0)
    await s.settle([arm.GRIPPER_ID], 1.0)
    while True:
        a = await ask("hold an object between the jaws, Enter closes (q quits) ")
        if a.lower() == "q":
            return
        target = g["open"]
        contact = None
        while (target - g["closed"]) * direction < 0:
            target = target + direction * min(CONTACT_STEP, abs(g["closed"] - target))
            await s.move({arm.GRIPPER_ID: target}, 0.1)
            got = (await s.settle([arm.GRIPPER_ID], 0.1))[arm.GRIPPER_ID]
            if abs(got - target) > CONTACT_LAG:
                contact = got
                break
        if contact is None:
            print(f"closed fully ({g['closed']}): nothing held")
        else:
            hold = contact + direction * SQUEEZE
            await s.move({arm.GRIPPER_ID: hold}, 0.3)
            got = (await s.settle([arm.GRIPPER_ID], 0.3))[arm.GRIPPER_ID]
            print(f"contact at {contact} (~{width_at(g, contact):.0f} mm); holding with command {hold}, "
                  f"reached {got}. Pull gently: is it held firmly?")
        if YES:
            print("holding 10 s, then opening")
            await asyncio.sleep(10)
        else:
            await ask("Enter opens ")
        await s.move({arm.GRIPPER_ID: g["open"]}, 0.8)
        await s.settle([arm.GRIPPER_ID], 0.8)
        if YES:
            return  # one try per run: nobody presses Enter for the next


async def gripper_to(s, pulse):
    await s.move({arm.GRIPPER_ID: pulse}, 0.8)
    got = (await s.settle([arm.GRIPPER_ID], 0.8))[arm.GRIPPER_ID]
    print(f"gripper: command {pulse}, reached {got}")


# ---------- joints: commanded vs reached ----------

# the test poses: around the survey and look-down poses, the arm stretched forward (most load on joints 2 and 3),
# pan and wrist roll to both sides. Each is approached from below and from above (DELTA on every arm joint).
JOINT_POSES = {
    "survey": SURVEY_POSE,
    "look down": {**SURVEY_POSE, 4: 170},  # navigation's LOOK_DOWN_POSE; the start pose (4: 100) approached
                                           # from below brings the gripper within the chassis margin
    "forward": {1: 500, 2: 500, 3: 150, 4: 500, 5: 500},
    "pan left": {**SURVEY_POSE, 1: 650},
    "pan right": {**SURVEY_POSE, 1: 350},
    "roll": {**SURVEY_POSE, 5: 700},
}
DELTA = 40
MOVE_S = 1.5


async def joints(s):
    plan = []
    for name, pose in JOINT_POSES.items():
        for side in (-1, 1):
            pre = {i: p + side * DELTA for i, p in pose.items()}
            for label, P in ((f"{name} (before, {'below' if side < 0 else 'above'})", pre), (name, pose)):
                hit = arm.collision(arm.pulses_to_angles(P))
                if hit:
                    raise SystemExit(f"pose {label} {P}: {hit}; not moving")
                plan.append((label, P, side))
    for name, pose in JOINT_POSES.items():
        tip = arm.fk(arm.pulses_to_angles(pose))["end_effector_link"][:3, 3]
        print(f"  {name:10s} {pose}  gripper tip at {np.round(tip, 2)} m")
    print(f"{len(plan)} moves of {MOVE_S} s, then back to the survey pose. Keep clear of the arm.")
    await ask("Enter to start ")
    rows = []
    for label, P, side in plan:
        await s.move(P, MOVE_S)
        got = await s.settle(arm.ARM_IDS, MOVE_S)
        if label in JOINT_POSES:
            err = {i: got[i] - P[i] for i in arm.ARM_IDS}
            rows.append({"pose": label, "from": "below" if side < 0 else "above", "command": P, "reached": got,
                         "error": err})
            print(f"  {label:10s} from {rows[-1]['from']:5s}: reached - commanded = "
                  + " ".join(f"{i}:{e:+d}" for i, e in err.items()))
    await s.move(SURVEY_POSE, MOVE_S)
    await s.settle(arm.ARM_IDS, MOVE_S)
    print("\nper servo: mean error (sag / offset), |below - above| (play), in pulses (1 pulse = 0.24 deg)")
    for i in arm.ARM_IDS:
        e = [r["error"][i] for r in rows]
        play = [abs(a["error"][i] - b["error"][i]) for a, b in zip(rows[::2], rows[1::2])]
        print(f"  servo {i}: mean {np.mean(e):+.1f}, max |error| {max(map(abs, e))}, play max {max(play)}")
    save_report("joints", {"delta": DELTA, "rows": rows})


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("check", choices=["health", "gripper", "grip", "joints", "gripper-to"])
    p.add_argument("pulse", type=int, nargs="?", help="gripper-to: the position (0-1000)")
    p.add_argument("--yes", action="store_true", help="no Enter before motions (gripper: no widths asked)")
    a = p.parse_args()
    global YES
    YES = a.yes
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        if a.check == "gripper-to":
            if a.pulse is None or not 0 <= a.pulse <= 1000:
                raise SystemExit("usage: gripper-to <pulse 0-1000>")
            return await gripper_to(s, a.pulse)
        await {"health": health, "gripper": gripper, "grip": grip, "joints": joints}[a.check](s)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
