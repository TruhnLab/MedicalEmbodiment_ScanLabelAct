"""Picking an object up from the floor and putting it down, steered by the camera: the core behind grasp.py (scripts)
and manipulation_tools.py (the chat, inside the navigator). No process of its own: it talks to the robot through a
Robot (below), so the navigator lends its rosbridge connection, odometry and detector instead of a second set.

    r = StandaloneRobot(Servos(client))      # or NavRobot(nav) in manipulation_tools.py
    await r.start()
    result = await pick(r, prompts=["red block", "red cube"])   # follow it in, grasp, lift, carry in the survey pose
    await put_down(r)                        # in front of the robot, then the survey pose

The camera sits on the gripper (only the wrist roll between), so "gripper straight above the object" always looks the
same in the picture: the object at one spot (config/grasp_reference.json, taught with grasp.py down / teach). A pick
moves the arm until the object sits there, goes down along the gripper's axis (DOWN_DEG), closes until the gripper
stalls, lifts. Measured 2026-10-12 with a 30 mm block (grasp.py): centred in 3-4 steps to 4-8 px, contact at its width.
"""

import asyncio
import json
import math
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import arm
from check_arm import REPO, REPORTS, Blocked, Stopped, guarded_move
from landerpi.robot import ROBOT_HOST, SURVEY_POSE, TOPICS

REFERENCE = REPO / "config" / "grasp_reference.json"
SNAPSHOT = f"http://{ROBOT_HOST}:8080/snapshot?topic={TOPICS['rgb'][0]}"  # '/' literal (web_video_server)

DOWN_DEG = 60          # 70 deg reached only up to 7 cm above the floor at 24 cm ahead, 60 deg up to 11 cm (2026-10-12)
GRASP_Z = 0.015        # jaw tips at half the 30 mm block's height
RETREAT = 0.10         # pre-grasp: this far back along the gripper's axis (~9 cm higher, the object fully in view)
OPEN, CLOSED = 201, 545  # config/gripper.json
FREE_STEP = 15         # pulses per guarded step (30 tripped the lag stop under load, 2026-10-12)
SURVEY_MOVE_S = 1.2    # s for the smooth move back to P
FREE_LAG = 30          # pulses: in low stretched poses servo 2 sags 21-23 while moving (10 at rest; a stop at 20
                       # tripped twice in free air, 2026-10-12)
PIXEL_TOL = 8          # px from the reference spot: close enough (~4 mm on the floor at this distance)
GAIN = 0.7             # part of the measured error corrected per step: the full error overshot and swung
                       # 273 / 246 / 267 / 250 px around 259 (servo deadband, 2026-10-12)
OBJECT_PLANE = 0.02    # m: the object's visible top and front are ~2 cm up (the 30 mm block)
CONTACT_LAG, CONTACT_STEP, SQUEEZE = 8, 8, 15  # check_arm.py grip: a 29.5 mm block held firmly at a 14-pulse lag;
                                               # steps of 5 took ~10 s to close (2026-10-12)
REACH = np.array([0.232, 0.0])  # where the base puts an object before grasping: the middle of the 60 deg reach
REACH_X_MAX = 0.245    # grasp pose reachable at 60 deg up to ~0.24 m ahead (T4-T6 yes, T7-T9 no, 2026-10-12)
REACH_Y_MAX = 0.06
GRASP_X_MAX = 0.240    # at 0.245 servo 2 sagged 23 pulses going down (2026-10-12): drive closer first
SCORE_MIN = 0.05       # as perception.SCORE_MIN


class GraspError(Exception):
    """A failure the chat can pass on as it is."""


# ---------- the robot, as the grasp needs it ----------

class Robot:
    """What a grasp uses; subclasses provide the connection. servos: check_arm.Servos on the robot's rosbridge."""
    max_drive = 0.30       # m the base may drive during one pick (approach_found stops ~20 cm in front: the
                           # object ~0.37 m from the centre, the grasp needs it at 0.23)

    def __init__(self, servos, detector=None):
        self.servos = servos
        self.detector = detector
        self.detector_lock = None  # a threading.Lock if another thread uses the same detector
        self.arm_now = None        # the servos' latest measured positions (approach: for blocked())
        self.pick_log = None       # PickLog of the running pick
        self.K = self.D = None

    _intrinsics = None     # (K, D), read once per process: every subscribe adds a callback to the client

    async def start(self):
        """The colour camera's intrinsics (one camera_info message)."""
        if Robot._intrinsics is not None:
            self.K, self.D = Robot._intrinsics
            return
        got = asyncio.get_running_loop().create_future()
        c = self.servos.c
        await c.subscribe(*TOPICS["rgb_info"], lambda m, r: got.done() or got.set_result(m))
        try:
            m = await asyncio.wait_for(got, 5.0)
        except asyncio.TimeoutError:
            raise GraspError("no camera_info from the robot's colour camera")
        self.K, self.D = Robot._intrinsics = np.array(m["k"]).reshape(3, 3), np.array(m["d"])

    def log(self, text):
        print(text, flush=True)

    def stopped(self):
        return False

    def show(self, found=None, label=""):
        """The console's camera overlay (NavRobot): the object's box in the picture just taken, or nothing (None)
        while the arm moves."""

    async def twist(self, vx, vy):
        await self.servos.c.publish(*TOPICS["cmd_vel"], {"linear": {"x": vx, "y": vy, "z": 0.0},
                                                         "angular": {"x": 0.0, "y": 0.0, "z": 0.0}})

    def accepts(self, label, prompts):
        """Whether a detection's label means the object: one of the prompts, or a longer word containing one. With
        navigation's prompts kept (they contained ours) the cube came as "red cube toy" (0.92) and only "red cube"
        counted: three picks lost it (live, 2026-10-12 16:24)."""
        label = label.lower()
        return any(p.lower() == label or p.lower() in label for p in prompts)

    def context(self):
        """What else to record with a pick's progress (NavRobot: the navigator's localization)."""
        return {}

    def blocked(self, vx, vy, obj):
        """Whether driving at (vx, vy) would hit something that is not the object at obj (NavRobot: the lidar)."""
        return False

    async def frame(self):
        """(the newest colour picture, when it was taken (time.monotonic()), its source). Here a snapshot from the
        robot's web_video_server: its newest frame, taken about when asked for."""
        t = time.monotonic()
        return await self.snapshot(), t, "snapshot"

    async def snapshot(self):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, lambda: urllib.request.urlopen(SNAPSHOT, timeout=5).read())
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    async def find(self, img, prompts):
        """(centre x, centre y, width, height, area, mask or None, box, score) in pixels, or None: the detector's best
        match for the prompts (centre of its mask), or without a detector the largest red blob."""
        if self.detector is None or not prompts:
            b = find_red(img)
            if b is None:
                return None
            cx, cy, w, h, area = b
            return (*b, None, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1.0)
        det, lock = self.detector, self.detector_lock

        def detect():
            # the detector's own prompts kept if they contain ours: switching takes ~2 s (YOLOE loads its text
            # encoder anew), the first picture after it ~1 s; navigation's search had ours plus a synonym, and every
            # pick switched (2026-10-12)
            if not set(prompts) <= set(det.prompts):
                det.set_prompts(prompts)
            return [d for d in det.detect(img) if self.accepts(d["label"], prompts)]

        def run():
            if lock is None:
                return detect()
            with lock:  # the navigator's detection thread shares the detector (live.detector_lock)
                return detect()
        dets = [d for d in await asyncio.get_running_loop().run_in_executor(None, run) if d["score"] >= SCORE_MIN]
        if not dets:
            return None
        d = dets[0]
        x0, y0, x1, y1 = d["box"]
        if d["mask"] is not None and d["mask"].sum() > 100:
            ys, xs = np.nonzero(d["mask"])
            cx, cy, area = float(xs.mean()), float(ys.mean()), int(d["mask"].sum())
        else:
            cx, cy, area = (x0 + x1) / 2, (y0 + y1) / 2, int((x1 - x0) * (y1 - y0))
        return cx, cy, int(x1 - x0), int(y1 - y0), area, d["mask"], d["box"], d["score"]


class StandaloneRobot(Robot):
    """For scripts: its own rosbridge client."""
    max_drive = 0.10       # the robot is on a cable during the calibration sessions (user, 2026-10-12)


# ---------- geometry ----------

def axis_point(target, back):
    """The jaw tips' position `back` m back along the approach axis from target (towards the robot and up)."""
    return np.asarray(target, float) - back * arm.approach_dir(target, DOWN_DEG)


def pose_at(point, q0=None):
    """Arm pulses (servos 1-4) with the jaw tips at point, gripper DOWN_DEG down; None if not reachable. A seed q0
    near the answer (the arm following) is tried alone first."""
    best = arm.ik(point, DOWN_DEG, q0, iterations=80) if q0 is not None else None
    if best is None or best[1] > 0.002:
        seeds = arm.planar_ik(point, DOWN_DEG)[:1] + ([q0] if q0 is not None else [])
        best = min((arm.ik(point, DOWN_DEG, s, iterations=80) for s in seeds), key=lambda r: r[1], default=None)
    if best is None or best[1] > 0.002 or arm.collision(best[0], chassis=False):
        return None
    return {i: v for i, v in arm.angles_to_pulses(best[0]).items() if i in arm.ARM_IDS[:4]}


def floor_point(px, q, K, D, z=OBJECT_PLANE):
    """Where the camera ray through pixel px meets the plane at height z (base_footprint)."""
    C = arm.camera(q)
    u = cv2.undistortPoints(np.array([[px]], np.float64), K, D).ravel()
    ray = C[:3, :3] @ np.array([u[0], u[1], 1.0])
    o = C[:3, 3]
    return o + (z - o[2]) / ray[2] * ray


def find_red(img):
    """The largest red blob: (centre x, centre y, width, height, area) in pixels, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 150:
        return None
    x, y, w, h = stats[i, :4]
    return float(cent[i][0]), float(cent[i][1]), int(w), int(h), int(stats[i, cv2.CC_STAT_AREA])


def width_mm(pulse):
    g = json.loads((REPO / "config" / "gripper.json").read_text())
    w = np.array(sorted(g["widths_mm"]))
    return float(np.interp(pulse, w[:, 0], w[:, 1])) + 3.5  # the jaws give ~3.5 mm at contact (gripper.json)


# ---------- motions ----------

async def go(r, pose, max_step=FREE_STEP):
    """Guarded move of servos 1-4, checked for collisions on the way first; through the survey pose if the direct
    way would bring the gripper over the robot's front (after navigation's look_at the camera tilt is far down: the
    first chat pick was refused there, 2026-10-12)."""
    now = await r.servos.read(arm.ARM_IDS)
    hit = arm.path_collision(arm.pulses_to_angles(now), arm.pulses_to_angles({**pose, 5: now[5]}), chassis=False)
    if hit:
        await survey(r)
        now = await r.servos.read(arm.ARM_IDS)
        hit = arm.path_collision(arm.pulses_to_angles(now), arm.pulses_to_angles({**pose, 5: now[5]}), chassis=False)
    if hit:
        raise GraspError(f"the arm cannot move there without touching the robot ({hit}); this is no matter of the "
                         f"robot's position: report it")
    try:
        return await guarded_move(r.servos, pose, max_step=max_step, lag_limit=FREE_LAG, stop=r.stopped)
    except Blocked as e:
        raise GraspError(f"the arm stopped: something is in its way ({e})")


async def move_along(r, target, back_from, back_to, step=0.02):
    """Jaw tips from back_from to back_to (m back along the axis) in straight-line steps."""
    n = max(1, int(round(abs(back_to - back_from) / step)))
    q = None
    for k in range(1, n + 1):
        b = back_from + (back_to - back_from) * k / n
        pose = pose_at(axis_point(target, b), q)
        if pose is None:
            raise GraspError(f"the arm cannot reach {b * 100:.0f} cm above the object along its way down")
        q = arm.pulses_to_angles({**pose, 5: 500})
        try:  # hitting the object's top lags far more than the sag
            await guarded_move(r.servos, pose, max_step=20, lag_limit=FREE_LAG, stop=r.stopped)
        except Blocked as e:
            raise GraspError(f"the gripper hit something on its way down ({e})")


async def gripper(r, pulse, duration=0.8):
    await r.servos.move({arm.GRIPPER_ID: pulse}, duration)
    return (await r.servos.settle([arm.GRIPPER_ID], duration))[arm.GRIPPER_ID]


async def close_on_object(r):
    """Closes in small steps until the gripper stalls: lagging its command AND not advancing for two steps. A lag
    alone was a jaw pushing the off-centre block across the paper (first grasp, 2026-10-12: "contact" at 43 mm on a
    30 mm block); pushing on centres it between the jaws. Squeezes; returns the contact position or None."""
    target, prev, still = OPEN, None, 0
    while target < CLOSED:
        if r.stopped():
            raise Stopped()
        target = min(CLOSED, target + CONTACT_STEP)
        await r.servos.move({arm.GRIPPER_ID: target}, 0.1)
        got = (await r.servos.settle([arm.GRIPPER_ID], 0.1))[arm.GRIPPER_ID]
        still = still + 1 if prev is not None and got - prev < 2 and target - got > CONTACT_LAG else 0
        prev = got
        if still >= 2:
            await gripper(r, got + SQUEEZE, 0.3)
            return got
    return None


async def holding(r, contact):
    """Still pressing on something (the command is contact + SQUEEZE)."""
    g = (await r.servos.read([arm.GRIPPER_ID]))[arm.GRIPPER_ID]
    return contact + SQUEEZE - g > 5


async def survey(r):
    """Straight back to the survey pose (P; navigation drives with it) in one smooth move, its way checked first. No
    overshoot: going 40 pulses past P tilted the camera at the ceiling for a moment (user, 2026-10-12). Guarded steps
    of 15 pulses took ~11 s; free air needs no contact check on the way, only whether it got there."""
    p = {i: SURVEY_POSE[i] for i in range(1, 5)}
    now = await r.servos.read(arm.ARM_IDS)
    if arm.path_collision(arm.pulses_to_angles(now), arm.pulses_to_angles({**p, 5: now[5]}), chassis=False):
        try:  # the safe way: guarded steps
            await guarded_move(r.servos, p, max_step=FREE_STEP, lag_limit=FREE_LAG, stop=r.stopped)
        except Blocked as e:
            r.log(f"survey: {e}")
        return
    if r.stopped():
        raise Stopped()
    await r.servos.move(p, SURVEY_MOVE_S)
    got = await r.servos.settle(list(p), SURVEY_MOVE_S)
    if any(abs(got[i] - p[i]) > FREE_LAG for i in p):
        r.log(f"survey: the arm stopped short of P ({got})")


# ---------- following the object while getting there ----------

class PickLog:
    """One pick, for analysis and presentation videos (pick_video.py): every tracker picture as a JPEG and one JSON
    line per picture, observation, control step and event in log.jsonl, in data/manipulation/picks/<time>/; the last
    KEEP are kept (~5-10 MB each). The live run of 2026-10-12 14:43 could only be reconstructed from the user's
    description; this records what the controller saw and did. One folder per pick (it was overwritten by the next
    one): only the last of the day's picks was left for a demo video (2026-10-13). DIR set: that folder, overwritten
    (the simulation)."""
    ROOT = REPO / "data" / "manipulation" / "picks"
    KEEP = 30
    DIR = None

    def __init__(self):
        import shutil
        if self.DIR is not None:
            self.dir = Path(self.DIR)
            shutil.rmtree(self.dir, ignore_errors=True)
        else:
            self.dir = self.ROOT / time.strftime("%Y%m%d_%H%M%S")
            old = sorted(p for p in self.ROOT.glob("2*") if p.is_dir()) if self.ROOT.exists() else []
            for p in old[:max(0, len(old) - self.KEEP + 1)]:
                shutil.rmtree(p, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "log.jsonl", "a", encoding="utf-8")
        self.n = 0
        self.t0 = time.monotonic()
        self.line("clock", wall=time.time())  # the pick's t = 0 on the wall clock (the navigator's log, other recordings)

    def line(self, kind, **kw):
        kw = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
        self.f.write(json.dumps({"t": round(time.monotonic() - self.t0, 3), "kind": kind, **kw}, default=str) + "\n")
        self.f.flush()

    async def picture(self, img):
        if not isinstance(img, np.ndarray):  # a simulated picture (test_pick_sim.py)
            return None
        self.n += 1
        name = f"{self.n:04d}.jpg"
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80]))
        return name

    def close(self):
        self.f.close()


class Tracker:
    """Detection in the background, as fast as the detector goes: every picture's box goes to the console at once
    (user, 2026-10-12: "the detection box needs to be drawn in all stages"), the newest result to the controller with
    the moment its picture was taken (t_cap, time.monotonic()). An error is logged and the loop goes on: a silently
    dead tracker would leave the controller with an old box."""

    def __init__(self, r, prompts, log):
        self.r, self.prompts, self.log = r, prompts, log
        self.label = prompts[0] if prompts else "red block"
        self.latest = None     # {"t_cap", "det", "det_ms", "src"}
        self.n = 0
        self.errors = 0
        self.task = None

    async def run(self):
        while True:
            try:
                img, t_cap, src = await self.r.frame()
                t1 = time.monotonic()
                det = await self.r.find(img, self.prompts)
                det_ms = (time.monotonic() - t1) * 1000
                self.r.show(det, self.label)
                self.latest = {"t_cap": t_cap, "det": det, "det_ms": det_ms, "src": src}
                self.n += 1
                name = await self.log.picture(img)
                self.log.line("picture", n=self.n, file=name, src=src, age_ms=round((t1 - t_cap) * 1000),
                              det_ms=round(det_ms), box=None if det is None else [round(v) for v in det[6]],
                              centre=None if det is None else [round(det[0]), round(det[1])],
                              score=None if det is None else round(det[7], 2))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.errors += 1
                self.log.line("tracker_error", error=f"{type(e).__name__}: {e}")
                self.r.log(f"tracker: {type(e).__name__}: {e}")
                await asyncio.sleep(0.5)

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.r.show(None)


PRE_X_MAX = 0.28       # m: the farthest pre-grasp the arm reaches (T11 at 0.29 yes, T10/T12 at 0.29 +-6 cm no)
FAR_LOOK = np.array([0.27, 0.0, GRASP_Z])  # the first look: the camera sees ~0.2-0.5 m ahead (approach_found stops
                                           # ~20 cm in front: the object ~0.37 m from the centre, 2026-10-12)
BASE_TOL = 0.012       # m: the object this close to where the base should put it
BASE_GAIN = 0.8        # 1/s
BASE_MIN, BASE_MAX = 0.03, 0.06  # m/s: at the end of an odometry drive the wheels hardly turned (/odom said 10 cm,
                                 # the camera 3 cm, 2026-10-12)
MAX_AGE = 1.5          # s: the base drives only on a picture this fresh, moved on by what it drove since. Without
                       # that it drove on for 1.5 s with an old box and pushed the cube out of view (live run 2026-10-12
                       # 14:44, user: "the detections stopped quite early, causing the robot to overshoot"); 0.8 s let it
                       # never move with a detector at 0.8 s per picture (test_pick_sim.py)
ARM_STEP = 12          # pulses per servo per control step while following
ARM_STEP_TRANSIT = 20  # ... on the way from P / look_at to the lookout
FAR_VIEW_X = 0.38      # m: the lookout sees the floor up to ~0.42 from the centre (servo 2's sag takes a bit): further
                       # out the arm keeps its view and the base brings the object closer
TILT_ROW = 200         # px: where the camera's tilt keeps a far object (the picture's middle row is 192)
TILT_GAIN = 0.35       # pulses per px: 1 px = 0.14 deg, 1 pulse = 0.24 deg; 0.6 of the full correction per step
TILT_STEP = 15
TILT_DEADBAND = 40     # px
LOOK_START_S = 5.0     # s (and LOOK_START_PICTURES) without the object from where the arm started before it looks
LOOK_START_PICTURES = 12  # elsewhere: after 2 s it swung to the lookout although the block was in view from P (small
                          # there, the first pictures missed it) and lost it (live, 2026-10-12)
NOT_SEEN_FAIL_S = 30.0
GOAL_MOVE = 0.004      # m: the arm's pose is recomputed when its goal moved more than this (inverse kinematics ~20 ms);
                       # above the detections' noise (~2 px, 1-2 mm): at 2 mm the arm never stood still for a picture
CONTROL_S = 0.15
FIRST_RESULT_S = 20.0  # s for the detector's first result: switching YOLOE's prompts takes ~2 s (10 s the first time
                       # in a process), the first picture after it ~1 s
LOST_FAIL_S = 5.0      # s without the object in a fresh picture
LOOK_WAIT_S = 2.5      # s at each look (~10 pictures) before the next
LOOK_UP_PULSES = 40    # servo 4: ~10 deg
SETTLE_S = 0.4         # s after the last arm command / base stop before a picture counts as taken standing still
NOTE_S = 1.0           # s between progress notes
APPROACH_TIMEOUT_S = 90.0


def _q_at(hist, t):
    """The servo positions measured closest to time t (hist: [(t, positions)])."""
    return min(hist, key=lambda h: abs(h[0] - t))[1]


def _driven(vhist, t0, t1):
    """How far the base was commanded to move between t0 and t1 (vhist: [(t, vx, vy)], the command from t on), in
    the robot's frame. From the commands, not /odom (it over-reported 3x at these speeds): if the wheels do less, the
    prediction says "there" too early and the base stops short, never past the object."""
    d = np.zeros(2)
    for k, (t, vx, vy) in enumerate(vhist):
        t_end = vhist[k + 1][0] if k + 1 < len(vhist) else t1
        a, b = max(t, t0), min(t_end, t1)
        if b > a:
            d += np.array([vx, vy]) * (b - a)
    return d


async def _step_arm(r, cmd, target, meas, step, last_cmd_t):
    """One step of servos 1-4 towards target: (new command, its time). All four move together: on the way from
    navigation's look_at pose to the lookout the object may leave the picture for 1-2.5 s (test_pick_sim.py; keeping
    the tilt aimed at it on the way lost it for good twice). A step that would bring the gripper over the robot's front
    (look_at's tilt far down) goes through the survey pose first."""
    new = {i: cmd[i] + int(np.clip(target[i] - cmd[i], -step, step)) for i in cmd}
    if new == cmd:
        return cmd, last_cmd_t
    if arm.collision(arm.pulses_to_angles({**new, 5: meas[5]}), chassis=False):
        await survey(r)
        now = await r.servos.read(arm.ARM_IDS)
        cmd = {i: now[i] for i in cmd}
        new = {i: cmd[i] + int(np.clip(target[i] - cmd[i], -step, step)) for i in cmd}
        hit = arm.collision(arm.pulses_to_angles({**new, 5: meas[5]}), chassis=False)
        if hit:
            raise GraspError(f"the arm cannot move there without touching the robot ({hit})")
    await r.servos.move(new, CONTROL_S * 1.3)
    return new, time.monotonic()


async def approach(r, prompts, ref, trk):
    """The base drives and the arm follows until the gripper stands over the object at a comfortable reach, with the
    object on the reference spot. Returns the grasp target (base_footprint). trk: the running Tracker (pick() stops
    it after the lift: its boxes show in the console through all stages)."""
    log = r.pick_log
    vhist = [(time.monotonic(), 0.0, 0.0)]
    base_moving = False

    async def base(vx, vy):
        nonlocal base_moving
        if (vx, vy) == vhist[-1][1:]:
            if base_moving:
                await r.twist(vx, vy)  # the robot keeps a command only for a while: repeat it
            return
        vhist.append((time.monotonic(), vx, vy))
        base_moving = bool(vx or vy)
        await r.twist(vx, vy)
        log.line("base", vx=round(vx, 3), vy=round(vy, 3))

    try:
        # the arm starts where it is (P, or aimed at the object by navigation's look_at): it looks from there first
        # and the object stays in view while the arm moves on. Swinging straight to the lookout lost a cube 63 cm
        # ahead at the picture's top on the way (live, 2026-10-12), with no word of where it was
        far_pose = pose_at(axis_point(FAR_LOOK, RETREAT))
        # where to look when the start pose does not show it: the far lookout (approach_found stops ~20 cm in front),
        # the near one (straight over the grasp distance: an object put down there sat at the far lookout's bottom
        # edge behind the jaws, 2026-10-12), then the far one ~10 deg up
        looks = [("the lookout", far_pose), ("closer in front", pose_at(axis_point(np.array([*REACH, GRASP_Z]), RETREAT))),
                 ("further ahead", {**far_pose, 4: min(1000, far_pose[4] + LOOK_UP_PULSES)})]
        start = time.monotonic()
        now = await r.servos.read(arm.ARM_IDS)
        hist = [(start, now)]
        cmd = {i: now[i] for i in arm.ARM_IDS[:4]}
        last_cmd_t, base_stop_t, last_note = 0.0, 0.0, 0.0
        goal_done = None    # the goal the arm's pose was computed for
        pre = None
        obs = None          # the newest observation: {"p", "t", "det", "q", "still"}
        used = 0.0
        fine = None         # once the base is there: the target corrected by the picture (reference spot)
        travelled = 0.0
        first_t, look_i, look_t = None, -1, None   # the look being tried, since when it stands there
        fine_pose = None    # the pre-grasp pose over `fine`: the reference spot means something only from there
        first_x = None      # where the object was first seen (x): the base's real progress
        while True:
            t = time.monotonic()
            if r.stopped():
                raise Stopped()
            if t - start > APPROACH_TIMEOUT_S:
                raise GraspError("the gripper did not get over the object in time")
            meas = await r.servos.read(arm.ARM_IDS)
            t = time.monotonic()
            hist = [h for h in hist if t - h[0] < 5.0] + [(t, meas)]
            r.arm_now = meas
            if any(abs(cmd[i] - meas[i]) > FREE_LAG + 10 for i in cmd):
                raise GraspError("the arm stopped following: something is in its way")
            # the newest picture
            L = trk.latest
            if L is not None and L["t_cap"] > used:
                used = L["t_cap"]
                det = L["det"]
                if det is not None and L["t_cap"] >= start:
                    q = arm.pulses_to_angles(_q_at(hist, L["t_cap"]))
                    still = (L["t_cap"] - last_cmd_t > SETTLE_S and not base_moving
                             and L["t_cap"] - base_stop_t > SETTLE_S)
                    obs = {"p": floor_point((det[0], det[1]), q, r.K, r.D), "t": L["t_cap"], "det": det, "q": q,
                           "still": still, "new": True}
                    log.line("obs", p=obs["p"], age_ms=round((t - L["t_cap"]) * 1000), still=still,
                             centre=[round(det[0]), round(det[1])])
            if trk.n == 0:
                if t - start > FIRST_RESULT_S:
                    raise GraspError("the camera's object detection did not start")
                await asyncio.sleep(CONTROL_S)
                continue
            if first_t is None:
                first_t, first_n = t, trk.n  # the detector's first result: the "not in view" clock starts here
            if obs is None:
                await base(0.0, 0.0)
                if look_i == -1 and t - first_t > LOOK_START_S and trk.n - first_n >= LOOK_START_PICTURES:
                    look_i = 0
                if look_i >= 0:
                    name, pose = looks[look_i]
                    if look_t is None and all(abs(cmd[i] - pose[i]) <= 1 for i in cmd):
                        look_t = t
                        r.log(f"not in view yet: looking {name}")
                        log.line("look", name=name)
                    if look_t is not None and t - look_t > LOOK_WAIT_S:
                        if look_i + 1 == len(looks):
                            raise GraspError(f"the {trk.label} is not in view in front of the robot")
                        look_i, look_t = look_i + 1, None
                    else:
                        cmd, last_cmd_t = await _step_arm(r, cmd, pose, meas, ARM_STEP_TRANSIT, last_cmd_t)
                if t - first_t > NOT_SEEN_FAIL_S:
                    raise GraspError(f"the {trk.label} is not in view in front of the robot")
                await asyncio.sleep(CONTROL_S)
                continue
            if t - obs["t"] > LOST_FAIL_S:
                await base(0.0, 0.0)
                raise GraspError(f"lost sight of the {trk.label} (last seen {(obs['p'][0] - 0.16) * 100:.0f} cm ahead "
                                 f"of the robot's front, {obs['p'][1] * 100:+.0f} cm to the left)")
            age = t - obs["t"]
            if first_x is None:
                first_x = obs["p"][0]
                if first_x - REACH[0] > r.max_drive + 0.03:  # say it at once, with where it is (+ the estimate's error)
                    raise GraspError(f"the {trk.label} is {(first_x - 0.16) * 100:.0f} cm in front of the robot's "
                                     f"front: too far for the gripper (it reaches ~7 cm): approach it first")
            p = obs["p"].copy()
            p[:2] -= _driven(vhist, obs["t"], t)  # where it is now, after the base moved on since the picture
            # the base: the object to REACH (sideways only as far as the arm needs)
            ex = p[0] - REACH[0]
            ey = 0.0 if abs(p[1]) <= REACH_Y_MAX - 0.02 else p[1] - math.copysign(REACH_Y_MAX - 0.02, p[1])
            e = math.hypot(ex, ey)
            if t - last_note > NOTE_S:
                last_note = t
                ctx = r.context()
                if ctx:
                    log.line("context", **ctx)
                r.log(f"{'centring the gripper' if fine is not None else 'following the ' + trk.label}: "
                      f"{max(0.0, ex) * 100:.0f} cm to drive, {p[1] * 100:+.0f} cm to the side; "
                      f"{trk.n} pictures, the newest {age:.1f} s old")
            if fine is None:
                if age > MAX_AGE or e <= BASE_TOL:
                    if base_moving:
                        await base(0.0, 0.0)
                        base_stop_t = time.monotonic()
                    if obs["new"] and obs["still"] and p[0] <= PRE_X_MAX:
                        if e <= BASE_TOL * 1.5:  # measured standing still: there
                            fine = obs["p"].copy()
                            fine[2] = GRASP_Z
                            goal_done, fine_pose = None, None  # the pre-grasp pose over it: computed now
                            r.log(f"base there: object at ({fine[0]:.3f}, {fine[1]:+.3f}) m; centring the gripper")
                            log.line("there", p=fine)
                else:
                    v = min(BASE_MAX, max(BASE_MIN, BASE_GAIN * e))
                    vx, vy = v * ex / e, v * ey / e
                    if r.blocked(vx, vy, p):
                        raise GraspError("something is in the way to the object (lidar)")
                    # the limit on what the camera saw (the commands overstate it: the wheels did ~a third)
                    travelled += v * CONTROL_S
                    if first_x - obs["p"][0] > r.max_drive + 0.05:  # + the first estimate's error
                        raise GraspError(f"the object is more than {r.max_drive * 100:.0f} cm beyond the arm's "
                                         f"reach: approach it first")
                    if travelled > 4 * r.max_drive:
                        raise GraspError("the base hardly moves (blocked, or the battery low?)")
                    await base(vx, vy)
            # the fine correction: pictures taken with arm and base standing still, the arm in the pre-grasp pose
            # over the target. From the lookout the spot means nothing: an object found there already at the right
            # distance had its target moved 5 cm towards the robot, out of reach, and the pick looped (live,
            # 2026-10-12)
            at_fine = fine_pose is not None and all(cmd[i] == fine_pose[i] for i in cmd)
            if fine is not None and at_fine and obs["new"] and obs["still"]:
                det = obs["det"]
                err = math.hypot(det[0] - ref["x"], det[1] - ref["y"])
                log.line("fine", err_px=round(err, 1), fine=fine)
                if err <= PIXEL_TOL:
                    r.log(f"centred: {err:.0f} px off the reference spot")
                    return fine
                fine[:2] += GAIN * (floor_point((det[0], det[1]), obs["q"], r.K, r.D)
                                    - floor_point((ref["x"], ref["y"]), obs["q"], r.K, r.D))[:2]
                # the arm moves to the new pose before the next correction (else the same picture's error is added
                # again while a move under GOAL_MOVE never starts)
                goal_done, fine_pose = None, None
                r.log(f"centring the gripper: {err:.0f} px off")
            obs_new = obs["new"]
            obs["new"] = False
            # the arm: the gripper over the object, as far as it reaches
            # following: at most as far out as the lookout. Near PRE_X_MAX a few mm more reach meant 40 pulses on
            # servo 2: the camera tilted down and lost an object 35 cm out at the picture's top (test_pick_sim.py)
            goal = fine if fine is not None else np.array(
                [min(p[0], FAR_LOOK[0]), float(np.clip(p[1], -REACH_Y_MAX, REACH_Y_MAX)), GRASP_Z])
            if fine is None and p[0] > FAR_VIEW_X:
                # further than the lookout sees: the base brings it closer and the camera's tilt (servo 4 alone)
                # keeps it in the picture, as navigation's look_at aims. Held still in P it slid out at the bottom
                # 45 cm ahead (P sees the floor from ~0.68 m, the lookout up to ~0.42; live 2026-10-12)
                goal = None
                if obs_new and obs["t"] > last_cmd_t + 0.2:
                    row = obs["det"][1]
                    d4 = int(np.clip(-TILT_GAIN * (row - TILT_ROW), -TILT_STEP, TILT_STEP))
                    if abs(row - TILT_ROW) > TILT_DEADBAND and d4:
                        tilt = {**cmd, 4: max(0, min(1000, cmd[4] + d4))}
                        if not arm.collision(arm.pulses_to_angles({**tilt, 5: meas[5]}), chassis=False):
                            cmd, last_cmd_t = tilt, time.monotonic()
                            await r.servos.move(cmd, CONTROL_S * 1.3)
                            log.line("tilt", servo4=cmd[4], row=round(row))
            if fine is None and not base_moving and e <= 2 * BASE_TOL:
                goal = goal_done  # the base stopped near: the arm holds still for the picture that says "there"
            if goal is None:
                pre = None
            elif goal_done is None or np.linalg.norm(goal[:2] - goal_done[:2]) > GOAL_MOVE:
                pre = pose_at(axis_point(goal, RETREAT), arm.pulses_to_angles({**cmd, 5: meas[5]}))
                goal_done = goal.copy()
                fine_pose = pre if fine is not None else None
                if pre is None and fine is not None:
                    fine, goal_done = None, None  # beyond the reach after all: the base again
                    log.line("fine_unreachable")
            if pre is not None:
                far = max(abs(pre[i] - cmd[i]) for i in cmd) > 3 * ARM_STEP  # still on the way from P / look_at
                cmd, last_cmd_t = await _step_arm(r, cmd, pre, meas, ARM_STEP_TRANSIT if far else ARM_STEP,
                                                  last_cmd_t)
            await asyncio.sleep(CONTROL_S)
    finally:
        if base_moving or vhist[-1][1:] != (0.0, 0.0):
            for _ in range(3):
                await r.twist(0.0, 0.0)
                await asyncio.sleep(0.05)


async def pick(r, prompts=None):
    """Finds the object (prompts for the detector; none: the red block), drives the base and follows with the arm
    until the gripper stands over it, grasps it and lifts it straight into the survey pose. Returns {"width_mm",
    "target", "seconds"}; GraspError if not."""
    ref = json.loads(REFERENCE.read_text())["block_px"]
    t0 = time.monotonic()
    r.pick_log = PickLog()
    r.pick_log.line("start", prompts=prompts, **r.context())
    await gripper(r, OPEN, 0.5)
    trk = Tracker(r, prompts, r.pick_log)
    trk.start()
    try:
        return await _grasp(r, prompts, ref, trk, t0)
    except (GraspError, Stopped) as e:
        r.pick_log.line("failed", error=str(e) or type(e).__name__, **r.context())
        if isinstance(e, GraspError):
            # back to P: left in the lookout, the next try could not see a cube 63 cm ahead (2026-10-12), and the
            # navigator drives with P. After STOP the arm stays where it is
            await survey(r)
        raise
    finally:
        await trk.stop()
        r.pick_log.line("tracker", pictures=trk.n, errors=trk.errors)
        r.pick_log.close()  # left open, the next pick's log could not be removed and was appended (Windows)


async def _grasp(r, prompts, ref, trk, t0):
    """Approach (the tracker running), down, close, lift, P."""
    target = await approach(r, prompts, ref, trk)
    r.log(f"over it ({target[0]:.3f}, {target[1]:+.3f}); going down")
    await move_along(r, target, RETREAT, 0.0)
    r.log("closing the gripper")
    contact = await close_on_object(r)
    if contact is None:
        await gripper(r, OPEN)
        await move_along(r, target, 0.0, RETREAT)
        await survey(r)
        raise GraspError("the gripper closed on nothing (the object may have moved, or it is narrower than 15 mm)")
    r.log("lifting it")
    await move_along(r, target, 0.0, RETREAT)
    await survey(r)
    if not await holding(r, contact):
        await gripper(r, OPEN)
        raise GraspError("the object slipped out of the gripper while lifting")
    w = width_mm(contact)
    r.log(f"holding it: contact at {contact} (~{w:.0f} mm), {time.monotonic() - t0:.0f} s")
    r.pick_log.line("holding", contact=contact, width_mm=round(w), seconds=round(time.monotonic() - t0),
                    **r.context())
    return {"width_mm": round(w), "target": [round(float(v), 3) for v in target],
            "seconds": round(time.monotonic() - t0), "contact": contact}


async def put_down(r, target=None):
    """Puts the held object down in front of the robot (target: base_footprint point, default the middle of the
    reach), opens, back to the survey pose."""
    target = np.array([REACH[0], REACH[1], GRASP_Z] if target is None else target, float)
    pre = pose_at(axis_point(target, RETREAT))
    if pre is None or pose_at(target) is None:
        raise GraspError(f"({target[0]:.2f}, {target[1]:+.2f}) is out of the arm's reach")
    await go(r, pre)
    await move_along(r, target, RETREAT, 0.0)
    await gripper(r, OPEN)
    await move_along(r, target, 0.0, RETREAT)
    await survey(r)
