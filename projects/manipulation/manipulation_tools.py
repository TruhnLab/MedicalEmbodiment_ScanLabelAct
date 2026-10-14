"""The chat's manipulation tools (landerpi/tools.py rules): pick an object up from the floor in front of the robot, put
it down. They run in the navigator (projects/navigation/navigator.py; this module is listed in assistant.TOOL_MODULES)
and use its rosbridge connection, odometry, lidar scan and detector (NavRobot) for pick.py.

The sequence the chat follows: find it (watch_for / search_here) -> approach_found (stops ~20 cm in front) ->
pick_up_object (follows it with the camera: the base drives the last ~15 cm, the arm
keeps the gripper over it) -> drive_to_object / drive_to_point -> put_down_object.

While a tool runs it owns the base and the arm (the navigator is not driving: refused otherwise), live detection is
stopped and the detector shared under live.detector_lock (the tool uses it with its own prompts), and the arm ends in the survey pose. What the robot holds
is nav.held: {"label", "width_mm", "since", "mask"}; mask: the held object's pixels in the colour picture (bool HxW,
grown by HELD_MARGIN_PX). The camera and the gripper are rigid (only the wrist roll between), so the mask stays valid
in every arm pose: depth obstacles and detection skip those pixels (navigation's side).
"""

import asyncio
import math
import time

import cv2
import numpy as np

import arm
import pick as core
from check_arm import Servos, Stopped
from landerpi.tools import ToolError, tool

TOOL_TIMEOUT_S = 240.0
HELD_MARGIN_PX = 15        # the held object's mask grown by this (the depth camera's stray pixels at its edges)
CORRIDOR_HALF_W = 0.15     # m: the base's half width plus margin, for the last centimetres' lidar check
NEAR_TARGET_M = 0.08       # lidar points this close to the object are the object (a tall one is seen by the lidar)
HALF_LENGTH = 0.16         # m, robot centre to front (goto.py)
STREAM_MAX_AGE = 0.3       # s: an older stream frame is replaced by a snapshot
OWN_ARM_M = 0.08           # lidar points this close to the gripper or camera (above the floor) are the arm itself


class NavRobot(core.Robot):
    """pick.py's robot through the navigator: its client, odometry, scan and detector."""
    max_drive = 0.50       # m during one pick, lidar-checked: a cube ~50 cm ahead was followed in and picked live
                           # (2026-10-12); approach_found normally leaves ~15 cm

    def __init__(self, nav, stopped):
        super().__init__(Servos(nav.client))
        self.nav = nav
        self._stopped = stopped
        self.stop_count = nav.stop_count
        self._last_frame = None
        self._loc_seen = None      # the localization status at the last note (context)

    def log(self, text):
        print(f"manipulation: {text}", flush=True)
        self.nav.note("manipulation", text)

    def stopped(self):
        return self._stopped() or self.nav.stop_count != self.stop_count

    def show(self, found=None, label=""):
        """Through live.py's overlay (the console draws it over the camera picture); clear_overlay while moving."""
        live = self.nav.live
        if live is None:
            return
        if found is None:
            live.clear_overlay()
        else:
            live.set_overlay([{"label": label, "score": float(found[7]), "box": [float(v) for v in found[6]]}], 640, 400)
        if hasattr(live, "fed_until"):  # the console polls the overlay fast while it is fed (board, 2026-10-12 14:50)
            live.fed_until = time.time() + 1.0

    def accepts(self, label, prompts):
        """Also the words navigation's search used for the same item (live.alias: detector word -> item looked for):
        its prompts are kept when they contain ours, and the cube then came as "red cube toy" (2026-10-12)."""
        if super().accepts(label, prompts):
            return True
        alias = getattr(self.nav.live, "alias", {}) or {}
        mine = {alias.get(p.lower()) for p in prompts} | {p.lower() for p in prompts}
        return alias.get(label.lower()) in mine

    def context(self):
        """The navigator's localization with every progress note: it turned "uncertain" during a pick (2026-10-12
        16:20) and nothing recorded why."""
        nav = self.nav
        pose = nav.current_pose() if hasattr(nav, "current_pose") else None
        ctx = {"loc": getattr(nav, "status", None), "loc_score": None if getattr(nav, "score", None) is None
               else round(float(nav.score), 3), "pose": None if pose is None else [round(float(v), 3) for v in pose],
               "odom": list(nav.odom[-1]) if getattr(nav, "odom", None) else None}
        pts = getattr(nav, "last_scan_robot", None)
        if pts is not None and len(pts):
            p = np.asarray(pts)[:, :2]
            ctx["scan_near_front"] = int(np.sum((p[:, 0] > 0) & (np.hypot(p[:, 0], p[:, 1]) < 0.45)))
            if ctx["loc"] != self._loc_seen:  # the localization changed: the whole scan of that moment (navigation's
                ctx["scan"] = np.round(p, 3).tolist()  # suggestion, board 2026-10-12 17:00)
        self._loc_seen = ctx["loc"]
        return ctx

    def ensure_detector(self):
        """live.py's detector (created if it has none yet), shared under its detector_lock."""
        live = self.nav.live
        self.detector_lock = getattr(live, "detector_lock", None)
        if live.detector is None:
            from landerpi.detector import Detector
            live.detector = Detector()
        self.detector = live.detector

    async def frame(self):
        """(picture, when it was taken (time.monotonic()), source): the navigator's video stream if its newest frame
        is under STREAM_MAX_AGE old (its robot stamp against the newest odometry's), else a snapshot from the robot.
        No waiting for a new frame: that loop took seconds while the event loop was busy (2026-10-12)."""
        f = self.nav.live.frame
        if f is not None and self.nav.odom:
            age = max(0.0, self.nav.odom[-1][0] - f[1])
            if age < STREAM_MAX_AGE and f is not self._last_frame:
                self._last_frame = f
                img = cv2.imdecode(np.frombuffer(f[0], np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img, time.monotonic() - age, "stream"
        t = time.monotonic()
        return await self.snapshot(), t, "snapshot"

    def blocked(self, vx, vy, obj):
        """Something in the way of the base's next ~12 cm that is not the object itself (the lidar; a tall object is
        seen by it, a block is below its plane)."""
        pts = self.nav.last_scan_robot
        if pts is None or not len(pts):
            return False
        p = np.asarray(pts)[:, :2]
        v = math.hypot(vx, vy)
        d = np.array([vx, vy]) / max(v, 1e-6)
        along = p @ d
        across = np.abs(p @ np.array([-d[1], d[0]]))
        ahead = (along > HALF_LENGTH - 0.02) & (along < HALF_LENGTH + 0.12) & (across < CORRIDOR_HALF_W)
        near_obj = np.hypot(p[:, 0] - obj[0], p[:, 1] - obj[1]) < NEAR_TARGET_M
        # the arm itself: in the lookout poses the gripper hangs ~10 cm above the floor, 20 cm ahead, in the lidar's
        # plane (~9 cm): it stopped two picks as "something in the way" (2026-10-12)
        q = arm.pulses_to_angles(self.arm_now or {})
        own = np.zeros(len(p), bool)
        for f in ("link4", "link5", "end_effector_link", "jaw_tips", "rgb_camera_link"):
            c = arm.frame(f, q)[:2, 3]
            own |= np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < OWN_ARM_M
        return bool(np.any(ahead & ~near_obj & ~own))


def _nav(ctx):
    return ctx.services["navigator"]


def _run(ctx, make):
    """Runs make(robot) (a coroutine) on the navigator's event loop and waits; STOP ends it."""
    if arm.MISSING:
        raise ToolError("the arm cannot be used here: its model (the robot's URDF, config/landerpi.urdf) is missing "
                        "on this computer")
    nav = _nav(ctx)
    if not nav.connected or nav.client is None:
        raise ToolError("the robot is not connected: the user has to press Connect")
    g = nav.goto
    if g is not None and g.state in ("planning", "driving", "docking", "turning"):
        raise ToolError("the robot is driving: wait until the drive has ended")
    if nav.live is not None:
        nav.live.stop()  # the tool uses the detector with its own prompts
        nav.live.clear_overlay()  # the boxes of before (the approach) no longer fit once the arm moves
        nav.live.video(True)  # its pictures for the tool's own live detection

    async def job():
        settle = getattr(nav, "arm_settle", None)
        if settle is not None and not settle.done():
            # navigator.arm_to's second step (after approach_found's aim) still to come: it moved the camera ~20 deg
            # during the first second of a pick (2026-10-12): cancelled, then wait until the arm stands still
            settle.cancel()
        r = NavRobot(nav, ctx.stopped)
        prev = None
        for _ in range(20):
            now = await r.servos.read([1, 2, 3, 4, 5])
            if prev is not None and all(abs(now[i] - prev[i]) <= 2 for i in now if now[i] is not None and prev[i] is not None):
                break
            prev = now
            await asyncio.sleep(0.15)
        await r.start()
        return r, await make(r)
    async def job_and_clear():
        try:
            return await job()
        finally:
            if nav.live is not None:
                nav.live.clear_overlay()  # no box left over from the tool's last picture
                nav.live.video(nav.live.recorder is not None)  # the video only while recording (live.stop)
    fut = asyncio.run_coroutine_threadsafe(job_and_clear(), nav.loop)
    t0 = time.time()
    while not fut.done():
        if time.time() - t0 > TOOL_TIMEOUT_S:
            fut.cancel()
            raise ToolError(f"the arm did not finish within {TOOL_TIMEOUT_S:.0f} s")
        time.sleep(0.2)
    try:
        return fut.result()
    except Stopped:
        raise ToolError("stopped (STOP in the console): the arm stays where it is")
    except core.GraspError as e:
        raise ToolError(str(e))


async def _held_mask(r, prompts):
    """The held object's pixels in the colour picture (survey pose), grown by HELD_MARGIN_PX, or None."""
    import cv2
    await asyncio.sleep(0.5)
    img = await r.snapshot()
    b = await r.find(img, prompts)
    if b is None or b[5] is None:
        return None
    k = 2 * HELD_MARGIN_PX + 1
    return cv2.dilate(b[5].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


@tool("pick_up_object",
      "Pick up a small object lying on the floor in front of the robot with the gripper (15-45 mm wide, block-sized). "
      "Approach it first (approach_found: the robot stops ~20 cm in front); this tool keeps it in the camera, drives "
      "the last ~15 cm itself while the arm follows it, grasps it and lifts it. The robot then carries it (in the survey pose) until "
      "put_down_object.",
      {"item": {"type": "string", "description": "what to pick up, as the detector should look for it, e.g. 'red block'"},
       "synonyms": {"type": "array", "items": {"type": "string"},
                    "description": "2-3 other words for it (e.g. 'red cube'), looked for in the same pass"}},
      required=["item"], moves_robot=True)
def pick_up_object(ctx, item, synonyms=None):
    nav = _nav(ctx)
    if getattr(nav, "held", None):
        raise ToolError(f"the robot already holds the {nav.held['label']}: put it down first (put_down_object)")
    prompts = [item] + [w for w in (synonyms or []) if w and w.strip()][:3]

    async def make(r):
        r.ensure_detector()
        res = await core.pick(r, prompts)
        res["mask"] = await _held_mask(r, prompts)
        return res
    r, res = _run(ctx, make)
    nav.held = {"label": item, "width_mm": res["width_mm"], "since": time.time(), "mask": res["mask"]}
    return {"result": f"picked up the {item} (about {res['width_mm']} mm wide) and holding it",
            "seconds": res["seconds"]}


@tool("put_down_object",
      "Put the object the robot holds down on the floor right in front of it (~7 cm ahead of the robot) and open the "
      "gripper. Drive to the place first (drive_to_object / drive_to_point); the floor in front must be free.",
      {}, moves_robot=True)
def put_down_object(ctx):
    nav = _nav(ctx)
    held = getattr(nav, "held", None)
    if not held:
        raise ToolError("the robot holds nothing")

    async def make(r):
        await core.put_down(r)
    _run(ctx, make)
    nav.held = None
    return {"result": f"put the {held['label']} down in front of the robot"}
