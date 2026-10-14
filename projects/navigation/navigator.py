"""Live localization in a known room and driving to its objects: the task console's backend.

    python projects/navigation/navigator.py           # the helper starts it for the task console

Talks to the robot's rosbridge directly: /scan, /odom, the depth images (PNG, from the robot's depth
compressor, started over SSH like the recorder does) and /tf in; /controller/cmd_vel out (the turn on the
spot that localization makes, and driving to an object). No ROS container needed; SLAM / the explorer must
not run alongside (nothing else may drive the robot meanwhile). Dry run without motion: set
LANDERPI_CMD_OUT=/landerpi/cmd_vel_dryrun before starting it (commands go to an unused topic).

HTTP on 127.0.0.1:8767 (POSTs need the header X-LanderPi-Console, bodies are JSON):
    GET  /state                      status, pose in the room, the latest scan in the room's frame
    POST /room      {"name": ...}    use data/rooms/<name> (not localized until /localize)
    POST /localize                   localize: one turn on the spot, then the search (the console needs Armed)
    POST /goto      {"object": id}   drive to an object of the room's scene.json (goto.py); needs "localized"
                    {"x": .., "y": ..} ... or to a point on the floor (pin & go)
    POST /stop                       stop turning / driving
    POST /quit                       stop and exit (the helper's stop button)
    GET  /chat                       the chat so far (assistant.py)
    POST /chat      {"message": .., "armed": bool}   one message to the chat; answers when it is done acting
    POST /chat/reset                 start the conversation over

Localization: localize.py, always after one turn on the spot: the global search gets a full circle (the
lidar's rear ~155 deg are filtered out of /scan). Afterwards every scan is tracked from the odometry's
prediction. Driving to objects is only possible while "localized".

Driving (goto.py, 10 Hz): stops if the scans or odometry are older than STALE_S, or if localization stops
being "localized"; the lidar points plus the depth camera's obstacles (depth_view.py) are the collision
check, and both go into the plan as obstacles the room map doesn't have.
"""

# torch (live detection, live.py) must load before numpy's MKL has run: afterwards Windows found MKL's
# libiomp5md.dll of the conda environment first and torch failed with "WinError 127 ... shm.dll" (real run,
# 2026-10-08: the search never saw anything). Both OpenMP runtimes are allowed (detector.py says why).
import os  # noqa: E402
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except ImportError:
    pass  # no detection installed: finding things is off, the rest works

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "manipulation"))  # its chat tools (assistant.py)
from landerpi.robot import LIDAR_IN_BASE, ROSBRIDGE_URL, TOPICS  # noqa: E402
from landerpi.room import Room  # noqa: E402
from landerpi.robot import ROBOT_HOST  # noqa: E402
from landerpi.rosbridge import RosbridgeClient  # noqa: E402
from depth_view import DepthObstacles, TfTable, planar  # noqa: E402
from goto import GAP, HALF_LENGTH, GoTo  # noqa: E402
from live import LiveDetection  # noqa: E402
from taskrun import RUNS, TaskRecorder  # noqa: E402
from localize import Localizer, compose, relative, scan_points, stitch, thin, wrap  # noqa: E402

PORT = 8767


ARM_ZONE_M, ARM_ZONE_DEG = 0.45, 70  # lidar points this close in front can be the arm or what it holds


def beyond_arm(pts):
    """The scan without what the arm can reach (base frame, x ahead): the localization matches walls, not the arm.
    Localization went "uncertain" 2 s into a pick and the pose ended 0.8 m off (real run 2026-10-12 16:20), as the
    arm swung forward into its lookout pose. The collision check still gets every point."""
    d = pts - np.array(LIDAR_IN_BASE[:2])
    near = (np.hypot(d[:, 0], d[:, 1]) < ARM_ZONE_M) & (np.abs(np.degrees(np.arctan2(d[:, 1], d[:, 0]))) < ARM_ZONE_DEG)
    return pts[~near] if (~near).sum() >= 20 else pts


def code_version():
    """Newest change of the navigator's code. The helper replaces a running navigator with older code: one
    started before a fix kept running through Disconnect / Connect, and the fix never arrived (2026-10-08)."""
    repo = Path(__file__).resolve().parents[2]
    files = [*Path(__file__).parent.glob("*.py"), *(repo / "landerpi").glob("*.py"), repo / "offboard_ros" / "explore_core.py",
             repo / "offboard_ros" / "depth_obstacles.py", repo / "offboard_ros" / "coverage3d.py",
             *(repo / "projects" / "manipulation").glob("*.py")]  # its chat tools run in the navigator too
    return int(max(f.stat().st_mtime for f in files if f.exists()))


CODE_VERSION = code_version()
CMD_OUT = os.environ.get("LANDERPI_CMD_OUT", TOPICS["cmd_vel"][0])
STOP = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}

# the explorer's start turn (0.7 x 0.6 rad/s, explore_core.py): its scans localized 4/4 runs in the
# offline test (test_localize.py, "start turn"), so the same speed here; ~15 s per turn
TURN_RATE = 0.42
TURN_TIMEOUT_S = 45.0
ARM_OVERSHOOT = 12        # pulses above the pose first (arm_to): past the servos' ~5-pulse deadband; 40 tilted the
                          # camera ~20 deg up for a moment (servos 2 and 4 together; user: "looks into the sky")
ARM_OVERSHOOT_SHARE = 0.7 # of the move's time to get there, the rest down to the pose
STALE_S = 0.5            # no scan/odometry for this long while turning: stop (as the explorer does)
# Scores (mean likelihood of the scan points, localize.py). A global result is checked on the last
# VERIFY_SCANS single scans, not on the stitched cloud: stitched over a few seconds of driving even the
# right pose scored only 0.34-0.53 (odometry drift, points through doors beyond the map), single scans
# at correct poses 0.85-0.99 (replayed run 20261006_161813 in room 20261006_154055, 2026-10-07)
VERIFY_SCANS = 10
GLOBAL_MIN = 0.60
TRACK_MIN = 0.40
TRACK_BAD_SCANS = 20     # 2 s of scans below TRACK_MIN: "uncertain"
SCAN_DISPLAY = 0.08      # m: the scan sent to the console, thinned to this
DRIVE_HZ = 10.0
DEPTH_THROTTLE_MS = 250  # ~4 depth images per second, as the relay forwards them to the explorer
NEAR_DEPTH_M = 2.0       # depth obstacles closer than this go into the collision check
# Things the detector placed on the floor are obstacles too: 3 cm blocks are below the lidar and below the depth
# obstacles' band (10-45 cm), and the robot drove over the two blocks next to the one it went to (real run
# 2026-10-09). Sure sightings whose bottom is lower than SEEN_LOW_M, as a disc of their size; not the target.
SEEN_LOW_M = 0.15
SEEN_MIN_R, SEEN_MAX_R = 0.03, 0.15  # m: the disc's radius (half the sighting's size, within these)
TARGET_SELF_M = 0.15                 # a sighting this close to the drive's target is the target itself


def stamp(msg):
    return msg["header"]["stamp"]["sec"] + msg["header"]["stamp"]["nanosec"] * 1e-9


def yaw_of(q):
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


class Navigator:
    def __init__(self):
        self.loop = None
        self.client = None
        self.connected = False
        self.room = None
        self.loc = None
        self.status = "no room"
        self.message = "Choose a room."
        self.pose = None            # robot base in the room's map (x, y, yaw)
        self.pose_odom = None       # the odometry pose at the moment of self.pose
        self.pose_t = None          # that moment (robot time)
        self.score = None
        self.runner_up = None
        self.bad_scans = 0
        # (robot time, x, y, yaw): 60 s at 30 Hz, longer than a turn (TURN_TIMEOUT_S), so a turn's scans
        # are stitched with odometry interpolated at their own time, after the turn. A 13 s history
        # clamped the first half of a 28 s turn to one pose (score 0.24); odometry taken when a scan
        # arrives lags it by up to ~30 ms, ~1 deg while turning (score 0.38; 2026-10-07, replayed run)
        self.odom = deque(maxlen=1800)
        self.scans = deque(maxlen=100)  # (robot time, points in the robot frame), 10 s at 10 Hz
        self.scan_view = None           # latest scan in the room frame, for the console
        self.last_odom_recv = self.last_scan_recv = 0.0
        self.turn = None                # {"turned", "last", "since", "scans"} while turning
        self.job = None                 # the running localization (asyncio task)
        self.goto = None                # GoTo for the room (goto.py)
        self.drive = None               # the running drive (asyncio task)
        self.last_scan_robot = None     # latest scan points in the robot frame (collision check)
        self.tf = TfTable()
        self.depth = DepthObstacles(self.tf, held=self.held_view)
        self.servo = {}                 # arm servo id -> position (servo_states)
        self.stop_count = 0             # +1 at every stop: a running chat tool sees it (nav_tools.py)
        self.last_stop = None           # why the last stop came
        self.arm_settle = None          # the second step of an arm move (arm_to)
        # what the gripper holds, set by projects/manipulation's tools: None or {"label", "width_mm", "since", "mask"}
        # (mask: its pixels in the colour picture; the depth obstacles and the detection skip them)
        self.held = None
        self.last_note = None
        self.assistant = None           # the chat (assistant.py), created at the first message
        self.rgb_K = None               # colour camera intrinsics (fx, fy, cx, cy)
        self.live = None                # live detection (live.py), created with the event loop

    # ---------- sensor input ----------

    def odom_at(self, ts):
        """Odometry poses at robot times ts (scalar -> 3, list -> Nx3), interpolated; clamped to the
        newest (a scan usually arrives before the odometry of its moment)."""
        o = np.array(self.odom)
        t = np.asarray(ts, float)
        return np.stack([np.interp(t, o[:, 0], o[:, 1]), np.interp(t, o[:, 0], o[:, 2]),
                         wrap(np.interp(t, o[:, 0], np.unwrap(o[:, 3])))], -1)

    def on_odom(self, msg, recv):
        p = msg["pose"]["pose"]
        self.odom.append((stamp(msg), p["position"]["x"], p["position"]["y"], yaw_of(p["orientation"])))
        self.last_odom_recv = recv
        if self.turn is not None:
            yaw = self.odom[-1][3]
            if self.turn["last"] is not None:
                self.turn["turned"] += abs(wrap(yaw - self.turn["last"]))
            self.turn["last"] = yaw

    def on_scan(self, msg, recv):
        t, pts = stamp(msg), scan_points(msg, LIDAR_IN_BASE)
        self.last_scan_recv = recv
        if len(pts) < 20 or not self.odom:
            return
        self.scans.append((t, pts))
        self.last_scan_robot = pts
        if self.turn is not None:
            self.turn["scans"].append((t, pts))
        if self.pose is None or self.loc is None or self.status == "searching":
            return
        o = self.odom_at(t)
        predicted = compose(self.pose, relative(self.pose_odom, o))
        self.pose, info = self.loc.track(beyond_arm(pts), predicted)  # ~4 ms
        self.pose_odom, self.pose_t, self.score = o, t, info["score"]
        self.bad_scans = self.bad_scans + 1 if self.score < TRACK_MIN else 0
        if self.status == "localized" and self.bad_scans >= TRACK_BAD_SCANS:
            self.set_status("uncertain", "The scans no longer fit the room well: localize again.")
        view = thin(pts, SCAN_DISPLAY)
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.scan_view = np.round(view @ np.array([[c, s], [-s, c]]) + self.pose[:2], 3).tolist()

    def set_status(self, status, message):
        self.status, self.message = status, message
        self.note("status", f"{status}: {message}")

    def note(self, kind, text):
        """A line in the log, and an event in an open recording (taskrun.py)."""
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        self.last_note = {"kind": kind, "text": text, "time": time.time()}  # the console's live line (tasks.js)
        live = self.live
        for r in (live.recorder, live.last_search) if live is not None else ():
            if r is not None:
                r.event(kind, text)

    # ---------- recording (the console's ● Record) ----------

    def start_record(self):
        if self.live.recorder is not None:
            return
        name = time.strftime("%Y%m%d_%H%M%S")
        self.live.recorder = TaskRecorder(RUNS / name, room=self.room.name if self.room else None)
        self.live.video(True)
        self.note("record", f"recording {name}")

    def stop_record(self):
        rec = self.live.recorder
        if rec is None:
            return
        self.live.recorder = None
        self.live.video(self.live.watcher.active)
        rec.stop()
        self.note("record", f"recording {rec.path.name} stopped ({rec.n} images); making its video")

    # ---------- localization ----------

    def use_room(self, name):
        room = Room(name)
        scene = room.session_path / "scene" / "scene.json"
        area = json.loads(scene.read_text()).get("area_polygon") if scene.exists() else None
        self.cancel()
        self.stop_drive("room changed")
        if self.live is not None:
            self.live.stop()
            self.live.watcher.forget()  # finds and sightings belong to a room
            self.live.clear_overlay()
        self.room, self.loc = room, Localizer(room.grid, room.resolution, room.origin, area=area)
        from scan_cells import ScanCells
        ScanCells.ready(room)  # loads the room's scan cells in the background (live.py checks places with them)
        self.goto = GoTo(room.grid, room.resolution, room.origin, walls=room.nogo_lines())
        self.depth = DepthObstacles(self.tf, held=self.held_view)  # obstacles of another room are no use here
        self.pose = self.score = self.runner_up = self.scan_view = None
        self.set_status("idle", f"Room {name} loaded. Localize: one turn on the spot (Armed, ⟳ Localize).")

    def cancel(self):
        if self.job is not None and not self.job.done():
            self.job.cancel()
        self.job = None

    def clear_overlay(self):
        if self.live is not None:
            self.live.clear_overlay()

    def start_localize(self):
        if self.loc is None:
            raise ValueError("choose a room first")
        self.clear_overlay()
        self.cancel()
        self.stop_drive("localizing again")
        self.depth.mem.forget()  # a fresh start: nothing remembered from before
        self.job = self.loop.create_task(self.localize())

    async def localize(self):
        """Always with one turn on the spot: it sees the whole room, so a wrong but confident result is
        much less likely than from the ~205 deg standing still (user, 2026-10-07: localizing standing still
        dropped; driving is only unlocked by this)."""
        try:
            if not self.connected:
                self.set_status("failed", "Not connected to the robot.")
                return
            scans = await self.turn_on_spot()
            if scans is None:
                return
            if len(scans) < 5:
                self.set_status("failed", "No lidar scans arrive from the robot.")
                return
            self.status = "searching"
            odom = self.odom_at([t for t, _ in scans])
            t_last, odom_last = scans[-1][0], odom[-1]
            pts = stitch([(p, o) for (_, p), o in zip(scans, odom)])
            t0 = time.time()
            pose, info = await self.loop.run_in_executor(None, self.loc.global_search, pts)
            how = "after one turn"
            # finish on the last second of scans alone: the search's cloud carries the odometry's drift
            # over the whole turn (replayed with driving: 10-20 cm / up to 7 deg off before this)
            recent = stitch([(p, o) for (_, p), o in zip(scans[-VERIFY_SCANS:], odom[-VERIFY_SCANS:])])
            for window in ((0.30, math.radians(10)), (0.15, math.radians(6))):
                pose, _ = self.loc.track(recent, pose, window=window)
            found = info["score"]
            # Already tracking: keep that pose if it explains the same scans better than the search's
            # result (e.g. an ambiguous search in a room whose halves look alike)
            if self.pose is not None:
                tracked = compose(self.pose, relative(self.pose_odom, odom_last))
                tracked, s = self.loc.refine(thin(pts), tracked)
                if s >= info["score"]:
                    pose, info = tracked, {**info, "score": s, "ambiguous": False}
                    how += f", kept the tracked pose (the search's best fitted worse: {found:.2f})"
            # the result is the pose at the last scan; tracking takes it from there
            self.pose, self.pose_odom, self.pose_t = pose, odom_last, t_last
            self.runner_up, self.bad_scans = info["runner_up"], 0
            self.score = fit = float(np.mean([
                self.loc.score(self.loc.fine, p, compose(pose, relative(odom_last, o))[None])[0]
                for (_, p), o in zip(scans[-VERIFY_SCANS:], odom[-VERIFY_SCANS:])]))
            details = (f"{how}; scans fit {fit:.2f}, search score {info['score']:.2f} (next best elsewhere "
                       f"{info['runner_up']:.2f}), {time.time() - t0:.0f} s")
            if info["ambiguous"] or fit < GLOBAL_MIN:
                self.set_status("ambiguous", f"Not sure where the robot is ({details}). Is it the right room? "
                                "Otherwise drive it a bit and localize again.")
            else:
                self.set_status("localized", f"Localized {details}.")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep serving; the console shows the message
            self.set_status("failed", f"Localization failed: {e}")

    async def turn_on_spot(self):
        """One full turn at TURN_RATE; returns its scans, or None if stopped."""
        self.turn = {"turned": 0.0, "last": None, "since": time.time(), "scans": []}
        self.set_status("turning", "Turning once on the spot to see the whole room …")
        twist = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": TURN_RATE}}
        try:
            while self.turn["turned"] < 2 * math.pi + 0.1:
                now = time.time()
                if now - max(self.last_odom_recv, 0) > STALE_S or now - self.last_scan_recv > STALE_S:
                    self.set_status("failed", "Stopped turning: no scans or odometry from the robot.")
                    return None
                if now - self.turn["since"] > TURN_TIMEOUT_S:
                    self.set_status("failed", "Stopped turning: no full turn within "
                                    f"{TURN_TIMEOUT_S:.0f} s (blocked?).")
                    return None
                await self.client.publish(CMD_OUT, TOPICS["cmd_vel"][1], twist)
                await asyncio.sleep(0.1)
            return self.turn["scans"]
        finally:
            self.turn = None
            await self.send_stop()

    async def send_stop(self):
        for _ in range(3):  # the robot keeps the last command; make sure the stop arrives
            try:
                await self.client.publish(CMD_OUT, TOPICS["cmd_vel"][1], STOP)
            except Exception:
                return
            await asyncio.sleep(0.05)

    async def stop(self, reason="stopped"):
        """reason: who stopped (the console says: STOP, manual driving, disarmed, Cancel button, ...); in the
        log and the drive's message, so a chat tool can say why (2026-10-08: a search stopped twice and nobody
        knew by what)."""
        self.stop_count += 1
        self.last_stop = reason
        self.note("stop", f"stop: {reason}")
        self.cancel()
        self.stop_drive(f"stopped ({reason})")
        await self.send_stop()
        if self.status in ("turning", "searching"):
            self.set_status("idle" if self.pose is None else "uncertain", "Stopped.")

    # ---------- arm (for the chat's survey pose, nav_tools.py) ----------

    def on_servo_state(self, msg, recv):
        for s in msg.get("servo_state", []):
            self.servo[s["id"]] = s["position"]

    def arm_positions(self):
        return dict(self.servo)

    async def arm_to(self, pose, duration):
        """Servo positions {id: pulse} in duration seconds (as the console's H / P keys do), reached from above:
        first ARM_OVERSHOOT pulses higher, then the pose. The servos stop short of a pose by their deadband,
        reached from below by 5-13 pulses (the camera 5-7 deg off from what /tf says: /tf shows the commanded
        positions), from above by 0-5 (~1.5 deg; manipulation's measurement 2026-10-12, agent board). Same total
        time; returns at once (callers wait), a newer command cancels the second step."""
        if self.arm_settle is not None:
            self.arm_settle.cancel()
        high = {i: min(1000, int(p) + ARM_OVERSHOOT) for i, p in pose.items()}
        await self.servo_command(high, ARM_OVERSHOOT_SHARE * duration)

        async def settle():
            await asyncio.sleep(ARM_OVERSHOOT_SHARE * duration)
            await self.servo_command(pose, (1 - ARM_OVERSHOOT_SHARE) * duration)
        self.arm_settle = self.loop.create_task(settle())

    async def servo_command(self, pose, duration):
        await self.client.publish(TOPICS["servo_cmd"][0], TOPICS["servo_cmd"][1], {
            "duration": float(duration), "position_unit": "pulse",
            "position": [{"id": int(i), "position": int(p)} for i, p in pose.items()]})

    # ---------- driving to an object ----------

    def pose_matrix_at(self, t):
        """4x4 robot pose in the map at robot time t (tracked pose + odometry), or None."""
        if self.pose is None or not self.odom:
            return None
        return planar(compose(self.pose, relative(self.pose_odom, self.odom_at(t))))

    def on_rgb_info(self, msg, recv):
        k = msg["k"]
        self.rgb_K = (k[0], k[4], k[2], k[5])

    def on_found(self, label):
        """Live detection confirmed a find (live.py): a running drive or turn stops right there."""
        g = self.goto
        if g is not None and g.stop_on_find and g.state in ("planning", "driving", "docking", "turning"):
            f = self.live.watcher.found.get(label, {})
            g.found(label, f.get("position"))

    def held_view(self):
        """The held item's mask in the colour picture and the colour camera's intrinsics, or None."""
        h = self.held
        if not h or h.get("mask") is None or self.rgb_K is None:
            return None
        return h["mask"], self.rgb_K

    def current_pose(self):
        """The tracked pose moved on by the odometry since its scan."""
        if self.pose is None or not self.odom:
            return None
        return compose(self.pose, relative(self.pose_odom, np.array(self.odom[-1][1:])))

    def start_goto(self, object_id=None, point=None):
        """To an object of the room (its id) or to a point on the floor (pin & go: (x, y))."""
        if self.goto is None:
            raise ValueError("choose a room first")
        if self.status != "localized":
            raise ValueError("the robot is not localized")
        self.cancel()
        self.stop_drive("new goal")
        self.clear_overlay()
        # Every new goal starts without remembered depth obstacles: if the way was blocked and is clear now,
        # the next try finds it (user, 2026-10-07). The robot faces its travel direction, so the camera sees
        # what is still there again, and the collision check stops it in any case.
        self.depth.mem.forget()
        if point is not None:
            self.goto.start_point(float(point[0]), float(point[1]), time.monotonic())
            what = f"pin at x {point[0]:.2f} y {point[1]:.2f}"
        else:
            objs = {o["id"]: o for o in self.room.objects()}
            if object_id not in objs:
                raise ValueError(f"no object {object_id} in this room")
            self.goto.start(objs[object_id], time.monotonic())
            what = f"#{object_id} {objs[object_id]['label']}"
        self.note("drive", f"goto {what}")
        self.drive = self.loop.create_task(self.drive_loop())


    def start_goto_target(self, obj):
        """Drive to a target that is not in the room's list (a find of the live detection): stops in front
        of it like at an object; seeing it again doesn't end this drive."""
        if self.goto is None:
            raise ValueError("choose a room first")
        if self.status != "localized":
            raise ValueError("the robot is not localized")
        self.cancel()
        self.stop_drive("new goal")
        self.clear_overlay()
        self.depth.mem.forget()
        self.goto.start(obj, time.monotonic())
        self.goto.stop_on_find = False
        self.goto.is_find = True
        pose = self.current_pose()
        if pose is not None:
            # only the side facing the robot: with other things around a find (blocks beside it), the near side's
            # stop was blocked and the robot drove half a metre away towards another side, then came back (real
            # run 2026-10-09 17:30). If the direct stop is blocked, the plan goes as close as it fits (goto.py).
            c = np.array(obj["center"][:2], float)
            n = np.asarray(pose[:2], float) - c
            if np.hypot(*n) > 0.05:
                n = n / np.hypot(*n)
                p = c + n * (max(obj["size"][:2]) / 2 + GAP + HALF_LENGTH)
                self.goto.spots = [(float(p[0]), float(p[1]), math.atan2(-n[1], -n[0]), 0.0)]
        self.note("drive", f"goto found {obj['label']} at ({obj['center'][0]:.2f}, {obj['center'][1]:.2f})")
        self.drive = self.loop.create_task(self.drive_loop())

    def start_turn(self, object_id=None, degrees=None):
        """Turn on the spot: to face an object of the room, or by degrees (positive = left)."""
        if self.goto is None:
            raise ValueError("choose a room first")
        if self.status != "localized":
            raise ValueError("the robot is not localized")
        pose = self.current_pose()
        if object_id is not None:
            objs = {o["id"]: o for o in self.room.objects()}
            if object_id not in objs:
                raise ValueError(f"no object {object_id} in this room")
            o = objs[object_id]
            heading = math.atan2(o["center"][1] - pose[1], o["center"][0] - pose[0])
            label, what = o["label"], f"#{object_id} {o['label']}"
        else:
            heading = wrap(pose[2] + math.radians(float(degrees)))
            label = what = f"the new heading ({float(degrees):+.0f} deg)"
        self.cancel()
        self.stop_drive("new goal")
        self.clear_overlay()
        self.goto.start_turn(heading, label, time.monotonic(), object_id)
        self.note("drive", f"turn to {what}")
        self.drive = self.loop.create_task(self.drive_loop())

    def stop_drive(self, why):
        if self.goto is not None:
            self.goto.stop(why)
        if self.drive is not None and not self.drive.done():
            self.drive.cancel()
        self.drive = None

    def obstacles_map(self, pose):
        """Latest scan + depth obstacles + things seen on the floor, in the map frame (planning)."""
        pts = [self.depth.points("kept"), self.seen_obstacles()]
        if self.last_scan_robot is not None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pts.append(self.last_scan_robot @ np.array([[c, s], [-s, c]]) + pose[:2])
        return np.concatenate(pts)

    def seen_obstacles(self):
        """The detector's sure sightings on the floor (any search of this chat, also rejected ones: a red cube is
        in the way all the same) as discs of points, map frame; the drive's own target left out."""
        if self.live is None:
            return np.zeros((0, 2))
        g = self.goto
        target = None if g is None or g.target is None or g.target.get("center") is None else g.target["center"][:2]
        pts = []
        for e in list(self.live.watcher.seen):
            p = e.get("position")
            if p is None or e.get("quality", "sure") != "sure" or e.get("missing_since"):
                continue
            size = e.get("size") or [0.05, 0.05, 0.05]
            if p[2] - size[2] / 2 > SEEN_LOW_M:
                continue  # on a table or cabinet, or on the wall (an outlet): not in the way
            if target is not None and math.hypot(p[0] - target[0], p[1] - target[1]) < TARGET_SELF_M:
                continue
            r = min(SEEN_MAX_R, max(SEEN_MIN_R, max(size[0], size[1]) / 2))
            a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            pts.append(np.stack([p[0] + r * np.cos(a), p[1] + r * np.sin(a)], 1))
            pts.append(np.array([[p[0], p[1]]]))
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def near_points(self, pose):
        """Latest scan + confirmed depth obstacles + things seen on the floor near the robot, robot frame
        (collision check)."""
        pts = [self.last_scan_robot] if self.last_scan_robot is not None else []
        d = np.concatenate([self.depth.points("confirmed"), self.seen_obstacles()])
        if len(d):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            local = (d - pose[:2]) @ np.array([[c, -s], [s, c]])
            pts.append(local[np.hypot(local[:, 0], local[:, 1]) < NEAR_DEPTH_M])
        return np.concatenate(pts) if pts else None

    async def drive_loop(self):
        g = self.goto
        planning = None
        try:
            while g.state in ("planning", "driving", "docking", "turning"):
                now = time.monotonic()
                if time.time() - self.last_scan_recv > STALE_S or time.time() - self.last_odom_recv > STALE_S:
                    g.stop("no scans or odometry from the robot: stopped")
                    break
                if self.status != "localized":
                    g.stop(f"localization {self.status}: stopped")
                    break
                pose = self.current_pose()
                if g.plan_due(now) and (planning is None or planning.done()):
                    g.set_obstacles(self.obstacles_map(pose))
                    planning = self.loop.run_in_executor(None, g.plan, pose, now)  # ~0.1-0.3 s
                    if g.state == "planning":
                        await planning
                        continue
                vx, vy, wz = g.step(now, pose, self.near_points(pose))
                await self.client.publish(CMD_OUT, TOPICS["cmd_vel"][1], {
                    "linear": {"x": float(vx), "y": float(vy), "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": float(wz)}})
                await asyncio.sleep(1.0 / DRIVE_HZ)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            g.stop(f"error ({type(e).__name__}: {e})")
        finally:
            await self.send_stop()
            self.note("drive", f"drive {g.state}: {g.message}")

    def on_depth(self, msg, recv):
        if self.pose is None:
            return
        try:  # never let the depth path stop the navigator: without it, driving uses the lidar alone
            t = stamp(msg)
            self.depth.on_depth(msg, planar(compose(self.pose, relative(self.pose_odom, self.odom_at(t)))))
        except Exception as e:
            if not getattr(self, "depth_error", False):
                self.depth_error = True
                print(f"depth obstacles off: {type(e).__name__}: {e}", flush=True)

    # ---------- robot link ----------

    async def run(self):
        self.loop = asyncio.get_running_loop()
        while True:
            try:
                self.client = await RosbridgeClient(ROSBRIDGE_URL).connect()
                await self.client.subscribe(TOPICS["odom"][0], TOPICS["odom"][1], self.on_odom)
                await self.client.subscribe(TOPICS["scan"][0], TOPICS["scan"][1], self.on_scan)
                await self.client.subscribe(TOPICS["servo_state"][0], TOPICS["servo_state"][1], self.on_servo_state,
                                            throttle_ms=500)
                await self.client.subscribe(TOPICS["rgb_info"][0], TOPICS["rgb_info"][1], self.on_rgb_info,
                                            throttle_ms=5000)
                await self.client.subscribe(TOPICS["tf"][0], TOPICS["tf"][1], self.tf.add, queue_length=20)
                await self.client.subscribe(TOPICS["tf_static"][0], TOPICS["tf_static"][1], self.tf.add, queue_length=20)
                # through a lambda: use_room replaces self.depth, and a bound method would keep feeding the
                # old one (the new one then never got the intrinsics: no depth obstacles after a room change)
                await self.client.subscribe(TOPICS["depth_info"][0], TOPICS["depth_info"][1],
                                            lambda m, r: self.depth.on_info(m, r), throttle_ms=5000)
                await self.client.subscribe(TOPICS["depth_png"][0], TOPICS["depth_png"][1], self.on_depth,
                                            throttle_ms=DEPTH_THROTTLE_MS)
                if ROBOT_HOST not in ("127.0.0.1", "localhost"):  # not for the fake robot
                    self.loop.run_in_executor(None, start_depth_compressor)
                self.connected = True
                print(f"connected to {ROSBRIDGE_URL} (commands to {CMD_OUT})", flush=True)
                await self.client.closed.wait()
            except Exception as e:
                print(f"robot link: {type(e).__name__}: {e}", flush=True)
            self.stop_drive("lost the robot link")
            if self.connected and self.status == "turning":
                self.set_status("failed", "Lost the robot link while turning.")
            self.cancel()
            self.connected = False
            await asyncio.sleep(2)

    def state(self):
        now = time.time()
        return {
            "connected": self.connected, "room": self.room.name if self.room else None, "code": CODE_VERSION,
            "status": self.status, "message": self.message,
            "pose": None if self.pose is None else [round(float(v), 4) for v in self.pose], "pose_t": self.pose_t,
            "score": self.score, "runner_up": self.runner_up,
            "turned_deg": round(math.degrees(self.turn["turned"])) if self.turn else None,
            "scan": self.scan_view, "scan_age": round(now - self.last_scan_recv, 2) if self.last_scan_recv else None,
            "cmd_topic": CMD_OUT, "depth_images": self.depth.count, "depth_kept": len(self.depth.mem.kept()),
            "drive": self.drive_state(),
            "live": self.live.state() if self.live else None,
            "last_note": self.last_note,
            "recording": None if not self.live or self.live.recorder is None else {
                "name": self.live.recorder.path.name, "images": self.live.recorder.n,
                "seconds": round(time.time() - self.live.recorder.start_time)},
        }

    def drive_state(self):
        g = self.goto
        if g is None or g.target is None:
            return None
        path = g.path[::max(1, len(g.path) // 80)] if g.path else []
        return {"state": g.state, "message": g.message, "target": {"id": g.target["id"], "label": g.target["label"]},
                "goal": None if g.goal is None else [round(v, 3) for v in g.goal],
                "path": [[round(x, 3), round(y, 3)] for x, y in path]}


def start_depth_compressor():
    """The robot's PNG depth compressor (light: compression only), as the recorder starts it."""
    from landerpi.robot_ssh import ensure_depth_compressor
    try:
        print(f"depth compressor: {ensure_depth_compressor()}", flush=True)
    except Exception as e:
        print(f"depth compressor: {e} (driving uses the lidar alone)", flush=True)


def make_handler(nav):
    def call(coro_or_fn, *args):
        """Runs on the event loop and waits for it (the HTTP server has its own threads)."""
        async def wrapper():
            r = coro_or_fn(*args)
            return await r if asyncio.iscoroutine(r) else r
        return asyncio.run_coroutine_threadsafe(wrapper(), nav.loop).result(timeout=30)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            origin = self.headers.get("Origin", "")
            if origin.startswith(("http://127.0.0.1:", "http://localhost:")):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "X-LanderPi-Console, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):  # CORS preflight of the console's POSTs
            self.send_response(204)
            origin = self.headers.get("Origin", "")
            if origin.startswith(("http://127.0.0.1:", "http://localhost:")):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "X-LanderPi-Console, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.end_headers()

        def do_GET(self):
            if self.path == "/chat":
                return self._reply(200, {"messages": nav.assistant.log if nav.assistant else []})
            if self.path == "/overlay":  # the boxes alone, often (tasks.js): read without the event loop
                return self._reply(200, {"overlay": nav.live.overlay_now() if nav.live is not None else None})
            if self.path != "/state":
                return self._reply(404, {"error": "not found"})
            self._reply(200, call(nav.state))

        def chat(self, body):
            """Runs in this HTTP thread (the tools call the event loop themselves; a drive takes a while)."""
            if nav.assistant is None:
                from assistant import Assistant
                nav.assistant = Assistant({"navigator": nav, "call": call})
            stops = nav.stop_count
            try:
                new = nav.assistant.send(str(body["message"])[:2000], armed=bool(body.get("armed")),
                                         stopped=lambda: nav.stop_count != stops)
            except RuntimeError as e:
                return self._reply(409, {"ok": False, "error": str(e)})
            except Exception as e:  # a bug: the console shows it instead of a dropped connection
                import traceback
                traceback.print_exc()
                return self._reply(500, {"ok": False, "error": f"the chat failed: {type(e).__name__}: {e}"})
            self._reply(200, {"ok": True, "messages": new})

        def do_POST(self):
            # the header forces a CORS preflight, so other websites can't make the robot turn
            if "X-LanderPi-Console" not in self.headers:
                return self._reply(403, {"error": "missing header"})
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if self.path == "/chat":
                return self.chat(body)
            if self.path == "/chat/reset":
                if nav.assistant:
                    nav.assistant.reset()
                if nav.live is not None:  # the sightings belong to the conversation (user, 2026-10-09)
                    nav.live.stop()
                    nav.live.watcher.forget()
                    nav.live.clear_overlay()
                return self._reply(200, {"ok": True})
            try:
                if self.path == "/room":
                    call(nav.use_room, str(body["name"]))
                elif self.path == "/localize":
                    call(nav.start_localize)
                elif self.path == "/goto":
                    if "x" in body:
                        call(nav.start_goto, None, (float(body["x"]), float(body["y"])))
                    else:
                        call(nav.start_goto, int(body["object"]))
                elif self.path == "/record/start":
                    call(nav.start_record)
                elif self.path == "/record/stop":
                    call(nav.stop_record)
                elif self.path == "/stop":
                    call(nav.stop, str(body.get("reason") or "stop from the console")[:80])
                elif self.path == "/quit":
                    call(nav.stop)
                    self._reply(200, {"ok": True})
                    os._exit(0)
                else:
                    return self._reply(404, {"error": "not found"})
            except (KeyError, ValueError, FileNotFoundError) as e:
                return self._reply(400, {"ok": False, "error": str(e)})
            self._reply(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--room", help="load this room at the start")
    args = p.parse_args()

    nav = Navigator()

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], ConnectionError):  # a closed browser tab: nothing to report
                super().handle_error(request, client_address)

    async def start():
        nav.loop = asyncio.get_running_loop()
        nav.live = LiveDetection(nav)
        server = Server(("127.0.0.1", args.port), make_handler(nav))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"navigator: http://127.0.0.1:{args.port}/state", flush=True)
        if args.room:
            nav.use_room(args.room)
        await nav.run()

    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
