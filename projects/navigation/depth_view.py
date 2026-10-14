"""Obstacles above the lidar's slice, from the depth camera, for driving to objects (goto.py). No ROS.

The rules are the explorer's (offboard_ros/depth_obstacles.py: 10-45 cm above the floor, confirmed over
several images; coverage3d.FloorHeight: the floor's height from the images), imported unchanged. The
camera sits on the arm, so its pose comes from the robot's /tf and /tf_static (TfTable) and the measured
mounting correction in the survey pose (config/camera_correction.json), as explorer.py does it.

    tf = TfTable(); tf.add(msg)                       # every /tf and /tf_static message
    d = DepthObstacles(tf); d.on_info(camera_info)
    d.on_depth(compressed_png_msg, T_map_base)        # 4x4, the robot's pose at the image's time
    d.points("confirmed")                             # Nx2 map points: current obstacles (collision check)
    d.points("kept")                                  # ... seen in several images (planning)
"""

import base64
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offboard_ros"))
import coverage3d as c3  # noqa: E402
import depth_obstacles as do  # noqa: E402

CAMERA_CORRECTION = REPO / "config" / "camera_correction.json"
HELD_NEAR_M = 0.6         # depth pixels nearer than this may show the held item (it is ~0.10-0.15 m away)
# Low things near the robot: the explorer's band starts 10 cm above the floor (depth_obstacles.py, floor noise),
# the lidar sees ~20 cm up; a 3 cm block was in neither, the detector hardly boxed it (not what was searched for),
# and the robot nearly drove into it (real run 2026-10-09). Here a cell counts if its top stands LOW_STEP above
# the lowest floor around it (its 3x3 neighbourhood), within LOW_RANGE: a local step, not a height above one
# global floor, which a camera tilted by 1.5 deg turned into a whole stretch of "obstacle" (synthetic test).
LOW_STEP = 0.02           # m: the top (90 % of its points) this much above the floor around it
LOW_TOP = 0.15            # m: ... and lower than this above the floor (higher: the explorer's band has it)
LOW_RANGE = 1.0           # m from the camera: beyond it a 3 cm block has too few depth points
LOW_MIN_POINTS = 8        # sampled depth points (every 2nd pixel) for a cell's heights


def low_cells(depth_mm, K, T_map_cam, floor_z):
    """Cells (do.CELL grid) whose top stands LOW_STEP above the floor around them, within LOW_RANGE."""
    fx, fy, cx, cy = K
    z = depth_mm[::do.STEP, ::do.STEP].astype(np.float32) / 1000.0
    v, u = np.nonzero((z > do.MIN_RANGE) & (z < LOW_RANGE))
    if len(v) < LOW_MIN_POINTS:
        return set()
    zz = z[v, u]
    u, v = u * do.STEP, v * do.STEP
    w = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], 1) @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
    h = w[:, 2] - floor_z
    ok = h < LOW_TOP + 0.05
    ij = np.floor(w[ok, :2] / do.CELL).astype(np.int64)
    h = h[ok]
    if not len(h):
        return set()
    order = np.lexsort((h, ij[:, 1], ij[:, 0]))  # by cell, then height
    ij, h = ij[order], h[order]
    cells, start, count = np.unique(ij, axis=0, return_index=True, return_counts=True)
    good = count >= LOW_MIN_POINTS
    cells, start, count = cells[good], start[good], count[good]
    low = h[start + (0.1 * (count - 1)).astype(int)]   # each cell's floor (10 %)
    top = h[start + (0.9 * (count - 1)).astype(int)]   # ... and top (90 %)
    floor_at = {tuple(c): f for c, f in zip(cells, low)}
    out = set()
    for c, t in zip(cells, top):
        if t > LOW_TOP:
            continue
        around = [floor_at.get((c[0] + di, c[1] + dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        ground = min(f for f in around if f is not None)
        if t - ground > LOW_STEP:
            out.add((int(c[0]), int(c[1])))
    return out


def matrix(t):
    q, p = t["rotation"], t["translation"]
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def planar(pose):
    """(x, y, yaw) -> 4x4."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = pose[:2]
    return T


class TfTable:
    """The latest transform per child frame (the arm moves rarely: no interpolation in time)."""

    def __init__(self):
        self.parent = {}  # child -> (parent, 4x4)

    def add(self, msg, recv=None):
        for t in msg.get("transforms", []):
            if t["header"]["frame_id"] == "odom":
                continue  # odometry: the navigator has the pose
            self.parent[t["child_frame_id"]] = (t["header"]["frame_id"], matrix(t["transform"]))

    def chain(self, frame, root="base_footprint"):
        """frame -> root as 4x4, or None if a link is missing."""
        T = np.eye(4)
        for _ in range(30):
            if frame == root:
                return T
            if frame not in self.parent:
                return None
            frame, M = self.parent[frame]
            T = M @ T
        return None


class ClearingMemory(do.DepthMemory):
    """The explorer's DepthMemory, but cells can be forgotten again. Kept for good, cells boxed the robot in
    for the rest of the session: after a drive ending close to a cabinet front no plan found a way out any
    more, even after localizing again, until the navigator was restarted (real run, 2026-10-07 14:51). Now:
      - a kept cell the camera looks at again (in view, CLEAR_RANGE) without seeing it in CLEAR_MISSES
        images in a row is forgotten (the user cleared the way; a person walked off)
      - a cell nobody saw again for FORGET_S is forgotten (the collision check sees it again if it's there)"""
    CLEAR_RANGE = (0.35, 1.3)        # m from the camera: where the depth band (10-45 cm) is reliably in view
    CLEAR_HALF_ANGLE = math.radians(28)  # of the camera's 74 deg, with margin for the image edges
    CLEAR_MISSES = 3
    FORGET_S = 120.0

    def __init__(self):
        super().__init__()
        self.misses = {}

    def add_view(self, cells, now, cam_xy, cam_dir):
        """One image's detections plus where the camera looked (map x, y and the view direction)."""
        self.add(cells, now)  # (the navigator also forgets everything at every new goal)
        for c in cells:
            self.misses[c] = 0
        kept = self.kept()
        if not kept:
            return
        centres = self.centres(kept)
        d = centres - cam_xy
        dist = np.hypot(d[:, 0], d[:, 1])
        ang = np.abs(np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - cam_dir))))
        in_view = (dist > self.CLEAR_RANGE[0]) & (dist < self.CLEAR_RANGE[1]) & (ang < self.CLEAR_HALF_ANGLE)
        for c, v in zip(kept, in_view):
            if v and c not in cells:
                self.misses[c] = self.misses.get(c, 0) + 1
                if self.misses[c] >= self.CLEAR_MISSES:
                    self.forget(c)
        for c in [c for c, t in self.seen.items() if now - t[-1] > self.FORGET_S]:
            self.forget(c)

    def forget(self, c=None):
        """One cell, or everything (c=None)."""
        if c is None:
            for d in (self.seen, self.count, self.misses):
                d.clear()
            return
        for d in (self.seen, self.count, self.misses):
            d.pop(c, None)


class DepthObstacles:
    def __init__(self, tf, held=None):
        self.tf = tf
        self.held = held   # () -> (mask in the colour picture, colour K) of what the gripper holds, or None
        self.K = None
        self.mem = ClearingMemory()
        self.floor = c3.FloorHeight()
        self.count = 0
        self.last = None   # (depth mm, T_map_cam, robot time) of the latest image
        self.fix = None
        try:
            c = json.loads(CAMERA_CORRECTION.read_text())
            rx, ry = math.radians(c["roll_deg"]), math.radians(c["pitch_deg"])
            Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
            Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
            self.fix = (Ry @ Rx, np.array(c["base_to_depth_rotation"]))
        except (OSError, ValueError, KeyError):
            pass  # without the correction: 2.8 deg pitch error, ~10 cm at 2 m (explorer.py)

    def on_info(self, msg, recv=None):
        k = msg["k"]
        self.K = (k[0], k[4], k[2], k[5])

    def camera_in_base(self, frame):
        """A camera frame in base_footprint, with the measured mounting correction while the arm is in the
        survey pose (as calibrated; reconstruct.py applies it the same way to depth and colour)."""
        B = self.tf.chain(frame)
        if B is None:
            return None
        D = self.tf.chain("depth_camera_link")
        if self.fix is not None and D is not None:
            R, ref = self.fix
            cos = (np.trace(ref.T @ D[:3, :3]) - 1) / 2
            if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= 3:  # survey pose
                B = B.copy()
                B[:3, :3] = R @ B[:3, :3]
        return B

    def on_depth(self, msg, T_map_base):
        if self.K is None or T_map_base is None:
            return False
        B = self.camera_in_base(msg["header"]["frame_id"])
        if B is None:
            return False
        data = msg["data"]
        raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
        depth = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            return False
        T_map_cam = T_map_base @ B
        hv = self.held() if self.held is not None else None
        if hv is not None:  # the carried item is no obstacle (manipulation's request, 2026-10-12)
            depth = self.without_held(depth, B, *hv)
        st = msg["header"]["stamp"]
        # the latest image with its pose, for 3D positions of detections (perception.py)
        self.last = (depth, T_map_cam, st["sec"] + st["nanosec"] * 1e-9)
        self.floor.add(depth, self.K, T_map_cam)
        forward = T_map_cam[:3, 2]  # the optical axis
        cells = do.obstacle_cells(depth, self.K, T_map_cam, self.floor.z) | \
            low_cells(depth, self.K, T_map_cam, self.floor.z)  # and low things near the robot (blocks, cables)
        self.mem.add_view(cells, time.monotonic(), T_map_cam[:2, 3], math.atan2(forward[1], forward[0]))
        self.count += 1
        return True

    def without_held(self, depth, B_depth, mask, K_rgb):
        """The depth image with the pixels that show the held item blanked: near pixels (HELD_NEAR_M) projected
        into the colour picture, those landing on its mask. The colour camera and the gripper are rigid, so the
        mask holds in every arm pose."""
        B_rgb = self.camera_in_base("rgb_camera_link")
        if B_rgb is None or self.K is None:
            return depth
        z = depth.astype(np.float32) / 1000.0
        v, u = np.nonzero((z > 0) & (z < HELD_NEAR_M))
        if not len(v):
            return depth
        fx, fy, cx, cy = self.K
        zz = z[v, u]
        p = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz, np.ones_like(zz)], 1)
        q = (np.linalg.inv(B_rgb) @ B_depth @ p.T).T  # depth camera -> base -> colour camera
        rx, ry, rcx, rcy = K_rgb
        ok = q[:, 2] > 0.01
        pu = np.full(len(q), -1, int)
        pv = np.full(len(q), -1, int)
        pu[ok] = np.round(q[ok, 0] * rx / q[ok, 2] + rcx).astype(int)
        pv[ok] = np.round(q[ok, 1] * ry / q[ok, 2] + rcy).astype(int)
        h, w = mask.shape
        inside = ok & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        hit = np.zeros(len(q), bool)
        hit[inside] = mask[pv[inside], pu[inside]]
        out = depth.copy()
        out[v[hit], u[hit]] = 0
        return out

    def points(self, which="confirmed"):
        cells = self.mem.confirmed(time.monotonic()) if which == "confirmed" else self.mem.kept()
        return do.DepthMemory.centres(cells) if cells else np.zeros((0, 2))
