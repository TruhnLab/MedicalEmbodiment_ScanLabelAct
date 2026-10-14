"""M3: 3D reconstruction of a recorded mapping session, on the laptop.

    python projects/mapping/reconstruct.py                       # newest session
    python projects/mapping/reconstruct.py data/sessions/<name> --voxel 0.01 --max-depth 3

Writes <session>/reconstruction/: mesh.ply (TSDF surface, coloured), cloud.ply (coloured points),
poses.csv (camera pose in the map for every keyframe) and preview.png.

Camera pose of an image taken at time t (the camera sits on the arm):
    map -> odom            SLAM's drift correction      (streams/slam_map_odom.jsonl; identity without SLAM)
    odom -> base_footprint wheel/IMU odometry           (streams/odom.jsonl)
    base_footprint -> ... -> link1..link5 -> depth_camera_link
                           robot + arm transforms        (calib/tf_static.json, streams/tf.jsonl)
depth_camera_link and rgb_camera_link are optical frames (z forward, x right, y down), matching the
camera_info of the depth and colour images. Colour and depth are separate captures (up to ~50 ms
apart): each gets its own pose, and colour is projected onto the depth points accordingly.
"""

import argparse
import csv
import json
import math
import os
import time
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

REPO = Path(__file__).resolve().parents[2]
DEPTH_FRAME = "depth_camera_link"
RGB_FRAME = "rgb_camera_link"
# The colour images show the scene ~0.1 s before their stamps (the camera driver stamps them when it
# publishes, not at exposure; the MJPEG's X-Timestamp passes that on). Measured by the scene
# understanding side (2026-10-06): best fit -0.09 to -0.12 s in 20261005_142856 and _160227, with
# projects/scene_understanding/check_camera_timing.py (depth edges on colour edges while turning).
# Depth and poses fit at every turn rate. Uncorrected, at 30 deg/s the colour was ~3 deg (~20 px) off
# and one wall outlet became two objects 19 cm apart.
RGB_TIME_OFFSET = -0.10
BASE_FRAME = "base_footprint"


# ---------- rigid transforms ----------

def matrix(translation, rotation):
    x, y, z, w = rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return T


def stamp(header):
    return header["stamp"]["sec"] + header["stamp"]["nanosec"] * 1e-9


class Series:
    """A transform over time; lookups return the sample nearest to t (sources run at 10-30 Hz)."""

    def __init__(self):
        self.t, self.T = [], []

    def add(self, t, T):
        self.t.append(t)
        self.T.append(T)

    def finish(self):
        order = np.argsort(self.t)
        self.t = [self.t[i] for i in order]
        self.T = [self.T[i] for i in order]
        return self

    def at(self, t):
        i = bisect_left(self.t, t)
        if i == 0:
            return self.T[0], abs(self.t[0] - t)
        if i == len(self.t):
            return self.T[-1], abs(self.t[-1] - t)
        j = i if self.t[i] - t < t - self.t[i - 1] else i - 1
        return self.T[j], abs(self.t[j] - t)


class TransformTree:
    """Static + time-varying transforms of one session; chains them from 'map' to any frame."""

    def __init__(self, session):
        self.parent = {}   # child -> parent
        self.static = {}   # child -> 4x4
        self.dynamic = {}  # child -> Series
        for t in json.loads((session / "calib" / "tf_static.json").read_text())["transforms"]:
            self.parent[t["child_frame_id"]] = t["header"]["frame_id"]
            self.static[t["child_frame_id"]] = matrix(t["transform"]["translation"], t["transform"]["rotation"])
        for line in open(session / "streams" / "tf.jsonl", encoding="utf-8"):
            for t in json.loads(line)["msg"]["transforms"]:
                child = t["child_frame_id"]
                self.parent[child] = t["header"]["frame_id"]
                self.dynamic.setdefault(child, Series()).add(
                    stamp(t["header"]), matrix(t["transform"]["translation"], t["transform"]["rotation"]))
        # odom -> base_footprint from the odometry stream (the same as in tf, but at the full rate)
        odom = Series()
        for line in open(session / "streams" / "odom.jsonl", encoding="utf-8"):
            m = json.loads(line)["msg"]
            odom.add(stamp(m["header"]), matrix(m["pose"]["pose"]["position"], m["pose"]["pose"]["orientation"]))
        self.parent[BASE_FRAME] = "odom"
        self.dynamic[BASE_FRAME] = odom
        # map -> odom from SLAM, if it was running
        slam = session / "streams" / "slam_map_odom.jsonl"
        self.has_slam = slam.exists() and slam.stat().st_size > 0
        if self.has_slam:
            s = Series()
            for line in open(slam, encoding="utf-8"):
                t = json.loads(line)["msg"]
                s.add(stamp(t["header"]), matrix(t["transform"]["translation"], t["transform"]["rotation"]))
            self.dynamic["odom"] = s
        else:
            self.static["odom"] = np.eye(4)
        self.parent["odom"] = "map"
        for s in self.dynamic.values():
            s.finish()
        self.borrowed = self._borrow_static(session)

    def _borrow_static(self, session):
        """The robot's fixed transforms (camera mounts etc.) never change, but the recorder sometimes
        misses one at startup. Fill frames this session lacks from other sessions of the same robot."""
        borrowed = []
        for other in sorted(session.parent.iterdir(), key=lambda d: d.name, reverse=True):
            f = other / "calib" / "tf_static.json"
            if other == session or not f.exists():
                continue
            for t in json.loads(f.read_text())["transforms"]:
                child = t["child_frame_id"]
                if child not in self.parent:
                    self.parent[child] = t["header"]["frame_id"]
                    self.static[child] = matrix(t["transform"]["translation"], t["transform"]["rotation"])
                    borrowed.append(f"{child} (from {other.name})")
        return borrowed

    def gap(self, frame, t):
        """Largest time between t and the robot/arm transform samples used for `frame` (excl. SLAM)."""
        g = 0.0
        while frame not in ("map", "odom"):
            if frame in self.dynamic:
                g = max(g, self.dynamic[frame].at(t)[1])
            frame = self.parent[frame]
        return g

    def map_to(self, frame, t):
        """4x4 pose of `frame` in the map at time t, and the largest time gap used in the chain."""
        T, gap = np.eye(4), 0.0
        chain = []
        while frame != "map":
            if frame not in self.parent:
                raise KeyError(f"no transform to '{frame}' (missing in calib/tf_static.json?)")
            chain.append(frame)
            frame = self.parent[frame]
        for child in reversed(chain):
            if child in self.dynamic:
                M, g = self.dynamic[child].at(t)
                gap = max(gap, g)
            else:
                M = self.static[child]
            T = T @ M
        return T, gap


class CameraCorrection:
    """Calibrated camera rotation in the survey pose (config/camera_correction.json, written by
    camera_calibration.py): the robot model computes the camera pose from the commanded servo
    positions, the real camera looks a few degrees up and is slightly rolled. Applied in the base
    frame, only to keyframes whose camera sits like in the calibration (the survey pose)."""

    def __init__(self, session):
        self.ok = False
        f = REPO / "config" / "camera_correction.json"
        if not f.exists():
            return
        c = json.loads(f.read_text())
        rx, ry = np.radians(c["roll_deg"]), np.radians(c["pitch_deg"])
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        self.R = Ry @ Rx
        self.pan = 0.0
        before = c.get("pan_deg_before")
        if before and session.name < before["session"]:  # recorded before the servo-1 zero fix
            self.pan = before["pan_deg"]
            rz = np.radians(self.pan)
            self.R = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]) @ self.R
        self.ref = np.array(c["base_to_depth_rotation"])
        self.text = f"pitch {c['pitch_deg']:+.1f}, roll {c['roll_deg']:+.1f}" + (
            f", pan {self.pan:+.1f} (before the servo fix)" if self.pan else "") + " deg"
        self.applied = self.skipped = 0
        self.ok = True

    def apply(self, tree, T_map_depth, T_map_rgb, t_depth, t_rgb):
        out = []
        for T, t in ((T_map_depth, t_depth), (T_map_rgb, t_rgb)):
            T_base = tree.map_to(BASE_FRAME, t)[0]
            B = np.linalg.inv(T_base) @ T
            out.append((T_base, B))
        # survey pose? (the depth camera's orientation on the robot as in the calibration, +-3 deg)
        cos = (np.trace(self.ref.T @ out[0][1][:3, :3]) - 1) / 2
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > 3:
            self.skipped += 1
            return T_map_depth, T_map_rgb
        self.applied += 1
        result = []
        for T_base, B in out:
            B = B.copy()
            B[:3, :3] = self.R @ B[:3, :3]  # orientation only: the camera's position is fine
            result.append(T_base @ B)
        return result


# ---------- images ----------

def intrinsics(info):
    k = info["k"]
    return info["width"], info["height"], k[0], k[4], k[2], k[5]


def depth_points(depth_mm, info, min_d, max_d, edge_px=1):
    """Depth image -> Nx3 points in the depth optical frame, plus the pixel indices used.
    Pixels at depth discontinuities ("flying pixels" between an object edge and the background,
    which smear objects into fringes) and next to holes are dropped."""
    w, h, fx, fy, cx, cy = intrinsics(info)
    z = depth_mm.astype(np.float32) / 1000.0
    kernel = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)  # edge_px pixels dropped at edges
    has_depth = (z > 0).astype(np.uint8)
    local_max = cv2.dilate(z, kernel)
    local_min = cv2.erode(np.where(z > 0, z, 1e3).astype(np.float32), kernel)
    edge = (local_max - local_min) > (0.02 + 0.03 * z)  # jump > 2 cm + 3 % of the distance
    valid = (z > min_d) & (z < max_d) & ~edge & (cv2.erode(has_depth, kernel) > 0)
    v, u = np.nonzero(valid)
    zz = z[v, u]
    pts = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], axis=1)
    return pts, v, u


def colour_for(points_rgb_cam, rgb, info):
    """Colour of each point (already in the colour camera's optical frame); NaN where not visible."""
    w, h, fx, fy, cx, cy = intrinsics(info)
    z = points_rgb_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(points_rgb_cam[:, 0] * fx / z + cx).astype(np.int64)
        v = np.round(points_rgb_cam[:, 1] * fy / z + cy).astype(np.int64)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    colours = np.full((len(z), 3), np.nan, dtype=np.float32)
    colours[ok] = rgb[v[ok], u[ok]][:, ::-1] / 255.0  # BGR -> RGB
    return colours


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def planar(T):
    """Keeps only the ground-plane part of a correction (x, y, yaw): the robot drives on a flat
    floor, so a correction may not tilt or lift the camera (best practice for ground robots)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    P = np.eye(4)
    P[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    P[:2, 3] = T[:2, 3]
    return P


# ---------- pose refinement and floor levelling ----------

ICP_VOXEL = 0.03          # downsampling for alignment
ICP_MAX_DIST = 0.08       # correspondences further apart are ignored
ICP_MAX_SHIFT = 0.06      # accept corrections up to 6 cm ...
ICP_MAX_TURN_DEG = 4.0    # ... and 4 degrees per keyframe
MODEL_RADIUS = 4.0        # align against the model within this distance of the camera


def refine_poses(frames, progress=lambda *a, **k: None):
    """Frame-to-model ICP: aligns every keyframe's depth points to the model built from the frames
    before it, correcting small local errors (arm servos, timing, SLAM jitter). SLAM stays in charge
    of the global position: each correction is relative to the keyframe's own recorded pose, small
    and planar, and corrections do NOT accumulate. (Accumulating them let ICP slide along flat walls
    and floors: up to 92 cm of drift in a 2-minute session.) frames: list of dicts with 'pts'
    (depth frame, downsampled) and 'T0' (recorded map pose); adds 'C' (correction)."""
    model = o3d.geometry.PointCloud()
    accepted = 0
    for i, f in enumerate(frames):
        progress("Aligning (ICP)", i + 1, len(frames))
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(f["pts"]))
        # normals once per keyframe, on its own small cloud (facing the camera); they travel with
        # the points into the model, so the model never needs a full normal estimation
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        src.orient_normals_towards_camera_location(np.zeros(3))
        C = np.eye(4)
        T_init = f["T0"]
        if len(model.points) > 2000 and len(src.points) > 300:
            cam = T_init[:3, 3]
            box = o3d.geometry.AxisAlignedBoundingBox(cam - MODEL_RADIUS, cam + MODEL_RADIUS)
            local = model.crop(box)  # C++ crop: no per-point Python work on the whole model
            if len(local.points) > 2000:
                result = o3d.pipelines.registration.registration_icp(
                    src, local, ICP_MAX_DIST, T_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                delta = planar(result.transformation @ np.linalg.inv(T_init))
                shift = np.linalg.norm(delta[:2, 3])
                turn = abs(np.degrees(np.arctan2(delta[1, 0], delta[0, 0])))
                if (result.fitness > 0.4 and result.inlier_rmse < 0.03
                        and shift < ICP_MAX_SHIFT and turn < ICP_MAX_TURN_DEG):
                    C = delta
                    accepted += 1
        f["C"] = C.copy()
        model += src.transform(C @ f["T0"])
        if i % 5 == 4:
            model = model.voxel_down_sample(ICP_VOXEL)
    return accepted


def level_floor(frames):
    """Rigid correction that puts the floor at z = 0 and makes it level (removes e.g. the robot
    model's base-height offset). Uses the dominant low horizontal plane of all keyframes."""
    pts = np.concatenate([transform(f["C"] @ f["T0"], f["pts"]) for f in frames])
    hist, edges = np.histogram(pts[:, 2], bins=np.arange(-0.5, 1.0, 0.02))
    floor_z = edges[np.argmax(hist)] + 0.01
    low = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[np.abs(pts[:, 2] - floor_z) < 0.12]))
    if len(low.points) < 500:
        return np.eye(4), None
    (a, b, c, d), inliers = low.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    n = np.array([a, b, c]) * np.sign(c)
    d *= np.sign(c)
    # rotate the floor normal onto +z (Rodrigues), then shift the plane to z = 0
    axis = np.cross(n, [0, 0, 1])
    s, cth = np.linalg.norm(axis), n[2]
    R = np.eye(3)
    if s > 1e-9:
        k = axis / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * Kx + (1 - cth) * (Kx @ Kx)
    L = np.eye(4)
    L[:3, :3] = R
    L[2, 3] = d  # plane n.x + d = 0 lies at height -d before levelling
    return L, (np.degrees(np.arccos(min(1.0, cth))), -d, len(inliers))


# ---------- mesh clean-up and colouring from the images ----------

def clean_mesh(mesh, min_triangles=300, smooth_iterations=5):
    """Removes small floating fragments (depth noise, reflections) and smooths the surface gently
    (Taubin smoothing keeps the volume, unlike plain Laplacian smoothing, which shrinks it)."""
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[clusters] < min_triangles)
    mesh.remove_unreferenced_vertices()
    if smooth_iterations:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()
    return mesh, int((sizes < min_triangles).sum())


def bilinear(img, x, y):
    """Image values at sub-pixel positions (x, y inside the image); cv2.remap is limited to 32k points."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, img.shape[1] - 1)
    y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    ax = (x - x0)[:, None]
    ay = (y - y0)[:, None]
    img = img.astype(np.float32)
    return ((img[y0, x0] * (1 - ax) + img[y0, x1] * ax) * (1 - ay)
            + (img[y1, x0] * (1 - ax) + img[y1, x1] * ax) * ay)


class VertexGrid:
    """Vertices grouped in 1 m columns, so a keyframe only looks at those within its range."""

    def __init__(self, V, cell=1.0):
        self.cell = cell
        keys = np.floor(V[:, :2] / cell).astype(np.int64)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse.ravel(), kind="stable")
        bounds = np.searchsorted(inverse.ravel()[order], np.arange(len(uniq) + 1))
        self.groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
        self.centres = (uniq + 0.5) * cell

    def near(self, T_map_cam, max_depth):
        """Indices of the vertices that can be in front of this camera within max_depth."""
        c = T_map_cam[:2, 3]
        fwd = T_map_cam[:2, 2]
        d = self.centres - c
        reach = max_depth + self.cell
        ok = (np.hypot(d[:, 0], d[:, 1]) < reach) & (d @ fwd > -self.cell)
        sel = [self.groups[i] for i in np.nonzero(ok)[0]]
        return np.concatenate(sel) if sel else np.zeros(0, np.int64)


def observe(V, N, view, session, depth_info, rgb_info, max_depth, grid=None):
    """Which vertices a keyframe saw, their colour in its image and a quality weight.
    Visibility: the vertex must lie on the keyframe's measured depth surface (not behind it).
    Weight: frontal and close views count most (sharper, less smeared by pose errors).
    grid: a VertexGrid of V, to look only at vertices near the camera (much faster)."""
    if grid is not None:
        cand = grid.near(view[1], max_depth)
        if not len(cand):
            return None
        o = observe(V[cand], N[cand], view, session, depth_info, rgb_info, max_depth)
        return None if o is None else (cand[o[0]],) + o[1:]
    k, T_map_depth, T_map_rgb = view
    depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(session / k["rgb"]))
    if depth is None or rgb is None:
        return None
    w, h, fx, fy, cx, cy = intrinsics(depth_info)
    Pd = transform(np.linalg.inv(T_map_depth), V)
    z = Pd[:, 2]
    near = np.nonzero((z > 0.2) & (z < max_depth))[0]
    z = z[near]
    u = np.round(Pd[near, 0] * fx / z + cx).astype(np.int64)
    v = np.round(Pd[near, 1] * fy / z + cy).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    near, z, u, v = near[inside], z[inside], u[inside], v[inside]
    d = depth[v, u].astype(np.float32) / 1000.0
    vis = (d > 0) & (np.abs(z - d) < 0.05 + 0.02 * z)  # tolerant of the remaining few-cm pose errors
    idx = near[vis]
    if not len(idx):
        return None
    Pr = transform(np.linalg.inv(T_map_rgb), V[idx])
    rw, rh, rfx, rfy, rcx, rcy = intrinsics(rgb_info)
    ur = Pr[:, 0] * rfx / Pr[:, 2] + rcx
    vr = Pr[:, 1] * rfy / Pr[:, 2] + rcy
    ok = (Pr[:, 2] > 0) & (ur >= 0) & (ur <= rw - 1) & (vr >= 0) & (vr <= rh - 1)
    idx, ur, vr = idx[ok], ur[ok], vr[ok]
    col = srgb_to_linear(bilinear(rgb, ur, vr)[:, ::-1] / 255.0)  # BGR -> RGB, linear light
    ray = V[idx] - T_map_rgb[:3, 3]
    dist = np.linalg.norm(ray, axis=1)
    cos = np.abs(np.einsum("ij,ij->i", ray, N[idx])) / np.maximum(dist, 1e-6)
    # squared distance from the image centre, 1 at the corners (for vignetting)
    r2 = (((ur - rcx) / rcx) ** 2 + ((vr - rcy) / rcy) ** 2) / 2
    # edges of the image are more distorted and vignetted than the centre
    weight = cos ** 2 / np.maximum(dist, 0.3) ** 2 * np.clip(1.0 - 0.7 * r2, 0.2, 1.0)
    return idx, col.astype(np.float32), weight.astype(np.float32), r2.astype(np.float32)


def observe_views(views, fn, workers=None):
    """fn(n, view) for every keyframe, several at once (image decoding and numpy release the GIL);
    yields (n, result) in order, with only a few results in memory at a time."""
    workers = workers or min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        for n, view in enumerate(views):
            pending.append((n, pool.submit(fn, n, view)))
            if len(pending) >= 2 * workers:
                m, f = pending.popleft()
                yield m, f.result()
        while pending:
            m, f = pending.popleft()
            yield m, f.result()


def srgb_to_linear(c):
    """Camera pixel values are gamma-encoded (sRGB); light adds up and scales only in linear units."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def luminance(c):
    return c @ np.array([0.2126, 0.7152, 0.0722])


def vignetting(r2, v):
    """Brightness falloff towards the image corners (1 in the centre), r2 = 1 at the corners."""
    return 1.0 + v[0] * r2 + v[1] * r2 ** 2


def colour_mesh(mesh, views, session, depth_info, rgb_info, max_depth, progress):
    """Colours the mesh directly from the keyframe images instead of the TSDF average. The camera
    changes exposure and white balance all the time and its image corners are darker, which made
    the colours patchy. Model, in linear light (where brightness really is multiplicative):
        observed colour = gain of the keyframe (r, g, b) x vignetting(position in the image)
                          x true colour of the surface
    1) gains and vignetting from surfaces seen by several keyframes (alternating updates)
    2) colour: weighted average of the corrected observations (frontal, close views count most)
    3) again, without observations far from that average: people walking by, reflections,
       highlights, badly aligned views
    Returns the gains (keyframe index -> [r, g, b]; None if there was too little overlap), the
    vignetting coefficients and the fraction of vertices coloured from the images."""
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(V), min(len(V), 150_000), replace=False))
    obs = []  # per keyframe: (sample positions, colours, weights, r2)
    Vs, Ns = V[sample], N[sample]
    for n, o in observe_views(views, lambda n, view: observe(Vs, Ns, view, session, depth_info, rgb_info, max_depth)):
        progress("Colours: exposure", n + 1, len(views))
        obs.append(o)
    gains = np.ones((len(views), 3))
    vig = np.zeros(2)
    seen = [i for i, o in enumerate(obs) if o is not None and len(o[0]) > 200]
    if len(seen) < 3:
        return None, None, 0.0
    for _ in range(8):
        num = np.zeros((len(sample), 3))
        den = np.zeros(len(sample))
        for i in seen:
            idx, col, wgt, r2 = obs[i]
            num[idx] += wgt[:, None] * col / (gains[i] * vignetting(r2, vig)[:, None])
            den[idx] += wgt
        ratios, radii = [], []
        for i in seen:
            idx, col, wgt, r2 = obs[i]
            # compare with what the OTHER keyframes saw there (leave-one-out): most surfaces are
            # seen by only a few keyframes, so the plain average is mostly the keyframe itself
            own = wgt[:, None] * col / (gains[i] * vignetting(r2, vig)[:, None])
            other_den = den[idx] - wgt
            other = (num[idx] - own) / np.maximum(other_den, 1e-9)[:, None]
            ok = (other_den > 0.2 * wgt) & (other.min(axis=1) > 0.005)  # dark surfaces: noisy ratios
            if ok.sum() > 100:
                gains[i] = np.clip(np.median(col[ok] / (other[ok] * vignetting(r2[ok], vig)[:, None]), axis=0),
                                   0.2, 5.0)
                ratios.append(luminance(col[ok]) / luminance(other[ok] * gains[i]))
                radii.append(r2[ok])
        gains /= np.median(gains[seen], axis=0)  # overall brightness: like the typical keyframe
        # vignetting: observed / (gain x surface colour) is what is left, as a function of the
        # position in the image; least squares fit of 1 + a r2 + b r2^2
        # on the medians of 10 radius bins (robust against people, reflections, bad matches)
        if not ratios:
            break
        ratio, r2 = np.concatenate(ratios), np.concatenate(radii)
        edges = np.linspace(0, 1, 11)
        centres, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (r2 >= lo) & (r2 < hi) & (ratio > 0.2) & (ratio < 5)
            if sel.sum() > 200:
                centres.append((lo + hi) / 2)
                medians.append(np.median(ratio[sel]))
        if len(centres) < 3:
            break
        c = np.array(centres)
        # (relative to the centre bin: the overall level belongs to the gains)
        fit = np.linalg.lstsq(np.stack([c, c ** 2], 1), np.array(medians) / medians[0] - 1.0, rcond=None)[0]
        vig = fit if 0.4 <= vignetting(1.0, fit) <= 1.2 else vig  # physically plausible falloff only

    grid = VertexGrid(V)

    def accumulate(stage, reference=None):
        num = np.zeros((len(V), 3))
        den = np.zeros(len(V))
        for n, o in observe_views(views, lambda n, view: observe(V, N, view, session, depth_info, rgb_info,
                                                                  max_depth, grid)):
            progress(stage, n + 1, len(views))
            if o is None:
                continue
            idx, col, wgt, r2 = o
            col = col / (gains[n] * vignetting(r2, vig)[:, None])
            wgt = wgt ** 2  # strongly prefer the best views: sharper than a broad average
            if reference is not None:  # drop observations far from the first average
                ref = reference[idx]
                wgt = wgt * (np.abs(luminance(col) - luminance(ref)) < 0.35 * luminance(ref) + 0.02)
            num[idx] += wgt[:, None] * col
            den[idx] += wgt
        return num, den

    num, den = accumulate("Colours: best views")
    first = num / np.maximum(den, 1e-12)[:, None]
    num2, den2 = accumulate("Colours: without outliers", first)
    have = den > 0
    robust = den2 > 0
    linear = np.where(robust[:, None], num2 / np.maximum(den2, 1e-12)[:, None], first)
    colours = np.asarray(mesh.vertex_colors).copy()
    colours[have] = linear_to_srgb(linear[have])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colours, 0, 1))
    return ({views[i][0]["index"]: gains[i].round(4).tolist() for i in range(len(views))}, vig, have.mean())


def colour_points_from_mesh(cloud, mesh, max_dist):
    """Gives each point the (corrected) colour of the nearest mesh vertex, so Points and Mesh show
    the same unified colours. Points further than max_dist from the mesh keep their own colour."""
    nns = o3c.nns.NearestNeighborSearch(o3c.Tensor(np.asarray(mesh.vertices), dtype=o3c.float32))
    nns.knn_index()
    idx, d2 = nns.knn_search(o3c.Tensor(np.asarray(cloud.points), dtype=o3c.float32), 1)
    idx, d2 = idx.numpy()[:, 0], d2.numpy()[:, 0]
    near = d2 < max_dist ** 2
    colours = np.asarray(cloud.colors).copy()
    colours[near] = np.asarray(mesh.vertex_colors)[idx[near]]
    cloud.colors = o3d.utility.Vector3dVector(colours)
    return near.mean()


def export_cameras(path, views, gains, rgb_info, depth_info, rgb_time_offset=0.0):
    """Final camera poses and intrinsics per keyframe: the input for the scene labelling, the presentation
    video and other reconstruction methods. Poses are camera -> map, optical frames.
    rgb_time_offset_s: the colour-time correction already in T_map_rgb (absent in files written
    before 2026-10-06, which had none): readers must not correct again."""
    data = {"rgb_intrinsics": intrinsics(rgb_info), "depth_intrinsics": intrinsics(depth_info),
            "rgb_distortion": rgb_info.get("d"), "depth_distortion": depth_info.get("d"),
            "intrinsics_order": "width, height, fx, fy, cx, cy", "depth_scale": 1000.0,
            "rgb_time_offset_s": rgb_time_offset, "frames": []}
    for k, T_d, T_r in views:
        data["frames"].append({"index": int(k["index"]), "rgb": k["rgb"], "depth": k["depth"],
                               "T_map_rgb": T_r.round(6).tolist(), "T_map_depth": T_d.round(6).tolist(),
                               "gain": (gains or {}).get(k["index"])})
    path.write_text(json.dumps(data, indent=1))


def viewer_mesh(mesh, max_triangles):
    """A lighter copy for the browser (the full mesh is several 100 MB for a large room)."""
    if len(mesh.triangles) <= max_triangles:
        return mesh
    # vertex clustering (merge vertices per voxel, colours averaged): ~4 s instead of ~80 s for
    # quadric decimation of a 5 M triangle room, and plenty for the browser view
    voxel = 0.015
    while True:
        light = mesh.simplify_vertex_clustering(voxel_size=voxel,
                                                contraction=o3d.geometry.SimplificationContraction.Average)
        if len(light.triangles) <= max_triangles:
            return light
        voxel *= 1.25


# ---------- progress for the console (reconstruction/progress.json, read by the helper) ----------

class Progress:
    STAGES = ["Reading keyframes", "Aligning (ICP)", "Levelling the floor", "Fusing (TSDF)",
              "Building the mesh", "Colours: exposure", "Colours: best views", "Colours: without outliers",
              "Writing files",
              "Preview image"]

    def __init__(self, path):
        self.path, self.last = path, 0.0
        self.stage, self.since, self.t0 = None, time.monotonic(), time.monotonic()

    def finish(self):
        """Prints how long the last stage and the whole reconstruction took."""
        self.__call__("done")

    def __call__(self, stage, done=0, total=0, force=False):
        now = time.monotonic()
        if stage != self.stage:  # log how long each stage took (to see where the time goes)
            if self.stage is not None:
                print(f"  [time] {self.stage}: {now - self.since:.0f} s", flush=True)
            if stage == "done":
                print(f"  [time] total: {now - self.t0:.0f} s", flush=True)
                return
            self.stage, self.since, force = stage, now, True
        if not force and done != total and now - self.last < 0.5:
            return
        self.last = now
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"stage": stage, "step": self.STAGES.index(stage) + 1,
                                       "steps": len(self.STAGES), "done": done, "total": total}))
            os.replace(tmp, self.path)
        except OSError:
            # Windows refuses to replace a file while the helper (or a virus scanner) reads it;
            # progress is only informational, the next update tries again
            self.last = 0.0


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, nargs="?", help="session folder (default: newest)")
    p.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size in metres")
    p.add_argument("--min-depth", type=float, default=0.25, help="ignore depth closer than this (gripper)")
    p.add_argument("--max-depth", type=float, default=3.0, help="ignore depth further than this (noisy)")
    p.add_argument("--no-icp", action="store_true", help="use the recorded poses as they are")
    p.add_argument("--no-level", action="store_true", help="don't move the floor to z = 0")
    p.add_argument("--no-camera-correction", action="store_true",
                   help="ignore config/camera_correction.json (calibrated camera tilt/roll)")
    p.add_argument("--smooth", type=int, default=5, help="Taubin smoothing iterations (0 = off)")
    p.add_argument("--keyframe-step", type=float, default=0.10,
                   help="use a keyframe every this many metres of travel ... (0 = all keyframes)")
    p.add_argument("--keyframe-turn", type=float, default=10.0, help="... or degrees of turning")
    p.add_argument("--tsdf-colours", action="store_true",
                   help="keep the TSDF's averaged colours (no exposure correction / best-view colouring)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default <session>/reconstruction)")
    # tested on 20261005_142856 (2026-10-05): smoothing -2 % noise (invisible), near weighting +6 % floor
    # area but +5 % noise; the depth noise (4-8 mm) is below the 1 cm voxels, the poses limit sharpness
    p.add_argument("--rgb-time-offset", type=float, default=RGB_TIME_OFFSET,
                   help="seconds the colour images are shifted from their stamps (see RGB_TIME_OFFSET)")
    p.add_argument("--trunc", type=float, default=8.0,
                   help="TSDF band around each measured surface, in voxels (Open3D's default 8 = 8 cm at 1 cm)")
    p.add_argument("--strict-edges", action="store_true",
                   help="drop 2 pixels (not 1) at depth edges: fewer mixed pixels at thin parts (chair legs)")
    p.add_argument("--depth-smooth", action="store_true", help="edge-preserving smoothing of the depth images")
    p.add_argument("--near-weight", type=float, default=0.0,
                   help="depth closer than this (m) counts twice: more complete, slightly noisier (0 = off)")
    p.add_argument("--viewer-triangles", type=int, default=1_000_000, help="size of mesh_view.ply for the browser")
    args = p.parse_args()
    session = args.session or max((REPO / "data" / "sessions").iterdir(), key=lambda d: d.name)
    out = args.out or session / "reconstruction"
    out.mkdir(exist_ok=True)
    progress = Progress(out / "progress.json")
    progress("Reading keyframes", force=True)

    rgb_info = json.loads((session / "calib" / "rgb_camera_info.json").read_text())
    depth_info = json.loads((session / "calib" / "depth_camera_info.json").read_text())
    tree = TransformTree(session)
    with open(session / "keyframes.csv", encoding="utf-8") as f:
        keyframes = list(csv.DictReader(f))
    print(f"Session {session.name}: {len(keyframes)} keyframes, "
          f"poses {'SLAM-corrected (map)' if tree.has_slam else 'odometry only (no SLAM stream)'}")
    if args.keyframe_step > 0 and keyframes:
        # fast runs record ~3.5 keyframes per second, mostly near-duplicates: keep one per
        # keyframe_step metres or keyframe_turn degrees (by the recorded odometry)
        kept, last = [], None
        for k in keyframes:
            x, y, yaw = float(k["odom_x"]), float(k["odom_y"]), float(k["odom_yaw"])
            if last is None or math.hypot(x - last[0], y - last[1]) >= args.keyframe_step or \
                    abs(math.degrees(math.atan2(math.sin(yaw - last[2]), math.cos(yaw - last[2])))) >= args.keyframe_turn:
                kept.append(k)
                last = (x, y, yaw)
        print(f"  keyframes used: {len(kept)} of {len(keyframes)} (one per {100 * args.keyframe_step:.0f} cm or "
              f"{args.keyframe_turn:.0f} deg; --keyframe-step 0 uses all)")
        keyframes = kept
    if tree.borrowed:
        print(f"  fixed transforms missing in this session, taken from others: {', '.join(tree.borrowed)}")

    w, h, fx, fy, cx, cy = intrinsics(depth_info)
    K = o3c.Tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))

    # ---- pass 1: poses. Recorded pose chain, refined by ICP, then the floor levelled ----
    correction = CameraCorrection(session)
    frames, gaps = [], []
    for n, k in enumerate(keyframes):
        progress("Reading keyframes", n + 1, len(keyframes))
        # the colour image shows the scene RGB_TIME_OFFSET before its stamp: its pose is looked up there
        t_rgb, t_depth = float(k["rgb_stamp"]) + args.rgb_time_offset, float(k["depth_stamp"])
        depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        pts, _, _ = depth_points(depth, depth_info, args.min_depth, args.max_depth)
        small = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(ICP_VOXEL)
        T0, T0_rgb = tree.map_to(DEPTH_FRAME, t_depth)[0], tree.map_to(RGB_FRAME, t_rgb)[0]
        if correction.ok and not args.no_camera_correction:
            T0, T0_rgb = correction.apply(tree, T0, T0_rgb, t_depth, t_rgb)
        frames.append({"k": k, "t_rgb": t_rgb, "t_depth": t_depth, "pts": np.asarray(small.points),
                       "T0": T0, "T0_rgb": T0_rgb, "C": np.eye(4)})
        # time gap to the robot/arm pose samples (SLAM's map->odom changes slowly; not counted)
        gaps.append(max(tree.gap(DEPTH_FRAME, t_depth), tree.gap(RGB_FRAME, t_rgb)))
    if not frames:
        raise SystemExit("no usable keyframes")
    if correction.ok and not args.no_camera_correction:
        print(f"  camera correction ({correction.text}): {correction.applied} keyframes in the survey pose"
              + (f", {correction.skipped} in other arm poses left as they are" if correction.skipped else ""))
    if not args.no_icp:
        accepted = refine_poses(frames, progress)
        shifts = [np.linalg.norm(f["C"][:2, 3]) for f in frames]
        print(f"  ICP: {accepted}/{len(frames)} keyframes aligned to the model, "
              f"corrections median {100 * np.median(shifts):.1f} cm, max {100 * max(shifts):.1f} cm")
    L = np.eye(4)
    if not args.no_level:
        progress("Levelling the floor", force=True)
        L, floor = level_floor(frames)
        if floor:
            print(f"  floor before levelling: tilt {floor[0]:.1f} deg, height {floor[1]:+.3f} m "
                  f"({floor[2]:,} points) -> now level at z = 0")

    # ---- pass 2: fuse all keyframes with their final poses ----
    # sparse TSDF volume (Open3D's tensor API; the legacy ScalableTSDFVolume returns empty meshes in
    # Open3D 0.20). Blocks of 8^3 voxels are allocated only near observed surfaces.
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=args.voxel, block_resolution=8, block_count=60000,
        device=o3c.Device("CPU:0"))
    cloud_pts, cloud_col, poses, views = [], [], [], []
    empty_frames = 0
    for n, f in enumerate(frames):
        progress("Fusing (TSDF)", n + 1, len(frames))
        k = f["k"]
        depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(session / k["rgb"]))
        if rgb is None:
            continue
        T_map_depth = L @ f["C"] @ f["T0"]
        T_map_rgb = L @ f["C"] @ f["T0_rgb"]
        poses.append((k["index"], f["t_depth"], T_map_depth))
        views.append((k, T_map_depth, T_map_rgb))

        pts, v, u = depth_points(depth, depth_info, args.min_depth, args.max_depth, 2 if args.strict_edges else 1)
        pts_map = transform(T_map_depth, pts)
        colours = colour_for(transform(np.linalg.inv(T_map_rgb), pts_map), rgb, rgb_info)
        seen = ~np.isnan(colours[:, 0])
        cloud_pts.append(pts_map[seen])
        cloud_col.append(colours[seen])

        # TSDF needs colour on the depth pixel grid: draw the projected colours into a depth-sized image
        colour_img = np.zeros((h, w, 3), dtype=np.uint8)
        colour_img[v[seen], u[seen]] = (colours[seen] * 255).astype(np.uint8)
        depth_mm = np.zeros((h, w), dtype=np.uint16)  # TSDF input: uint16 millimetres
        depth_mm[v[seen], u[seen]] = depth[v[seen], u[seen]]
        if args.depth_smooth:
            # edge-preserving smoothing (depth edges are already removed): less noise on flat surfaces
            zf = depth_mm.astype(np.float32)
            sm = cv2.bilateralFilter(zf, 5, 30.0, 3.0)  # range sigma 30 mm, space sigma 3 px
            depth_mm = np.where(depth_mm > 0, np.round(sm), 0).astype(np.uint16)
        if not depth_mm.any():
            # nothing usable in range (e.g. the camera right in front of a wall or furniture): Open3D
            # aborts the whole fusion on an empty frame (run 20261006_142432)
            empty_frames += 1
            continue
        colour_t = o3d.t.geometry.Image(o3c.Tensor(colour_img))
        extrinsic = o3c.Tensor(np.linalg.inv(T_map_depth))  # map -> camera
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_mm))
        blocks = volume.compute_unique_block_coordinates(depth_t, K, extrinsic, 1000.0, args.max_depth)
        volume.integrate(blocks, depth_t, colour_t, K, extrinsic, 1000.0, args.max_depth, args.trunc)
        if args.near_weight:
            # depth noise grows with the square of the distance: close views count twice
            near = np.where(depth_mm < args.near_weight * 1000, depth_mm, 0).astype(np.uint16)
            if not near.any():
                continue
            depth_t = o3d.t.geometry.Image(o3c.Tensor(near))
            volume.integrate(blocks, depth_t, colour_t, K, extrinsic, 1000.0, args.max_depth, args.trunc)

    if empty_frames:
        print(f"  {empty_frames} keyframes without usable depth skipped")
    progress("Building the mesh", force=True)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(cloud_pts)))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(cloud_col).astype(np.float64))
    cloud = cloud.voxel_down_sample(args.voxel)
    # only surfaces seen in at least two keyframes: suppresses single-frame depth noise
    mesh = volume.extract_triangle_mesh(weight_threshold=1.5).to_legacy()
    mesh.compute_vertex_normals()
    # fused colours are 0..1, but averaging can overshoot slightly; clip (scale only if really 0..255)
    colours = np.asarray(mesh.vertex_colors)
    if len(colours):
        scale = 255.0 if colours.max() > 2.0 else 1.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colours / scale, 0, 1))
    mesh, fragments = clean_mesh(mesh, smooth_iterations=args.smooth)
    print(f"  mesh clean-up: {fragments:,} small fragments removed, {args.smooth} smoothing iterations")
    gains = None
    if not args.tsdf_colours:
        gains, vig, covered = colour_mesh(mesh, views, session, depth_info, rgb_info, args.max_depth, progress)
        if gains:
            g = np.array(list(gains.values())).mean(axis=1)
            print(f"  colours from the images: {100 * covered:.0f} % of the vertices; exposure gains per "
                  f"keyframe (linear) 5-95 %: {np.percentile(g, 5):.2f}..{np.percentile(g, 95):.2f}; "
                  f"image corners {100 * (vignetting(1.0, vig) - 1):+.0f} % brightness (vignetting)")
            near = colour_points_from_mesh(cloud, mesh, 0.10)  # raw depth points scatter (median 2 cm)
            print(f"  points: {100 * near:.0f} % coloured like the nearest mesh surface")
    progress("Writing files", force=True)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), cloud)
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    light = viewer_mesh(mesh, args.viewer_triangles)
    o3d.io.write_triangle_mesh(str(out / "mesh_view.ply"), light)
    export_cameras(out / "cameras.json", views, gains, rgb_info, depth_info, args.rgb_time_offset)

    with open(out / "poses.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "stamp"] + [f"T{r}{c}" for r in range(4) for c in range(4)])
        for idx, t, T in poses:
            wr.writerow([idx, f"{t:.6f}"] + [f"{x:.6f}" for x in T.ravel()])

    # sanity checks: the floor must be at z ~ 0 in the map, and the pose samples close in time
    xyz = np.asarray(cloud.points)
    hist, edges = np.histogram(xyz[:, 2], bins=np.arange(-0.5, 2.0, 0.02))
    floor_z = edges[np.argmax(hist)] + 0.01
    print(f"  points {len(xyz):,}, mesh {len(mesh.triangles):,} triangles, extent "
          f"{np.ptp(xyz[:, 0]):.1f} x {np.ptp(xyz[:, 1]):.1f} x {np.ptp(xyz[:, 2]):.1f} m")
    print(f"  most common height (should be the floor, ~0 m): {floor_z:+.2f} m")
    # floor plane: tilt shows a camera-angle error (arm servo calibration), offset a height error
    low = cloud.select_by_index(np.nonzero(np.abs(xyz[:, 2] - floor_z) < 0.15)[0])
    if len(low.points) > 500:
        (a, b, c, d), inliers = low.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=500)
        n = np.array([a, b, c]) * np.sign(c)
        tilt = np.degrees(np.arccos(min(1.0, n[2])))
        print(f"  floor plane: tilt {tilt:.1f} deg (0 = level), height at the origin {-d / c:+.3f} m, "
              f"{len(inliers):,} points")
    print(f"  median / largest time gap image <-> robot pose sample: "
          f"{1000 * np.median(gaps):.0f} / {1000 * max(gaps):.0f} ms")
    progress("Preview image", force=True)
    make_preview(out / "preview.png", xyz, np.asarray(cloud.colors), [T for _, _, T in poses], floor_z)
    progress.finish()
    for name in ("progress.json", "progress.tmp"):
        try:
            (out / name).unlink(missing_ok=True)
        except OSError:
            pass  # being read right now; the helper ignores it once the job has ended
    print(f"  written to {out}")


def make_preview(path, xyz, colours, poses, floor_z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = np.array([T[:3, 3] for T in poses])
    view = np.array([T[:3, 2] for T in poses])  # optical z = viewing direction
    if len(xyz) > 400_000:  # matplotlib scatter of millions of points takes many minutes
        keep = np.random.default_rng(0).choice(len(xyz), 400_000, replace=False)
        xyz, colours = xyz[keep], colours[keep]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    order = np.argsort(xyz[:, 2])  # draw low points first, so walls/furniture stay visible from above
    ax[0].scatter(xyz[order, 0], xyz[order, 1], c=colours[order], s=1, linewidths=0)
    ax[0].quiver(cams[:, 0], cams[:, 1], view[:, 0], view[:, 1], color="red", scale=25, width=0.003)
    ax[0].set_title("Top view (map frame), red = camera and viewing direction")
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[1].scatter(xyz[:, 0], xyz[:, 2], c=colours, s=1, linewidths=0)
    ax[1].axhline(floor_z, color="red", lw=0.8, ls="--", label=f"floor peak {floor_z:+.2f} m")
    ax[1].set_title("Side view (x vs height)")
    ax[1].set_aspect("equal")
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
