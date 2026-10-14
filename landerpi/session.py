"""A recorded session's keyframes with their final camera poses (reconstruction/cameras.json), for
projects that work on reconstructed sessions (scene understanding, later real2sim).

    from landerpi.session import Session
    s = Session("data/sessions/20261005_124116")
    f = s.frames[0]
    pts, v, u = s.depth_points(f)          # Nx3 in the map frame + the depth pixels they came from
    uv, ok = s.project(f, pts, "rgb")      # map points -> colour image pixels

cameras.json is written by projects/mapping/reconstruct.py: only the keyframes the reconstruction used
(one per ~10 cm / 10 deg), poses camera -> map in OpenCV optical axes (x right, y down, z forward),
floor at z = 0. Colour and depth are separate cameras with their own intrinsics and poses (captured
up to ~50 ms apart), so a pixel of one is found in the other by going through 3D.
The depth filtering is the same as reconstruct.py's depth_points (copied, that file stays untouched).
"""

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SESSIONS = REPO / "data" / "sessions"

# the gripper fills the bottom of every image at < ~25 cm (measured in reconstruct.py's runs)
MIN_DEPTH = 0.25
# beyond ~3 m the depth camera's noise is several cm (reconstruct.py's --max-depth default)
MAX_DEPTH = 3.0


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


class Frame:
    def __init__(self, d, session):
        self.index = int(d["index"])
        self.rgb_file = session.path / d["rgb"]
        self.depth_file = session.path / d["depth"]
        self.T_map_rgb = np.array(d["T_map_rgb"])
        self.T_map_depth = np.array(d["T_map_depth"])
        self.stamp = float(Path(d["rgb"]).stem)

    def rgb(self):
        return cv2.imread(str(self.rgb_file))

    def depth(self):
        """Depth in metres (0 = no measurement)."""
        return cv2.imread(str(self.depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def __repr__(self):
        return f"Frame({self.index}, {self.rgb_file.name})"


def correct_rgb_time(frames, offset):
    """Turns each colour camera pose to the moment the image really shows: offset seconds from its
    timestamp, at the robot's turn rate there (from the neighbouring keyframes). Yaw only; the robot
    moves < 2 cm in 0.1 s. The colour images are ~0.1 s late against depth and poses
    (measured with projects/scene_understanding/check_camera_timing.py, 2026-10-06)."""
    def yaw(T):
        return np.arctan2(T[1, 0], T[0, 0])
    rates = []
    for i, f in enumerate(frames):
        a, b = frames[max(i - 1, 0)], frames[min(i + 1, len(frames) - 1)]
        dt = b.stamp - a.stamp
        # only between close keyframes: across a gap the turn rate is unknown
        ok = 0 < dt < 1.5
        rates.append(((yaw(b.T_map_rgb) - yaw(a.T_map_rgb) + np.pi) % (2 * np.pi) - np.pi) / dt if ok else 0.0)
    for f, rate in zip(frames, rates):
        a = rate * offset
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        f.T_map_rgb = f.T_map_rgb.copy()
        f.T_map_rgb[:3, :3] = R @ f.T_map_rgb[:3, :3]


class Session:
    def __init__(self, path, rgb_time_offset=None):
        """rgb_time_offset: the total correction of the colour images against their stamps, in seconds
        (see correct_rgb_time); None = poses as in cameras.json. Reconstructions since 2026-10-06 already
        contain -0.1 s (cameras.json "rgb_time_offset_s", projects/mapping/reconstruct.py); only the
        rest is applied here, so old and new files end up corrected once."""
        p = Path(path)
        self.path = p if p.is_absolute() or p.exists() else SESSIONS / p
        self.name = self.path.name
        cams = json.loads((self.path / "reconstruction" / "cameras.json").read_text())
        # intrinsics as (width, height, fx, fy, cx, cy)
        self.K = {"rgb": tuple(cams["rgb_intrinsics"]), "depth": tuple(cams["depth_intrinsics"])}
        self.frames = [Frame(d, self) for d in cams["frames"]]
        # the correction already in T_map_rgb (absent in files written before 2026-10-06: none)
        self.rgb_time_offset = cams.get("rgb_time_offset_s") or 0.0
        if rgb_time_offset is not None and abs(rgb_time_offset - self.rgb_time_offset) > 1e-6:
            correct_rgb_time(self.frames, rgb_time_offset - self.rgb_time_offset)
            self.rgb_time_offset = rgb_time_offset

    def nogo_lines(self):
        """The no-go lines at the end of the run, [[x1, y1, x2, y2], ...] in the map frame, or None for
        runs recorded before 2026-10-05 (streams/nogo_lines.jsonl is written on every change)."""
        f = self.path / "streams" / "nogo_lines.jsonl"
        if not f.exists():
            return None
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1])["msg"]["lines"] if lines else []

    def mesh_file(self, view=False):
        return self.path / "reconstruction" / ("mesh_view.ply" if view else "mesh.ply")

    def depth_points(self, frame, min_d=MIN_DEPTH, max_d=MAX_DEPTH, frame_out="map"):
        """Depth image -> Nx3 points (map frame, or "camera" = depth optical frame) plus the pixel
        rows/columns used. Pixels at depth discontinuities ("flying pixels" between an object edge and
        the background) and next to holes are dropped."""
        z = frame.depth()
        kernel = np.ones((3, 3), np.uint8)
        has_depth = (z > 0).astype(np.uint8)
        local_max = cv2.dilate(z, kernel)
        local_min = cv2.erode(np.where(z > 0, z, 1e3).astype(np.float32), kernel)
        edge = (local_max - local_min) > (0.02 + 0.03 * z)  # jump > 2 cm + 3 % of the distance
        valid = (z > min_d) & (z < max_d) & ~edge & (cv2.erode(has_depth, kernel) > 0)
        v, u = np.nonzero(valid)
        w, h, fx, fy, cx, cy = self.K["depth"]
        zz = z[v, u]
        pts = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], axis=1)
        if frame_out == "map":
            pts = transform(frame.T_map_depth, pts)
        return pts, v, u

    def project(self, frame, pts_map, camera="rgb"):
        """Map points -> pixel coordinates (Nx2 float, u = column, v = row) in this frame's colour or
        depth image, and a mask of the points in front of the camera and inside the image."""
        T = frame.T_map_rgb if camera == "rgb" else frame.T_map_depth
        P = transform(np.linalg.inv(T), pts_map)
        w, h, fx, fy, cx, cy = self.K[camera]
        z = P[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([P[:, 0] * fx / z + cx, P[:, 1] * fy / z + cy], axis=1)
        ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        return uv, ok

    def points_in_mask(self, frame, mask, pts_map=None):
        """The depth points of this frame that fall inside a colour-image mask (HxW bool): lifts a 2D
        segmentation to 3D. Returns the points (map frame) and their indices into pts_map."""
        if pts_map is None:
            pts_map = self.depth_points(frame)[0]
        uv, ok = self.project(frame, pts_map, "rgb")
        idx = np.nonzero(ok)[0]
        uu = np.round(uv[idx, 0]).astype(np.int64)
        vv = np.round(uv[idx, 1]).astype(np.int64)
        idx = idx[mask[vv, uu]]
        return pts_map[idx], idx
