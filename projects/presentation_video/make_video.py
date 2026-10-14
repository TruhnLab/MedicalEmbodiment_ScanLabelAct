"""Presentation video of a mapping run: the 2D lidar map (left) and the 3D reconstruction building
up (right), in sync and sped up.

Made afterwards, on demand, from a finished recording and its reconstruction; nothing here runs
during a run and nothing extra is recorded for it.

    python projects/presentation_video/make_video.py data/sessions/<name>
    python projects/presentation_video/make_video.py data/sessions/<name> --preview 10    # 10 s test clip
    python projects/presentation_video/make_video.py data/sessions/<name> --still 0.5     # one frame as PNG

Inputs (session folder): reconstruction/mesh_view.ply and cameras.json (run reconstruct.py first),
streams/slam_robot_pose and scan.
Output: <session>/video/<session>_x<speed>.mp4 (H.264, 1920 x 1080, 30 fps)

Left: the occupancy map, built from the recorded scans along the final SLAM path (a new map every
2 s, as SLAM publishes it), with the path driven and the robot, in the console's colours.
Right: the final mesh, revealed at the moment the robot's camera first saw each part (newly seen
surfaces glow blue for a moment): the end result, built up in the order it was scanned.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects" / "mapping"))
import reconstruct  # noqa: E402  (the reconstruction's visibility test and transforms)

# colours (RGB), after the console's themes (console/style.css). Light: white background; free
# space is a light blue-grey there (the console's white would vanish into the background).
THEMES = {
    "light": dict(SURFACE=(255, 255, 255), BORDER=(225, 228, 234), GRID=(208, 213, 222), TEXT=(21, 24, 33),
                  ACCENT=(47, 107, 255), DANGER=(239, 68, 68), MAP_FREE=(232, 236, 243), MAP_OCC=(21, 24, 33),
                  CONE=(47, 107, 255), CONE_ALPHA=0.16,
                  BADGE=(241, 243, 246)),
    "dark": dict(SURFACE=(29, 33, 42), BORDER=(38, 43, 53), GRID=(38, 43, 53), TEXT=(231, 233, 238),
                 ACCENT=(91, 140, 255), DANGER=(239, 68, 68), MAP_FREE=(61, 70, 87), MAP_OCC=(231, 233, 238),
                 CONE=(80, 160, 255), CONE_ALPHA=0.22,
                 BADGE=(14, 16, 20)),
}


def set_theme(name):
    """Sets the colour constants below (module-wide: one theme per video)."""
    globals().update(THEMES[name])


set_theme("light")
GLOW = np.array([0.45, 0.72, 1.0])  # newly seen surfaces in 3D (sRGB)


# ---------- reading the recording ----------

def stamp(h):
    return h["stamp"]["sec"] + h["stamp"]["nanosec"] * 1e-9


def read_stream(session, name):
    out = []
    with open(session / "streams" / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line)["msg"])
            except json.JSONDecodeError:
                pass  # the last line of an interrupted recording
    return out


def static_transform(session, parent, child):
    """parent -> child from the recorded static transforms (chain of parents)."""
    tfs = {t["child_frame_id"]: t for t in json.loads((session / "calib" / "tf_static.json").read_text())["transforms"]}
    T, frame = np.eye(4), child
    while frame != parent:
        t = tfs[frame]
        T = reconstruct.matrix(t["transform"]["translation"], t["transform"]["rotation"]) @ T
        frame = t["header"]["frame_id"]
    return T


class Recording:
    def __init__(self, session):
        self.session = session
        poses = sorted(read_stream(session, "slam_robot_pose"), key=lambda m: stamp(m["header"]))
        if not poses:
            sys.exit("no SLAM poses (streams/slam_robot_pose.jsonl): record with SLAM running")
        self.pose_t = np.array([stamp(m["header"]) for m in poses])
        self.pose_xy = np.array([[m["pose"]["position"]["x"], m["pose"]["position"]["y"]] for m in poses])
        q = [m["pose"]["orientation"] for m in poses]
        self.pose_yaw = np.unwrap([math.atan2(2 * (o["w"] * o["z"] + o["x"] * o["y"]), 1 - 2 * (o["y"] ** 2 + o["z"] ** 2))
                                   for o in q])

        # lidar endpoints in the robot frame (the lidar sits ~7 cm in front of the robot's centre)
        T_lidar = static_transform(session, "base_footprint", "lidar_frame")
        self.lidar_xy = T_lidar[:2, 3]
        lidar_yaw = math.atan2(T_lidar[1, 0], T_lidar[0, 0])
        scans = sorted(read_stream(session, "scan"), key=lambda m: stamp(m["header"]))
        self.scan_t = np.array([stamp(m["header"]) for m in scans])
        self.scans = []
        for s in scans:
            r = np.array([np.nan if v is None else v for v in s["ranges"]], np.float32)
            a = s["angle_min"] + lidar_yaw + np.arange(len(r)) * s["angle_increment"]
            ok = np.isfinite(r) & (r > 0.05) & (r < 8.0)
            self.scans.append((np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], 1) + self.lidar_xy).astype(np.float32))

        # the reconstruction's camera poses (map frame, after ICP and floor levelling)
        cams = json.loads((session / "reconstruction" / "cameras.json").read_text())
        self.cam_frames = cams["frames"]
        self.cam_t = np.array([float(Path(f["rgb"]).stem) for f in self.cam_frames])
        # where the colour camera looks (its z axis in the map), for the field-of-view cone
        fwd = np.array([np.array(f["T_map_rgb"])[:2, 2] for f in self.cam_frames])
        self.cam_yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
        self.rgb_info = json.loads((session / "calib" / "rgb_camera_info.json").read_text())
        self.depth_info = json.loads((session / "calib" / "depth_camera_info.json").read_text())

        # no-go lines drawn in the console (recorded since 2026-10-05, only when they change; laptop
        # receive time -> robot time with the recording's clock offset)
        self.nogo_t, self.nogo = [], []
        nogo_file = session / "streams" / "nogo_lines.jsonl"
        if nogo_file.exists():
            offset = json.loads((session / "meta.json").read_text()).get("clock_offset_s", 0.0)
            for line in nogo_file.read_text(encoding="utf-8").splitlines():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.nogo_t.append(m["recv"] - offset)
                self.nogo.append(m["msg"]["lines"])

        self.t_start = max(self.cam_t[0], self.pose_t[0])
        self.t_end = min(self.cam_t[-1], self.pose_t[-1])

    def pose(self, t):
        t = np.clip(t, self.pose_t[0], self.pose_t[-1])
        return (np.interp(t, self.pose_t, self.pose_xy[:, 0]), np.interp(t, self.pose_t, self.pose_xy[:, 1]),
                np.interp(t, self.pose_t, self.pose_yaw))

    def nogo_lines(self, t):
        i = int(np.searchsorted(self.nogo_t, t, side="right")) - 1
        return self.nogo[i] if i >= 0 else []

    def camera_yaw(self, t):
        return float(np.interp(t, self.cam_t, self.cam_yaw))

    def trail(self, t):
        return self.pose_xy[:max(1, int(np.searchsorted(self.pose_t, t, side="right")))]


# ---------- 2D: the lidar map ----------

class ScanMap:
    """Occupancy map from the scans along the final SLAM path: free where the lidar saw through,
    occupied where it hit something. Rebuilt every 2 s of the run, as SLAM publishes its map."""

    RES = 0.05

    def __init__(self, rec):
        self.rec = rec
        self.ends, self.origins = [], []
        for t, p in zip(rec.scan_t, rec.scans):
            x, y, yaw = rec.pose(t)
            c, s = math.cos(yaw), math.sin(yaw)
            self.ends.append(np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], 1))
            lx, ly = rec.lidar_xy
            self.origins.append((x + c * lx - s * ly, y + s * lx + c * ly))
        allp = np.concatenate([e for e in self.ends if len(e)])
        lo, hi = np.percentile(allp, 0.2, axis=0) - 0.5, np.percentile(allp, 99.8, axis=0) + 0.5
        self.ox, self.oy = lo
        self.w, self.h = np.ceil((hi - lo) / self.RES).astype(int)
        self.bounds = (lo, hi)
        self.free = np.zeros((self.h, self.w), np.int32)
        self.hits = np.zeros((self.h, self.w), np.int32)
        self.done = 0
        self.image = None  # RGBA, row 0 = the map's bottom

    def cells(self, p):
        return ((np.asarray(p) - (self.ox, self.oy)) / self.RES).astype(np.int32)

    def at(self, t):
        """(RGBA image, changed?) of the map at time t."""
        t0 = self.rec.t_start
        t = t0 + math.floor((t - t0) / 2.0) * 2.0
        n = int(np.searchsorted(self.rec.scan_t, t, side="right"))
        if n < self.done:  # back in time: start over
            self.free[:] = 0
            self.hits[:] = 0
            self.done = 0
        if n == self.done and self.image is not None:
            return self.image, False
        mask = np.zeros((self.h, self.w), np.uint8)
        for k in range(self.done, n):
            e = self.ends[k]
            if len(e) < 3:
                continue
            c = self.cells(e)
            mask[:] = 0
            cv2.fillPoly(mask, [np.vstack([self.cells([self.origins[k]]), c])], 1)
            self.free += mask
            ok = (c[:, 0] >= 0) & (c[:, 0] < self.w) & (c[:, 1] >= 0) & (c[:, 1] < self.h)
            np.add.at(self.hits, (c[ok, 1], c[ok, 0]), 1)
        self.done = n
        occ = (self.hits >= 2) & (self.hits >= 0.3 * self.free)
        free = (self.free > 0) & ~occ
        self.image = np.zeros((self.h, self.w, 4), np.uint8)
        self.image[free] = (*MAP_FREE, 255)
        self.image[occ] = (*MAP_OCC, 255)
        return self.image, True


class MapPanel:
    """The console's map view: map, 1 m grid, path driven and the robot. Drawn at 2x and scaled
    down (smooth lines)."""

    SS = 2

    def __init__(self, rec, size):
        self.rec = rec
        self.W, self.H = size
        self.map = ScanMap(rec)
        lo, hi = self.map.bounds
        lo, hi = np.minimum(lo, rec.pose_xy.min(0)) - 0.3, np.maximum(hi, rec.pose_xy.max(0)) + 0.3
        self.cx, self.cy = (lo + hi) / 2
        self.scale = min(self.W / (hi[0] - lo[0]), self.H / (hi[1] - lo[1]))  # px per m (1x)
        self.base = None

    def px(self, xy):
        """Map -> canvas pixels (2x), int32 with 2 bits of sub-pixel precision for cv2."""
        s = self.scale * self.SS
        xy = np.asarray(xy, float).reshape(-1, 2)
        p = np.stack([self.W * self.SS / 2 + (xy[:, 0] - self.cx) * s, self.H * self.SS / 2 - (xy[:, 1] - self.cy) * s], 1)
        return np.round(p * 4).astype(np.int32)

    def base_layer(self, image):
        """Background, map and 1 m grid (changes only with a new map)."""
        SS = self.SS
        canvas = np.zeros((self.H * SS, self.W * SS, 3), np.uint8)
        canvas[:] = SURFACE
        s = self.scale * SS
        k = self.map.RES * s
        x0 = self.W * SS / 2 + (self.map.ox - self.cx) * s
        y0 = self.H * SS / 2 - (self.map.oy + self.map.h * self.map.RES - self.cy) * s
        M = np.array([[k, 0, x0 + 0.5 * k - 0.5], [0, k, y0 + 0.5 * k - 0.5]])
        warped = cv2.warpAffine(np.ascontiguousarray(image[::-1]), M, (canvas.shape[1], canvas.shape[0]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        m = warped[..., 3] > 0
        canvas[m] = warped[..., :3][m]
        lines = np.zeros(canvas.shape[:2], np.uint8)
        gx0, gx1 = self.cx - self.W / 2 / self.scale, self.cx + self.W / 2 / self.scale
        gy0, gy1 = self.cy - self.H / 2 / self.scale, self.cy + self.H / 2 / self.scale
        for x in range(math.ceil(gx0), math.floor(gx1) + 1):
            cv2.polylines(lines, [self.px([[x, gy0], [x, gy1]])], False, 255, SS, cv2.LINE_AA, shift=2)
        for y in range(math.ceil(gy0), math.floor(gy1) + 1):
            cv2.polylines(lines, [self.px([[gx0, y], [gx1, y]])], False, 255, SS, cv2.LINE_AA, shift=2)
        a = lines[..., None].astype(np.float32) / 255 * 0.5
        return (canvas * (1 - a) + np.array(GRID) * a).astype(np.uint8)

    def blend_poly(self, canvas, pts, colour, alpha):
        """Semi-transparent polygon, blended only within its bounding box."""
        x0, y0 = np.maximum(pts.min(0) // 4 - 2, 0)
        x1, y1 = np.minimum(pts.max(0) // 4 + 3, (canvas.shape[1], canvas.shape[0]))
        if x1 <= x0 or y1 <= y0:
            return
        roi = canvas[y0:y1, x0:x1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts - np.array([x0, y0]) * 4], 255, cv2.LINE_AA, shift=2)
        a = mask[..., None].astype(np.float32) / 255 * alpha
        roi[:] = (roi * (1 - a) + np.array(colour) * a).astype(np.uint8)

    def draw(self, t):
        image, changed = self.map.at(t)
        if changed or self.base is None:
            self.base = self.base_layer(image)
        canvas = self.base.copy()
        for x1, y1, x2, y2 in self.rec.nogo_lines(t):  # light: they explain gaps, not the main content
            line = np.zeros(canvas.shape[:2], np.uint8)
            cv2.polylines(line, [self.px([[x1, y1], [x2, y2]])], False, 255, 4 * self.SS, cv2.LINE_AA, shift=2)
            a = line[..., None].astype(np.float32) / 255 * 0.55
            canvas[:] = (canvas * (1 - a) + np.array(DANGER) * a).astype(np.uint8)
        trail = self.rec.trail(t)
        if len(trail) > 1:
            cv2.polylines(canvas, [self.px(trail)], False, ACCENT, 2 * self.SS, cv2.LINE_AA, shift=2)
        x, y, yaw = self.rec.pose(t)
        size = max(8, 0.18 * self.scale)  # the console's robot triangle
        c, s = math.cos(yaw), math.sin(yaw)
        tri = [[x + (c * u - s * v) / self.scale, y + (s * u + c * v) / self.scale]
               for u, v in ([size, 0], [-0.6 * size, 0.55 * size], [-0.6 * size, -0.55 * size])]
        # the camera's field of view (as in the console): 1.5 m long, where the camera really looked
        cam = self.rec.camera_yaw(t)
        half = math.atan(self.rec.rgb_info["width"] / 2 / self.rec.rgb_info["k"][0])
        fan = [[x, y]] + [[x + 1.5 * math.cos(cam + a), y + 1.5 * math.sin(cam + a)] for a in np.linspace(-half, half, 24)]
        self.blend_poly(canvas, self.px(fan), CONE, CONE_ALPHA)
        cv2.fillPoly(canvas, [self.px(tri)], DANGER, cv2.LINE_AA, shift=2)
        return cv2.resize(canvas, (self.W, self.H), interpolation=cv2.INTER_AREA)


# ---------- 3D: the reconstruction building up ----------

def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


class Scene3D:
    """The finished mesh, revealed in the order the camera saw it; rendered off-screen (Open3D /
    Filament, runs on the laptop's integrated graphics) at 2x and scaled down."""

    SS = 2

    def __init__(self, rec, size, glow_run_s, cache_dir):
        self.rec, self.glow = rec, glow_run_s
        mesh = o3d.io.read_triangle_mesh(str(rec.session / "reconstruction" / "mesh_view.ply"))
        mesh.compute_vertex_normals()
        V = np.asarray(mesh.vertices)
        T = np.asarray(mesh.triangles)
        self.first = self.first_seen(mesh, cache_dir)
        # triangles appear when most of their corners have been seen; never seen: at the end
        reveal = np.median(self.first[T], axis=1)
        reveal[~np.isfinite(reveal)] = rec.t_end
        order = np.argsort(reveal, kind="stable")
        self.T_sorted = T[order].astype(np.int32)
        self.reveal_sorted = reveal[order]
        # light shading baked into the colours (gives walls and furniture depth). With post-processing
        # off, Filament writes vertex colours unchanged, so they stay sRGB like the photos.
        light = np.array([0.35, -0.45, 0.82])
        light /= np.linalg.norm(light)
        shade = 0.72 + 0.28 * np.abs(np.asarray(mesh.vertex_normals) @ light)
        self.base = np.clip(np.asarray(mesh.vertex_colors) * shade[:, None], 0, 1)
        self.verts = o3d.utility.Vector3dVector(V)

        lo, hi = np.percentile(V, 0.5, axis=0), np.percentile(V, 99.5, axis=0)
        self.centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.25])
        self.half = (hi[:2] - lo[:2]) / 2
        self.renderer = None
        self.resize(size)

    def first_seen(self, mesh, cache_dir):
        """When the camera first saw each vertex (robot time; inf = never): the reconstruction's own
        visibility test (on the depth image, not occluded) over its keyframes, cached."""
        src = self.rec.session / "reconstruction" / "mesh_view.ply"
        cache = cache_dir / "first_seen.npz"
        if cache.exists():
            c = np.load(cache)
            if float(c["mtime"]) == src.stat().st_mtime and len(c["first"]) == len(mesh.vertices):
                return c["first"]
        print("When was each surface first seen (once per reconstruction) ...", flush=True)
        V, N = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
        grid = reconstruct.VertexGrid(V)
        rec = self.rec
        views = [(f, np.array(f["T_map_depth"]), np.array(f["T_map_rgb"])) for f in rec.cam_frames]
        first = np.full(len(V), np.inf)
        for n, o in reconstruct.observe_views(views, lambda n, v: reconstruct.observe(
                V, N, v, rec.session, rec.depth_info, rec.rgb_info, 3.0, grid)):
            if o is not None:
                first[o[0]] = np.minimum(first[o[0]], rec.cam_t[n])
        np.savez(cache, first=first, mtime=src.stat().st_mtime)
        return first

    def resize(self, size):
        self.W, self.H = size
        self.renderer = None  # release the previous one first
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.W * self.SS, self.H * self.SS)
        sc = self.renderer.scene
        sc.view.set_post_processing(False)  # exact colours: no tone mapping
        # Filament linearises the background colour (unlike the vertex colours): pre-compensate
        sc.set_background([*linear_to_srgb(np.array(SURFACE) / 255), 1.0])
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.full_shown = False

    def mesh_at(self, t):
        n = int(np.searchsorted(self.reveal_sorted, t, side="right"))
        col = self.base.copy()
        if self.glow > 0:
            age = t - self.first
            g = (age >= 0) & (age < self.glow)
            f = (0.75 * (1 - age[g] / self.glow))[:, None]
            col[g] = col[g] * (1 - f) + GLOW * f
        m = o3d.geometry.TriangleMesh(self.verts, o3d.utility.Vector3iVector(self.T_sorted[:n]))
        m.vertex_colors = o3d.utility.Vector3dVector(col)
        return m

    def look(self, azimuth_deg, elevation_deg):
        """Camera around the room's centre, far enough to fit the whole room into the panel."""
        fov = 38.0
        a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
        across = abs(self.half[0] * math.sin(a)) + abs(self.half[1] * math.cos(a))
        along = abs(self.half[0] * math.cos(a)) + abs(self.half[1] * math.sin(a))
        need_h = max(across / (self.W / self.H), along * math.sin(e) + 0.6 * math.cos(e))
        dist = need_h / math.tan(math.radians(fov / 2)) * 1.2 + along * math.cos(e) * 0.5
        eye = self.centre + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        self.renderer.setup_camera(fov, self.centre, eye, [0, 0, 1])

    def render(self, t, azimuth, elevation, full=False):
        sc = self.renderer.scene
        if not full or not self.full_shown:
            if sc.has_geometry("mesh"):
                sc.remove_geometry("mesh")
            sc.add_geometry("mesh", self.mesh_at(self.rec.t_end + 1e9 if full else t), self.material)
            self.full_shown = full
        self.look(azimuth, elevation)
        img = np.asarray(self.renderer.render_to_image())
        return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)


# ---------- the video ----------

def font(bold, size):
    for f in (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else
              ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def mmss(seconds):
    s = int(max(0, seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class Video:
    W, H = 1920, 1080

    def __init__(self, rec, args):
        self.rec = rec
        dur = rec.t_end - rec.t_start
        nice = [1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 30, 32, 40, 50, 60, 64, 80, 100]
        self.speed = args.speed or (next((s for s in nice if dur / s <= args.target), nice[-1]) if args.target else 8)
        self.half = self.W // 2
        self.map_panel = MapPanel(rec, (self.half, self.H))
        out_dir = rec.session / "video"
        out_dir.mkdir(exist_ok=True)
        self.scene = Scene3D(rec, (self.W - self.half, self.H), args.glow * self.speed, out_dir)
        self.label_font, self.badge_font = font(True, 22), font(True, 24)
        self.az0, self.az1, self.elev = -105.0, -75.0, 52.0  # 3D view: a slow pan during the run

    def azimuth(self, t):
        f = (t - self.rec.t_start) / max(1e-6, self.rec.t_end - self.rec.t_start)
        return self.az0 + (self.az1 - self.az0) * (0.5 - 0.5 * math.cos(math.pi * f))

    def badge(self, draw, text):
        """Fast-forward symbol, speed and run time, bottom left."""
        f = self.badge_font
        w = draw.textlength(text, font=f) + 66
        x, y, h = 24, self.H - 24 - 44, 44
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=BADGE)
        cy = y + h / 2
        for dx in (16, 28):
            draw.polygon([(x + dx, cy - 9), (x + dx + 12, cy), (x + dx, cy + 9)], fill=ACCENT)
        draw.text((x + 50, cy), text, font=f, fill=TEXT, anchor="lm")

    def run_frame(self, t):
        frame = np.empty((self.H, self.W, 3), np.uint8)
        frame[:, :self.half] = self.map_panel.draw(t)
        frame[:, self.half:] = self.scene.render(t, self.azimuth(t), self.elev)
        frame[:, self.half - 1:self.half + 1] = BORDER
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        draw.text((24, 22), "Lidar map", font=self.label_font, fill=TEXT, anchor="lt")
        draw.text((self.half + 24, 22), "3D reconstruction", font=self.label_font, fill=TEXT, anchor="lt")
        self.badge(draw, f"×{self.speed}   {mmss(t - self.rec.t_start)}")
        return np.asarray(pil)

    def orbit_frame(self, k, n):
        """Full width: one turn around the finished room."""
        e = 0.5 - 0.5 * math.cos(math.pi * k / max(1, n - 1))
        return self.scene.render(self.rec.t_end, self.az1 + 360.0 * e, self.elev - 14.0 * math.sin(math.pi * e), full=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--speed", type=int, default=0, help="time-lapse factor (default 8; or see --target)")
    p.add_argument("--target", type=float, default=0, help="instead of a fixed speed: about this video length in seconds")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--theme", choices=list(THEMES), default="light", help="white (light) or dark background")
    p.add_argument("--glow", type=float, default=1.2, help="seconds (video) newly seen surfaces glow (0 = off)")
    p.add_argument("--orbit", type=float, default=0.0, help="append a full-width orbit around the room (seconds)")
    p.add_argument("--preview", type=float, default=0, help="only this many seconds (test clip)")
    p.add_argument("--at", type=float, default=0.3, help="where the preview starts (fraction of the run)")
    p.add_argument("--still", type=float, default=None, help="write one frame at this fraction of the run as PNG")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    set_theme(args.theme)
    session = args.session.resolve()
    if not (session / "reconstruction" / "cameras.json").exists():
        sys.exit("no reconstruction: run projects/mapping/reconstruct.py on this session first")
    t0 = time.monotonic()
    print("Reading the recording ...", flush=True)
    rec = Recording(session)
    video = Video(rec, args)
    dur = rec.t_end - rec.t_start
    print(f"Run {mmss(dur)} min, shown x{video.speed}", flush=True)

    if args.still is not None:
        out = args.out or session / "video" / f"{session.name}_still_{args.still:.2f}.png"
        cv2.imwrite(str(out), cv2.cvtColor(video.run_frame(rec.t_start + args.still * dur), cv2.COLOR_RGB2BGR))
        print(f"wrote {out}")
        return

    import imageio_ffmpeg
    name = f"{session.name}_preview" if args.preview else f"{session.name}_x{video.speed}"
    out = args.out or session / "video" / f"{name}.mp4"
    fps = args.fps
    step = video.speed / fps  # robot seconds per video frame
    if args.preview:
        t_a = rec.t_start + args.at * dur
        times = np.arange(t_a, min(rec.t_end, t_a + args.preview * video.speed), step)
    else:
        times = np.append(np.arange(rec.t_start, rec.t_end, step), rec.t_end)
    n_orbit = 0 if args.preview else int(args.orbit * fps)
    total = len(times) + fps + n_orbit
    writer = imageio_ffmpeg.write_frames(str(out), (video.W, video.H), fps=fps, codec="libx264", quality=None,
                                         output_params=["-crf", "18", "-preset", "medium"],
                                         macro_block_size=1, ffmpeg_log_level="error")
    writer.send(None)
    done = 0

    def emit(img):
        nonlocal done
        writer.send(np.ascontiguousarray(img).tobytes())
        done += 1
        if done % fps == 0 or done == total:
            print(f"\r  frame {done}/{total}  ({time.monotonic() - t0:.0f} s)", end="", flush=True)

    for t in times:
        img = video.run_frame(t)
        emit(img)
    for _ in range(fps):  # hold the end for a second
        emit(img)
    if n_orbit:
        video.scene.resize((video.W, video.H))
        last = img
        for k in range(n_orbit):
            img = video.orbit_frame(k, n_orbit)
            if k < fps:  # crossfade from the two panels
                a = (k + 1) / (fps + 1)
                img = (last * (1 - a) + img * a).astype(np.uint8)
            emit(img)
    writer.close()
    print(f"\nwrote {out}  ({done / fps:.0f} s of video, {time.monotonic() - t0:.0f} s to make)")


if __name__ == "__main__":
    main()
