"""Where is the robot in a known room? Lidar scans matched against the room's 2D map. No ROS, numpy only,
so it runs on the laptop, in a container or offline on recordings (test_localize.py).

    loc = Localizer(room.grid, room.resolution, room.origin, area=polygon)
    pose, info = loc.global_search(points)          # points: Nx2 in the robot frame (e.g. one turn on the spot)
    pose, info = loc.track(points, predicted_pose)  # every scan afterwards, around the odometry's prediction

Poses are (x, y, yaw) of the robot base in the map. Scans become points with scan_points() (lidar
mount from the robot's static TF) and are stitched over a turn with stitch() (odometry).

Score of a pose: the mean over the scan points of exp(-d^2 / 2 sigma^2), d = distance to the nearest
wall of the map (a likelihood field). Points on things that are not in the map (people, moved chairs)
just score 0, so they don't pull the pose; the walls dominate.

Global search, coarse to fine:
1. every reachable position on a 20 cm grid x every 3 deg, against a blurred field (sigma 15 cm), so
   the true pose scores well although the grid misses it by up to 10 cm / 1.5 deg;
2. the 40 best distinct candidates are refined in 3 steps down to 1 cm / 0.25 deg (sigma 5 cm);
3. the best one wins; it is unambiguous if no candidate > 0.5 m or 20 deg away scores nearly as well.
A room with near-symmetric halves can look alike; then `ambiguous` is set and more of the room has to
be seen (drive a bit, turn again).
"""

import math

import cv2
import numpy as np

COARSE_STEP = 0.20     # m, positions of the global search
COARSE_YAW = math.radians(3)
COARSE_SIGMA = 0.15    # m, wide enough that a 10 cm / 1.5 deg miss still scores (see above)
FINE_SIGMA = 0.05      # m, the map's 5 cm cells
CANDIDATES = 40
DISTINCT = (0.5, math.radians(20))  # other candidates must be this far away to count as an alternative
POINT_SPACING = 0.05   # m, scan points thinned to this (a turn gives ~10 k points, most redundant)
MIN_RANGE, MAX_RANGE = 0.15, 10.0   # m; the LD19's own body echoes and far, noisy returns
# points scored per pose in the global search: a turn gives ~3500 after thinning, and the search time
# grows with them (24 s for 3552 points on 30 m2, 2026-10-07); a few hundred spread over the turn
# pin the pose down as well
MAX_COARSE_POINTS, MAX_FINE_POINTS = 300, 800


def pose_matrix(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0, 0, 1]])


def matrix_pose(T):
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])


def compose(a, b):
    return matrix_pose(pose_matrix(a) @ pose_matrix(b))


def relative(a, b):
    """b seen from a (a^-1 * b)."""
    return matrix_pose(np.linalg.inv(pose_matrix(a)) @ pose_matrix(b))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def scan_points(msg, lidar_in_base=(0.0, 0.0, 0.0)):
    """rosbridge LaserScan dict -> Nx2 points in the robot base frame."""
    r = np.array([np.nan if v is None else v for v in msg["ranges"]], float)
    a = msg["angle_min"] + msg["angle_increment"] * np.arange(len(r))
    ok = np.isfinite(r) & (r > max(MIN_RANGE, msg["range_min"])) & (r < MAX_RANGE)
    pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], 1)
    return pts @ pose_matrix(lidar_in_base)[:2, :2].T + lidar_in_base[:2]


def thin(pts, spacing=POINT_SPACING):
    """One point per spacing x spacing cell."""
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / spacing).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(idx)]


def spread(pts, n):
    """At most n of pts, evenly spread over the list (scans are stitched in time order, so over the turn)."""
    return pts if len(pts) <= n else pts[np.linspace(0, len(pts) - 1, n).astype(int)]


def stitch(scans):
    """[(points in the robot frame, odometry pose)] -> all points in the frame of the last pose.
    Over one turn on the spot odometry's error is small (EKF with the IMU's yaw)."""
    ref = np.linalg.inv(pose_matrix(scans[-1][1]))
    out = [pts @ (ref @ pose_matrix(p))[:2, :2].T + (ref @ pose_matrix(p))[:2, 2] for pts, p in scans]
    return thin(np.concatenate(out))


class Localizer:
    def __init__(self, grid, resolution, origin, area=None):
        """grid: occupancy (-1 unknown, 0..100, row = y). area: polygon [(x, y), ...] the robot can be
        in (scene.json's area_polygon); default: all free cells."""
        self.res = resolution
        self.origin = np.asarray(origin, float)
        self.shape = grid.shape
        occ = (grid >= 65).astype(np.uint8)
        # distance to the nearest wall, in m (cells outside the map count as far)
        dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5).astype(np.float32) * resolution
        self.fine = np.exp(-dist ** 2 / (2 * FINE_SIGMA ** 2)).astype(np.float32)
        self.coarse = np.exp(-dist ** 2 / (2 * COARSE_SIGMA ** 2)).astype(np.float32)
        # where the robot can stand: free, 15 cm from walls (its half width), inside the area
        free = (grid == 0) & (dist > 0.15)
        if area is not None and len(area) >= 3:
            mask = np.zeros(self.shape, np.uint8)
            poly = ((np.asarray(area, float)[:, :2] - self.origin) / resolution).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
            free &= mask.astype(bool)
        self.free = free

    def score(self, field, pts, poses):
        """Mean likelihood of pts (Nx2, robot frame) at each pose (Px3) -> P scores."""
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        x = poses[:, None, 0] + c[:, None] * pts[None, :, 0] - s[:, None] * pts[None, :, 1]
        y = poses[:, None, 1] + s[:, None] * pts[None, :, 0] + c[:, None] * pts[None, :, 1]
        col = ((x - self.origin[0]) / self.res).astype(np.int32)
        row = ((y - self.origin[1]) / self.res).astype(np.int32)
        inside = (col >= 0) & (col < self.shape[1]) & (row >= 0) & (row < self.shape[0])
        v = field[np.clip(row, 0, self.shape[0] - 1), np.clip(col, 0, self.shape[1] - 1)]
        return np.where(inside, v, 0.0).mean(1)

    def search(self, field, pts, poses, chunk=200_000):
        """Scores of many poses, in chunks (P x N lookups)."""
        n = max(1, chunk // max(1, len(pts)))
        return np.concatenate([self.score(field, pts, poses[i:i + n]) for i in range(0, len(poses), n)])

    def refine(self, pts, pose, steps=((0.10, math.radians(2)), (0.03, math.radians(0.75)),
                                       (0.01, math.radians(0.25)))):
        """Local grid search around pose, finer each step (+-2 steps per axis)."""
        best = np.asarray(pose, float)
        k = np.arange(-2, 3)
        for dxy, dyaw in steps:
            dx, dy, da = np.meshgrid(k * dxy, k * dxy, k * dyaw, indexing="ij")
            cand = best + np.stack([dx.ravel(), dy.ravel(), da.ravel()], 1)
            sc = self.score(self.fine, pts, cand)
            best = cand[np.argmax(sc)]
        return best, float(self.score(self.fine, pts, best[None])[0])

    def global_search(self, pts):
        """pts: Nx2 in the robot frame, ideally a full turn (the LD19's rear ~155 deg is filtered out
        of /scan). Returns (pose, info) with info: score, runner_up, ambiguous, candidates."""
        n = len(thin(pts))
        pts = spread(thin(pts), MAX_FINE_POINTS)
        coarse_pts = spread(thin(pts, COARSE_STEP / 2), MAX_COARSE_POINTS)
        rows, cols = np.nonzero(self.free)
        xy = np.stack([cols, rows], 1) * self.res + self.origin + self.res / 2
        xy = xy[np.unique(np.floor(xy / COARSE_STEP).astype(np.int64), axis=0, return_index=True)[1]]
        yaws = np.arange(-math.pi, math.pi, COARSE_YAW)
        poses = np.concatenate([np.column_stack([xy, np.full(len(xy), a)]) for a in yaws])
        sc = self.search(self.coarse, coarse_pts, poses)

        # best distinct candidates (non-maximum suppression over position and yaw)
        cands = []
        for i in np.argsort(-sc):
            p = poses[i]
            if all(np.hypot(*(p[:2] - q[:2])) > 2 * COARSE_STEP or abs(wrap(p[2] - q[2])) > 4 * COARSE_YAW
                   for q in cands):
                cands.append(p)
                if len(cands) == CANDIDATES:
                    break
        refined = sorted((self.refine(pts, p) for p in cands), key=lambda r: -r[1])
        best, best_score = refined[0]
        runner = max((s for p, s in refined[1:]
                      if np.hypot(*(p[:2] - best[:2])) > DISTINCT[0] or abs(wrap(p[2] - best[2])) > DISTINCT[1]),
                     default=0.0)
        return best, {"score": best_score, "runner_up": runner, "points": n,
                      "ambiguous": runner > 0.85 * best_score,
                      "candidates": [(p.tolist(), s) for p, s in refined[:5]]}

    def track(self, pts, predicted, window=(0.15, math.radians(6))):
        """One scan, around the pose odometry predicts. Searches +-window, then refines."""
        pts = thin(pts)
        k = np.linspace(-1, 1, 7)
        dx, dy, da = np.meshgrid(k * window[0], k * window[0], k * window[1], indexing="ij")
        cand = np.asarray(predicted, float) + np.stack([dx.ravel(), dy.ravel(), da.ravel()], 1)
        start = cand[np.argmax(self.score(self.fine, pts, cand))]
        pose, s = self.refine(pts, start, steps=((0.02, math.radians(0.75)), (0.01, math.radians(0.25))))
        return pose, {"score": s}
