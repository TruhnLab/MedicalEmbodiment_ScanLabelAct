"""3D camera coverage from depth images: which surfaces the camera has seen well enough.

The exploration steers by it: every depth image is counted into a voxel grid, a voxel's observation is
"good" if it is close enough and not at a grazing angle, and a surface counts as covered after GOOD_VIEWS
good views (the reconstruction keeps surfaces seen at least twice). grid_status turns that into the
exploration's targets on the 2D map. projects/coverage3d/README.md has the measurements behind the values.

ROS-free (numpy), used live by the explorer and offline by projects/coverage3d/evaluate.py.
"""

from collections import deque

import numpy as np

VOXEL = 0.05
MAX_RANGE = 2.5          # m: default (evaluate.py); the exploration passes its own (2 m)
MIN_COS = 0.5            # viewing angle at most 60 deg from head-on
# the floor is seen at a flat angle from 0.3 m camera height (1 m away: 73 deg from head-on), but the
# reconstruction builds it fine from such views (2026-10-05): its own limit (tuned with evaluate.py)
MIN_COS_FLOOR = 0.15
# ... but the floor (shiny, seen from 0.26 m height) only gives depth up close: share of valid floor pixels
# by distance from the camera in run 20261006_142432: 0.7-1.3 m 72-85 %, 1.5-1.7 m 24 %, beyond 1.7 m
# 0-4 %. Floor seen only from further away stayed a hole in the mesh while counting as covered.
FLOOR_MAX_RANGE = 1.3
FLOOR_Z = 0.08           # points below this (map frame, floor at z = 0) are floor
MIN_RANGE = 0.25         # closer: the gripper
GOOD_VIEWS = 2
# exploration targets (grid_status): a voxel seen at least SURFACE_VIEWS times (even poorly) is a real
# surface; a map column counts as seen when SEEN_SHARE of its surface voxels have GOOD_VIEWS good
# views; up to Z_MAX (the camera, 0.25 m high and looking level, sees up to ~1.2 m at 2 m distance)
SURFACE_VIEWS = 2
SEEN_SHARE = 0.8
# once seen, a column turns unseen again only below this share (hysteresis): single far or grazing looks
# add 1-2 voxels that are only "observed" and flipped it back, each flip a new viewpoint (run
# 20261007_093027, keyframes replayed: 7.0 m2 of flips; half the flipped columns had <= 2 such voxels).
# A latch without a lower limit hid real gaps: the never-good voxels were mostly 0.6-1.25 m high.
KEEP_SHARE = 0.6
KEEP_MISSING = 2         # ... or while at most this many of its voxels are not seen well (a floor column with
                         # one noise voxel above it is at 50 %: the share alone is too coarse for small columns)
Z_MAX = 1.2
STEP = 3                 # every 3rd pixel in both directions (5 cm voxels at <= 2.5 m: still dense)


class FloorHeight:
    """The floor's height in the map frame, from the depth images: live it sits 3.6-5.4 cm above z = 0
    (robot model; the reconstruction levels it afterwards, 6 runs 2026-10-06). The most common height near
    the ground within 1.5 m (where the floor gives good depth), as the median over the last frames: one
    frame of a table top or a wall's foot doesn't move it. Shared by the coverage and the depth obstacles."""

    def __init__(self):
        self.z = 0.0
        self._samples = deque(maxlen=60)

    def update(self, z, rng):
        near = (rng < 1.5) & (z > -0.15) & (z < 0.20)
        if near.sum() < 300:
            return
        hist, edges = np.histogram(z[near], bins=35, range=(-0.15, 0.20))
        if hist.max() < 150:
            return
        k = int(np.argmax(hist))
        self._samples.append(0.5 * (edges[k] + edges[k + 1]))
        self.z = float(np.median(self._samples))

    def add(self, depth_mm, K, T_map_cam):
        """One depth image (as VoxelCoverage.add)."""
        fx, fy, cx, cy = K
        z = depth_mm[::STEP, ::STEP].astype(np.float32) / 1000.0
        v, u = np.nonzero((z > MIN_RANGE) & (z < 1.5))
        if not len(v):
            return
        zz = z[v, u]
        pts = np.stack([(u * STEP - cx) * zz / fx, (v * STEP - cy) * zz / fy, zz], 1)
        self.update(pts @ T_map_cam[2, :3] + T_map_cam[2, 3], zz)


class VoxelCoverage:
    """Good-view counts per voxel in a fixed box around the map origin (sparse growth not needed for
    a room: 40 x 40 x 2.5 m at 5 cm is 32 MB of uint8)."""

    def __init__(self, origin=(-20.0, -20.0, -0.3), size=(40.0, 40.0, 2.5), max_range=MAX_RANGE, floor=None):
        self.origin = np.array(origin)
        # heights are measured from the floor (FloorHeight). Without it the floor's noise spread into the
        # 5-10 cm layer, judged as a wall (grazing: never good), and floor seen well from close up stayed
        # "unseen" (run 20261006_161813: 8.1 of the 10.2 m2 left; live 69 %, levelled 84 %).
        # floor: a shared estimate its owner updates (explorer.py); otherwise this one keeps its own.
        self.floor = floor if floor is not None else FloorHeight()
        self._own_floor = floor is None
        self.min_cos_floor = MIN_COS_FLOOR
        self.max_range = max_range
        self.shape = tuple(int(s / VOXEL) for s in size)
        self.good = np.zeros(self.shape, np.uint8)  # good views
        # columns that once counted as seen (grid_status): they stay seen down to KEEP_SHARE. Live, flips
        # back were 0.1-0.4 m2 per minute in run 20261007_101108, each a new viewpoint; coverage was flat
        # at 85 % for the last 7.5 of its 16.8 min
        self.seen_cols = np.zeros(self.shape[:2], bool)
        self.any = np.zeros(self.shape, np.uint8)   # any view (also far or grazing)

    def add(self, depth_mm, K, T_map_cam):
        """One depth image (uint16 mm), intrinsics (fx, fy, cx, cy), optical camera -> map."""
        fx, fy, cx, cy = K
        z = depth_mm[::STEP, ::STEP].astype(np.float32) / 1000.0
        # surface orientation from the depth gradients: cos of the angle between ray and normal
        gx = np.gradient(z, axis=1) * fx / STEP
        gy = np.gradient(z, axis=0) * fy / STEP
        v, u = np.nonzero((z > MIN_RANGE) & (z < 4.0))
        if not len(v):
            return
        zz = z[v, u]
        cos = 1.0 / np.sqrt(1.0 + (gx[v, u] / zz) ** 2 + (gy[v, u] / zz) ** 2)
        pts = np.stack([(u * STEP - cx) * zz / fx, (v * STEP - cy) * zz / fy, zz], 1)
        w = pts @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
        if self._own_floor:
            self.floor.update(w[:, 2], zz)
        w[:, 2] -= self.floor.z
        floor = w[:, 2] < FLOOR_Z
        rng = np.sqrt((pts ** 2).sum(1))
        good = (np.where(floor, rng < FLOOR_MAX_RANGE, zz < self.max_range)
                & (cos > np.where(floor, self.min_cos_floor, MIN_COS)))
        idx = np.floor((w - self.origin) / VOXEL).astype(np.int64)
        inside = ((idx >= 0) & (idx < self.shape)).all(1)
        flat = np.ravel_multi_index(idx[inside].T, self.shape)
        # each voxel counts once per image (many pixels hit the same voxel)
        self._count(self.any, np.unique(flat))
        self._count(self.good, np.unique(flat[good[inside]]))

    @staticmethod
    def _count(grid, flat):
        g = grid.reshape(-1)
        g[flat] = np.minimum(g[flat].astype(np.int16) + 1, 255).astype(np.uint8)

    def covered(self):
        return self.good >= GOOD_VIEWS

    def column_index(self, x, y):
        return (np.floor((np.asarray(x) - self.origin[0]) / VOXEL).astype(int),
                np.floor((np.asarray(y) - self.origin[1]) / VOXEL).astype(int))

    def z_index(self, z):
        return int(np.floor((z - self.origin[2]) / VOXEL))

    def grid_status(self, shape, res, origin_x, origin_y, z_max=Z_MAX):
        """The exploration's targets on a 2D map grid (row = y, col = x; 5 cm like the voxels), one rule
        for everything (floor, walls, furniture, obstacles):
          surface: the column holds a voxel the camera saw at least SURFACE_VIEWS times, even poorly
                   (a real surface, not a single flying pixel), between the floor and z_max
          seen:    at least SEEN_SHARE of those voxels have GOOD_VIEWS good views
        Returns (surface, seen) as bool grids."""
        h, w = shape
        rr, cc = np.mgrid[:h, :w]
        vx, vy = self.column_index(origin_x + (cc + 0.5) * res, origin_y + (rr + 0.5) * res)
        ok = (vx >= 0) & (vx < self.shape[0]) & (vy >= 0) & (vy < self.shape[1])
        vx, vy = vx.clip(0, self.shape[0] - 1), vy.clip(0, self.shape[1] - 1)
        a, f, b = self.z_index(-0.10), self.z_index(FLOOR_Z), self.z_index(z_max) + 1
        observed = self.any[vx, vy, a:b] >= SURFACE_VIEWS
        good = observed & (self.good[vx, vy, a:b] >= GOOD_VIEWS)
        # the floor is one surface, even if its noise spreads it over two voxel layers: the upper layer
        # also collects noisy points from far views (never good beyond FLOOR_MAX_RANGE), and counting the
        # layers separately kept the floor at 50 % "seen" wherever the robot had also looked from afar
        # (run 20261006_154055: 7.7 of the 10.1 m2 left unseen had valid floor depth from 0.7-1.3 m)
        k = f - a
        n_obs = observed[..., k:].sum(axis=2) + observed[..., :k].any(axis=2)
        n_good = good[..., k:].sum(axis=2) + good[..., :k].any(axis=2)
        surface = (n_obs > 0) & ok
        seen = surface & (n_good >= SEEN_SHARE * n_obs)
        self.seen_cols[vx[seen], vy[seen]] = True
        keep = (n_good >= KEEP_SHARE * n_obs) | (n_obs - n_good <= KEEP_MISSING)
        return surface, seen | (surface & self.seen_cols[vx, vy] & keep)

    def seen_on_grid(self, occupied, res, origin_x, origin_y, views=GOOD_VIEWS, grid=None):
        """The coverage on a 2D map grid (row = y, col = x; 5 cm like the voxels): a free cell is seen
        when its floor was seen well enough, an occupied cell when its wall/furniture 0.1-0.5 m high was
        (3x3 around it: lidar and depth place a wall a cell apart). views/grid: e.g. 1 and self.any
        for "seen at all". Checked against the reconstruction
        with projects/coverage3d/evaluate.py (96-97 % of its floor gaps predicted, 2026-10-05)."""
        from scipy import ndimage
        h, w = occupied.shape
        rr, cc = np.mgrid[:h, :w]
        vx, vy = self.column_index(origin_x + (cc + 0.5) * res, origin_y + (rr + 0.5) * res)
        ok = (vx >= 0) & (vx < self.shape[0]) & (vy >= 0) & (vy < self.shape[1])
        vx, vy = vx.clip(0, self.shape[0] - 1), vy.clip(0, self.shape[1] - 1)

        def covered(z0, z1):
            a, b = self.z_index(z0), self.z_index(z1) + 1
            return ((self.good if grid is None else grid)[vx, vy, a:b] >= views).any(axis=2) & ok

        floor = covered(-0.10, FLOOR_Z)
        wall = ndimage.maximum_filter(covered(0.10, 0.50), size=3)
        return np.where(occupied, wall, floor)
