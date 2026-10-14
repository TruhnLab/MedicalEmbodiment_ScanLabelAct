"""Obstacles the lidar can't see, from the depth camera: table tops, couch fronts, shelves.

The lidar measures one horizontal slice ~20 cm above the floor; the robot drove into a couch whose
front is higher than that (2026-10-05). The depth camera sees them. The lidar stays the main sensor;
depth only ADDS obstacles, and only certain ones:
  - height above the floor between MIN_H (floor noise, pose errors) and the robot's own height MAX_H
    (higher: the robot passes under it); range 0.25 m (gripper) to MAX_RANGE
  - per image: a 5 cm cell counts only with MIN_PIXELS depth pixels in it (no specks)
  - over time (DepthMemory): a cell slows/stops the robot only after CONFIRM_FRAMES images within
    CONFIRM_S, and is kept for planning after KEEP_FRAMES images
ROS-free (numpy), used by explorer.py and checked offline on recordings.
"""

import numpy as np

CELL = 0.05
MIN_H, MAX_H = 0.10, 0.45
MIN_RANGE, MAX_RANGE = 0.25, 1.8
MIN_PIXELS = 25
STEP = 2               # every 2nd pixel in both directions: 4x fewer points, the same cells
CONFIRM_FRAMES, CONFIRM_S = 2, 1.0
KEEP_FRAMES = 3


def obstacle_cells(depth_mm, K, T_world_cam, floor_z=0.0):
    """Depth image (uint16 mm) -> set of (i, j) cells (CELL grid of the world frame's x, y) holding
    obstacles at robot height. T_world_cam: optical camera -> a frame with z up; floor_z: the floor's
    height in it (live 3.6-5.4 cm, coverage3d.FloorHeight; with 0 the band started ~5 cm above the floor)."""
    fx, fy, cx, cy = K
    z = depth_mm[::STEP, ::STEP].astype(np.float32) / 1000.0
    v, u = np.nonzero((z > MIN_RANGE) & (z < MAX_RANGE))
    if not len(v):
        return set()
    zz = z[v, u]
    u, v = u * STEP, v * STEP
    pts = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], 1)
    w = pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    keep = (w[:, 2] - floor_z > MIN_H) & (w[:, 2] - floor_z < MAX_H)
    if not keep.any():
        return set()
    ij = np.floor(w[keep, :2] / CELL).astype(np.int64)
    cells, counts = np.unique(ij, axis=0, return_counts=True)
    return {tuple(c) for c in cells[counts * STEP * STEP >= MIN_PIXELS]}


class DepthMemory:
    """Which cells are confirmed obstacles (for the collision check) and which are kept (planning)."""

    def __init__(self):
        self.seen = {}   # cell -> list of times seen (recent ones)
        self.count = {}  # cell -> images it was seen in, in total

    def add(self, cells, now):
        for c in cells:
            t = self.seen.setdefault(c, [])
            t.append(now)
            if len(t) > 8:
                del t[:-8]
            self.count[c] = self.count.get(c, 0) + 1

    def confirmed(self, now):
        """Cells seen in CONFIRM_FRAMES images within the last CONFIRM_S (current obstacles)."""
        return [c for c, t in self.seen.items() if sum(now - s <= CONFIRM_S for s in t) >= CONFIRM_FRAMES]

    def kept(self):
        """Cells seen in at least KEEP_FRAMES images during the run (permanent for planning)."""
        return [c for c, n in self.count.items() if n >= KEEP_FRAMES]

    @staticmethod
    def centres(cells):
        a = np.array(list(cells), float).reshape(-1, 2)
        return (a + 0.5) * CELL
