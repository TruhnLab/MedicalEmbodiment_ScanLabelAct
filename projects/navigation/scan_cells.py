"""The room's 3D scan as occupied 5 cm cells, to check where a detected item can be (perception.py, live.py).
Cups showed up 1.05 m above the floor where the scan has nothing but floor (real run 2026-10-09): depth from the
mask's edge mixed foreground and background. A sighting is "sure" only if it rests on something the scan knows
(floor, a cabinet top, ...); otherwise its place is estimated along the camera's line of sight through the
mask's bottom (where the item touches what it stands on), up to the first scanned surface.

    cells = ScanCells.of(room)                     # cached in data/rooms/<room>/scan_cells.npz (built once, ~30 s)
    cells.supported(position, size)                # position/size as perception.locate gives them (z above floor)
    cells.ray_hit(origin, direction)               # floor-relative map coordinates -> first hit point or None

Heights are above the floor (the reconstruction's floor is at z = 0, like perception.locate's positions).
Tall table tops are often missing from the scan (the camera is 0.3 m high): an item on one is not "supported";
it stays a vague sighting (only a direction), never a wrong place on the map.
"""

import threading

import numpy as np

CELL_M = 0.05
SUPPORT_XY_M = 0.10    # scan points this close sideways ...
SUPPORT_BELOW_M = 0.12  # ... and up to this far below the item's bottom carry it
FLOOR_M = 0.06         # an item's bottom this close to the floor stands on it (the scan's floor has holes)
RAY_FROM_M, RAY_TO_M, RAY_STEP_M = 0.3, 6.0, 0.03

_cache = {}
_loading = set()
_lock = threading.Lock()


class ScanCells:
    def __init__(self, occ, lo):
        self.occ, self.lo = occ, np.asarray(lo, float)  # occ[ix, iy, iz]; lo: the corner cell's minimum

    @classmethod
    def of(cls, room):
        """The room's cells, or None if the room has no reconstruction."""
        with _lock:
            if room.name not in _cache:
                _cache[room.name] = cls._load(room)
            return _cache[room.name]

    @classmethod
    def ready(cls, room):
        """The room's cells if loaded; otherwise starts loading them in the background and returns None (the
        live detection must not wait: the first build reads the 230 MB cloud)."""
        if room is None:
            return None
        if room.name in _cache:
            return _cache[room.name]
        with _lock:
            if room.name not in _loading:
                _loading.add(room.name)
                threading.Thread(target=cls.of, args=(room,), daemon=True, name="scan cells").start()
        return None

    @classmethod
    def _load(cls, room):
        cache = room.path / "scan_cells.npz"
        cloud = room.session_path / "reconstruction" / "cloud.ply"
        if cache.exists() and (not cloud.exists() or cache.stat().st_mtime >= cloud.stat().st_mtime):
            z = np.load(cache)
            return cls(z["occ"], z["lo"])
        if not cloud.exists():
            return None
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(cloud)).points)
        pts = pts[(pts[:, 2] > -0.1) & (pts[:, 2] < 2.5)]
        lo = np.floor(pts.min(0) / CELL_M) * CELL_M
        idx = np.floor((pts - lo) / CELL_M).astype(int)
        occ = np.zeros(idx.max(0) + 1, bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        np.savez_compressed(cache, occ=occ, lo=lo)
        return cls(occ, lo)

    def _any(self, lo_xyz, hi_xyz):
        a = np.clip(np.floor((np.asarray(lo_xyz) - self.lo) / CELL_M).astype(int), 0, np.array(self.occ.shape) - 1)
        b = np.clip(np.floor((np.asarray(hi_xyz) - self.lo) / CELL_M).astype(int), 0, np.array(self.occ.shape) - 1)
        return bool(self.occ[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].any())

    def supported(self, position, size):
        x, y, z = position
        bottom = z - (size[2] / 2 if size is not None else 0.0)
        if bottom <= FLOOR_M:
            return True
        r = SUPPORT_XY_M
        return self._any((x - r, y - r, bottom - SUPPORT_BELOW_M), (x + r, y + r, bottom + 0.03))

    def ray_hit(self, origin, direction):
        """The first point along the ray that hits the scan or the floor (floor-relative map coordinates)."""
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        for t in np.arange(RAY_FROM_M, RAY_TO_M, RAY_STEP_M):
            p = o + t * d
            if p[2] <= 0.0:
                return o + (-o[2] / d[2]) * d if d[2] < 0 else p
            i = np.floor((p - self.lo) / CELL_M).astype(int)
            if (i >= 0).all() and (i < self.occ.shape).all() and self.occ[tuple(i)]:
                return p
        return None
