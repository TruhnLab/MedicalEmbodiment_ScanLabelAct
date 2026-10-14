"""Autonomous exploration for the LanderPi: where to go, how to get there, where to look.

ROS-free (numpy/scipy only), so it can be tested on the laptop with a simulated robot
(offboard_ros/test_explore.py); offboard_ros/explorer.py wraps it as a ROS node.

The robot drives continuously; nothing stops to "scan":
  - goals, in one pass: the camera leads, the lidar map (which grows much faster) follows. First
    the viewpoints for the surfaces (walls, furniture) the camera has not seen yet, in the order of
    the shortest route through them (re-planned every second as the map grows), so the robot works
    its way through the room and does not have to come back. Only when no known surface is left:
    the nearest frontier of the lidar map (free space next to unknown)
  - path: Dijkstra over the occupancy grid, preferring clearance from obstacles; re-planned every
    second while driving
  - following: holonomic pure pursuit (the mecanum base moves in any direction)
  - gaze: independently of the travel direction the base turns the camera towards the most
    unseen surfaces nearby, within +-GAZE_MAX of the travel direction: the lidar only sees the
    front ~205 deg, so the robot never drives towards its blind rear
  - safety: stops if the lidar sees anything close in the travel direction (people, chairs that
    are not in the map yet), then re-plans around it
Poses are (x, y, yaw) in the map frame; velocities (vx, vy, wz) in the robot frame.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

# robot and behaviour (metres, radians, seconds)
ROBOT_RADIUS = 0.20       # half the diagonal of the chassis
MIN_CLEARANCE = 0.26      # paths keep at least this from obstacles (robot radius + margin)
COMFORT_CLEARANCE = 0.55  # ... and prefer this much
# One exploration mode (2026-10-06; before: Quick / Normal / Thorough / Quick 3D, which differed in speed,
# camera range and viewpoints per surface). The lidar discovers (frontiers, walls the camera has not seen
# at all); RGB + depth map: whatever the camera has seen but not yet well enough is a target, whatever it
# is (coverage3d.py). Without depth images (the simulator) a 2D estimate on the lidar map stands in; live,
# explorer.py pauses instead.
#   speed/turn: 0.3 m/s, 0.6 rad/s; look_turn 0.3 rad/s while facing surfaces at a viewpoint (sharper
#   images); camera_range 2.0 m: labelling needs depth within ~1.5-2 m (agent board, 2026-10-06)
#   views: 2D estimate's viewpoints per surface; tries: visits before a surface is given up;
#   min_surface: smallest gap visited (cells)
MODES = {
    "3d": {"speed": 0.30, "turn": 0.6, "camera_range": 2.0, "views": 1, "max_angle": 60, "coverage3d": True,
           "tries": 2, "min_surface": 5, "look_turn": 0.3},
}
MODE = "3d"
# a surface counts as seen (= usable for the 3D reconstruction) only from the mode's number of
# different viewpoints ("views"), closer than its camera_range and not at a grazing angle (the
# reconstruction keeps surfaces seen >= 2 times; depth gets noisy with distance and at flat angles)
NEW_VIEW_M, NEW_VIEW_RAD = 0.25, math.radians(15)  # a new viewpoint: moved/turned at least this much
TURN_ACCEL = 1.2          # rad/s^2: smooth turning (sharper images, no swinging back and forth)
GAZE_EVERY_S = 1.5        # how often the viewing direction is re-chosen
LOOKAHEAD = 0.35
GOAL_TOLERANCE = 0.25
GAZE_MAX = math.radians(35)        # camera direction vs travel direction while driving. Was 70: the
                                   # robot then drove mostly sideways/diagonally, where mecanum wheels
                                   # partly cancel (2-3 of 4 push) and it got stuck on a floor ledge
                                   # (2026-10-05); bigger look-arounds happen at the viewpoints
CAMERA_FOV = math.radians(74)      # horizontal field of view of the depth camera
LIDAR_SECTOR = (math.radians(-100), math.radians(95))  # what the lidar sees (robot frame)
STOP_DISTANCE = 0.27      # obstacle this close (from the robot centre) in the travel direction: stop
SLOW_DISTANCE = 0.55      # ... this close: half speed (instead of the old hard stop at 0.32 m, which
                          # stopped the robot ~10 times a minute next to mapped walls, 2026-10-05)
MIN_FRONTIER = 10         # cells (50 cm at 5 cm resolution): smaller gaps in the lidar map don't count
NO_PROGRESS_S = 15.0      # goal given up if the path length does not shrink for this long
SPIN_TIMEOUT_S = 45.0     # a full turn takes ~20 s; give up after this
MIN_GOAL_DIST = 0.5       # goals closer than this are ignored (frontiers of the robot's own blind spot)
SWITCH_GAIN = 1.5         # a new goal must be this much better than the current one to switch
STALL_S = 30.0            # no new map or camera coverage for this long: give the current goal area up
DONE_AFTER_S = 10.0       # no goal for this long (after looking around) before the room counts as done
STUCK_S = 4.0             # within this window: told to drive >= STUCK_DRIVE_S but the SLAM pose moved
STUCK_DRIVE_S = 2.5       # less than STUCK_M net: stuck (e.g. a floor ledge or transition). Turning on
STUCK_M = 0.08            # the spot in between doesn't reset it (it did: the robot jiggled 15 s at a ledge)
PUSH_S = 2.0              # then: face the travel direction and push forward this long at full speed
LOOK_S = 4.0              # at a coverage viewpoint: face the unseen surfaces this long
SLIDE_M = 0.5             # ... modes needing several views: slide this far along the surface meanwhile
SLIDE_SPEED = 0.08        # m/s (slow: sharp images)
LEG_GROUP_M = 0.3         # small unseen pieces this close to each other count as one (furniture legs)
# unseen areas larger than this are split into pieces of this size, one viewpoint each: the camera sees
# ~2 m, so one viewpoint 1.2 m from the centre of a 4 x 6 m area (run 20261006_141517: floor and walls
# on the left) left most of it unseen, and the run ended with it red
TILE_M = 1.5
# floor areas: the camera (0.26 m high, CAMERA_AHEAD of the robot's centre) gets floor depth only 0.7-1.3 m
# away (valid floor pixels in run 20261006_142432: 0.5-0.7 m 41 %, 0.7-1.3 m 72-85 %, beyond 1.7 m < 5 %).
# With the walls' 1.2 m, two floor patches were only seen from 1.0-2.5 m and stayed holes in the mesh; a
# stop 0.9 m from a piece's centre put its near edge ~0.45 m from the camera (too close). So the whole
# piece has to lie in that band: small pieces, the camera ~1.0 m from the centre, and no stop whose near
# edge is closer than FLOOR_NEAR_M. In a narrow corridor that rules out looking across (the near edge is
# too close) and leaves looking along it.
CAMERA_AHEAD = 0.09
FLOOR_TILE_M = 0.6
FLOOR_BAND_M = (0.7, 1.3)        # floor distances from the camera with good depth
FLOOR_NEAR_M = 0.6               # hard limits for a stop: the piece's near edge / far edge (camera)
FLOOR_FAR_M = 1.45
WALL_VIEW_M = (0.7, 1.2, 1.6)    # walls/furniture: min, ideal, max distance of the viewpoint (robot centre)
FLOOR_RANGE = FLOOR_BAND_M[1] + CAMERA_AHEAD   # floor this far from the robot's centre can count as seen
MIN_SURFACE = 8           # cells (40 cm): smaller unseen surface clusters are not visited
CHECKED_RADIUS = 1.0      # m: a viewpoint looked from is not chosen again within this distance ...
# ... for the same target (a piece whose centre is this close to the one looked at then). For ANY target,
# 29 stops' circles covered the whole drivable room and the run ended at 69 % with 19 viewpoints gone at
# once (run 20261006_161813, replayed). Looking at the same surfaces again in vain is what the tries
# (attempts per cell) already limit.
SAME_TARGET_M = 0.6
ROUTE_MAX = 30            # viewpoints in the route (the nearest; keeps planning fast)
ROUTE_CELL = 3            # route distances on a coarser grid: 3 x 5 cm cells
ROUTE_SWITCH = 0.85       # a newly computed route replaces the kept one only if it is 15 % shorter
COVERAGE_COMPLETE = 0.98  # every mode aims at the whole room; the modes differ in quality (views,
                          # distance, angle), not in how much of the room they cover


FREE, UNKNOWN, OCC = 0, 1, 2


@dataclass
class Grid:
    """Occupancy grid: cell (row, col) covers x = origin_x + col*res, y = origin_y + row*res."""
    cells: np.ndarray          # FREE / UNKNOWN / OCC (uint8)
    res: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_ros(cls, data, width, height, res, ox, oy, occupied=65, free=25):
        raw = np.asarray(data, dtype=np.int16).reshape(height, width)
        cells = np.full(raw.shape, UNKNOWN, np.uint8)
        cells[(raw >= 0) & (raw <= free)] = FREE
        cells[raw >= occupied] = OCC
        return cls(cells, res, ox, oy)

    def to_cell(self, x, y):
        return int(math.floor((y - self.origin_y) / self.res)), int(math.floor((x - self.origin_x) / self.res))

    def to_xy(self, r, c):
        return self.origin_x + (c + 0.5) * self.res, self.origin_y + (r + 0.5) * self.res

    def inside(self, r, c):
        return 0 <= r < self.cells.shape[0] and 0 <= c < self.cells.shape[1]


@dataclass
class Status:
    state: str = "idle"        # idle | exploring | paused | done | stopped
    phase: str = ""            # frontiers | coverage
    goal: tuple = None
    path: list = field(default_factory=list)
    message: str = ""
    mode: str = "normal"
    remaining: int = 0         # coverage phase: surface clusters still to visit
    coverage: float = 0.0
    frontiers: int = 0
    blocked: bool = False


class Explorer:
    def __init__(self):
        self.status = Status()
        self.grid = None
        self.clearance = None
        self.seen = None           # bool grid: surface cells seen from enough viewpoints (mode's "views")
        self.views = None          # uint8 grid: from how many viewpoints each surface cell was seen
        self.last_view = None      # pose of the last counted viewpoint
        self.blacklist = []        # goals given up (not reachable / no progress): (x, y), for every purpose
        self.checked = []          # viewpoints reached and looked from: (x, y, target x, target y); only this spot is
                                   # not chosen again (CHECKED_RADIUS): neighbouring viewpoints stay available
        self.goal_since = 0.0
        self.best_remaining = math.inf
        self.last_plan = -math.inf
        self.gaze = None
        # look around: one full turn on the spot (at the start: the lidar is blind to the rear;
        # and before calling the room done when no goal is reachable)
        self.spin = None             # {"turned": rad, "last": yaw, "since": time} while turning
        self.p = MODES[MODE]
        self.walls = []              # no-go lines [(x1, y1, x2, y2)] in map coordinates
        self.depth_obstacles = []    # [(x, y)] map cells the depth camera confirmed (explorer.py)
        self.seen3d = None           # 3D: bool grid like seen, from the depth images (explorer.py)
        self.surface3d = None        # 3D: cells holding a surface the camera saw at all (even poorly)
        self._depth_mask = None
        self.raw = None              # the last lidar map, before the no-go lines are drawn in
        self._region = None          # free area connected to the robot, widened (see update_region)
        self._free_region = None     # ... not widened
        self._wall_mask = None       # cells of the no-go lines
        self.wz = 0.0                # last turn rate (for smooth turning)
        self.progress = (0, 0, 0.0)  # (known cells, seen surface cells, time of the last increase)
        self.home = None             # where the run started: no-go lines keep the robot on this side
        self.events = []             # texts for the console's event history (taken by the ROS node)
        self.no_goal_since = None    # time of the first plan without a goal (after looking around)
        self.attempts = None         # uint8 grid: unsuccessful visits per unseen surface cell (given up
                                     # after as many as the mode needs views, each from another viewpoint)
        self.goal_cluster = None     # (rows, cols) of the surface cluster the current coverage goal is for
        self.goal_views = None       # ... and its cells' view counts when the goal was chosen
        self.route = []              # [(x, y)] the planned order of the coverage viewpoints (kept between plans)
        self.unreached = []          # surface clusters (x, y) whose viewpoint could not be reached once
        self.drive_log = []          # [(time, x, y)] while driving, for the stuck check
        self.unstick = None          # {"phase": "turn"/"push", "since", "start": (x, y)} while getting free
        self.look = None             # at a viewpoint: {"until": time, "target": (x, y), "slide": (x, y) or None}
        self.looked_around = False   # a full turn since the last reachable goal

    # ---------- map ----------

    def event(self, text):
        """Something the user should know about (shown in the console's history)."""
        self.events.append(text)

    def given_up_cells(self):
        if self.grid is None or self.attempts is None:
            return 0
        return int((self.surfaces() & ~self.seen & (self.attempts >= self.tries())).sum())

    def tries(self):
        return self.p.get("tries", self.p["views"])

    def min_surface(self):
        return self.p.get("min_surface", MIN_SURFACE)

    def coverage_visible(self):
        """3D mode: the share of the surfaces the camera saw well enough, among those it saw at all
        (the rest is mostly hidden: under the table, behind legs). None without 3D data."""
        if self.surface3d is None or self.seen is None or self.surface3d.shape != self.seen.shape:
            return None
        s = self.surfaces() & self.surface3d
        return float((s & self.seen).sum()) / max(1, s.sum())

    def set_walls(self, walls):
        """No-go lines from the console: the robot never crosses them, and what lies behind them
        is not explored."""
        self.walls = [tuple(float(v) for v in w) for w in walls if len(w) == 4]
        if self.raw is not None:
            self.set_map(self.raw)

    def draw_walls(self, grid):
        cells = grid.cells.copy()
        mask = np.zeros(cells.shape, bool)
        for x1, y1, x2, y2 in self.walls:
            n = int(math.hypot(x2 - x1, y2 - y1) / (grid.res * 0.5)) + 1
            for t in np.linspace(0, 1, n + 1):
                r, c = grid.to_cell(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True  # 15 cm thick: no gaps
        mask &= np.ones(cells.shape, bool)
        cells[mask] = OCC
        return Grid(cells, grid.res, grid.origin_x, grid.origin_y), mask

    def set_map(self, grid):
        """New lidar map: keep the camera coverage, shifted if the map's origin moved."""
        self.raw = grid
        virtual = np.zeros(grid.cells.shape, bool)
        if self.walls:
            grid, virtual = self.draw_walls(grid)
        self._wall_mask = virtual
        # obstacles only the depth camera sees (above the lidar's slice; depth_obstacles.py): real
        # obstacles for driving and distance keeping, but no surfaces to look at (coverage unchanged)
        depth = np.zeros(grid.cells.shape, bool)
        for x, y in self.depth_obstacles:
            r, c = grid.to_cell(x, y)
            if grid.inside(r, c):
                depth[r, c] = True
        if depth.any():
            cells = grid.cells.copy()
            cells[depth] = OCC
            grid = Grid(cells, grid.res, grid.origin_x, grid.origin_y)
        if self.views is not None and self.grid is not None:
            old, views = self.grid, self.views
            dr = int(round((old.origin_y - grid.origin_y) / grid.res))
            dc = int(round((old.origin_x - grid.origin_x) / grid.res))
            new = np.zeros(grid.cells.shape, np.uint8)
            h, w = views.shape
            r0, c0 = max(0, dr), max(0, dc)
            r1, c1 = min(new.shape[0], dr + h), min(new.shape[1], dc + w)
            if r1 > r0 and c1 > c0:
                new[r0:r1, c0:c1] = views[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            self.views = new
            if self.attempts is not None:
                a2 = np.zeros(grid.cells.shape, np.uint8)
                if r1 > r0 and c1 > c0:
                    a2[r0:r1, c0:c1] = self.attempts[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
                self.attempts = a2
        else:
            self.views = np.zeros(grid.cells.shape, np.uint8)
        if self.attempts is None or self.attempts.shape != grid.cells.shape:
            self.attempts = np.zeros(grid.cells.shape, np.uint8)
        self.seen = self.views >= self.p["views"]
        self.grid = grid
        # distance to the nearest obstacle (unknown counts as obstacle: don't plan through it)
        self.clearance = ndimage.distance_transform_edt(grid.cells == FREE) * grid.res
        # ... and to real obstacles only: the robot's own footprint is "unknown" in the lidar map (the
        # lidar can't see under itself), so near the robot only obstacles count
        self.clearance_occ = ndimage.distance_transform_edt(grid.cells != OCC) * grid.res
        # ... and to real obstacles without the no-go lines: a line may be drawn right next to the
        # robot, it must still be able to drive away from it (just not through it)
        self.clearance_real = ndimage.distance_transform_edt((self.raw.cells != OCC) & ~depth) * grid.res
        # surface normals (pointing out of the obstacles), for the viewing-angle check
        occ = ndimage.gaussian_filter((grid.cells == OCC).astype(float), 1.5)
        gy, gx = np.gradient(occ)
        norm = np.hypot(gx, gy)
        self.normals = np.where(norm[..., None] > 1e-6, -np.stack([gx, gy], -1) / np.maximum(norm, 1e-9)[..., None], 0.0)
        # surface cells worth seeing: obstacles next to free space (not the far side of walls)
        # the exploration area stays defined between plans (status and coverage use it)
        self._region = self._free_region = None
        if self.home is not None:
            self.update_region(grid.to_cell(*self.home))
        self._depth_mask = depth
        self.update_targets()

    def update_targets(self):
        """What is still to be mapped. With 3D data (explorer.py sets surface3d / seen3d from the depth
        images): one rule for everything, floor, walls, furniture, obstacles alike: whatever the camera
        has seen at least twice is a surface to map, done once seen well (coverage3d.grid_status). For
        discovery, also the lidar's walls the camera has not seen at all yet (the lidar sees far ahead
        and around corners; frontiers are separate). Without depth images: the lidar's walls, with the
        2D estimate of what the camera saw."""
        g = self.grid
        lidar = (g.cells == OCC) & ndimage.binary_dilation(g.cells == FREE, iterations=2) & ~self._wall_mask
        if self.p.get("coverage3d") and self.surface3d is not None and self.surface3d.shape == g.cells.shape:
            camera = ndimage.binary_dilation(self.surface3d, iterations=1)  # lidar and depth: a cell apart
            # discovery by the lidar, for what the camera has not seen at all: its walls, and the floor it
            # knows (free cells); the floor between table and chairs stayed unmapped otherwise, since it
            # was never a target (run 20261006_140356)
            unseen_by_camera = ((lidar | (g.cells == FREE)) & ~camera)
            self._surfaces = (self.surface3d | unseen_by_camera) & ~self._wall_mask
        else:
            self._surfaces = lidar & ~self._depth_mask
        self.apply_seen3d()

    def apply_seen3d(self):
        """3D: 'seen' comes from the depth images' 3D coverage (set by explorer.py)."""
        if self.p.get("coverage3d") and self.seen3d is not None and self.views is not None                 and self.seen3d.shape == self.views.shape:
            self.seen = self.seen3d.copy()

    # ---------- camera coverage ----------

    def update_coverage(self, pose):
        """Counts the surface cells the camera sees from this pose (2D ray casting in the lidar
        map), once per viewpoint: a cell counts as seen after the mode's number of viewpoints, each
        within the camera range and not at a grazing angle."""
        if self.grid is None:
            return
        x, y, yaw = pose
        if self.last_view is not None:
            lx, ly, lyaw = self.last_view
            if (math.hypot(x - lx, y - ly) < NEW_VIEW_M
                    and abs(math.atan2(math.sin(yaw - lyaw), math.cos(yaw - lyaw))) < NEW_VIEW_RAD):
                return  # same viewpoint as the last counted one
        self.last_view = (x, y, yaw)
        g = self.grid
        steps = int(self.p["camera_range"] / (g.res * 0.7))
        hits = set()
        for a in np.linspace(yaw - CAMERA_FOV / 2, yaw + CAMERA_FOV / 2, 60):
            dx, dy = math.cos(a) * g.res * 0.7, math.sin(a) * g.res * 0.7
            px, py = x, y
            for i in range(steps):
                px += dx
                py += dy
                r, c = g.to_cell(px, py)
                if not g.inside(r, c):
                    break
                if g.cells[r, c] == UNKNOWN:
                    if i * g.res * 0.7 < 0.35:
                        continue  # the robot's own footprint (the lidar can't see under itself)
                    break
                if g.cells[r, c] == OCC:
                    nx, ny = self.normals[r, c]
                    if nx or ny:  # grazing view of a surface: hardly any usable depth
                        if -(math.cos(a) * nx + math.sin(a) * ny) < math.cos(math.radians(self.p["max_angle"])):
                            break
                    hits.update((rr, cc) for rr in range(max(0, r - 1), r + 2) for cc in range(max(0, c - 1), c + 2))
                    break
        if hits:
            rr, cc = np.array(list(hits)).T
            ok = (rr < g.cells.shape[0]) & (cc < g.cells.shape[1])
            v = self.views[rr[ok], cc[ok]]
            self.views[rr[ok], cc[ok]] = np.minimum(v.astype(np.int16) + 1, 255).astype(np.uint8)
            self.seen[rr[ok], cc[ok]] = self.views[rr[ok], cc[ok]] >= self.p["views"]
        self.apply_seen3d()

    def surfaces(self):
        """Surfaces to look at: next to the free area the robot is in (not behind no-go lines or
        in places it can't get to)."""
        return self._surfaces & self._region if self._region is not None else self._surfaces

    def component_at(self, labels, rc, radius=0.5):
        """Label of the free area at (or nearest to) a cell; 0 if there is none within radius."""
        g = self.grid
        r, c = rc
        k = int(radius / g.res)
        win = labels[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
        rr, cc = np.nonzero(win)
        if len(rr) == 0:
            return 0
        i = np.argmin((rr + max(0, r - k) - r) ** 2 + (cc + max(0, c - k) - c) ** 2)
        return win[rr[i], cc[i]]

    def update_region(self, robot_rc):
        """The exploration area: the free area connected to the START (no-go lines count as walls),
        so a line drawn mid-run keeps the start's side; plus the same widened by 2 cells.
        Returns True if the robot is inside it (False: it has to come back across a line)."""
        g = self.grid
        # the area the robot can DRIVE in (clearance), not all connected free cells: a gap too narrow
        # to drive through (e.g. a no-go line ending 0.4 m short of a wall) let the whole room behind
        # it count, and the run ended at 44 % with those surfaces unreachable (2026-10-05)
        labels, _ = ndimage.label((g.cells == FREE) & (self.clearance >= MIN_CLEARANCE))
        anchor = g.to_cell(*self.home) if self.home is not None else robot_rc
        lab = self.component_at(labels, anchor, 1.0) or self.component_at(labels, robot_rc)
        if not lab:
            self._region = self._free_region = None
            return True
        drivable = labels == lab
        # plus the free space up to the walls next to it (the robot's centre keeps MIN_CLEARANCE away)
        reach = int(math.ceil(MIN_CLEARANCE / g.res)) + 1
        self._free_region = ndimage.binary_dilation(drivable, iterations=reach, mask=g.cells == FREE) | drivable
        # (2 cells: the wall face and the cell behind it, which the camera rays mark too; not the
        # cells deep inside noisy walls next to tiny free specks, which no camera can see)
        self._region = ndimage.binary_dilation(self._free_region, iterations=2)
        return self.component_at(labels, robot_rc) == lab

    def coverage(self):
        s = self.surfaces()
        return float((s & self.seen).sum()) / max(1, s.sum())

    # ---------- planning ----------

    def cost_field(self, start_rc, ignore_walls=False):
        """Dijkstra from the robot over cells with enough clearance (scipy's compiled graph search);
        returns the cost to every cell (inf = unreachable) and each cell's predecessor.
        ignore_walls: to come back to the start's side: no-go lines may be crossed, but each line cell
        costs a lot, so the way back crosses as few lines as possible (one or several, as needed)."""
        g = self.grid
        clearance, clearance_occ = self.clearance, self.clearance_real
        line_cost = None
        if ignore_walls and self.raw is not None and self.raw.cells.shape == g.cells.shape:
            g = self.raw
            clearance = ndimage.distance_transform_edt(g.cells == FREE) * g.res
            clearance_occ = self.clearance_real
            line_cost = self._wall_mask * 300.0
        h, w = g.cells.shape
        ok = clearance >= MIN_CLEARANCE
        # near the robot: its own footprint is unknown in the lidar map, and it may stand closer to
        # something than the planner likes (e.g. after a manual drive): let it leave, avoiding only
        # real obstacles
        r0, c0 = start_rc
        k = int(0.6 / g.res)
        win = (slice(max(0, r0 - k), r0 + k + 1), slice(max(0, c0 - k), c0 + k + 1))
        ok[win] |= (g.cells[win] != OCC) & (clearance_occ[win] >= ROBOT_RADIUS + 0.02)
        # the robot's own spot always: it may stand on a line that was just drawn
        ok[max(0, r0 - 1):r0 + 2, max(0, c0 - 1):c0 + 2] = True
        penalty = np.clip((COMFORT_CLEARANCE - clearance) / (COMFORT_CLEARANCE - MIN_CLEARANCE), 0, 1) * 4.0
        if line_cost is not None:
            penalty = penalty + line_cost
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, weights = [], [], []
        for dr, dc, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a = (slice(max(0, -dr), h - max(0, dr)), slice(max(0, -dc), w - max(0, dc)))
            b = (slice(max(0, dr), h - max(0, -dr) if dr < 0 else h), slice(max(0, dc), w - max(0, -dc) if dc < 0 else w))
            both = ok[a] & ok[b]
            # undirected: cost of an edge = step x (1 + the larger of both cells' penalties)
            cost = step * (1.0 + np.maximum(penalty[a], penalty[b]))
            rows.append(idx[a][both])
            cols.append(idx[b][both])
            weights.append(cost[both])
        graph = sparse.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(h * w, h * w))
        dist, pred = csgraph.dijkstra(graph, directed=False, indices=r0 * w + c0, return_predecessors=True)
        return dist.reshape(h, w), pred.reshape(h, w)

    def path_to(self, parent, goal_rc):
        g = self.grid
        w = g.cells.shape[1]
        h_w = g.cells.size
        r, c = goal_rc
        cells = []
        while parent[r, c] >= 0 and len(cells) < h_w:
            cells.append((r, c))
            r, c = divmod(int(parent[r, c]), w)
        cells.reverse()
        # every second cell is plenty for following (10 cm spacing)
        return [g.to_xy(r, c) for r, c in cells[1::2]] + ([g.to_xy(*cells[-1])] if cells else [])

    def frontier_goals(self, dist, robot_rc):
        """Reachable cells next to each frontier (free cells bordering unknown space). Frontiers
        right at the robot are its own blind spot / footprint, not unexplored space: skipped."""
        g = self.grid
        free = g.cells == FREE
        unknown = g.cells == UNKNOWN
        frontier = free & ndimage.binary_dilation(unknown, structure=np.ones((3, 3)))
        if self._free_region is not None:
            frontier &= self._free_region  # not behind no-go lines (nor anywhere else it can't get to)
        labels, n = ndimage.label(frontier, structure=np.ones((3, 3)))
        goals = []
        reach = np.isfinite(dist)
        for i in range(1, n + 1):
            cells = np.argwhere(labels == i)
            if len(cells) < MIN_FRONTIER:
                continue
            # the frontier itself is often too close to walls: aim for the nearest reachable cell
            # within 0.6 m of its centre
            centre = cells.mean(0)
            if math.hypot(centre[0] - robot_rc[0], centre[1] - robot_rc[1]) * g.res < MIN_GOAL_DIST + 0.2:
                continue
            rr, cc = np.nonzero(reach)
            d2 = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2
            away = (rr - robot_rc[0]) ** 2 + (cc - robot_rc[1]) ** 2 >= (MIN_GOAL_DIST / g.res) ** 2
            near = (d2 <= (0.6 / g.res) ** 2) & away
            if not near.any():
                continue
            k = np.argmin(np.where(near, dist[rr, cc], np.inf))
            goals.append(((rr[k], cc[k]), len(cells)))
        return goals

    def used_spots(self, pts, radius):
        """Cells within radius of the given spots."""
        g = self.grid
        blocked = np.zeros(g.cells.shape, bool)
        h, w = g.cells.shape
        if pts:
            k = int(radius / g.res) + 2
            for px, py in pts:
                r0, c0 = g.to_cell(px, py)
                ys, xs = np.ogrid[max(0, r0 - k):min(h, r0 + k + 1), max(0, c0 - k):min(w, c0 + k + 1)]
                # the exact distance from the cell's centre, as choose_goal's final check measures it:
                # counting in whole cells left a ring 0.94-1.0 m around each spot that passed here and
                # failed there; it is the cheapest ring to reach, so areas were dropped again (run
                # 20261006_154055 ended at 69 % with 'no goal'; replayed: 42 viewpoints, 6 left)
                cx, cy = g.to_xy(ys, xs)
                blocked[ys, xs] |= np.hypot(cx - px, cy - py) <= radius
        return blocked

    def coverage_goals(self, dist, robot_rc):
        """Viewpoints for clusters of unseen surfaces: a reachable cell ~1.2 m in front of them.
        Surfaces tried as often as the mode needs views, without success, are given up."""
        g = self.grid
        unseen = self.surfaces() & ~self.seen & (self.attempts < self.tries())
        if self.p.get("coverage3d") and self.surface3d is not None:
            # the 3D targets' "not yet seen" is speckled along the edges of what was seen well: single
            # cells dropped, gaps within ~20 cm merged, so one viewpoint serves one area (offline on
            # 20261006_125622: 63 open viewpoints at the end of the run without this)
            open_ = ndimage.binary_opening(unseen, structure=np.ones((2, 2)))
            unseen = ndimage.binary_closing(open_, structure=np.ones((5, 5))) & self.surfaces() & ~self.seen & (self.attempts < self.tries()) | open_
        labels, n = ndimage.label(unseen, structure=np.ones((3, 3)))
        # furniture legs (stools, chairs, tables) are many tiny separate dots in the lidar map, each
        # below MIN_SURFACE, so a row of stools was never visited (2026-10-05). Small pieces within
        # LEG_GROUP_M of each other form one group, if there are at least 3 of them.
        sizes = ndimage.sum(unseen, labels, range(1, n + 1))
        small = np.isin(labels, np.nonzero(sizes < self.min_surface())[0] + 1)
        if small.any():
            near = ndimage.binary_dilation(small, iterations=int(LEG_GROUP_M / g.res / 2) + 1)
            groups, m = ndimage.label(near, structure=np.ones((3, 3)))
            for j in range(1, m + 1):
                pieces = np.unique(labels[small & (groups == j)])
                if len(pieces) >= 3:
                    labels[np.isin(labels, pieces)] = pieces[0]  # one cluster from now on
        reach = np.isfinite(dist)
        # spots already used (looked from, or given up as unreachable) are excluded BEFORE a cluster's
        # viewpoint is chosen: excluding the chosen one afterwards dropped the whole cluster, since the
        # same nearest spot came up again (run 20261006_141517 ended with 'no goal' and ~30 % unseen;
        # offline with its visited viewpoints: 1 of 19 viewpoints left)
        # (given up as unreachable: for every target; looked from: per target, below)
        reach &= ~self.used_spots(self.blacklist, 1.0)
        rr, cc = np.nonzero(reach)
        goals = []
        away = np.hypot(rr - robot_rc[0], cc - robot_rc[1]) * g.res >= MIN_GOAL_DIST
        for i in np.unique(labels[labels > 0]):
            area = np.argwhere(labels == i)
            if len(area) < self.min_surface():
                continue  # small bits: mostly lidar artefacts, not worth a detour
            floor = (g.cells[area[:, 0], area[:, 1]] == FREE).mean() >= 0.5
            for cells in self.tiles(area, FLOOR_TILE_M if floor else TILE_M):
                centre = cells.mean(0)
                d = np.sqrt((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2) * g.res
                tx, ty = g.to_xy(*centre)
                looked = [(x, y) for x, y, ax, ay in self.checked if math.hypot(ax - tx, ay - ty) < SAME_TARGET_M]
                fresh = ~self.used_spots(looked, CHECKED_RADIUS)[rr, cc] if looked else True
                if floor:
                    # facing the piece's centre: its near and far edge as seen from the camera
                    r = np.percentile(np.hypot(*(cells - centre).T), 90) * g.res
                    cam = d - CAMERA_AHEAD
                    ok = (cam - r >= FLOOR_NEAR_M) & (cam + r <= FLOOR_FAR_M) & away & fresh
                    ideal = sum(FLOOR_BAND_M) / 2 + CAMERA_AHEAD
                else:
                    near, ideal, far = WALL_VIEW_M
                    ok = (d > near) & (d < far) & away & fresh
                if not ok.any():
                    # no free spot at the usual distance (all used, or walls around): anywhere the
                    # camera still sees well enough, rather than dropping the area
                    ok = (d > 0.5) & (d < (FLOOR_RANGE if floor else self.p["camera_range"])) & away & fresh
                if not ok.any():
                    continue
                k = np.argmin(np.where(ok, dist[rr, cc] + 20 * np.abs(d - ideal) / g.res, np.inf))
                goals.append(((rr[k], cc[k]), len(cells), (cells[:, 0], cells[:, 1])))
        return goals

    def tiles(self, cells, size):
        """An unseen area split into pieces of about size m (fixed squares of the map, so the pieces stay
        put between re-plans); pieces below min_surface join the nearest larger one."""
        t = int(size / self.grid.res)
        if np.ptp(cells, axis=0).max() <= t:
            return [cells]
        keys, inv = np.unique(cells // t, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        counts = np.bincount(inv, minlength=len(keys))
        big = np.nonzero(counts >= self.min_surface())[0]
        if len(big) <= 1:
            return [cells]
        centres = np.array([cells[inv == j].mean(0) for j in range(len(keys))])
        for j in np.nonzero(counts < self.min_surface())[0]:
            inv[inv == j] = big[np.argmin(np.hypot(*(centres[big] - centres[j]).T))]
        return [cells[inv == j] for j in big]

    def choose_goal(self, pose):
        g = self.grid
        start = g.to_cell(pose[0], pose[1])
        if not g.inside(*start):
            return None, None
        dist, parent = self.cost_field(start)
        if not self.update_region(start) and self._free_region is not None:
            # on the wrong side of a no-go line drawn meanwhile: back to the start's side (crossing
            # the line once), to the nearest spot there with enough room
            dist, parent = self.cost_field(start, ignore_walls=True)
            # well inside (>= 45 cm from the line and obstacles), not right at the line
            back = np.where(self._free_region & (self.clearance >= 0.45), dist, np.inf)
            if not np.isfinite(back).any():
                back = np.where(self._free_region & (self.clearance >= MIN_CLEARANCE), dist, np.inf)
            if np.isfinite(back).any():
                rc = np.unravel_index(np.argmin(back), back.shape)
                self.status.phase = "returning"
                return (int(rc[0]), int(rc[1])), self.path_to(parent, (int(rc[0]), int(rc[1])))

        # committed to a viewpoint: keep it until it is reached, its surfaces have been seen on the
        # way, or it is given up. (Re-choosing every second made the robot swing between two
        # viewpoints: each one drops out of the candidates when the robot gets close to it.)
        st = self.status
        if st.phase == "coverage" and st.goal is not None and self.goal_cluster is not None:
            rr, cc = self.goal_cluster
            ok = (rr < g.cells.shape[0]) & (cc < g.cells.shape[1])
            grc = g.to_cell(*st.goal)
            if (~self.seen[rr[ok], cc[ok]]).sum() >= MIN_SURFACE // 2 and g.inside(*grc) and np.isfinite(dist[grc]):
                return grc, self.path_to(parent, grc)

        def allowed(rc, checked=True):
            x, y = g.to_xy(*rc)
            return (all(math.hypot(x - bx, y - by) > 1.0 for bx, by in self.blacklist)
                    and not (checked and any(math.hypot(x - c[0], y - c[1]) <= CHECKED_RADIUS for c in self.checked)))

        frontiers = [f for f in self.frontier_goals(dist, start) if allowed(f[0])]
        self.status.frontiers = len(frontiers)
        self.status.coverage = self.coverage()
        views = []
        if self.status.coverage < COVERAGE_COMPLETE:
            # (coverage_goals already left out the spots looked from at the same target)
            views = [v for v in self.coverage_goals(dist, start) if allowed(v[0], checked=False)]
        self.status.remaining = len(views)
        if not frontiers and not views:
            return None, None
        clusters = {v[0]: v[2] for v in views}
        if views:
            # the camera first: all known unseen surfaces, along one route; its first viewpoint is next
            route = sorted(views, key=lambda v: dist[v[0]])[:ROUTE_MAX]
            first = self.plan_route(start, [v[0] for v in route])
            cands = [(first, 0.0)]
            kinds = {first: "coverage"}
        else:
            # every known surface seen: on to the nearest frontier (large ones count more)
            cands = [(f[0], dist[f[0]] / (1.0 + 0.1 * f[1])) for f in frontiers]
            kinds = {f[0]: "frontiers" for f in frontiers}
        best = min(cands, key=lambda c: c[1])
        # hysteresis: keep heading for (a candidate near) the current goal unless the best is clearly
        # better; the frontiers change with every map update and the robot would zig-zag otherwise
        cur = self.status.goal
        if cur is not None:
            near_cur = [c for c in cands if math.hypot(g.to_xy(*c[0])[0] - cur[0], g.to_xy(*c[0])[1] - cur[1]) < 1.0]
            if near_cur:
                keep = min(near_cur, key=lambda c: c[1])
                if keep[1] <= SWITCH_GAIN * best[1]:
                    best = keep
        self.status.phase = kinds[best[0]]
        self.goal_cluster = clusters.get(best[0]) if self.status.phase == "coverage" else None
        if self.goal_cluster is not None:  # views before the visit: what the visit adds is not a failure
            rr, cc = self.goal_cluster
            self.goal_views = self.views[rr, cc].copy()
        return best[0], self.path_to(parent, best[0])

    def slide_point(self, x, y, target):
        """Where to slide to at a viewpoint (several views of the same surfaces): SLIDE_M sideways,
        across the line of sight, to the side with room for it; None if neither side is free."""
        g = self.grid
        tx, ty = target
        d = math.hypot(tx - x, ty - y)
        if d < 1e-6:
            return None
        px, py = -(ty - y) / d, (tx - x) / d  # perpendicular to the line of sight
        best = None
        for sgn in (1, -1):
            ex, ey = x + sgn * SLIDE_M * px, y + sgn * SLIDE_M * py
            ok = True
            for t in np.linspace(0.2, 1.0, 5):
                r, c = g.to_cell(x + t * (ex - x), y + t * (ey - y))
                if not g.inside(r, c) or self.clearance[r, c] < MIN_CLEARANCE:
                    ok = False
                    break
            if ok:
                r, c = g.to_cell(ex, ey)
                if best is None or self.clearance[r, c] > best[0]:
                    best = (self.clearance[r, c], (ex, ey))
        return best[1] if best else None

    def route_distances(self, start, cells):
        """Driving distances (m) between the robot (row 0) and the viewpoints (rows 1..), on a grid of
        ROUTE_CELL x ROUTE_CELL cells (a block is passable if any of its cells is): one Dijkstra from
        every point at once, around walls instead of straight lines."""
        g, k = self.grid, ROUTE_CELL
        h, w = g.cells.shape
        H, W = -(-h // k), -(-w // k)
        ok_f = np.zeros((H * k, W * k), bool)
        ok_f[:h, :w] = self.clearance >= MIN_CLEARANCE
        ok = ok_f.reshape(H, k, W, k).any(axis=(1, 3))
        pts = [(r // k, c // k) for r, c in [start] + list(cells)]
        for r, c in pts:
            ok[r, c] = True  # the robot's own block (its footprint is unknown in the lidar map)
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, wts = [], [], []
        for dr, dc, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a_ = (slice(max(0, -dr), H - max(0, dr)), slice(max(0, -dc), W - max(0, dc)))
            b_ = (slice(max(0, dr), H - max(0, -dr) if dr < 0 else H), slice(max(0, dc), W - max(0, -dc) if dc < 0 else W))
            both = ok[a_] & ok[b_]
            rows.append(idx[a_][both])
            cols.append(idx[b_][both])
            wts.append(np.full(both.sum(), step * k * g.res))
        graph = sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))
        flat = [r * W + c for r, c in pts]
        d = csgraph.dijkstra(graph, directed=False, indices=flat)
        return d[:, flat]

    def plan_route(self, start, cells):
        """Order of the viewpoints: an open route from the robot through all of them (nearest first,
        then 2-opt: reverse segments while that shortens it), with real driving distances. The
        previous route is kept (new viewpoints inserted where they cost least) unless a new one is
        clearly shorter, so the order doesn't flip as the map grows. Returns the first viewpoint."""
        g = self.grid
        D = self.route_distances(start, cells)
        D[~np.isfinite(D)] = 1e3
        n = len(cells)

        def length(order):
            return D[0, order[0] + 1] + sum(D[a + 1, b + 1] for a, b in zip(order, order[1:]))

        def two_opt(order):
            improved = True
            while improved:
                improved = False
                for i in range(len(order) - 1):
                    for j in range(i + 1, len(order)):
                        new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                        if length(new) < length(order) - 1e-6:
                            order, improved = new, True
            return order

        # fresh: nearest neighbour from the robot, then 2-opt
        order, cur, rest = [], 0, set(range(n))
        while rest:
            j = min(rest, key=lambda k: D[cur, k + 1])
            order.append(j)
            rest.remove(j)
            cur = j + 1
        fresh = two_opt(order)
        # kept: the previous route's viewpoints that still exist (matched within 0.5 m), in their order
        xy = [g.to_xy(*c) for c in cells]
        kept = []
        for px, py in getattr(self, "route", []):
            m = min(range(n), key=lambda k: math.hypot(xy[k][0] - px, xy[k][1] - py))
            if math.hypot(xy[m][0] - px, xy[m][1] - py) < 0.5 and m not in kept:
                kept.append(m)
        for j in [k for k in range(n) if k not in kept]:  # new viewpoints: cheapest insertion
            best = min(range(len(kept) + 1), key=lambda i: length(kept[:i] + [j] + kept[i:]))
            kept.insert(best, j)
        kept = two_opt(kept)
        order = fresh if length(fresh) < ROUTE_SWITCH * length(kept) else kept
        self.route = [xy[k] for k in order]
        self.route_length = length(order)
        return cells[order[0]]

    # ---------- control ----------

    def gaze_target(self, pose, travel):
        """Direction to look: most unseen surface cells within reach, near the travel direction."""
        g = self.grid
        x, y, _ = pose
        r0, c0 = g.to_cell(x, y)
        k = int(self.p["camera_range"] / g.res)
        sl = (slice(max(0, r0 - k), r0 + k + 1), slice(max(0, c0 - k), c0 + k + 1))
        unseen = self.surfaces()[sl] & ~self.seen[sl]
        rr, cc = np.nonzero(unseen)
        if len(rr) == 0:
            return travel
        px, py = g.origin_x + (cc + sl[1].start + 0.5) * g.res, g.origin_y + (rr + sl[0].start + 0.5) * g.res
        d = np.hypot(px - x, py - y)
        keep = (d < self.p["camera_range"]) & (d > 0.3)
        if not keep.any():
            return travel
        ang = np.arctan2(py[keep] - y, px[keep] - x)
        rel = np.angle(np.exp(1j * (ang - travel)))
        ok = np.abs(rel) <= GAZE_MAX + CAMERA_FOV / 2
        if not ok.any():
            return travel
        # the window of one field of view with the most unseen cells
        cands = np.linspace(-GAZE_MAX, GAZE_MAX, 15)
        counts = np.array([(np.abs(rel[ok] - a) < CAMERA_FOV / 2).sum() for a in cands])
        best = travel + cands[int(np.argmax(counts))]
        # hysteresis: keep looking where we look unless another direction is clearly better (two
        # equally good directions made the robot swing back and forth)
        if self.gaze is not None:
            prev_rel = math.atan2(math.sin(self.gaze[0] - travel), math.cos(self.gaze[0] - travel))
            if abs(prev_rel) <= GAZE_MAX:
                prev_count = (np.abs(rel[ok] - prev_rel) < CAMERA_FOV / 2).sum()
                if prev_count >= 0.75 * counts.max():
                    return self.gaze[0]
        return best

    def blocked_ahead(self, scan_xy, direction, distance=STOP_DISTANCE):
        """Lidar points (robot frame) close to the robot in the given travel direction. Points inside
        the robot's own outline are lidar noise (seen: 3 cm at +104 deg, stopping sideways drives)."""
        if scan_xy is None or len(scan_xy) == 0:
            return False
        d = np.hypot(scan_xy[:, 0], scan_xy[:, 1])
        keep = d > ROBOT_RADIUS
        scan_xy, d = scan_xy[keep], d[keep]
        a = np.arctan2(scan_xy[:, 1], scan_xy[:, 0])
        rel = np.abs(np.angle(np.exp(1j * (a - direction))))
        # a corridor as wide as the robot, STOP_DISTANCE long
        lateral = d * np.sin(np.minimum(rel, math.pi / 2))
        return bool(np.any((rel < math.pi / 2) & (d * np.cos(rel) < distance) & (lateral < ROBOT_RADIUS + 0.03)))

    def give_up_goal(self, reached=False, pose=None):
        """The current goal is done with. Not reached: never again this viewpoint. Reached (looked
        at its surfaces): not this spot again (neighbouring ones may follow). In the coverage phase, the surface cells the
        visit added no view to count one more attempt (given up after as many as the mode needs views)."""
        if self.status.goal is not None:
            if reached:
                if self.goal_cluster is not None:
                    tx, ty = self.grid.to_xy(np.mean(self.goal_cluster[0]), np.mean(self.goal_cluster[1]))
                else:
                    tx, ty = self.status.goal
                self.checked.append((*self.status.goal, tx, ty))
            else:
                self.blacklist.append(self.status.goal)
        if self.status.phase == "coverage" and self.goal_cluster is not None and self.attempts is not None:
            rr, cc = self.goal_cluster
            if not reached:
                # not reached (stuck, blocked): the surfaces get a second try from another viewpoint
                # before they count as given up (Quick gave a surface up after one stuck attempt)
                cx, cy = self.grid.to_xy(rr.mean(), cc.mean())
                if not any(math.hypot(cx - ux, cy - uy) < 1.0 for ux, uy in self.unreached):
                    self.unreached.append((cx, cy))
                    self.goal_cluster = None
                    return
            ok = (rr < self.attempts.shape[0]) & (cc < self.attempts.shape[1])
            three_d = self.p.get("coverage3d") and self.seen3d is not None
            if self.goal_views is not None and len(self.goal_views) == len(rr) and not three_d:
                # only the cells the visit added no view to count as a failed attempt (the others are
                # on their way to the mode's number of views). Not in 3D: these 2D ray counts grow at
                # every visit near a wall or furniture, so those cells never failed, were never given up
                # and drew the robot back forever (run 20261007_093027: 113 visits in 10 min, 0.3 m2
                # given up, stopped by hand). There a cell still unseen after a visit that had it in
                # view is one failed try.
                ok &= self.views[np.minimum(rr, self.views.shape[0] - 1), np.minimum(cc, self.views.shape[1] - 1)] <= self.goal_views
            if reached and pose is not None:
                # ... and only those the camera could have seen from here (in range, in the picture
                # while facing the cluster): the rest of a long wall gets its own viewpoint
                g = self.grid
                px, py = g.origin_x + (cc + 0.5) * g.res, g.origin_y + (rr + 0.5) * g.res
                x, y = pose[0], pose[1]
                facing = math.atan2(py.mean() - y, px.mean() - x)
                rel = np.angle(np.exp(1j * (np.arctan2(py - y, px - x) - facing)))
                cells = g.cells[np.minimum(rr, g.cells.shape[0] - 1), np.minimum(cc, g.cells.shape[1] - 1)]
                reach = np.where(cells == FREE, FLOOR_RANGE, self.p["camera_range"])
                ok &= (np.hypot(px - x, py - y) <= reach) & (np.abs(rel) <= CAMERA_FOV / 2)
            self.attempts[rr[ok], cc[ok]] = np.minimum(self.attempts[rr[ok], cc[ok]].astype(np.int16) + 1, 255).astype(np.uint8)
        self.goal_cluster = None

    def plan_due(self, now):
        return self.status.state == "exploring" and self.grid is not None and self.spin is None and (
            now - self.last_plan > (0.5 if self.status.goal is None else 1.0))

    def plan(self, now, pose):
        """Chooses the goal and the path (the slow part: ~0.1-0.8 s; the ROS node runs it in a thread)."""
        st = self.status
        self.last_plan = now
        goal, path = self.choose_goal(pose)
        if st.state != "exploring":
            return  # paused/stopped meanwhile
        if goal is not None:
            self.no_goal_since = None
        if goal is None:
            if not self.looked_around:
                self.spin = {"turned": 0.0, "last": None, "since": None}
                st.goal, st.path, st.phase = None, [], "looking around"
                st.message = "no goal in sight: turning once to look around"
                self.event("no goal in sight: turning once on the spot to check")
                return
            # wait a few map updates before calling it done (the map may lag behind, e.g. right
            # after the look-around turn)
            if self.no_goal_since is None:
                self.no_goal_since = now
            if now - self.no_goal_since < DONE_AFTER_S:
                st.goal, st.path, st.message = None, [], "no goal yet: waiting for the map to catch up"
                return
            st.state, st.goal, st.path = "done", None, []
            gone = self.given_up_cells() * self.grid.res
            # 3D also gives up floor (an area): report m2 there, metres of wall otherwise (2D stand-in)
            gone_scale, gone_unit = (self.grid.res, "m2") if self.p.get("coverage3d") else (1.0, "m")
            vis = self.coverage_visible()
            st.message = (f"room done: no unexplored areas left; the camera saw {100 * self.coverage():.0f} % of the surfaces"
                          + (f" ({100 * vis:.0f} % of what was in view)" if vis is not None else "")
                          + (f", {gone * gone_scale:.1f} {gone_unit} given up (not seen well enough from its viewpoint, "
                             f"or not reachable)" if gone >= 0.1 else "")
                          + f"; the rest are small pieces (< {100 * self.min_surface() * self.grid.res:.0f} cm)")
            self.event(st.message)
            return
        self.looked_around = False
        gx, gy = self.grid.to_xy(*goal)
        if st.goal is None or math.hypot(gx - st.goal[0], gy - st.goal[1]) > 1.0:
            self.goal_since, self.best_remaining = now, math.inf
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([pose[:2]] + path[:-1], path))
            what = {"frontiers": "new area", "coverage": "viewpoint for unseen surfaces",
                    "returning": "back to the start's side"}.get(st.phase, st.phase)
            self.event(f"next goal: {what}, {length:.1f} m away")
        st.goal, st.path = (gx, gy), path

    def step(self, now, pose, scan_xy, plan_inline=True):
        """One control cycle (~10 Hz). Returns (vx, vy, wz) in the robot frame. With
        plan_inline=False the caller runs plan() itself (in a thread) when plan_due()."""
        st = self.status
        if st.state != "exploring" or self.grid is None:
            return 0.0, 0.0, 0.0
        if self.home is None:
            self.home = (pose[0], pose[1])
        self.update_coverage(pose)
        x, y, yaw = pose
        if self.spin is not None:
            return self.turn_around(yaw, now)
        if plan_inline and self.plan_due(now):
            self.plan(now, pose)
            if self.spin is not None:
                return self.turn_around(yaw, now)
        if st.state != "exploring" or st.goal is None:  # (an empty path at the goal: the arrival below)
            return 0.0, 0.0, 0.0
        # stall watchdog: moving but learning nothing (e.g. dithering in a narrow spot)
        known, seen = int((self.grid.cells != UNKNOWN).sum()), int((self.seen & self._surfaces).sum())
        if known > self.progress[0] + 20 or seen > self.progress[1] + 5 or self.progress[2] == 0.0:
            self.progress = (known, seen, now)
        elif now - self.progress[2] > STALL_S:
            self.give_up_goal()
            self.progress = (known, seen, now)
            st.message = "no progress here: trying elsewhere"
            self.event("no new map or coverage for 30 s here: giving this spot up")
            st.goal = None
            return 0.0, 0.0, 0.0
        # progress check: give a goal up if we stop getting closer
        remaining = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([(x, y)] + st.path[:-1], st.path))
        if remaining < self.best_remaining - 0.05:
            self.best_remaining, self.goal_since = remaining, now
        elif now - self.goal_since > NO_PROGRESS_S:
            self.give_up_goal()
            st.message = "goal not reachable, trying another"
            self.event("goal not reachable (no progress for 15 s): giving it up")
            st.goal = None
            return 0.0, 0.0, 0.0
        if math.hypot(st.goal[0] - x, st.goal[1] - y) < GOAL_TOLERANCE or not st.path:
            if st.phase == "coverage" and self.goal_cluster is not None:
                # at the viewpoint: face the surfaces for a moment, then give the rest of the cluster up
                if self.look is None:
                    rr, cc = self.goal_cluster
                    target = self.grid.to_xy(rr.mean(), cc.mean())
                    slide = self.slide_point(x, y, target) if self.p["views"] > 1 else None
                    until = now + (max(LOOK_S, SLIDE_M / SLIDE_SPEED + 2.0) if slide else LOOK_S)
                    self.look = {"until": until, "target": target, "slide": slide}
                if now < self.look["until"]:
                    tx, ty = self.look["target"]
                    bearing = math.atan2(ty - y, tx - x)
                    err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
                    st.message = "looking at surfaces not seen yet"
                    vx = vy = 0.0
                    if self.look["slide"] is not None:
                        # several views of the same surfaces: slide along them, camera on them
                        sx, sy = self.look["slide"]
                        d = math.hypot(sx - x, sy - y)
                        rel = math.atan2(sy - y, sx - x) - yaw
                        lo, hi = LIDAR_SECTOR
                        rel = math.atan2(math.sin(rel), math.cos(rel))
                        if d < 0.05 or not (lo <= rel <= hi) or self.blocked_ahead(scan_xy, rel):
                            self.look["slide"] = None  # there, or not safe: finish looking from here
                            self.look["until"] = min(self.look["until"], now + 1.0)
                        else:
                            v = min(SLIDE_SPEED, 0.5 * d)
                            vx, vy = v * math.cos(rel), v * math.sin(rel)
                    return vx, vy, self.smooth_turn(1.5 * err, limit=self.p.get("look_turn"))
                self.look = None
            if st.phase == "coverage":
                self.give_up_goal(reached=True, pose=pose)  # looked at it; not again from here
                self.event("viewpoint checked")
            st.goal = None
            return 0.0, 0.0, 0.0
        # pure pursuit: first path point at least LOOKAHEAD away
        target = st.path[-1]
        for px, py in st.path:
            if math.hypot(px - x, py - y) >= LOOKAHEAD:
                target = (px, py)
                break
        travel = math.atan2(target[1] - y, target[0] - x)
        dist_goal = math.hypot(st.goal[0] - x, st.goal[1] - y)
        speed = min(self.p["speed"], 0.6 * dist_goal + 0.05)
        # heading: look at unseen surfaces, but keep the travel direction inside the lidar's view
        if self.gaze is None or now - self.gaze[1] > GAZE_EVERY_S:
            self.gaze = (self.gaze_target(pose, travel), now)
        want = self.gaze[0]
        err = math.atan2(math.sin(want - yaw), math.cos(want - yaw))
        wz = 1.2 * err
        rel = math.atan2(math.sin(travel - yaw), math.cos(travel - yaw))  # travel direction, robot frame
        lo, hi = LIDAR_SECTOR
        margin = math.radians(15)
        if not (lo + margin <= rel <= hi - margin):
            speed = 0.0  # turn first: never drive towards the blind rear
            wz = 1.5 * rel
        elif abs(err) > math.radians(40):
            speed *= 0.5  # big turns: slower, sharper images
        if speed > 0.1 and self.blocked_ahead(scan_xy, rel, SLOW_DISTANCE):
            speed = max(0.1, 0.5 * speed)  # something close ahead: slower, stop only if really close
        if self.unstick is not None or self.is_stuck(now, x, y, speed > 0):
            return self.get_free(now, x, y, yaw, travel, scan_xy)
        wz = self.smooth_turn(wz)
        if speed > 0 and self.blocked_ahead(scan_xy, rel):
            if not st.blocked:
                self.event("obstacle ahead: stopped, re-planning around it")
            st.blocked = True
            st.message = "obstacle ahead: waiting / re-planning"
            self.last_plan = -math.inf  # re-plan now (the map will show the obstacle soon)
            return 0.0, 0.0, wz
        st.blocked = False
        st.message = {"frontiers": "exploring new areas", "coverage": "looking at surfaces not seen yet",
                      "returning": "back to the start's side of the no-go line"}.get(st.phase, "exploring")
        return speed * math.cos(rel), speed * math.sin(rel), wz

    def is_stuck(self, now, x, y, driving):
        """Over the last STUCK_S: told to drive for at least STUCK_DRIVE_S, but the SLAM pose (not the
        wheels: slipping wheels report motion) stayed within STUCK_M of where it was."""
        log = self.drive_log
        if log and now - log[-1][0] > 1.0:
            log.clear()  # a gap (looking at a viewpoint, a new goal): start over
        log.append((now, x, y, driving))
        while log and now - log[0][0] > STUCK_S:
            log.pop(0)
        if now - log[0][0] < STUCK_S - 0.3:
            return False
        drove = sum(b[0] - a[0] for a, b in zip(log, log[1:]) if a[3])
        return drove >= STUCK_DRIVE_S and max(math.hypot(px - x, py - y) for _, px, py, _ in log) < STUCK_M

    def get_free(self, now, x, y, yaw, travel, scan_xy):
        """Stuck (e.g. at a floor transition): turn to face the travel direction (mecanum wheels
        push hardest forward), push forward at full speed for PUSH_S; still stuck: give the goal up."""
        st = self.status
        u = self.unstick
        if u is None:
            u = self.unstick = {"phase": "turn", "since": now, "start": (x, y)}
            self.event("stuck (wheels slipping?): turning to push forward")
        self.drive_log.clear()
        err = math.atan2(math.sin(travel - yaw), math.cos(travel - yaw))
        st.message = "stuck: trying to get free"
        if u["phase"] == "turn":
            if abs(err) > math.radians(10) and now - u["since"] < 8.0:
                return 0.0, 0.0, self.smooth_turn(1.5 * err)
            u.update(phase="push", since=now, start=(x, y))
        if math.hypot(x - u["start"][0], y - u["start"][1]) > 0.10:
            self.unstick = None  # free again
            self.event("free again")
            return 0.0, 0.0, 0.0
        if now - u["since"] > PUSH_S or self.blocked_ahead(scan_xy, err):
            self.unstick = None
            self.give_up_goal()
            st.goal = None
            self.event("still stuck: giving this goal up")
            return 0.0, 0.0, 0.0
        v = self.p["speed"]
        return v * math.cos(err), v * math.sin(err), self.smooth_turn(1.5 * err)

    def turn_around(self, yaw, now):
        """One slow full turn on the spot (no translation: safe even next to obstacles). Gives up
        after SPIN_TIMEOUT_S, e.g. if the robot can't turn."""
        sp = self.spin
        if sp["since"] is None:
            sp["since"] = now
        if sp["last"] is not None:
            sp["turned"] += abs(math.atan2(math.sin(yaw - sp["last"]), math.cos(yaw - sp["last"])))
        sp["last"] = yaw
        if sp["turned"] >= 2 * math.pi - 0.15 or now - sp["since"] > SPIN_TIMEOUT_S:
            self.spin, self.looked_around = None, True
            self.last_plan = -math.inf
            self.status.goal = None
            return 0.0, 0.0, 0.0
        self.status.phase = "looking around"
        return 0.0, 0.0, self.smooth_turn(0.7 * self.p["turn"])

    def smooth_turn(self, wz, dt=0.1, limit=None):
        """Turn rate limited to the mode's maximum (or a lower limit) and changed by at most TURN_ACCEL
        per second."""
        top = min(self.p["turn"], limit) if limit else self.p["turn"]
        wz = max(-top, min(top, wz))
        step = TURN_ACCEL * dt
        self.wz = max(self.wz - step, min(self.wz + step, wz))
        return self.wz

    # ---------- commands ----------

    def command(self, cmd):
        st = self.status
        word, _, arg = cmd.partition(" ")
        if word == "walls":
            import json
            self.set_walls(json.loads(arg or "[]"))
            return
        if word == "start":
            self.p = MODES[MODE]  # one mode; older consoles may still send "start quick" etc.
            cmd = "start"
        if cmd == "start":
            if st.state in ("done", "stopped", "idle"):  # a fresh run: forget goals and coverage
                self.status, self.blacklist, self.gaze, self.home = Status(), [], None, None
                self.checked = []
                self.unstick, self.drive_log, self.unreached = None, [], []
                self.route = []
                self.goal_cluster = None
                if self.grid is not None:
                    self.attempts = np.zeros(self.grid.cells.shape, np.uint8)
                if self.grid is not None:
                    self.views = np.zeros(self.grid.cells.shape, np.uint8)
                    self.seen = np.zeros(self.grid.cells.shape, bool)
                self.last_view = None
                self.seen3d = self.surface3d = None  # the node starts a fresh 3D coverage too
                self.spin, self.looked_around = {"turned": 0.0, "last": None, "since": None}, False  # fill the blind rear first
            self.status.state, self.status.message = "exploring", "exploring"
            self.event("started: first a turn on the spot")
            self.status.mode = MODE
        elif cmd == "pause" and st.state == "exploring":
            st.state, st.message = "paused", "paused"
            self.event("paused")
        elif cmd == "resume" and st.state == "paused":
            st.state, st.message, st.goal = "exploring", "exploring", None
            self.event("resumed")
            if self.spin is not None:
                self.spin["last"] = self.spin["since"] = None  # continue the turn from where it is now
        elif cmd == "stop":
            st.state, st.goal, st.path, st.message = "stopped", None, [], "stopped"
            self.event("stopped (new map or SLAM stopped)")
