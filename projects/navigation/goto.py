"""Driving to an object in a known room: where to stop, the path there, following it. ROS-free (numpy/scipy),
so the navigator (navigator.py) and the simulator (test_goto.py) run the same code.

    g = GoTo(room.grid, room.resolution, room.origin, walls=room.nogo_lines())
    g.start(obj, pose)                       # obj: a scene.json object
    g.set_obstacles(points_map)              # lidar + depth points not in the room map (moved chairs, people)
    g.plan(pose)                             # ~0.1-0.3 s: the caller runs it in a thread, every PLAN_EVERY_S
    vx, vy, wz = g.step(now, pose, near_xy)  # ~10 Hz; near_xy: lidar + depth points in the robot frame
    g.state, g.message                       # planning | driving | docking | arrived | failed | stopped

Planner, safety check and turn smoothing are the explorer's (offboard_ros/explore_core.py, imported
unchanged): Dijkstra over the occupancy grid keeping MIN_CLEARANCE from obstacles and preferring
COMFORT_CLEARANCE, holonomic pure pursuit, stop when the lidar sees something close in the travel
direction. What is new here:
  - the goal: a spot GAP in front of one face of the object's box, facing it; of all faces and spots
    along them the one with the shortest path (an operable part, e.g. a drawer: only the faces pointing
    away from its cabinet)
  - heading: the travel direction while driving (the lidar sees ahead; mecanum wheels push best
    forwards), turning towards the object over the last APPROACH_M, so it arrives facing it
  - docking: the last GOAL_TOLERANCE straight to the pose, slowly; it ends early when the lidar/depth
    points straight ahead are GAP from the robot's front (the box from the scan can be 10-20 cm off,
    scene_understanding README)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "offboard_ros"))
import explore_core as ec  # noqa: E402

TOP_SPEED = 0.40       # m/s; the explorer drives 0.3, manual driving goes up to 0.5 (user: as fast as
                       # possible, 2026-10-07): raise after real runs
TURN_RATE = 0.8        # rad/s; tracking corrects ~6 deg per scan around odometry's prediction (localize.py)
GAP = 0.20             # m between the robot's front and the object (user, 2026-10-07)
HALF_LENGTH = 0.16     # m, robot centre to front (chassis 0.31 x 0.25 m: half diagonal 0.20, explore_core)
APPROACH_M = 1.0       # from this far on the robot turns to face the object
GOAL_TOLERANCE = 0.25  # m: then docking
DOCK_SPEED = 0.12
ARRIVED_M, ARRIVED_RAD = 0.04, math.radians(4)
SPOT_STEP = 0.20       # m between candidate spots along a face
SIDE_PENALTY = 1.0     # m of path: the ends of an elongated object are a worse front than its long sides
CLOSEST_M = 2.0        # no side reachable: stop at the nearest reachable spot within this of the object
CREEP_MAX = 0.8        # m: at the stop, an object seen straight ahead closer than this is driven up to (GAP)
FREE_AROUND_M = 0.30   # around the robot, live and remembered points never block the plan
NEAR_M = 0.6         # blocked this close to the stop: dock instead of waiting; stuck: as close as it got
PLAN_EVERY_S = 1.0
NO_PROGRESS_S = 15.0   # the path doesn't get shorter for this long: give up (explorer's value)
DOCK_TIMEOUT_S = 10.0
TURN_TIMEOUT_S = 20.0   # a half turn at TURN_RATE takes ~5 s
MIN_TURN_RATE = 0.15   # rad/s: slower and the wheels don't overcome friction near the end of a turn


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def approach_spots(obj):
    """Candidate stops (x, y, heading) in front of the faces of an object's upright box, facing it."""
    cx, cy = obj["center"][:2]
    length, width = obj["size"][:2]
    yaw = math.radians(obj["yaw_deg"])
    u = np.array([math.cos(yaw), math.sin(yaw)])   # along size[0]
    v = np.array([-u[1], u[0]])
    # Operable parts (drawers, cabinet doors; scene.json part_of) need no rule of their own: they are thin,
    # their long sides are front and back, and the back lies inside the whole, where the robot doesn't fit
    # (plan() drops such spots). Directions from the whole's box failed: a drawer near the end of a 4.6 m
    # kitchen counter's box was approached from the counter's end (2026-10-07 simulation)
    spots = []
    for n, depth, along, extent in ((u, length / 2, v, width / 2), (-u, length / 2, v, width / 2),
                                    (v, width / 2, u, length / 2), (-v, width / 2, u, length / 2)):
        # elongated objects (cabinets, drawers, wall fixtures) face the room with a long side: the ends
        # count SIDE_PENALTY metres of extra path (a power outlet on a wall was approached from its end,
        # along the wall, 2026-10-07 simulation)
        side = 2 * extent < 0.67 * max(length, width)
        k = max(1, int(2 * extent / SPOT_STEP))
        offsets = np.linspace(-extent, extent, k + 2)[1:-1] if k > 1 else [0.0]
        for t in offsets:
            p = np.array([cx, cy]) + n * (depth + GAP + HALF_LENGTH) + along * t
            spots.append((float(p[0]), float(p[1]), math.atan2(-n[1], -n[0]), SIDE_PENALTY if side else 0.0))
    return spots


class GoTo:
    def __init__(self, grid, resolution, origin, walls=()):
        """grid: the room's occupancy (-1 unknown, 0..100), rows = y."""
        cells = np.full(grid.shape, ec.UNKNOWN, np.uint8)
        cells[(grid >= 0) & (grid <= 25)] = ec.FREE
        cells[grid >= 65] = ec.OCC
        self.base = ec.Grid(cells, resolution, origin[0], origin[1])
        self.ex = ec.Explorer()
        self.ex.p = dict(self.ex.p, speed=TOP_SPEED, turn=TURN_RATE)  # own copy: MODES stays untouched
        self.ex.walls = [tuple(float(v) for v in w) for w in walls if len(w) == 4]
        self.extra = np.zeros(cells.shape, bool)  # obstacles not in the room map (set_obstacles)
        self.ex.set_map(self.base)
        self.clear()

    def clear(self):
        self.state, self.message = "idle", ""
        self.target, self.spots, self.goal, self.path = None, [], None, []
        self.since = self.best_left = self.progress_time = self.last_plan = None
        self.dock_since = None
        self.closest_only = False  # no side reachable: going as close as possible (closest_spot)
        self.creeping = False      # past the planned stop, towards the object the sensors see ahead
        self.is_point = False      # pin & go (start_point)
        self.turn_heading = None   # turning on the spot (start_turn)
        self.stop_on_find = True   # live detection's find ends this drive (not when approaching the find)
        # a find of the live detection (navigator.start_goto_target): small things on the floor are invisible to
        # the lidar and below the depth obstacles' band, so the distance comes from where it was seen; creeping
        # up to "something ahead" took a wall 2.4 m behind a pencil as the target's face (simulation, 2026-10-08)
        self.is_find = False

    # ---------- obstacles and planning ----------

    def set_obstacles(self, points):
        """Points in the map frame (latest lidar scan, depth obstacles): cells the room map shows free
        become obstacles for the next plan."""
        g = self.base
        mask = np.zeros(g.cells.shape, bool)
        if points is not None and len(points):
            p = np.asarray(points, float)
            c = np.floor((p[:, 0] - g.origin_x) / g.res).astype(int)
            r = np.floor((p[:, 1] - g.origin_y) / g.res).astype(int)
            ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
            mask[r[ok], c[ok]] = True
        self.extra = mask

    def start(self, obj, now):
        self.clear()
        self.target = obj
        self.spots = approach_spots(obj)
        self.state, self.message = "planning", f"planning the way to {obj['label']}"
        self.since = self.progress_time = now
        self.best_left = math.inf

    def start_turn(self, heading, label, now, object_id=None):
        """Turn on the spot to a heading (map frame), e.g. to face an object across the room after a drive
        (user, 2026-10-07: the robot could only drive somewhere, not face something else). Same safety as
        driving (navigator.drive_loop); nothing is planned, the robot doesn't move away."""
        self.clear()
        self.target = {"id": object_id, "label": label}
        self.turn_heading = heading
        self.state, self.message = "turning", f"turning to face {label}"
        self.since = now

    def start_point(self, x, y, now):
        """Pin & go: to a point on the floor; arrives facing its travel direction (no object to face, no
        creeping up to anything)."""
        self.start({"id": None, "label": "the pin", "center": [x, y, 0.0], "size": [0.0, 0.0, 0.0], "yaw_deg": 0.0}, now)
        self.spots = [(x, y, None, 0.0)]
        self.is_point = True

    def plan(self, pose, now):
        """Chooses the spot with the shortest path and the path to it."""
        g = ec.Grid(self.base.cells.copy(), self.base.res, self.base.origin_x, self.base.origin_y)
        rr, cc = g.to_cell(pose[0], pose[1])
        extra = self.extra.copy()
        # The robot's own surroundings never count as blocked by live or remembered points: inside its outline
        # they are lidar noise (explore_core.blocked_ahead), and just around it a remembered depth obstacle
        # boxed it in for good (real run 2026-10-07 14:51: every later plan failed). What is really there,
        # the collision check sees while driving.
        k = int(math.ceil(FREE_AROUND_M / g.res))
        extra[max(0, rr - k):rr + k + 1, max(0, cc - k):cc + k + 1] = False
        g.cells[extra & (g.cells == ec.FREE)] = ec.OCC
        self.ex.set_map(g)
        dist, parent = self.ex.cost_field((rr, cc))
        best = None
        for x, y, heading, penalty in self.spots:
            r, c = g.to_cell(x, y)
            if not g.inside(r, c) or self.ex.clearance_real[r, c] < ec.ROBOT_RADIUS + 0.02:
                continue  # the robot doesn't fit there (e.g. against the wall next to a wall fixture)
            if not np.isfinite(dist[r, c]):
                # not reachable itself (too close to something): the nearest reachable cell within 0.3 m;
                # docking covers the rest
                k2 = int(0.3 / g.res)
                sub = dist[max(0, r - k2):r + k2 + 1, max(0, c - k2):c + k2 + 1]
                if not np.isfinite(sub).any():
                    continue
                i = np.unravel_index(np.argmin(np.where(np.isfinite(sub), sub, np.inf)), sub.shape)
                r, c = max(0, r - k2) + i[0], max(0, c - k2) + i[1]
            cost = dist[r, c] + penalty / g.res  # dist counts cells
            if best is None or cost < best[0]:
                best = (cost, (r, c), (x, y, heading))
        self.last_plan = now
        self.closest_only = best is None
        if best is None:
            best = self.closest_spot(g, dist)
        if best is None:
            reach = int(np.isfinite(dist).sum()) * g.res ** 2
            self.state = "failed"
            self.message = (f"no way to {self.target['label']}: " + (
                f"the robot is boxed in (it can reach only {reach:.2f} m2 from here; clear the way and try "
                "again)" if reach < 0.5 else "nothing near it is reachable"))
            return
        _, rc, goal = best
        path = self.ex.path_to(parent, rc) + [goal[:2]]
        if goal[2] is None:  # a pin: arrive facing the way it came
            a = path[-2] if len(path) >= 2 else pose[:2]
            goal = (goal[0], goal[1], math.atan2(goal[1] - a[1], goal[0] - a[0]) if math.hypot(
                goal[0] - a[0], goal[1] - a[1]) > 0.02 else pose[2])
        self.goal = goal
        self.path = path
        if self.state == "planning":
            self.state = "driving"

    def closest_spot(self, g, dist):
        """No side of the object can be reached (a stool in the middle of others, a shoe under the table:
        17 of 80 simulated drives in room 20261007_101108): the reachable spot nearest to its box within
        CLOSEST_M, facing it. The robot then stops there and says how far it got."""
        o = self.target
        yaw = math.radians(o["yaw_deg"])
        rows, cols = np.nonzero(np.isfinite(dist))
        if not len(rows):
            return None
        x = g.origin_x + (cols + 0.5) * g.res
        y = g.origin_y + (rows + 0.5) * g.res
        dx, dy = x - o["center"][0], y - o["center"][1]
        lx, ly = np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy
        gap = np.hypot(np.maximum(np.abs(lx) - o["size"][0] / 2, 0), np.maximum(np.abs(ly) - o["size"][1] / 2, 0))
        near = gap < CLOSEST_M
        if not near.any():
            return None
        # nearest to the object first; among equally near spots (5 cm) the shorter path
        score = np.where(near, np.round(gap / 0.05) * 1e6 + dist[rows, cols], np.inf)
        i = int(np.argmin(score))
        heading = math.atan2(o["center"][1] - y[i], o["center"][0] - x[i])
        return score[i], (rows[i], cols[i]), (float(x[i]), float(y[i]), heading)

    def turn_step(self, now, pose):
        err = wrap(self.turn_heading - pose[2])
        if abs(err) < ARRIVED_RAD:
            self.state, self.message = "arrived", f"facing {self.target['label']}"
            return 0.0, 0.0, 0.0
        if now - self.since > TURN_TIMEOUT_S:
            self.state = "failed"
            self.message = f"could not turn to face {self.target['label']} ({math.degrees(err):.0f} deg left): blocked?"
            return 0.0, 0.0, 0.0
        wz = self.ex.smooth_turn(1.5 * err)
        if abs(wz) < MIN_TURN_RATE and abs(err) > ARRIVED_RAD:
            wz = math.copysign(MIN_TURN_RATE, err)
        self.message = f"turning to face {self.target['label']}, {math.degrees(abs(err)):.0f} deg to go"
        return 0.0, 0.0, wz

    def plan_due(self, now):
        return self.state in ("planning", "driving") and (self.last_plan is None or now - self.last_plan > PLAN_EVERY_S)

    def remaining(self, pose):
        pts = [pose[:2]] + [p[:2] for p in self.path]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    # ---------- driving ----------

    def front_gap(self, near_xy):
        """Distance from the robot's front to the nearest point straight ahead (robot frame, a corridor as
        wide as the robot), or inf."""
        if near_xy is None or not len(near_xy):
            return math.inf
        p = np.asarray(near_xy)
        ahead = (p[:, 0] > HALF_LENGTH - 0.02) & (np.abs(p[:, 1]) < 0.15)
        return float(p[ahead, 0].min() - HALF_LENGTH) if ahead.any() else math.inf

    def step(self, now, pose, near_xy):
        """One control cycle: (vx, vy, wz) in the robot frame."""
        if self.state == "turning":
            return self.turn_step(now, pose)
        if self.state not in ("driving", "docking") or self.goal is None:
            return 0.0, 0.0, 0.0
        x, y, yaw = pose
        gx, gy, gh = self.goal
        dist_goal = math.hypot(gx - x, gy - y)
        name = self.target["label"]
        # docking at the stop, or where the path had to end short of it (the stop itself is closer to
        # something than the planner's MIN_CLEARANCE; pushing towards it stalled at a wall, simulation)
        end = self.path[-2] if len(self.path) >= 2 else self.path[-1]
        at_end = math.hypot(end[0] - x, end[1] - y) < 0.10 and dist_goal < 0.5
        if self.state == "driving" and (dist_goal < GOAL_TOLERANCE or at_end):
            self.state, self.dock_since = "docking", now
        if self.state == "docking":
            return self.dock(now, pose, near_xy)

        left = self.remaining(pose)
        if left < self.best_left - 0.05:
            self.best_left, self.progress_time = left, now
        elif now - self.progress_time > NO_PROGRESS_S:
            if dist_goal < NEAR_M:
                self.state = "arrived"
                self.message = f"as close to {name} as it got: {self.box_distance(pose):.1f} m (something in the way)"
            else:
                self.state, self.message = "failed", f"no way to {name}: no progress for {NO_PROGRESS_S:.0f} s"
            return 0.0, 0.0, 0.0
        target = self.path[-1]
        for px, py in self.path:
            if math.hypot(px - x, py - y) >= ec.LOOKAHEAD:
                target = (px, py)
                break
        travel = math.atan2(target[1] - y, target[0] - x)
        # heading: the travel direction, blending into the goal heading over the last APPROACH_M
        w = min(1.0, max(0.0, (APPROACH_M - dist_goal) / (APPROACH_M - GOAL_TOLERANCE)))
        want = travel + w * wrap(gh - travel)
        err = wrap(want - yaw)
        rel = wrap(travel - yaw)  # travel direction in the robot frame
        speed = min(TOP_SPEED, 0.6 * dist_goal + 0.08)
        lo, hi = ec.LIDAR_SECTOR
        margin = math.radians(15)
        wz = 1.5 * err
        if not (lo + margin <= rel <= hi - margin):
            speed, wz = 0.0, 1.5 * rel  # turn first: never drive towards the blind rear
        elif abs(err) > math.radians(40):
            speed *= 0.5
        if speed > 0.1 and self.ex.blocked_ahead(near_xy, rel, ec.SLOW_DISTANCE + 0.5 * speed):
            speed = max(0.1, 0.5 * speed)  # something ahead: slower (the stop below if really close)
        wz = self.ex.smooth_turn(wz)
        # stopping distance grows with speed: the explorer's 0.27 m are for its 0.3 m/s
        stop = ec.STOP_DISTANCE + 0.4 * max(0.0, speed - 0.3)
        if speed > 0 and self.ex.blocked_ahead(near_xy, rel, stop):
            if dist_goal < NEAR_M:
                # close to the stop (next to other furniture: a shoe beside a stool leg waited until it gave
                # up, simulation): the slow docking, which goes only as far as nothing is in the way
                self.state, self.dock_since = "docking", now
                return 0.0, 0.0, wz
            self.message = f"obstacle ahead on the way to {name}: waiting / re-planning"
            self.last_plan = None  # re-plan now
            return 0.0, 0.0, wz
        self.message = f"driving to {name}, {left:.1f} m left"
        return speed * math.cos(rel), speed * math.sin(rel), wz

    def dock(self, now, pose, near_xy):
        """The last GOAL_TOLERANCE: straight to the stop and its heading, slowly; done there or when the
        object is GAP ahead."""
        x, y, yaw = pose
        gx, gy, gh = self.goal
        name = self.target["label"]
        herr = wrap(gh - yaw)
        dx, dy = gx - x, gy - y
        d = math.hypot(dx, dy)
        gap = self.front_gap(near_xy)
        facing = abs(herr) < math.radians(15)
        close = facing and gap <= GAP + 0.01  # the object is GAP ahead: no closer, only finish turning
        timeout = now - self.dock_since > DOCK_TIMEOUT_S
        # The box only gives the direction; the sensors give the distance (user, 2026-10-07: boxes from the
        # scan can be 10-20 cm off). At the planned stop, facing it, with something straight ahead but
        # further than GAP: creep forward until it is GAP ahead. Nothing ahead within CREEP_MAX (e.g. a
        # fixture above the lidar and the depth band): the planned stop is it.
        if not self.is_point and not self.is_find and not timeout and not close and abs(herr) < ARRIVED_RAD and (d < ARRIVED_M or self.creeping) \
                and GAP + 0.01 < gap < CREEP_MAX:
            self.creeping = True
            self.message = f"arriving at {name}: {gap:.2f} m to go"
            return min(DOCK_SPEED, gap - GAP + 0.02), 0.0, self.ex.smooth_turn(1.5 * herr)
        if ((d < ARRIVED_M or close or self.creeping) and abs(herr) < ARRIVED_RAD) or timeout:
            self.state = "arrived"
            if self.closest_only:
                o = self.target
                self.message = (f"as close to {name} as the robot fits: {self.box_distance(pose):.1f} m "
                                "(every side is blocked)")
            elif timeout and not close:
                self.message = f"near {name} ({d:.2f} m from the planned stop)"
            elif self.is_find:
                self.message = f"at {name}, {self.box_distance(pose):.2f} m in front of where it was seen"
            else:
                self.message = f"at {name}" + (f", {gap:.2f} m in front of it" if math.isfinite(gap) and not self.is_point else "")
            return 0.0, 0.0, 0.0
        if close:
            return 0.0, 0.0, self.ex.smooth_turn(1.5 * herr)
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy  # the stop in the robot frame
        v = min(DOCK_SPEED, 1.2 * d)
        if facing and math.isfinite(gap):
            v = min(v, max(0.0, gap - GAP) + 0.02)  # don't run into it
        vx, vy = (v * lx / d, v * ly / d) if d > 1e-3 else (0.0, 0.0)
        if d > 0.02 and self.ex.blocked_ahead(near_xy, math.atan2(vy, vx), ec.ROBOT_RADIUS + 0.03) \
                and not (facing and abs(math.atan2(vy, vx)) < math.radians(30)):
            vx = vy = 0.0  # sideways into something: only turn
        self.message = f"arriving at {name}"
        return vx, vy, self.ex.smooth_turn(1.5 * herr)

    def box_distance(self, pose):
        """From the robot's front to the object's box footprint."""
        o = self.target
        yaw = math.radians(o["yaw_deg"])
        fx, fy = pose[0] + HALF_LENGTH * math.cos(pose[2]), pose[1] + HALF_LENGTH * math.sin(pose[2])
        dx, dy = fx - o["center"][0], fy - o["center"][1]
        lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        return math.hypot(max(abs(lx) - o["size"][0] / 2, 0), max(abs(ly) - o["size"][1] / 2, 0))

    def found(self, label, position=None):
        """Live detection saw what is being looked for: stop here (navigator.on_found)."""
        if self.state in ("planning", "driving", "docking", "turning"):
            where = f" at ({position[0]:.2f}, {position[1]:.2f})" if position else ""
            self.state, self.message = "found", f"stopped: {label} seen{where}"

    def stop(self, why="stopped"):
        if self.state in ("planning", "driving", "docking", "turning"):
            self.state, self.message = "stopped", f"{why} on the way to {self.target['label']}"
