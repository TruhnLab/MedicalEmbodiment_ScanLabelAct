"""Autonomous exploration node (offboard container): wraps explore_core.Explorer.

In:  /map (slam_toolbox), /scan, TF map -> base_footprint, /explore/command (std_msgs/String:
     start | pause | resume | stop, from the console's Map view)
Out: /cmd_vel_auto (Twist, robot frame; relay.py forwards it to the robot's /controller/cmd_vel),
     /explore/status (String, JSON, 2 Hz), /explore/path (Path), /explore/coverage (OccupancyGrid:
     100 = surface the camera saw, 50 = surface not seen yet)

Safety: zero velocity whenever the pose or scan is older than 0.5 s, when paused/stopped, and
from the core's lidar stop. relay.py additionally sends zero if these commands stop arriving.
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

import coverage3d as c3
import depth_obstacles as do
import explore_core as ec

CAMERA_CORRECTION = "/config/camera_correction.json"  # repo's config/, mounted read-only

RATE_HZ = 10.0
STALE_S = 0.5
NO_DEPTH_S = 15.0  # exploring without depth images this long (from the start or resume): pause
POSE_STALE_S = 1.0  # SLAM pose: a new time stamp every ~0.1 s, at most 0.37 s apart (measured 2026-10-05)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ExplorerNode(Node):
    def __init__(self):
        super().__init__("explorer")
        self.ex = ec.Explorer()
        self.lock = threading.Lock()       # held by the control cycle and while swapping maps
        self.pending_map = None
        self.scan_xy, self.scan_time = None, 0.0
        self.pose, self.pose_time = None, 0.0
        self.lidar_offset = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(String, "/explore/command", self.on_command, 10)
        # depth camera (relay.py, ~4 Hz): obstacles above the lidar's slice (depth_obstacles.py)
        self.depth_K, self.depth_mem, self.depth_count = None, do.DepthMemory(), 0
        self.depth_fix = self.load_camera_correction()
        self.floor = c3.FloorHeight()  # the floor's height, from every depth image (coverage + obstacles)
        self.cov3d = c3.VoxelCoverage(floor=self.floor)  # Quick 3D mode: what the camera saw well enough, in 3D
        self.create_subscription(CameraInfo, "/depth/camera_info", self.on_depth_info, 1)
        self.create_subscription(CompressedImage, "/depth/compressed", self.on_depth, 1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_auto", 10)
        self.status_pub = self.create_publisher(String, "/explore/status", 10)
        self.path_pub = self.create_publisher(Path, "/explore/path", 10)
        self.cov_pub = self.create_publisher(OccupancyGrid, "/explore/coverage", 1)
        self.event_pub = self.create_publisher(String, "/explore/events", 50)
        self.create_timer(0.5, self.publish_events)
        self.create_timer(1.0 / RATE_HZ, self.control)
        self.create_timer(0.5, self.publish_status)
        self.create_timer(2.0, self.publish_coverage)
        self.create_timer(5.0, self.log_progress)
        self.was_moving = False
        self.last_cmd = (0.0, 0.0, 0.0)
        self.explore_since = time.monotonic()
        threading.Thread(target=self.planner, daemon=True).start()

    # ---------- inputs ----------

    def on_map(self, msg):
        info = msg.info
        self.pending_map = ec.Grid.from_ros(msg.data, info.width, info.height, info.resolution,
                                            info.origin.position.x, info.origin.position.y)

    def on_scan(self, msg):
        if self.lidar_offset is None:
            try:
                t = self.buffer.lookup_transform("base_footprint", msg.header.frame_id, Time())
                self.lidar_offset = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
            except TransformException:
                return
        r = np.array(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < min(msg.range_max, 6.0))
        ox, oy, oyaw = self.lidar_offset
        self.scan_xy = np.stack([ox + r[ok] * np.cos(a[ok] + oyaw), oy + r[ok] * np.sin(a[ok] + oyaw)], 1)
        self.scan_time = time.monotonic()

    # ---------- depth obstacles ----------

    def load_camera_correction(self):
        """The camera's measured mounting error (pitch/roll, survey pose), as reconstruct.py applies it:
        2.8 deg of pitch is 10 cm of height at 2 m, enough to see the floor as an obstacle."""
        try:
            c = json.load(open(CAMERA_CORRECTION))
        except (OSError, ValueError):
            self.get_logger().warning(f"no {CAMERA_CORRECTION}: depth obstacles without camera correction")
            return None
        rx, ry = math.radians(c["roll_deg"]), math.radians(c["pitch_deg"])
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        return Ry @ Rx, np.array(c["base_to_depth_rotation"])

    def on_depth_info(self, msg):
        self.depth_K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    @staticmethod
    def matrix(t):
        q, p = t.transform.rotation, t.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def on_depth(self, msg):
        # never let the depth path take the explorer down: without it the robot drives on lidar alone
        try:
            self.process_depth(msg)
        except Exception as e:
            if not getattr(self, "depth_error_logged", False):
                self.depth_error_logged = True
                self.get_logger().error(f"depth obstacles disabled for this image stream: {type(e).__name__}: {e}")

    def process_depth(self, msg):
        if self.depth_K is None:
            return
        import cv2
        depth = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            return
        try:  # the camera's pose at the image's time (it sits on the arm)
            stamp = Time.from_msg(msg.header.stamp)
            B = self.matrix(self.buffer.lookup_transform("base_footprint", msg.header.frame_id, stamp))
            T_map_base = self.matrix(self.buffer.lookup_transform("map", "base_footprint", stamp))
        except TransformException:
            return
        if self.depth_fix is not None:
            R, ref = self.depth_fix
            cos = (np.trace(ref.T @ B[:3, :3]) - 1) / 2
            if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= 3:  # survey pose, as calibrated
                B = B.copy()
                B[:3, :3] = R @ B[:3, :3]
        T_map_cam = T_map_base @ B
        self.floor.add(depth, self.depth_K, T_map_cam)
        cells = do.obstacle_cells(depth, self.depth_K, T_map_cam, self.floor.z)
        self.depth_mem.add(cells, time.monotonic())
        self.depth_count += 1
        self.depth_time = time.monotonic()
        if self.ex.p.get("coverage3d") and self.ex.status.state == "exploring":
            self.cov3d.add(depth, self.depth_K, T_map_cam)  # ~13 ms (measured offline)

    def depth_points(self, now):
        """Confirmed depth obstacles near the robot, in the robot frame (added to the lidar points)."""
        if self.pose is None:
            return None
        cells = self.depth_mem.confirmed(now)
        if not cells:
            return None
        xy = do.DepthMemory.centres(cells)
        x, y, yaw = self.pose
        dx, dy = xy[:, 0] - x, xy[:, 1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.stack([c * dx + s * dy, -s * dx + c * dy], 1)
        return local[np.hypot(local[:, 0], local[:, 1]) < 2.0]

    def on_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd.startswith("start") and self.ex.status.state in ("done", "stopped", "idle"):
            self.depth_mem = do.DepthMemory()  # a fresh run: forget the old depth obstacles
            self.ex.depth_obstacles = []
            mode = ec.MODES[ec.MODE]
            self.cov3d = c3.VoxelCoverage(max_range=mode["camera_range"], floor=self.floor)  # the mode's camera range, as in 2D
        if cmd.startswith("walls"):
            # no-go lines: stored now, drawn into the map by the planning thread (the only one that
            # swaps the map)
            self.ex.walls = [tuple(float(v) for v in w) for w in json.loads(cmd[5:].strip() or "[]") if len(w) == 4]
            if self.ex.raw is not None and self.pending_map is None:
                self.pending_map = self.ex.raw
            self.get_logger().info(f"{len(self.ex.walls)} no-go lines: "
                                   f"{[[round(v, 2) for v in w] for w in self.ex.walls]} (x1, y1, x2, y2 in m)")
            return
        with self.lock:
            self.ex.command(cmd)
        if cmd.startswith("start") or cmd == "resume":
            self.explore_since = time.monotonic()  # the depth check counts from here
        self.get_logger().info(f"command {cmd}: {self.ex.status.state}")
        if self.ex.status.state != "exploring":
            self.send(0.0, 0.0, 0.0)
        self.publish_status()

    def update_pose(self):
        try:
            t = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return
        tr = t.transform
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if getattr(self, "pose_stamp", None) == stamp:
            return  # nothing new: the lookup returns the last pose even when SLAM has stopped
        self.pose_stamp = stamp
        self.pose = (tr.translation.x, tr.translation.y, yaw_of(tr.rotation))
        # fresh only when its time stamp changed: a frozen SLAM keeps answering lookups with its
        # last transform (for the TF cache's length) and the robot drove blind on it (2026-10-05)
        self.pose_time = time.monotonic()

    # ---------- planning (thread) and control (timer) ----------

    def planner(self):
        while rclpy.ok():
            time.sleep(0.1)
            if self.pending_map is not None:
                grid, self.pending_map = self.pending_map, None
                kept = self.depth_mem.kept()
                with self.lock:
                    self.ex.depth_obstacles = [tuple(p) for p in do.DepthMemory.centres(kept)] if kept else []
                    self.ex.set_map(grid)
                    if self.ex.p.get("coverage3d") and self.depth_count:
                        g = self.ex.grid
                        self.ex.surface3d, self.ex.seen3d = self.cov3d.grid_status(g.cells.shape, g.res, g.origin_x, g.origin_y)
                        self.ex.update_targets()
            if self.pose is not None and self.ex.plan_due(time.monotonic()):
                try:
                    self.ex.plan(time.monotonic(), self.pose)
                    self.publish_path()
                except Exception as e:  # keep the node alive; the control cycle sends zero meanwhile
                    self.get_logger().error(f"planning failed: {e}")

    def control(self):
        self.update_pose()
        now = time.monotonic()
        with self.lock:
            if self.ex.status.state != "exploring":
                if self.was_moving:
                    self.send(0.0, 0.0, 0.0)
                return
            if now - max(getattr(self, "depth_time", 0.0), self.explore_since) > NO_DEPTH_S:
                # mapping needs the depth images (2026-10-06: no silent 2D estimate live); the helper
                # starts the robot's depth compressor with SLAM, so this means something is broken
                self.ex.command("pause")
                self.ex.event(f"no depth images for {NO_DEPTH_S:.0f} s: paused. Check the depth camera / "
                              "compressor on the robot (a recording starts it), then Resume")
                self.ex.status.message = "paused: no depth images"
                self.send(0.0, 0.0, 0.0)
                return
            if self.pose is None or now - self.pose_time > POSE_STALE_S or now - self.scan_time > STALE_S:
                self.ex.status.message = "waiting for pose / lidar"
                self.send(0.0, 0.0, 0.0)
                return
            obstacles = self.scan_xy
            extra = self.depth_points(now)
            if extra is not None and len(extra):
                obstacles = np.vstack([self.scan_xy, extra])
            vx, vy, wz = self.ex.step(now, self.pose, obstacles, plan_inline=False)
        self.send(vx, vy, wz)

    def send(self, vx, vy, wz):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(t)
        self.was_moving = bool(vx or vy or wz)
        self.last_cmd = (vx, vy, wz)

    # ---------- outputs for the console ----------

    def publish_status(self):
        # under the lock, as the control cycle: the planner thread swaps maps under it, and reading the
        # grids meanwhile (one already grown by a column, the other not) raised and killed the whole
        # node mid-run (run 20261007_140425, 6 min). An error here is logged, not fatal.
        with self.lock:
            try:
                self._publish_status()
            except Exception as e:
                self.get_logger().error(f"publish_status failed: {type(e).__name__}: {e}")

    def _publish_status(self):
        st = self.ex.status
        # log every change of state / phase with its reason (for checking runs afterwards)
        key = (st.state, st.phase)
        if key != getattr(self, "logged", None):
            self.logged = key
            cov = self.ex.coverage() if self.ex.grid is not None else 0.0
            self.get_logger().info(f"state {st.state} / {st.phase or '-'}: {st.message} "
                                   f"(camera coverage {100 * cov:.0f} %, frontiers {st.frontiers})")
        self.status_pub.publish(String(data=json.dumps({
            "state": st.state, "mode": st.mode, "phase": st.phase, "message": st.message, "frontiers": st.frontiers,
            "walls": len(self.ex.walls), "lines": [[round(v, 2) for v in w] for w in self.ex.walls],
            "remaining": st.remaining, "given_up_m2": round(self.ex.given_up_cells() * self.ex.grid.res ** 2, 1)
            if self.ex.grid is not None else 0.0,
            "coverage": round(self.ex.coverage(), 3) if self.ex.grid is not None else 0.0,
            "coverage_visible": (lambda v: None if v is None else round(v, 3))(self.ex.coverage_visible())
            if self.ex.grid is not None else None,
            "blocked": st.blocked, "goal": st.goal,
            # diagnostics: ages of the inputs (s), path length, last command
            "debug": {"pose_age": round(time.monotonic() - self.pose_time, 2) if self.pose else None,
                      "scan_age": round(time.monotonic() - self.scan_time, 2) if self.scan_xy is not None else None,
                      "path": len(st.path), "pose": [round(v, 2) for v in self.pose] if self.pose else None,
                      "cmd": [round(v, 3) for v in self.last_cmd]}})))

    def publish_events(self):
        while self.ex.events:
            text = self.ex.events.pop(0)
            self.get_logger().info(f"event: {text}")
            self.event_pub.publish(String(data=json.dumps({"time": time.time(), "text": text})))

    def log_progress(self):
        # (see publish_status)
        with self.lock:
            try:
                self._log_progress()
            except Exception as e:
                self.get_logger().error(f"log_progress failed: {type(e).__name__}: {e}")

    def _log_progress(self):
        """Every 5 s while exploring: what it is doing, so unexpected stops can be traced."""
        st = self.ex.status
        if st.state != "exploring":
            return
        pose = "-" if self.pose is None else f"({self.pose[0]:.2f}, {self.pose[1]:.2f}, {math.degrees(self.pose[2]):.0f} deg)"
        goal = "-" if st.goal is None else f"({st.goal[0]:.2f}, {st.goal[1]:.2f})"
        self.get_logger().info(
            f"progress: {st.phase or '-'} | {st.message} | goal {goal}, path {len(st.path)} | pose {pose} | "
            f"cmd ({self.last_cmd[0]:.2f}, {self.last_cmd[1]:.2f}, {self.last_cmd[2]:.2f}) | blocked {st.blocked} | "
            f"pose age {time.monotonic() - self.pose_time:.1f} s, scan age {time.monotonic() - self.scan_time:.1f} s | "
            f"depth images {self.depth_count} (floor {100 * self.floor.z:+.1f} cm), obstacle cells now {len(self.depth_mem.confirmed(time.monotonic()))}, "
            f"kept {len(self.depth_mem.kept())}")

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in list(self.ex.status.path):
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y = float(x), float(y)
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def publish_coverage(self):
        # (see publish_status)
        with self.lock:
            try:
                self._publish_coverage()
            except Exception as e:
                self.get_logger().error(f"publish_coverage failed: {type(e).__name__}: {e}")

    def _publish_coverage(self):
        """100 seen / 50 not yet / 25 given up (surfaces to look at); 75: an obstacle only the depth
        camera sees (furniture above the lidar's slice), drawn into the console's map."""
        g, seen = self.ex.grid, self.ex.seen
        if g is None:
            return
        data = np.full(g.cells.shape, -1, np.int8)
        for x, y in self.ex.depth_obstacles:
            r, c = g.to_cell(x, y)
            if g.inside(r, c):
                data[r, c] = 75
        if seen is not None and seen.shape == g.cells.shape:
            surf = self.ex.surfaces()
            data[surf & ~seen] = 50
            data[surf & seen] = 100
            if self.ex.attempts is not None and self.ex.attempts.shape == seen.shape:
                data[surf & ~seen & (self.ex.attempts >= self.ex.tries())] = 25  # given up
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = g.res
        msg.info.width, msg.info.height = g.cells.shape[1], g.cells.shape[0]
        msg.info.origin.position.x, msg.info.origin.position.y = g.origin_x, g.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = data.ravel().tolist()
        self.cov_pub.publish(msg)


def main():
    rclpy.init()
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0, 0.0)  # never leave the robot moving
        rclpy.shutdown()


if __name__ == "__main__":
    main()
