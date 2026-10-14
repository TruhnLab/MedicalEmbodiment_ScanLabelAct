"""Streams the robot's sensors to the laptop and records a mapping session.

Transport (measured 2026-09-30, see README):
  colour  MJPEG from the robot's web_video_server   ~14 fps, ~10 ms latency, ~1.4 MB/s
  depth   16-bit PNG via rosbridge                   ~10 fps, ~120 ms latency, ~53 KB/frame
  small   scan/odom/imu/tf/joints/battery via rosbridge, at full rate

Colour and depth arrive separately and are paired by timestamp; only pairs that add a new
viewpoint are kept as keyframes (the robot or the arm moved enough). Images are stored exactly
as received. Small topics are recorded completely.
"""

import asyncio
import base64
import json
import math
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from landerpi.messages import angle_diff, odom_pose, stamp
from landerpi.robot import ARM_JOINTS, BATTERY_CRITICAL_V, TOPICS

# colour runs at ~14 Hz, depth at ~10 Hz, so the nearest depth is at most ~35 ms away. Each image
# keeps its own stamp; its camera pose is later taken at exactly that time.
PAIR_TOLERANCE_S = 0.05
MAX_INFLIGHT = 8           # keyframes being written at once; more means the disk can't keep up
LAG_WARN_S = 0.5           # images arriving this late -> the Wi-Fi can't keep up, slow down

SMALL_STREAMS = ["scan", "odom", "imu", "tf", "joints", "battery"]
# from the offboard SLAM container (if it runs): the drift correction map -> odom and the robot pose in
# the map. With odom -> ... -> camera from the robot's tf, this gives every image's pose in the map.
SLAM_STREAMS = ["slam_map_odom", "slam_robot_pose"]
# the no-go lines drawn in the console (from the exploration's status), written only when they change:
# a few lines per run. They explain why the robot left parts out and can't be rebuilt from the other
# streams (projects/presentation_video draws them).
NOGO_STREAM = "nogo_lines"
# the exploration's view of the run, to replay why it chose a goal or ended (run 20261006_142432 ended at
# 74 % with 'no goal' and its state could not be rebuilt afterwards): the status when it changes, and
# every 5 s the SLAM map and the explorer's coverage grid (int8 cells, zlib + base64: a few KB each)
EXPLORE_STREAMS = ["explore_status", "explore_coverage", "slam_map"]
GRID_THROTTLE_MS = 5000


@dataclass
class KeyframePolicy:
    min_translation_m: float = 0.05
    min_rotation_deg: float = 5.0
    min_arm_change_rad: float = 0.02
    all_frames: bool = False   # keep every rgb/depth pair (for testing throughput)


@dataclass
class Limits:
    max_gb: float = 20.0
    max_minutes: float = 60.0


class Recorder:
    def __init__(self, clients, writer, policy=KeyframePolicy(), limits=Limits(), live=None, depth_hz=10.0):
        # separate rosbridge connections, so depth frames never delay the small sensor messages
        self.clients = clients          # {"sensors": ..., "depth": ...}
        self.writer = writer
        self.policy = policy
        self.limits = limits
        self.live = live
        self.depth_hz = depth_hz
        self.mjpeg = None               # set by the caller; feeds on_rgb_frame
        self.small_source = clients["sensors"]  # where scan/odom/imu/tf/joints come from (see start)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="keyframe")
        self.loop = None

        self.started = None
        self.stop_reason = None
        self.stopped = asyncio.Event()

        self.counts = {name: 0 for name in SMALL_STREAMS + SLAM_STREAMS + EXPLORE_STREAMS + [NOGO_STREAM, "rgb", "depth"]}
        self._nogo = None
        self.pairs = 0
        self.keyframes = 0
        self.skipped_still = 0     # pairs dropped because nothing moved
        self.dropped_backlog = 0   # keyframes dropped because the disk fell behind
        self.unpaired = 0
        self.gaps = 0              # keyframes taken after moving much further than planned
        self._inflight = 0

        self._rgb = deque(maxlen=10)     # (stamp, jpeg)
        self._depth = deque(maxlen=10)   # (stamp, png)
        self._odom = deque(maxlen=400)   # (stamp, x, y, yaw), ~14 s at 29 Hz
        self._arm = None
        self._last_kf = None             # (x, y, yaw, arm) of the last keyframe
        self._clock_offset = None        # min(laptop recv - robot stamp) on small topics
        self._lag = deque(maxlen=30)
        self._calib_done = set()
        self._static_tf = {}             # child frame -> static transform
        self.battery_v = None

    # ---------- setup ----------

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.started = time.time()
        sensors = self.clients["sensors"]
        # With SLAM running, the offboard container already receives every small robot stream (its
        # relay): read them there, so the robot's rosbridge serializes each stream only once.
        # (Reading them from the robot as well cut all rates to a third: 2.3 image pairs/s.)
        self.small_source = self.clients.get("slam") or sensors
        for name in SMALL_STREAMS:
            topic, msg_type = TOPICS[name]
            await self.small_source.subscribe(topic, msg_type, self._small(name), queue_length=10)
        # static transforms and calibration only need to be seen once
        topic, msg_type = TOPICS["tf_static"]
        await sensors.subscribe(topic, msg_type, self._on_tf_static)
        for name in ("rgb_info", "depth_info"):
            topic, msg_type = TOPICS[name]
            await sensors.subscribe(topic, msg_type, self._calib(name))
        # depth: throttled on the robot side, queue_length 1 so frames are dropped rather than queued
        topic, msg_type = TOPICS["depth_png"]
        await self.clients["depth"].subscribe(topic, msg_type, self._on_depth,
                                              throttle_ms=int(1000 / self.depth_hz), queue_length=1)
        slam = self.clients.get("slam")
        if slam is not None:
            await slam.subscribe("/tf", "tf2_msgs/msg/TFMessage", self._on_slam_tf, queue_length=20)
            await slam.subscribe("/robot_pose", "geometry_msgs/msg/PoseStamped", self._on_slam_pose, queue_length=10)
            await slam.subscribe("/explore/status", "std_msgs/msg/String", self._on_explore_status, queue_length=5)
            await slam.subscribe("/explore/coverage", "nav_msgs/msg/OccupancyGrid", self._grid("explore_coverage"),
                                 throttle_ms=GRID_THROTTLE_MS, queue_length=1)
            await slam.subscribe("/map", "nav_msgs/msg/OccupancyGrid", self._grid("slam_map"),
                                 throttle_ms=GRID_THROTTLE_MS, queue_length=1)

    def _on_slam_tf(self, msg, recv):
        # the SLAM container's /tf also carries the relayed robot transforms; keep only SLAM's own
        for t in msg["transforms"]:
            if t["header"]["frame_id"] == "map" and t["child_frame_id"] == "odom":
                self.counts["slam_map_odom"] += 1
                self.writer.append_stream("slam_map_odom", t, recv)

    def _grid(self, name):
        def callback(msg, recv):
            data = bytes(v & 0xFF for v in msg["data"])
            self.counts[name] += 1
            self.writer.append_stream(name, {"header": msg["header"], "info": msg["info"],
                                             "data_int8_zlib_b64": base64.b64encode(zlib.compress(data)).decode()}, recv)
        return callback

    def _on_explore_status(self, msg, recv):
        try:
            status = json.loads(msg["data"])
            lines = status.get("lines", [])
        except (ValueError, AttributeError):
            return
        key = tuple(json.dumps(status.get(k)) for k in ("state", "phase", "message", "goal", "remaining", "given_up_m2"))
        if key != getattr(self, "_status_key", None):
            self._status_key = key
            self.counts["explore_status"] += 1
            self.writer.append_stream("explore_status", {k: v for k, v in status.items() if k != "lines"}, recv)
        if lines != self._nogo:
            self._nogo = lines
            self.counts[NOGO_STREAM] += 1
            self.writer.append_stream(NOGO_STREAM, {"lines": lines}, recv)

    def _on_slam_pose(self, msg, recv):
        self.counts["slam_robot_pose"] += 1
        self.writer.append_stream("slam_robot_pose", msg, recv)

    def _small(self, name):
        def callback(msg, recv):
            self.counts[name] += 1
            self.writer.append_stream(name, msg, recv)
            if name == "odom":
                self._odom.append((stamp(msg), *odom_pose(msg)))
            elif name == "joints":
                pos = dict(zip(msg["name"], msg["position"]))
                if all(j in pos for j in ARM_JOINTS):
                    self._arm = [pos[j] for j in ARM_JOINTS]
            elif name == "battery":
                self.battery_v = msg["data"] / 1000
            if "header" in msg:
                self._update_clock_offset(recv - stamp(msg))
        return callback

    def _on_tf_static(self, msg, recv):
        # several robot nodes publish static transforms, each in its own message: merge them all
        # (keyed by child frame) and rewrite the file whenever something new arrives
        new = False
        for t in msg["transforms"]:
            if self._static_tf.get(t["child_frame_id"]) != t:
                self._static_tf[t["child_frame_id"]] = t
                new = True
        if new:
            self._calib_done.add("tf_static")
            self.writer.write_json("calib/tf_static.json", {"transforms": list(self._static_tf.values())})

    def _calib(self, name):
        def callback(msg, recv):
            if name in self._calib_done:
                return
            self._calib_done.add(name)
            file = "rgb_camera_info.json" if name == "rgb_info" else "depth_camera_info.json"
            self.writer.write_json(f"calib/{file}", msg)
            asyncio.ensure_future(self.clients["sensors"].unsubscribe(TOPICS[name][0]))
        return callback

    def _update_clock_offset(self, delta):
        # the smallest delay ever seen is our best estimate of the clock difference
        if self._clock_offset is None or delta < self._clock_offset:
            self._clock_offset = delta

    # ---------- images ----------

    def on_rgb_frame(self, jpeg, t, recv):
        """Called from the MJPEG reader thread."""
        self.loop.call_soon_threadsafe(self._on_rgb, jpeg, t, recv)

    def _on_rgb(self, jpeg, t, recv):
        self.counts["rgb"] += 1
        if self._clock_offset is not None:
            self._lag.append(recv - t - self._clock_offset)
        if self.live is not None:
            self.live.update_frame(jpeg)   # full-rate live video for the UI
        self._rgb.append((t, jpeg))
        self._pair()

    def _on_depth(self, msg, recv):
        self.counts["depth"] += 1
        self._depth.append((stamp(msg), base64.b64decode(msg["data"])))
        self._pair()

    def _pair(self):
        while self._rgb and self._depth:
            t_rgb, jpeg = self._rgb[0]
            match = min(self._depth, key=lambda d: abs(d[0] - t_rgb))
            if abs(match[0] - t_rgb) <= PAIR_TOLERANCE_S:
                self._rgb.popleft()
                self._depth.remove(match)
                self.pairs += 1
                self._on_pair(t_rgb, jpeg, match[0], match[1])
            elif match[0] > t_rgb:
                self._rgb.popleft()      # colour is faster than depth: frames without a partner are normal
                self.unpaired += 1
            else:
                break                    # its depth may still arrive

    def _on_pair(self, t_rgb, jpeg, t_depth, png):
        pose = self._pose_at(t_rgb)
        if pose is None:
            return
        if not self._is_new_view(pose):
            self.skipped_still += 1
            return
        if self._inflight >= MAX_INFLIGHT:
            self.dropped_backlog += 1
            return
        self._last_kf = (*pose, self._arm)
        self._inflight += 1
        index = self.keyframes
        self.keyframes += 1
        future = self.loop.run_in_executor(self.pool, self.writer.write_keyframe,
                                           index, t_rgb, jpeg, t_depth, png, pose)
        future.add_done_callback(self._keyframe_written)

    def _keyframe_written(self, future):
        self._inflight -= 1
        if future.exception() is not None:
            print(f"\nkeyframe write failed: {future.exception()}")

    def _pose_at(self, t):
        if not self._odom:
            return None
        _, x, y, yaw = min(self._odom, key=lambda o: abs(o[0] - t))
        return x, y, yaw

    def _is_new_view(self, pose):
        if self.policy.all_frames or self._last_kf is None:
            return True
        x, y, yaw = pose
        lx, ly, lyaw, larm = self._last_kf
        moved = math.hypot(x - lx, y - ly)
        turned = abs(math.degrees(angle_diff(yaw, lyaw)))
        arm_moved = (self._arm is not None and larm is not None
                     and max(abs(a - b) for a, b in zip(self._arm, larm)) >= self.policy.min_arm_change_rad)
        new = moved >= self.policy.min_translation_m or turned >= self.policy.min_rotation_deg or arm_moved
        # moving 3x further than planned between keyframes means coverage holes
        if new and (moved >= 3 * self.policy.min_translation_m or turned >= 3 * self.policy.min_rotation_deg):
            self.gaps += 1
        return new

    # ---------- status & stopping ----------

    @property
    def bytes_received(self):
        rx = sum(c.bytes_received for c in self.clients.values())
        return rx + (self.mjpeg.bytes_received if self.mjpeg else 0)

    def _snapshot(self):
        return {**self.counts, "pairs": self.pairs, "keyframes": self.keyframes}

    def status(self, dt, bytes_rx_delta, prev_counts):
        elapsed = time.time() - self.started
        lag = max(self._lag) if self._lag else None
        hints = []
        if lag is not None and lag > LAG_WARN_S:
            hints.append("slow down: Wi-Fi is behind")
        if self.mjpeg is not None and self.mjpeg.last_frame_time and time.time() - self.mjpeg.last_frame_time > 3:
            hints.append("no colour frames: video server stalled?")
        if self.dropped_backlog:
            hints.append("disk behind")
        if self.battery_v is not None and self.battery_v < 7.4:
            hints.append("battery low")
        slam = self.clients.get("slam")
        slam_ok = slam is not None and not slam.closed.is_set()
        if not slam_ok:
            hints.append("no SLAM: images get odometry poses only")
        return {
            "recording": self.stop_reason is None,
            "session": self.writer.root.name,
            "elapsed_s": round(elapsed, 1),
            "keyframes": self.keyframes,
            "pairs": self.pairs,
            "skipped_still": self.skipped_still,
            "dropped_backlog": self.dropped_backlog,
            "unpaired": self.unpaired,
            "gaps": self.gaps,
            "rx_mb_s": round(bytes_rx_delta / dt / 1e6, 2),
            "rates_hz": {k: round((v - prev_counts.get(k, 0)) / dt, 1) for k, v in self._snapshot().items()},
            "lag_s": None if lag is None else round(lag, 3),
            "written_gb": round(self.writer.bytes_written / 1e9, 3),
            "battery_v": self.battery_v,
            "slam": slam_ok,
            "hints": hints,
            "stop_reason": self.stop_reason,
        }

    def check_limits(self):
        if any(c.closed.is_set() for name, c in self.clients.items() if name != "slam"):
            return "connection to the robot lost"
        if self.small_source is self.clients.get("slam") and self.small_source.closed.is_set():
            return "SLAM stopped (odometry and the other sensor streams came through it)"
        if self.writer.bytes_written / 1e9 >= self.limits.max_gb:
            return f"data budget reached ({self.limits.max_gb} GB)"
        if time.time() - self.started >= self.limits.max_minutes * 60:
            return f"time limit reached ({self.limits.max_minutes} min)"
        if self.battery_v is not None and self.battery_v < BATTERY_CRITICAL_V:
            return f"battery critical ({self.battery_v:.2f} V)"
        return None

    async def run_status_loop(self, print_status=True):
        prev_bytes, prev_counts, prev_t = self.bytes_received, self._snapshot(), time.time()
        while self.stop_reason is None:
            await asyncio.sleep(1.0)
            now = time.time()
            st = self.status(now - prev_t, self.bytes_received - prev_bytes, prev_counts)
            prev_bytes, prev_counts, prev_t = self.bytes_received, self._snapshot(), now
            if self.live is not None:
                self.live.update_status(st)
            if print_status:
                r = st["rates_hz"]
                line = (f"\r{st['elapsed_s']:6.0f}s  kf {st['keyframes']:5d}  rgb {r['rgb']:4.1f}  "
                        f"depth {r['depth']:4.1f}  pairs {r['pairs']:4.1f}  scan {r['scan']:4.1f}  "
                        f"odom {r['odom']:4.1f} Hz  {st['rx_mb_s']:4.2f} MB/s  lag {1000 * (st['lag_s'] or 0):4.0f} ms  "
                        f"{st['written_gb']:6.3f} GB  {'  '.join(st['hints'])}")
                print(line.ljust(140), end="", flush=True)
            reason = self.check_limits()
            if reason:
                self.stop(reason)

    def stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stopped.set()

    # ---------- finish ----------

    def finish(self):
        """Waits for pending keyframes and writes meta.json."""
        self.pool.shutdown(wait=True)
        if self.live is not None:
            self.live.update_status({**self.live.status, "recording": False, "stop_reason": self.stop_reason})
        self.writer.close({
            "started": self.started,
            "duration_s": round(time.time() - self.started, 1),
            "stop_reason": self.stop_reason,
            "robot_url": self.clients["sensors"].url,
            "slam_url": self.clients["slam"].url if "slam" in self.clients else None,
            "small_streams_from": self.small_source.url,
            "sources": {"rgb": "web_video_server MJPEG" + (f" ({self.mjpeg.path})" if self.mjpeg else ""),
                        "depth": f"{TOPICS['depth_png'][0]} (16-bit PNG) at {self.depth_hz} Hz"},
            "clock_offset_s": self._clock_offset,
            "keyframe_policy": asdict(self.policy),
            "limits": asdict(self.limits),
            "counts": {**self.counts, "pairs": self.pairs, "keyframes": self.keyframes,
                       "skipped_still": self.skipped_still, "dropped_backlog": self.dropped_backlog,
                       "unpaired": self.unpaired, "gaps": self.gaps,
                       "video_stalls": self.mjpeg.stalls if self.mjpeg else None},
            "bytes_written": self.writer.bytes_written,
            "calibration_received": sorted(self._calib_done),
        })
