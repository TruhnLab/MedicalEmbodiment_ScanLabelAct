"""Replays a recorded session's lidar + TF into slam_toolbox (offline debugging, no robot).

    docker run --rm -v <repo>/data/sessions/<name>:/session landerpi-offboard-ros \
        bash -c "source /opt/ros/humble/setup.bash && python3 /offboard_ros/replay_slam.py /session"

Starts slam_toolbox with slam.yaml, publishes the recorded /scan, the robot's TF (odom ->
base_footprint, not the recorded map -> odom) and static transforms with their original time
stamps on /clock, and reports how slam_toolbox reacts: map updates (size, known cells) and its
map -> odom correction.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.set_message import set_message_fields
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from scan_fix import fixed_scan

STATIC_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def stamp(h):
    return h["stamp"]["sec"] + h["stamp"]["nanosec"] * 1e-9


TF_DELAY = float(sys.argv[sys.argv.index("--tf-delay") + 1]) if "--tf-delay" in sys.argv else 0.0
rng = __import__("random").Random(0)


def main():
    session = Path(sys.argv[1])
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    params = sys.argv[3] if len(sys.argv) > 3 else "/offboard_ros/slam.yaml"  # alternative settings
    events = []
    for line in open(session / "streams" / "scan.jsonl", encoding="utf-8"):
        m = json.loads(line)["msg"]
        if "--raw-scans" in sys.argv:  # as the robot sends them (500-505 rays)
            m["ranges"] = [float("inf") if r is None else r for r in m["ranges"]]
            m["intensities"] = [0.0 if i is None else i for i in m.get("intensities", [])]
        else:  # as relay.py publishes them now
            m = fixed_scan(m)
        events.append((stamp(m["header"]), "scan", m))
    for line in open(session / "streams" / "tf.jsonl", encoding="utf-8"):
        m = json.loads(line)["msg"]
        m["transforms"] = [t for t in m["transforms"] if t["child_frame_id"] != "odom"]  # SLAM's own output
        if m["transforms"]:
            # --tf-delay: positions arrive this late after their stamp, at random up to the value
            # (live over Wi-Fi: 30-330 ms; scans then wait for their position, see slam.yaml)
            late = TF_DELAY * rng.random() if TF_DELAY else 0.0
            events.append((stamp(m["transforms"][0]["header"]) + late, "tf", m))
    events.sort(key=lambda e: e[0])
    static = json.loads((session / "calib" / "tf_static.json").read_text())

    slam = subprocess.Popen(["ros2", "run", "slam_toolbox", "async_slam_toolbox_node", "--ros-args",
                             "--params-file", params, "-p", "use_sim_time:=true"])
    rclpy.init()
    node = Node("replay", parameter_overrides=[])
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    scan_pub = node.create_publisher(LaserScan, "/scan", 10)
    tf_pub = node.create_publisher(TFMessage, "/tf", 100)
    static_pub = node.create_publisher(TFMessage, "/tf_static", STATIC_QOS)
    maps = []
    node.create_subscription(OccupancyGrid, "/map", lambda m: maps.append(
        (time.time(), m.info.width, m.info.height, sum(1 for v in m.data if v >= 0))), STATIC_QOS)
    buf = Buffer()
    TransformListener(buf, node)
    time.sleep(4.0)  # slam_toolbox starting
    st = TFMessage()
    set_message_fields(st, static)
    static_pub.publish(st)

    gate = None
    if "--gate" in sys.argv:
        from scan_gate import ScanGate
        gate = ScanGate()
    t0, w0 = events[0][0], time.time()
    report_at = w0
    corr_samples = []  # map->odom (x, y, yaw deg) every 0.25 s: how much does SLAM jump?
    sample_at = w0
    for t, kind, m in events:
        while time.time() - w0 < (t - t0) / speed:
            rclpy.spin_once(node, timeout_sec=0.002)
        c = Clock()
        c.clock.sec, c.clock.nanosec = int(t), int((t % 1) * 1e9)
        clock_pub.publish(c)
        # --gate: scans only after their pose, as relay.py does it live (scan_gate.py)
        if gate is not None:
            out = gate.scan(stamp(m["header"]), m) if kind == "scan" else []
            if kind == "tf":
                out = gate.tf(m)  # before set_message_fields, which replaces the dicts in m
                tfm = TFMessage()
                set_message_fields(tfm, m)
                tf_pub.publish(tfm)
            out += gate.poll()
            for sm in out:
                msg = LaserScan()
                set_message_fields(msg, sm)
                scan_pub.publish(msg)
        else:
            msg = LaserScan() if kind == "scan" else TFMessage()
            set_message_fields(msg, m)
            (scan_pub if kind == "scan" else tf_pub).publish(msg)
        if time.time() > sample_at:
            sample_at = time.time() + 0.25
            try:
                mo = buf.lookup_transform("map", "odom", rclpy.time.Time())
                r = mo.transform.rotation
                corr_samples.append((mo.transform.translation.x, mo.transform.translation.y,
                                     math.degrees(math.atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y ** 2 + r.z ** 2)))))
            except Exception:
                pass
        if time.time() > report_at:
            report_at = time.time() + 2.0
            try:
                mo = buf.lookup_transform("map", "odom", rclpy.time.Time())
                r = mo.transform.rotation
                corr = (round(mo.transform.translation.x, 3), round(mo.transform.translation.y, 3),
                        round(math.degrees(math.atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y ** 2 + r.z ** 2))), 2))
            except Exception as e:
                corr = f"no map->odom ({type(e).__name__})"
            last = maps[-1] if maps else None
            if "--quiet" not in sys.argv:
                print(f"t+{t - t0:5.1f}s  map updates {len(maps)}, last {last[1:] if last else None} (w, h, known cells), "
                      f"map->odom {corr}", flush=True)
    end = time.time() + 5
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    print("final: map updates", len(maps), "last", maps[-1][1:] if maps else None, flush=True)
    jumps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(corr_samples, corr_samples[1:])]
    minutes = (events[-1][0] - t0) / 60
    if jumps:
        print(f"SLAM jumps > 10 cm: {sum(j > 0.10 for j in jumps) / minutes:.1f}/min, > 30 cm: "
              f"{sum(j > 0.30 for j in jumps) / minutes:.1f}/min, largest {100 * max(jumps):.0f} cm", flush=True)
    slam.terminate()


if __name__ == "__main__":
    main()
