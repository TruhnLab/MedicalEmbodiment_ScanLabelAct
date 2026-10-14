"""Relays the robot's small sensor topics into the offboard ROS container's graph.

The robot's rosbridge is the only link (no DDS over Wi-Fi). We subscribe there and republish
locally as real ROS messages, so slam_toolbox & co. run unchanged. /clock is published from the
robot's stamps (all nodes here use use_sim_time), so every timestamp lives on the robot clock.
"""

import asyncio
import base64
import json
import os
import threading
import time

import rclpy
import websockets
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import CameraInfo, CompressedImage

from scan_fix import fixed_scan
from scan_gate import ScanGate, stamp_s

ROBOT_URL = os.environ.get("ROBOT_URL", "ws://192.168.149.1:9090")
# the other direction: velocity commands of the exploration (explorer.py) -> the robot's drive
# (CMD_OUT=/landerpi/cmd_vel_dryrun for tests: commands go to an unused topic, the robot stays put)
CMD_IN, CMD_TYPE = "/cmd_vel_auto", "geometry_msgs/msg/Twist"
CMD_OUT = os.environ.get("CMD_OUT", "/controller/cmd_vel")
CMD_TIMEOUT_S = 0.5  # no command for this long while moving: send a stop (explorer crashed, ...)

# robot topic -> (type, local topic)
TOPICS = {
    "/scan": ("sensor_msgs/msg/LaserScan", "/scan"),
    "/odom": ("nav_msgs/msg/Odometry", "/odom"),
    "/tf": ("tf2_msgs/msg/TFMessage", "/tf"),
    "/tf_static": ("tf2_msgs/msg/TFMessage", "/tf_static"),
    "/joint_states": ("sensor_msgs/msg/JointState", "/joint_states"),
    # not needed for SLAM, but relayed so laptop tools (recorder) read everything from here and the
    # robot's rosbridge serializes each stream only once
    "/imu": ("sensor_msgs/msg/Imu", "/imu"),
    "/ros_robot_controller/battery": ("std_msgs/msg/UInt16", "/ros_robot_controller/battery"),
}
STATIC_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def sanitize(topic, msg):
    """rosbridge sends inf/NaN floats as null; ROS messages need real floats."""
    if topic == "/scan":
        # also: always the same number of rays (slam_toolbox drops scans with another count)
        return fixed_scan(msg)
    return msg


def stamp_of(msg):
    header = msg.get("header") or (msg.get("transforms") or [{}])[0].get("header")
    return header["stamp"] if header else None


class Relay(Node):
    def __init__(self):
        super().__init__("robot_relay")
        self.pubs = {}
        for robot_topic, (type_name, local_topic) in TOPICS.items():
            qos = STATIC_QOS if robot_topic == "/tf_static" else 10
            self.pubs[robot_topic] = (get_message(type_name), self.create_publisher(get_message(type_name), local_topic, qos))
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.static_tf = {}  # child frame -> transform dict; tf_static is re-sent merged
        self.latest = None
        self.counts = {t: 0 for t in TOPICS}
        self.create_timer(10.0, self.report)
        # scans only after their pose: avoids a tf2_ros deadlock that froze SLAM (scan_gate.py)
        self.gate, self.gate_lock = ScanGate(), threading.Lock()
        self.create_timer(0.05, self.release_late_scans)
        # velocity commands out: queued for the websocket task (set by pump() when connected)
        self.outbox = None
        self.loop = None
        self.last_cmd = 0.0
        self.moving = False
        self.create_subscription(Twist, CMD_IN, self.on_cmd, 10)
        # depth images for the explorer's obstacles above the lidar (depth_obstacles.py)
        self.depth_pub = self.create_publisher(CompressedImage, DEPTH_LOCAL, 1)
        self.depth_info_pub = self.create_publisher(CameraInfo, DEPTH_INFO_LOCAL, 1)
        self.depth_count = 0

    def on_depth(self, topic, msg):
        if topic == DEPTH_TOPIC:
            out = CompressedImage()
            set_message_fields(out.header, msg["header"])
            out.format = msg.get("format", "")
            out.data = base64.b64decode(msg["data"])  # rosbridge sends uint8[] as base64
            self.depth_pub.publish(out)
            self.depth_count += 1
        else:
            out = CameraInfo()
            set_message_fields(out, msg)
            self.depth_info_pub.publish(out)

    def on_cmd(self, msg):
        twist = {"linear": {"x": msg.linear.x, "y": msg.linear.y, "z": 0.0},
                 "angular": {"x": 0.0, "y": 0.0, "z": msg.angular.z}}
        self.last_cmd = time.monotonic()
        self.moving = bool(msg.linear.x or msg.linear.y or msg.angular.z)
        if self.outbox is not None:
            self.loop.call_soon_threadsafe(self.outbox.put_nowait, twist)

    def on_robot_message(self, topic, msg):  # not "handle": rclpy.Node uses that name
        with self.gate_lock:
            if topic == "/scan" and "header" in msg:
                for t, m in self.gate.scan(stamp_s(msg["header"]), ("/scan", msg)):
                    self.publish(t, m)
                return
            # (the gate reads the pose before publish(): converting the message for ROS replaces the
            # dicts in its transforms list with ROS objects)
            released = self.gate.tf(msg) if topic == "/tf" else []
            self.publish(topic, msg)
            for t, m in released:
                self.publish(t, m)

    def release_late_scans(self):
        with self.gate_lock:
            for t, m in self.gate.poll():
                self.publish(t, m)

    def publish(self, topic, msg):
        msg_type, pub = self.pubs[topic]
        if topic == "/tf_static":
            for t in msg["transforms"]:
                self.static_tf[t["child_frame_id"]] = t
            msg = {"transforms": list(self.static_tf.values())}
        s = stamp_of(msg)  # before conversion: set_message_fields replaces list items in place
        ros_msg = msg_type()
        set_message_fields(ros_msg, sanitize(topic, msg))
        pub.publish(ros_msg)
        self.counts[topic] += 1
        # the clock only moves forward: the newest robot stamp we have seen
        if s and topic != "/tf_static" and (self.latest is None or (s["sec"], s["nanosec"]) > self.latest):
            self.latest = (s["sec"], s["nanosec"])
            c = Clock()
            c.clock.sec, c.clock.nanosec = self.latest
            self.clock_pub.publish(c)

    def report(self):
        self.get_logger().info("relayed in 10 s: " + ", ".join(f"{t} {n}" for t, n in self.counts.items()))
        self.counts = {t: 0 for t in TOPICS}


async def pump(node):
    while rclpy.ok():
        try:
            async with websockets.connect(ROBOT_URL, max_size=None, ping_interval=10) as ws:
                for robot_topic, (type_name, _) in TOPICS.items():
                    await ws.send(json.dumps({"op": "subscribe", "topic": robot_topic, "type": type_name,
                                              "queue_length": 10}))
                await ws.send(json.dumps({"op": "subscribe", "topic": MOTOR_TOPIC, "type": MOTOR_TYPE, "queue_length": 10}))
                # depth: 4 per second, only the newest (queue 1); exists while the recorder's depth
                # compressor runs on the robot (an exploration always records)
                await ws.send(json.dumps({"op": "subscribe", "topic": DEPTH_TOPIC, "type": "sensor_msgs/msg/CompressedImage",
                                          "throttle_rate": DEPTH_THROTTLE_MS, "queue_length": 1}))
                await ws.send(json.dumps({"op": "subscribe", "topic": DEPTH_INFO_TOPIC, "type": "sensor_msgs/msg/CameraInfo",
                                          "throttle_rate": 5000, "queue_length": 1}))
                await ws.send(json.dumps({"op": "advertise", "topic": CMD_OUT, "type": CMD_TYPE}))
                node.loop, node.outbox = asyncio.get_running_loop(), asyncio.Queue()
                sender = asyncio.create_task(send_commands(node, ws))
                motors = asyncio.create_task(check_motors(node))
                node.get_logger().info(f"connected to {ROBOT_URL}")
                try:
                    await read_messages(node, ws)
                finally:
                    sender.cancel()
                    motors.cancel()
                    node.outbox = None
        except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
            # any connection problem (also a timeout while the robot boots or Wi-Fi hiccups): retry,
            # never exit (the relay is the only link to the robot)
            node.get_logger().warning(f"robot link lost ({type(e).__name__}: {e}); retrying in 2 s")
            await asyncio.sleep(2)


STOP = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}


async def send_commands(node, ws):
    """Forwards queued velocity commands; sends a stop if they stop arriving while moving."""
    while True:
        try:
            twist = await asyncio.wait_for(node.outbox.get(), timeout=0.1)
            await ws.send(json.dumps({"op": "publish", "topic": CMD_OUT, "msg": twist}))
        except asyncio.TimeoutError:
            if node.moving and time.monotonic() - node.last_cmd > CMD_TIMEOUT_S:
                node.moving = False
                node.get_logger().warning("velocity commands stopped arriving: stopping the robot")
                await ws.send(json.dumps({"op": "publish", "topic": CMD_OUT, "msg": STOP}))


# What reaches the motors (diagnostics only, not relayed): the motor board keeps the last speed target
# until the next command, so a non-zero target that stays is a robot that hums or creeps (2026-10-05)
MOTOR_TOPIC, MOTOR_TYPE = "/ros_robot_controller/set_motor", "ros_robot_controller_msgs/msg/MotorsState"
# the depth camera (16-bit PNG, republished on the robot by the recorder's depth compressor)
DEPTH_TOPIC = "/ascamera/camera_publisher/depth0/image_raw/compressed"
DEPTH_INFO_TOPIC = "/ascamera/camera_publisher/depth0/camera_info"
DEPTH_LOCAL, DEPTH_INFO_LOCAL = "/depth/compressed", "/depth/camera_info"
DEPTH_THROTTLE_MS = 250
MOTOR_HELD_S = 3.0


def watch_motors(node, msg):
    target = tuple(sorted((d["id"], round(d["rps"], 3)) for d in msg.get("data", [])))
    now = time.monotonic()
    if target != getattr(node, "motor_target", None):
        node.motor_target, node.motor_since, node.motor_warned = target, now, False
    elif held_still(node, now):
        node.motor_warned = True
        node.get_logger().warning(f"motors held at a non-zero target for {MOTOR_HELD_S:.0f} s while the robot "
                                  f"does not move: {node.motor_target} (id, rps)")


def held_still(node, now):
    """A non-zero target for MOTOR_HELD_S while the wheel odometry shows no motion: a stall or a
    leftover target (a long turn on the spot is a non-zero target too, but the robot moves)."""
    t = getattr(node, "motor_target", None)
    return (bool(t) and not node.motor_warned and any(rps for _, rps in t) and now - node.motor_since > MOTOR_HELD_S
            and now - getattr(node, "last_motion", 0.0) > MOTOR_HELD_S)


async def check_motors(node):
    """The board keeps its target, and set_motor is only sent on new commands: re-check periodically."""
    while True:
        await asyncio.sleep(1.0)
        if held_still(node, time.monotonic()):
            node.motor_warned = True
            node.get_logger().warning(f"motors held at a non-zero target for {MOTOR_HELD_S:.0f}+ s while the robot "
                                      f"does not move: {node.motor_target} (id, rps)")


async def read_messages(node, ws):
    async for raw in ws:
        m = json.loads(raw)
        if m.get("op") == "publish" and m["topic"] == MOTOR_TOPIC:
            watch_motors(node, m["msg"])
            continue
        if m.get("op") == "publish" and m["topic"] in (DEPTH_TOPIC, DEPTH_INFO_TOPIC):
            try:
                node.on_depth(m["topic"], m["msg"])
            except Exception as e:
                node.get_logger().warning(f"{m['topic']}: {e}")
            continue
        if m.get("op") == "publish" and m["topic"] == "/odom":
            tw = m["msg"]["twist"]["twist"]
            if abs(tw["linear"]["x"]) + abs(tw["linear"]["y"]) > 0.01 or abs(tw["angular"]["z"]) > 0.02:
                node.last_motion = time.monotonic()
        if m.get("op") == "publish" and m["topic"] in TOPICS:
            try:
                node.on_robot_message(m["topic"], m["msg"])
            except Exception as e:  # one bad message must not stop the relay
                node.get_logger().warning(f"{m['topic']}: {e}")


def main():
    rclpy.init()
    node = Relay()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        asyncio.run(pump(node))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
