"""Diagnostics for a frozen slam_toolbox (seen in 3 of 3 live runs on 2026-10-05, not reproducible by
replaying the same recording): if SLAM's map -> odom transform gets no new time stamp for FROZEN_S
while lidar scans keep arriving, write a stack trace of all slam_toolbox threads (gdb) to
/offboard_ros/maps/ (= data/maps on the laptop), once per freeze. It changes nothing; it only records
what SLAM is waiting for, so the cause can be fixed.
"""

import subprocess
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

FROZEN_S = 5.0


class Watchdog(Node):
    def __init__(self):
        super().__init__("slam_watchdog")
        self.map_odom_stamp, self.map_odom_time = None, time.monotonic()
        self.scan_time = 0.0
        self.dumped = False
        self.create_subscription(TFMessage, "/tf", self.on_tf, 50)
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan_time", time.monotonic()), 10)
        self.create_timer(1.0, self.check)

    def on_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                s = (t.header.stamp.sec, t.header.stamp.nanosec)
                if s != self.map_odom_stamp:
                    if self.dumped:
                        self.get_logger().info("SLAM publishes again")
                    self.map_odom_stamp, self.map_odom_time, self.dumped = s, time.monotonic(), False

    def check(self):
        now = time.monotonic()
        if (self.map_odom_stamp is None or self.dumped or now - self.scan_time > 2.0
                or now - self.map_odom_time < FROZEN_S):
            return
        self.dumped = True
        path = f"/offboard_ros/maps/slam_freeze_{datetime.now():%Y%m%d_%H%M%S}.txt"
        self.get_logger().error(f"SLAM frozen: no new map->odom for {now - self.map_odom_time:.0f} s while scans "
                                f"arrive; writing its threads' stack traces to {path}")
        try:
            pid = subprocess.run(["pgrep", "-f", "^/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node"], capture_output=True, text=True).stdout.split()[0]
            out = subprocess.run(["gdb", "-p", pid, "-batch", "-ex", "set pagination off",
                                  "-ex", "thread apply all bt"], capture_output=True, text=True, timeout=120)
            with open(path, "w") as f:
                f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
            self.get_logger().error(f"stack traces written: {path}")
        except Exception as e:
            self.get_logger().error(f"could not write stack traces: {e}")


def main():
    rclpy.init()
    rclpy.spin(Watchdog())


if __name__ == "__main__":
    main()
