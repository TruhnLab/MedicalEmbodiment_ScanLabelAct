"""Publishes the robot's pose in the map (/robot_pose) and its path (/robot_path) for the UI.

The UI would otherwise have to combine map->odom (from SLAM) and odom->base_footprint (from the
robot) itself; here tf2 does it.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

RATE_HZ = 5.0
PATH_STEP_M = 0.05   # add a path point every 5 cm


class RobotPose(Node):
    def __init__(self):
        super().__init__("robot_pose")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.path_pub = self.create_publisher(Path, "/robot_path", 10)
        self.path = Path()
        self.path.header.frame_id = "map"
        self.create_timer(1.0 / RATE_HZ, self.tick)

    def tick(self):
        try:
            t = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return  # SLAM hasn't published map->odom yet
        pose = PoseStamped()
        pose.header = t.header
        pose.header.frame_id = "map"
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.orientation = t.transform.rotation
        self.pose_pub.publish(pose)
        last = self.path.poses[-1].pose.position if self.path.poses else None
        if last is None or math.hypot(last.x - pose.pose.position.x, last.y - pose.pose.position.y) >= PATH_STEP_M:
            self.path.poses.append(pose)
            self.path.header.stamp = pose.header.stamp
            self.path_pub.publish(self.path)


def main():
    rclpy.init()
    rclpy.spin(RobotPose())


if __name__ == "__main__":
    main()
