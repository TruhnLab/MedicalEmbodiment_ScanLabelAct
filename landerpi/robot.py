"""Facts about our LanderPi (Mecanum, factory image), verified over rosbridge on 2026-09-29."""

import os

ROBOT_HOST = os.environ.get("LANDERPI_HOST", "192.168.149.1")
ROSBRIDGE_URL = f"ws://{ROBOT_HOST}:9090"

# topic -> message type
TOPICS = {
    "rgb": ("/ascamera/camera_publisher/rgb0/image", "sensor_msgs/msg/Image"),              # 640x400 bgr8, 14 Hz
    "rgb_info": ("/ascamera/camera_publisher/rgb0/camera_info", "sensor_msgs/msg/CameraInfo"),
    "depth": ("/ascamera/camera_publisher/depth0/image_raw", "sensor_msgs/msg/Image"),      # 640x400 mono16 (mm)
    # lossless 16-bit PNG of the above; only exists while our depth compressor runs (robot_ssh)
    "depth_png": ("/ascamera/camera_publisher/depth0/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
    "depth_info": ("/ascamera/camera_publisher/depth0/camera_info", "sensor_msgs/msg/CameraInfo"),
    "scan": ("/scan", "sensor_msgs/msg/LaserScan"),                                         # 10 Hz, 503 beams
    "odom": ("/odom", "nav_msgs/msg/Odometry"),                                             # 29 Hz, EKF-fused
    "imu": ("/imu", "sensor_msgs/msg/Imu"),                                                 # 45 Hz
    "tf": ("/tf", "tf2_msgs/msg/TFMessage"),
    "tf_static": ("/tf_static", "tf2_msgs/msg/TFMessage"),
    "joints": ("/joint_states", "sensor_msgs/msg/JointState"),
    "battery": ("/ros_robot_controller/battery", "std_msgs/msg/UInt16"),                    # millivolts
    "cmd_vel": ("/controller/cmd_vel", "geometry_msgs/msg/Twist"),
    # arm + gripper servos (ids 1-5 arm, 10 gripper), positions 0-1000 "pulse", 500 = centre (console/app.js)
    "servo_cmd": ("/servo_controller", "servo_controller_msgs/msg/ServosPosition"),
    "servo_state": ("/controller_manager/servo_states", "servo_controller_msgs/msg/ServoStateList"),
}

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]

# Arm pose for driving and mapping: camera straight ahead, tilted up from the start pose (console/app.js's
# default "survey" pose, key P; the depth camera's correction, config/camera_correction.json, is calibrated
# for it). The console can save another one per browser (Shift+P); tools use this one.
SURVEY_POSE = {1: 500, 2: 730, 3: 130, 4: 265, 5: 500}

# lidar_frame in base_footprint (x, y, yaw), from the robot's /tf_static (sessions of 2026-10-06):
# base_footprint -> base_link -> back_shell_middle_link -> lidar_frame, no rotation
LIDAR_IN_BASE = (0.073, 0.0, 0.0)

# 2S Li-ion pack; the controller's loud buzzer starts around 7.0 V
BATTERY_LOW_V = 7.4
BATTERY_CRITICAL_V = 7.1
