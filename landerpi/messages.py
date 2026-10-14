"""Converting rosbridge JSON messages (dicts) into numbers and numpy arrays."""

import base64
import math

import numpy as np


def stamp(msg):
    """Header stamp in seconds (robot clock)."""
    s = msg["header"]["stamp"]
    return s["sec"] + s["nanosec"] * 1e-9


def image_to_numpy(msg):
    """sensor_msgs/Image -> HxW(xC) array. Supports bgr8/rgb8 and mono16/16UC1 (depth in mm)."""
    data = base64.b64decode(msg["data"]) if isinstance(msg["data"], str) else bytes(msg["data"])
    h, w, step, enc = msg["height"], msg["width"], msg["step"], msg["encoding"]
    rows = np.frombuffer(data, dtype=np.uint8).reshape(h, step)
    if enc in ("bgr8", "rgb8"):
        img = rows[:, : w * 3].reshape(h, w, 3)
        return img if enc == "bgr8" else img[:, :, ::-1]
    if enc in ("mono16", "16UC1"):
        dtype = ">u2" if msg.get("is_bigendian") else "<u2"
        return rows[:, : w * 2].copy().view(dtype).reshape(h, w).astype(np.uint16)
    if enc in ("mono8", "8UC1"):
        return rows[:, :w].copy()
    raise ValueError(f"unsupported image encoding {enc!r}")


def yaw_from_quaternion(q):
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def odom_pose(msg):
    """nav_msgs/Odometry -> (x, y, yaw) of base_footprint in the odom frame."""
    p = msg["pose"]["pose"]
    return p["position"]["x"], p["position"]["y"], yaw_from_quaternion(p["orientation"])


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi
