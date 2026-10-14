"""The arm's kinematics on the laptop: servo pulses <-> joint angles, forward kinematics, a collision check
against the chassis and the floor. No ROS.

    import arm
    q = arm.pulses_to_angles({1: 500, 2: 730, 3: 130, 4: 265, 5: 500})
    frames = arm.fk(q)            # {"link1": 4x4, ..., "end_effector_link": 4x4, "depth_camera_link": 4x4}
    arm.collision(q)              # None, or a sentence saying what would hit what

The geometry is the robot's own URDF (config/landerpi.urdf, read from its robot_state_publisher on 2026-10-09):
base_footprint -> ... -> link1 (joint1, about z) -> link2 (joint2) -> link3 (joint3) -> link4 (joint4; the
camera is mounted here) -> link5 (joint5, wrist roll; the gripper) -> end_effector_link. joint2-4 turn about
-y, every axis tilted by ~0.5 deg (as in the URDF). test_arm.py checks fk() against a recording's /tf.
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "config" / "landerpi.urdf"
ARM_IDS = [1, 2, 3, 4, 5]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_ID = 10
# Hiwonder bus servos: 1000 pulses = 240 deg, 500 = joint angle 0. (p - 500) * 0.24 deg gives exactly the
# robot's /joint_states (survey pose 730 / 130 / 265 -> 0.963 / -1.550 / -0.984 rad, 2026-10-09); the URDF's
# limits (+-2.09 rad) are the servos' whole range
RAD_PER_PULSE = math.radians(240) / 1000

# what the arm must not reach into (base_footprint): the chassis is 0.31 x 0.25 m (explore_core.py), its top
# shell ends at ~0.13 m (back_shell_black_link, URDF); margins for the servos' and links' own size
CHASSIS = {"x": (-0.17, 0.17), "y": (-0.14, 0.14), "z_top": 0.15}
FLOOR_Z = 0.01
# the gripper's own body: tilted down over the robot (survey pose, servo 4 commanded 40) it got stuck at 73 with its
# tip at x 0.113, z 0.164 (2026-10-12): within the chassis outline the tip must stay higher than the links
GRIPPER_Z_MIN = 0.175
GRIPPER_X_MAX = 0.15   # ... over the robot only: the chassis' front is at ~0.16 (goto.py); the way to the grasp
                       # lookout passes x 0.17 at 10 cm height (2026-10-12)


# degrees per pulse per joint (config/arm_joints.json; nominal 0.24 fits the user's measurements, 2026-10-12: the
# camera sweeps' 0.232-0.265 put the gripper 32 mm off)
JOINTS_FILE = Path(__file__).resolve().parents[2] / "config" / "arm_joints.json"
try:
    _deg = json.loads(JOINTS_FILE.read_text())["deg_per_pulse"]
    RAD_PER_PULSE_JOINT = {i: math.radians(float(_deg[str(i)])) for i in ARM_IDS}
except (OSError, ValueError, KeyError):
    RAD_PER_PULSE_JOINT = {i: RAD_PER_PULSE for i in ARM_IDS}


def pulses_to_angles(pulses, nominal=False):
    """{servo id: pulse} -> [q1..q5] in rad (missing joints: 0). nominal: 0.24 deg per pulse, as the robot's /tf."""
    return [(pulses.get(i, 500) - 500) * (RAD_PER_PULSE if nominal else RAD_PER_PULSE_JOINT[i]) for i in ARM_IDS]


def angles_to_pulses(q):
    return {i: int(round(500 + a / RAD_PER_PULSE_JOINT[i])) for i, a in zip(ARM_IDS, q)}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _load(path=URDF):
    """child link -> (parent link, joint name, joint type, 4x4 origin, axis); {} if the file is missing (MISSING)."""
    if not path.exists():  # not in the repository (Hiwonder's file): projects/manipulation/README.md
        return {}
    tree = {}
    for j in ET.parse(path).getroot().findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        M = np.eye(4)
        M[:3, :3] = _rpy(*rpy)
        M[:3, 3] = xyz
        a = j.find("axis")
        axis = [float(v) for v in a.get("xyz").split()] if a is not None else [1, 0, 0]
        tree[j.find("child").get("link")] = (j.find("parent").get("link"), j.get("name"), j.get("type"), M, axis)
    return tree


TREE = _load()
# Without the URDF the module still imports and frame() says what is missing: the navigator's chat imports the
# manipulation tools, and an error at import stopped the whole chat (a public clone has no URDF, 2026-10-14)
MISSING = None if TREE else f"{URDF} is missing: export the robot's URDF (projects/manipulation/README.md)"
# the optical frames are not in the URDF: the camera driver publishes them on /tf_static (session 20261007_140425)
_OPTICAL = np.eye(4)
_OPTICAL[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # quaternion (-0.5, 0.5, -0.5, 0.5): z forward, x right
_OPTICAL[:3, 3] = [0.0, 0.02, 0.0]
TREE["depth_camera_link"] = ("depth_cam_link", "", "fixed", _OPTICAL, [1, 0, 0])
_RGB = np.eye(4)  # depth_camera_link -> rgb_camera_link (optical as well), quaternion (0, 0, -0.009, 1)
_RGB[:3, :3] = [[1, 0.018, 0], [-0.018, 1, 0], [0, 0, 1]]
_RGB[:3, 3] = [-0.0100, 0.0003, 0.0006]
TREE["rgb_camera_link"] = ("depth_camera_link", "", "fixed", _RGB, [1, 0, 0])
BASE_TO_JOINT1 = None  # set below (link1's origin in base_footprint)
# the jaw tips (closed to ~13 mm) reach 13 mm beyond the URDF's end_effector_link, along the gripper: held over T2
# by the nominal model, the user measured the tips 25 mm above the paper and over T5 (model: 37 mm, 3 mm behind T5;
# 2026-10-12, one clean reading: to be confirmed)
JAW_TIPS = 0.013
_TIPS = np.eye(4)
_TIPS[2, 3] = JAW_TIPS
TREE["jaw_tips"] = ("end_effector_link", "", "fixed", _TIPS, [1, 0, 0])
FRAMES = ["link1", "link2", "link3", "link4", "link5", "end_effector_link", "jaw_tips", "depth_camera_link",
          "gripper_link"]


def frame(name, q, root="base_footprint"):
    """One link in root (4x4) for arm joint angles q = [q1..q5] (other moving joints at 0)."""
    if MISSING:
        raise FileNotFoundError(MISSING)
    angles = dict(zip(ARM_JOINTS, q))
    T = np.eye(4)
    while name != root:
        parent, joint, kind, M, axis = TREE[name]
        J = np.eye(4)
        if kind == "revolute":
            J[:3, :3] = _axis_angle(axis, angles.get(joint, 0.0))
        T = M @ J @ T
        name = parent
    return T


def fk(q):
    """Joint angles [q1..q5] (rad) -> {link: 4x4 in base_footprint} for FRAMES."""
    return {f: frame(f, q) for f in FRAMES}


def collision(q, chassis=True):
    """None if the arm stays clear of the floor and the chassis in this pose, else a sentence. Checks points
    along the links (the links' own thickness is in the margins), not the camera's or gripper's body.
    chassis=False: only the floor and the gripper's measured contact with the robot's front; the CHASSIS box is a
    guess (2026-10-12), for moves that stop at contact (check_arm.Servos.guarded_move) instead."""
    f = fk(q)
    chain = ["link2", "link3", "link4", "link5", "end_effector_link", "jaw_tips"]
    for a, b in zip(chain, chain[1:]):
        for t in np.linspace(0, 1, 6):
            x, y, z = (1 - t) * f[a][:3, 3] + t * f[b][:3, 3]
            if z < FLOOR_Z:
                return f"{b} would touch the floor (z {z:.3f} m)"
            over = CHASSIS["x"][0] < x < CHASSIS["x"][1] and CHASSIS["y"][0] < y < CHASSIS["y"][1]
            if chassis and over and b in ("link4", "link5", "end_effector_link") and z < CHASSIS["z_top"]:
                return f"{b} would hit the chassis (at {x:.2f}, {y:.2f}, {z:.2f} m)"
            if over and x < GRIPPER_X_MAX and b == "end_effector_link" and t == 1 and z < GRIPPER_Z_MIN:
                return f"the gripper would hit the robot's front (tip at {x:.2f}, {y:.2f}, {z:.2f} m)"
    return None


def approach_dir(target, down_deg):
    """Unit vector pointing from the robot towards target (horizontally), tilted down_deg below horizontal."""
    az = math.atan2(target[1] - BASE_TO_JOINT1[1], target[0] - BASE_TO_JOINT1[0])
    d = math.radians(down_deg)
    return np.array([math.cos(d) * math.cos(az), math.cos(d) * math.sin(az), -math.sin(d)])


def ik(target, down_deg, q0=None, iterations=200):
    """Joint angles putting the jaw tips at target (base_footprint, m) with the gripper pointing down_deg
    below horizontal, towards the target; wrist roll 0. Damped least squares on the URDF's fk. Returns (q, position
    error in m, direction error in deg); the caller checks the error, the joint limits and collision()."""
    want = approach_dir(target, down_deg)
    limits = np.array([500 * RAD_PER_PULSE_JOINT[i] for i in ARM_IDS])  # the servos' 0-1000 range
    q = np.array(q0 if q0 is not None else pulses_to_angles({1: 500, 2: 730, 3: 130, 4: 265, 5: 500}), float)
    W = 0.05  # 1 rad of direction counts like 5 cm of position

    def residual(q):
        f = frame("jaw_tips", list(q))
        return np.concatenate([f[:3, 3] - target, W * (f[:3, 2] - want)])

    for _ in range(iterations):
        r = residual(q)
        J = np.zeros((6, 4))
        for j in range(4):
            dq = np.zeros(5)
            dq[j] = 1e-5
            J[:, j] = (residual(q + dq) - r) / 1e-5
        if np.linalg.norm(r[:3]) < 2e-5 and np.linalg.norm(r[3:]) < 2e-5:
            break  # 0.02 mm, ~0.02 deg: from a good seed after 3-5 steps (all 80 took ~18 ms each time, blocking the
                   # navigator's event loop while the arm followed an object, 2026-10-12)
        step = np.linalg.solve(J.T @ J + 1e-6 * np.eye(4), -J.T @ r)
        q[:4] += np.clip(step, -0.2, 0.2)
        q[:4] = np.clip(q[:4], -limits[:4], limits[:4])
        if np.linalg.norm(step) < 1e-7:
            break
    f = frame("jaw_tips", list(q))
    return q, float(np.linalg.norm(f[:3, 3] - target)), \
        float(np.degrees(np.arccos(np.clip(f[:3, 2] @ want, -1, 1))))


# starting poses for ik(): from the survey pose alone it found no floor pose; these reach floor points 20-30 cm
# ahead with the gripper 30-60 deg down (multi-start search, 2026-10-12)
SEEDS = [{1: 500, 2: 170, 3: 283, 4: 423}, {1: 500, 2: 164, 3: 185, 4: 590}, {1: 500, 2: 97, 3: 185, 4: 720},
         {1: 500, 2: 730, 3: 130, 4: 265}]


def planar_ik(target, down_deg):
    """Starting poses for ik(): the arm as a plane (pan, then three parallel joints; links 0.0741, 0.0741, then
    0.139 m from link4 to the jaw tips). joint2-4 turn about -y: a link's direction is (forward -sin, up cos) of the
    sum of the angles before it. Both elbow solutions; the URDF's small offsets are left to ik()."""
    sh = frame("link2", [0, 0, 0, 0, 0])[:3, 3]
    az = math.atan2(target[1] - BASE_TO_JOINT1[1], target[0] - BASE_TO_JOINT1[0])
    r = math.hypot(target[0] - BASE_TO_JOINT1[0], target[1] - BASE_TO_JOINT1[1]) - (sh[0] - BASE_TO_JOINT1[0])
    d = math.radians(down_deg)
    w_fwd, w_up = r - 0.139 * math.cos(d), target[2] + 0.139 * math.sin(d) - sh[2]
    L = 0.0741
    dist = math.hypot(w_fwd, w_up)
    if dist > 2 * L or dist < 1e-6:
        return []
    t = -math.atan2(w_fwd, w_up)
    beta = math.acos(min(1.0, dist / (2 * L)))
    t4 = math.atan2(-math.cos(d), -math.sin(d))
    out = []
    for sgn in (1, -1):
        q2, q3 = t + sgn * beta, -2 * sgn * beta
        q4 = math.remainder(t4 - q2 - q3, 2 * math.pi)
        out.append([az, q2, q3, q4, 0.0])
    return out


def reach(target, q0=None, max_err=0.002, chassis=True):
    """The steepest gripper direction (90 deg = straight down, in 5 deg steps down to 0) that reaches target without
    a collision: (q, down_deg) or (None, None)."""
    az = math.atan2(target[1] - BASE_TO_JOINT1[1], target[0] - BASE_TO_JOINT1[0])
    seeds = ([q0] if q0 is not None else []) + [[az] + pulses_to_angles(p)[1:] for p in SEEDS]  # turned to it
    for down in range(90, -1, -5):
        for s in planar_ik(target, down) + seeds:
            q, err, _ = ik(target, down, s, iterations=80)
            if err <= max_err and collision(q, chassis) is None:
                return q, down
    return None, None


def path_collision(q_from, q_to, steps=20, chassis=True):
    """The servos move all joints linearly from q_from to q_to: the first collision on the way, or None."""
    for t in np.linspace(0, 1, steps + 1)[1:]:
        hit = collision(list((1 - t) * np.asarray(q_from) + t * np.asarray(q_to)), chassis)
        if hit:
            return hit
    return None


BASE_TO_JOINT1 = None if MISSING else frame("link1", [0, 0, 0, 0, 0])[:3, 3]


# the colour camera's mounting correction (solve_camera.py; the URDF's camera is off by ~2-3 deg, 2026-10-12)
CAMERA_FILE = Path(__file__).resolve().parents[2] / "config" / "camera_mount.json"
try:
    CAMERA_X = np.array(json.loads(CAMERA_FILE.read_text())["X"])
except (OSError, ValueError, KeyError):
    CAMERA_X = np.eye(4)


def camera(q):
    """The colour camera's optical frame (rgb_camera_link, corrected) in base_footprint for joint angles q."""
    return frame("rgb_camera_link", q) @ CAMERA_X
