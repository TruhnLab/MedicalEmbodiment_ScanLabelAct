"""The colour camera's mounting correction from pictures of the still calibration sheet (collect_views.py).

    python projects/manipulation/solve_camera.py [views_dir ...] [--offsets] [--nominal] [--save]

Model: camera pose = arm.frame("rgb_camera_link", joints) @ X. X (rotation + offset in the camera frame) is one rigid
correction for all poses: the camera sits on link4, so a zero error of joint 4 is part of X. Per set of pictures the
sheet's place on the floor (x, y, turn; it lies flat: with its height and tilt free they traded against X's offset,
2026-10-12). Joint angles from the measured pulses with the measured degrees per pulse (config/arm_joints.json;
--nominal: 0.24 as the robot's /tf). --offsets adds zero errors of joints 1-3 (bounded to +-6 deg). Least squares on
the markers' corner pixels; --save writes X (and the offsets) to config/camera_mount.json.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import arm
import grid_sheet
from check_arm import REPO, REPORTS
from look_at_sheet import marker_corners

OUT = REPO / "config" / "camera_mount.json"
MAX_OFFSET = math.radians(6)


def load(views, nominal=False):
    """(K, D, [(name, joint angles, marker corners on the sheet, their pixels)]) of one set of pictures."""
    info = json.loads((views / "camera_info.json").read_text())
    K, D = np.array(info["k"]), np.array(info["d"])
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    data = []
    for js in sorted(views.glob("[0-9][0-9].json")):
        meta = json.loads(js.read_text())
        corners, ids, _ = det.detectMarkers(cv2.imread(str(js.with_suffix(".jpg"))))
        if ids is None:
            continue
        obj, pix = [], []
        for c, i in zip(corners, ids.flatten()):
            if int(i) in grid_sheet.MARKERS:
                obj.append(marker_corners(int(i)))
                pix.append(c.reshape(4, 2))
        if len(obj) >= 2:
            q = arm.pulses_to_angles({int(k): v for k, v in meta["measured"].items()}, nominal)
            data.append((js.stem, np.array(q), np.vstack(obj), np.vstack(pix)))
    return K, D, data


def T(rv, t):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    M[:3, 3] = t
    return M


def sheet_pose(s):
    return T([0, 0, s[2]], [s[0], s[1], 0])


def residuals(p, sets, offsets, per_view=False):
    X = T(p[0:3], p[3:6])
    dq = np.zeros(5)
    if offsets:
        dq[:3] = p[6:9]
    k = 9 if offsets else 6
    out = []
    for n, (K, D, data) in enumerate(sets):
        S = sheet_pose(p[k + 3 * n:k + 3 * n + 3])
        for _, q, obj, pix in data:
            M = np.linalg.inv(arm.frame("rgb_camera_link", list(q + dq)) @ X) @ S
            proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(M[:3, :3])[0], M[:3, 3], K, D)
            out.append((proj.reshape(-1, 2) - pix).ravel())
    if per_view:
        return [float(np.sqrt(np.mean(r ** 2))) for r in out]
    return np.concatenate(out)


def first_guess(K, D, data):
    """The sheet's place on the floor from one picture with the uncorrected camera."""
    name, q, obj, pix = data[len(data) // 2]
    ok, rv, tv = cv2.solvePnP(obj, pix, K, D)
    S = arm.frame("rgb_camera_link", list(q)) @ T(rv.ravel(), tv.ravel())
    return [S[0, 3], S[1, 3], math.atan2(S[1, 0], S[0, 0])]


def solve(sets, offsets, fix_x=False, sheet=None):
    n0 = 9 if offsets else 6
    p0 = np.concatenate([np.zeros(n0)] + [sheet if sheet is not None else first_guess(*s) for s in sets])
    lo, hi = np.full(len(p0), -np.inf), np.full(len(p0), np.inf)
    if offsets:
        lo[6:9], hi[6:9] = -MAX_OFFSET, MAX_OFFSET
    free = np.ones(len(p0), bool)
    if fix_x:
        free[:n0] = False
    if sheet is not None:
        free[n0:] = False

    def f(x):
        p = p0.copy()
        p[free] = x
        return residuals(p, sets, offsets)
    r = least_squares(f, p0[free], bounds=(lo[free], hi[free]))
    p = p0.copy()
    p[free] = r.x
    return p, math.sqrt(np.mean(r.fun ** 2))


def describe(p, sets, offsets):
    rpy = Rotation.from_rotvec(p[0:3]).as_euler("xyz", degrees=True)
    print(f"  camera correction (camera frame: x right, y down, z forward): rotation about x {rpy[0]:+.2f}, "
          f"y {rpy[1]:+.2f}, z {rpy[2]:+.2f} deg; offset {np.round(p[3:6] * 1000, 1)} mm")
    if offsets:
        print(f"  joint zero errors 1-3: {np.round(np.degrees(p[6:9]), 2)} deg")
    k = 9 if offsets else 6
    for n in range(len(sets)):
        s = p[k + 3 * n:k + 3 * n + 3]
        print(f"  sheet {n + 1}: edge centre at x {s[0]:.3f} y {s[1]:+.3f} m, turned {math.degrees(s[2]):+.2f} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views", nargs="*")
    ap.add_argument("--offsets", action="store_true", help="fit zero errors of joints 1-3 too")
    ap.add_argument("--nominal", action="store_true", help="0.24 deg per pulse for all joints")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--sheet", help="x,y,turn_deg: the sheet's place as the arm found it (one set of pictures); then "
                    "nothing trades against the camera's offset")
    a = ap.parse_args()
    dirs = [Path(v) for v in a.views] or [Path(sorted(glob.glob(str(REPORTS / "views_*")))[-1])]
    sets = [load(d, a.nominal) for d in dirs]
    for d, s in zip(dirs, sets):
        print(f"{d.name}: {len(s[2])} pictures, {sum(len(x[3]) for x in s[2])} corners")
    sheet = None
    if a.sheet:
        x, y, t = (float(v) for v in a.sheet.split(","))
        sheet = [x, y, math.radians(t)]
    p, rms = solve(sets, False, fix_x=True, sheet=sheet)
    print(f"camera as in the URDF{' (nominal 0.24 deg/pulse)' if a.nominal else ''}: corners off by {rms:.2f} px (rms)")
    p, rms = solve(sets, a.offsets, sheet=sheet)
    print(f"with the camera correction{' and joint zero errors' if a.offsets else ''}: {rms:.2f} px (rms)")
    describe(p, sets, a.offsets)
    pv = residuals(p, sets, a.offsets, per_view=True)
    names = [f"{d.name[-6:]}/{x[0]}" for d, s in zip(dirs, sets) for x in s[2]]
    print("  worst pictures:", ", ".join(f"{n} {v:.1f} px" for v, n in sorted(zip(pv, names), reverse=True)[:4]))
    if a.save:
        OUT.write_text(json.dumps({
            "description": "rgb_camera_link correction: camera pose = arm.frame('rgb_camera_link', joints) @ X, joints "
                           "with config/arm_joints.json (projects/manipulation/solve_camera.py)",
            "views": [d.name for d in dirs], "rms_px": round(rms, 3),
            "X": T(p[0:3], p[3:6]).round(6).tolist(),
            "joint_zero_errors_rad": p[6:9].round(5).tolist() if a.offsets else None,
        }, indent=1) + "\n")
        print(f"saved {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
