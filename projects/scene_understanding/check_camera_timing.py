"""Measures the time offset between the colour images and the depth images of a reconstructed session.

    python projects/scene_understanding/check_camera_timing.py 20261005_160227

While the robot turns, a colour image placed with the pose of the wrong moment is rotated against the
depth image. For keyframes taken while turning (>= 15 deg/s), the colour camera is turned by
turn rate x dt for a range of dt, and the script scores how close the depth image's object edges
(projected into the colour image) fall to the colour image's edges. The best dt is the offset:
negative = the colour image shows the scene dt seconds before its timestamp.

Result 2026-10-06: about -0.1 s in 20261005_142856 and 20261005_160227 (the camera driver stamps the colour
images when it publishes them, not at exposure); corrected since in projects/mapping/reconstruct.py
(RGB_TIME_OFFSET).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402

MIN_TURN = 15  # deg/s; slower turns move the image by less than a pixel per 10 ms


def yaw(T):
    return np.degrees(np.arctan2(T[1, 0], T[0, 0]))


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--range", type=float, default=0.3, help="dt from -range to +range seconds")
    ap.add_argument("--offset", type=float, default=None,
                    help="check with the colour poses corrected by this in total (s); default: as in cameras.json")
    a = ap.parse_args()
    s = Session(a.session, rgb_time_offset=a.offset)
    dts = np.round(np.arange(-a.range, a.range + 1e-6, 0.03), 3)
    score = np.zeros(len(dts))
    n = 0
    W, H, Fx, Fy, Cx, Cy = s.K["rgb"]
    w, h, fx, fy, cx, cy = s.K["depth"]
    for prev, f, nxt in zip(s.frames, s.frames[1:], s.frames[2:]):
        rate = ((yaw(nxt.T_map_rgb) - yaw(prev.T_map_rgb) + 180) % 360 - 180) / (nxt.stamp - prev.stamp)
        if abs(rate) < MIN_TURN:
            continue
        gray = cv2.cvtColor(f.rgb(), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        z = f.depth()
        depth_edges = cv2.Canny(np.clip(z / 3.0 * 255, 0, 255).astype(np.uint8), 20, 60)
        v, u = np.nonzero(depth_edges & ((z > 0.3) & (z < 2.5)).astype(np.uint8) * 255)
        if len(v) < 200:
            continue
        zz = z[v, u]
        P = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], 1) @ f.T_map_depth[:3, :3].T + f.T_map_depth[:3, 3]
        for k, dt in enumerate(dts):
            R = rotz(rate * dt) @ f.T_map_rgb[:3, :3]  # the colour camera turned about its vertical axis
            Pc = (P - f.T_map_rgb[:3, 3]) @ R
            ok = Pc[:, 2] > 0.1
            uu = Pc[ok, 0] * Fx / Pc[ok, 2] + Cx
            vv = Pc[ok, 1] * Fy / Pc[ok, 2] + Cy
            inside = (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
            score[k] += np.median(dist[vv[inside].astype(int), uu[inside].astype(int)])
        n += 1
    if not n:
        sys.exit("no keyframes taken while turning")
    score /= n
    best = dts[score.argmin()]
    print(f"{s.name}: {n} keyframes taken while turning (>= {MIN_TURN} deg/s), colour poses corrected by "
          f"{s.rgb_time_offset:+.2f} s")
    for dt, sc in zip(dts, score):
        print(f"  dt {dt:+.2f} s: depth edges {sc:.2f} px from colour edges" + ("  <- best" if dt == best else ""))
    print(f"colour images fit best {abs(best):.2f} s {'before' if best < 0 else 'after'} their timestamps")


if __name__ == "__main__":
    main()
