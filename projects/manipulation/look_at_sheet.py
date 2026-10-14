"""Where the calibration sheet (grid_sheet.py) lies, seen by the colour camera: the arm tilts the camera down at the
sheet, the ArUco markers give the sheet's pose in the camera, the measured servo positions (arm.py) the camera's
pose in base_footprint. Moves the arm (the survey pose with another tilt, approached from above): someone must watch the robot.

    python projects/manipulation/look_at_sheet.py [--tilt 40]

Prints the sheet's bottom edge centre in base_footprint (expected: x ~0.16 = the chassis front, y ~0, z ~0,
turned ~0 deg) and saves the picture with the markers to data/calibration/.
"""

import argparse
import asyncio
import json
import math
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import arm
import grid_sheet
from check_arm import REPORTS, Servos
from landerpi.robot import ROBOT_HOST, ROSBRIDGE_URL, SURVEY_POSE, TOPICS
from landerpi.rosbridge import RosbridgeClient

SNAPSHOT = f"http://{ROBOT_HOST}:8080/snapshot?topic={TOPICS['rgb'][0]}"  # '/' literal (web_video_server)
APPROACH = 40   # pulses above the target first: from above the servos stop 0-5 short, from below 5-13 (joints check)


def marker_corners(mid):
    """The four corners of a marker in the sheet frame (m), in ArUco's order: top-left, top-right, bottom-right,
    bottom-left of the printed image (page up = sheet x, page right = sheet -y)."""
    x, y = grid_sheet.MARKERS[mid]
    h = grid_sheet.MARKER_SIZE / 2
    return np.array([[x + h, y + h, 0], [x + h, y - h, 0], [x - h, y - h, 0], [x - h, y + h, 0]]) / 1000


async def camera_info(client):
    got = asyncio.get_running_loop().create_future()
    await client.subscribe(*TOPICS["rgb_info"], lambda m, r: got.done() or got.set_result(m))
    m = await asyncio.wait_for(got, 5.0)
    await client.unsubscribe(TOPICS["rgb_info"][0])
    return np.array(m["k"]).reshape(3, 3), np.array(m["d"])


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tilt", type=int, default=40, help="servo 4 (survey pose: 265; 40 looks at the sheet's middle)")
    a = p.parse_args()
    pose = {**SURVEY_POSE, 4: a.tilt}
    if arm.collision(arm.pulses_to_angles(pose)):
        raise SystemExit(arm.collision(arm.pulses_to_angles(pose)))
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        K, D = await camera_info(client)
        # the whole arm, not only the tilt: started from a low pose (after touch.py), tilting alone brought the
        # camera to 6 cm above the floor (2026-10-12)
        await s.move({**pose, 4: a.tilt + APPROACH}, 1.5)
        await s.settle(arm.ARM_IDS, 1.5)
        await s.move({4: a.tilt}, 0.8)
        measured = await s.settle(arm.ARM_IDS, 0.8)
        await asyncio.sleep(0.5)  # the picture must be newer than the move
        img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(SNAPSHOT, timeout=5).read(), np.uint8), cv2.IMREAD_COLOR)
    print("servos (measured):", measured)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)
    found = [int(i) for i in ids.flatten()] if ids is not None else []
    found_known = [i for i in found if i in grid_sheet.MARKERS]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shot = REPORTS / f"{stamp}_sheet.jpg"
    REPORTS.mkdir(parents=True, exist_ok=True)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imwrite(str(shot), img)
    print(f"markers seen: {sorted(found)}; picture: {shot}")
    if not found_known:
        raise SystemExit("no markers of the sheet in view: is it in front of the robot? (try --tilt 20 / 60)")
    obj = np.vstack([marker_corners(i) for i, c in zip(found, corners) if i in grid_sheet.MARKERS])
    pix = np.vstack([c.reshape(4, 2) for i, c in zip(found, corners) if i in grid_sheet.MARKERS])
    ok, rvec, tvec = cv2.solvePnP(obj, pix, K, D, flags=cv2.SOLVEPNP_IPPE if len(obj) == 4 else cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pix) ** 2, axis=1))))
    cam_sheet = np.eye(4)
    cam_sheet[:3, :3] = cv2.Rodrigues(rvec)[0]
    cam_sheet[:3, 3] = tvec.ravel()
    base_cam = arm.camera(arm.pulses_to_angles(measured))
    base_sheet = base_cam @ cam_sheet
    o, xa = base_sheet[:3, 3], base_sheet[:3, 0]
    tilt = math.degrees(math.acos(min(1.0, abs(base_sheet[2, 2]))))
    print(f"sheet's bottom edge centre in base_footprint: x {o[0]:.3f} m, y {o[1]:+.3f} m, z {o[2]:+.3f} m; "
          f"turned {math.degrees(math.atan2(xa[1], xa[0])):+.1f} deg; tilted {tilt:.1f} deg vs. the floor "
          f"(camera error: the sheet lies flat); markers fit to {reproj:.1f} px")
    print("expected: x ~0.16 (chassis front, goto.py HALF_LENGTH), y ~0, z ~0")
    (REPORTS / f"{stamp}_sheet.json").write_text(json.dumps({
        "tilt": a.tilt, "measured": measured, "markers": found, "reprojection_px": reproj,
        "base_sheet": base_sheet.round(5).tolist()}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
