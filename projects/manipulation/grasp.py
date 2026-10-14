"""Grasping from a script (the core is pick.py; the chat uses manipulation_tools.py). Moves the arm and the base (at
most 10 cm here: the robot is on a cable during these sessions): someone must watch.

    python projects/manipulation/grasp.py down [x y]     # open gripper at grasp height (default: target T5 of the sheet)
    python projects/manipulation/grasp.py teach          # (object centred between the jaws by hand) back out, picture,
                                                          # -> config/grasp_reference.json
    python projects/manipulation/grasp.py pick           # find the red block, follow it in (base + arm), grasp, lift
    python projects/manipulation/grasp.py pick --prompt "red block,red cube"   # found by YOLOE (navigation's detector)
    python projects/manipulation/grasp.py put [x y]      # put it down (default: the middle of the reach), open
    python projects/manipulation/grasp.py up             # back to the survey pose
"""

import sys

if "--prompt" in sys.argv:
    # navigation's open-vocabulary detector (YOLOE; importing another project's module is the repo's stopgap until it
    # moves to landerpi/). It must load torch before numpy's MKL first runs (detector.py), and arm.py computes at
    # import: so first
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "navigation"))
    import detector  # noqa: F401

import argparse
import asyncio
import json
import time

import cv2
import numpy as np

import pick as core
from check_arm import REPORTS, Servos
from landerpi.robot import ROSBRIDGE_URL
from landerpi.rosbridge import RosbridgeClient


def default_target():
    import touch
    from collect_views import latest_sheet
    x, y = touch.target_xy(5, latest_sheet())
    return np.array([x, y, core.GRASP_Z])


async def run(action, xy, prompts, max_drive=None):
    async with RosbridgeClient(ROSBRIDGE_URL) as client:
        s = Servos(client)
        await s.start()
        det = None
        if prompts:
            det = detector.Detector()
        r = core.StandaloneRobot(s, det)
        if max_drive is not None:
            r.max_drive = max_drive
        await r.start()
        if action == "down":
            target = np.array([xy[0], xy[1], core.GRASP_Z]) if len(xy) == 2 else default_target()
            await core.gripper(r, core.OPEN)
            await core.go(r, core.pose_at(core.axis_point(target, core.RETREAT)))
            await core.move_along(r, target, core.RETREAT, 0.0)
            (REPORTS / "grasp_target.json").write_text(json.dumps(target.tolist()))
            print("open gripper at grasp height. Centre the object between the jaws, then: grasp.py teach")
        elif action == "teach":
            target = np.array(json.loads((REPORTS / "grasp_target.json").read_text()))
            await core.move_along(r, target, 0.0, core.RETREAT)
            got = await s.read(core.arm.ARM_IDS)
            await asyncio.sleep(0.6)
            img = await r.snapshot()
            shot = REPORTS / f"grasp_teach_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(shot), img)
            b = await r.find(img, prompts)
            if b is None:
                raise SystemExit(f"nothing found in the picture {shot.name}")
            core.REFERENCE.write_text(json.dumps({
                "description": "where the object appears when the gripper is in the pre-grasp pose straight above it "
                               "(projects/manipulation/grasp.py teach)",
                "date": time.strftime("%Y-%m-%d %H:%M"), "found_by": ",".join(prompts) if prompts else "red filter",
                "down_deg": core.DOWN_DEG, "retreat_m": core.RETREAT, "grasp_z": core.GRASP_Z,
                "block_px": {"x": b[0], "y": b[1], "w": b[2], "h": b[3], "area": b[4]}, "servos": got,
                "picture": shot.name}, indent=1) + "\n")
            print(f"reference: object at ({b[0]:.0f}, {b[1]:.0f}) px, {b[2]} x {b[3]} px; picture {shot.name}")
        elif action == "pick":
            try:
                print(await core.pick(r, prompts))
            except core.GraspError as e:
                raise SystemExit(str(e))
        elif action == "put":
            target = [xy[0], xy[1], core.GRASP_Z] if len(xy) == 2 else None
            await core.put_down(r, target)
            print("put down")
        else:
            await core.survey(r)
            print("survey pose:", await s.read(core.arm.ARM_IDS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["down", "teach", "pick", "put", "up"])
    ap.add_argument("xy", nargs="*", type=float)
    ap.add_argument("--max-drive", type=float, help="m the base may drive during a pick (default 0.10: the cable)")
    ap.add_argument("--prompt", help='what to look for with the YOLOE detector, comma separated (default: the largest '
                    'red blob)')
    a = ap.parse_args()
    asyncio.run(run(a.action, a.xy, [p.strip() for p in a.prompt.split(",")] if a.prompt else None, a.max_drive))


if __name__ == "__main__":
    main()
