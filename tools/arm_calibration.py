"""Zero-point calibration of the arm's bus servos (stored in the servos themselves).

    python tools/arm_calibration.py read                  # stored offsets of servos 1-5
    python tools/arm_calibration.py survey                # arm to the survey pose (console key P)
    python tools/arm_calibration.py set 1 -12             # try an offset (lost at power-off)
    python tools/arm_calibration.py set 1 -12 --save      # ... and store it in the servo

A servo's offset shifts what "position 500" means physically, in steps of 0.24 degrees (-125..125).
So after calibration, 500 really is the centre: for our console, the robot's own joint states and
TF (which the 3D reconstruction uses) and Hiwonder's factory apps alike. Measure the error first,
e.g. with projects/mapping/camera_pan_check.py.
"""

import argparse
import asyncio

from landerpi.robot import ROBOT_HOST
from landerpi.rosbridge import RosbridgeClient

URL = f"ws://{ROBOT_HOST}:9090"
GET = "/ros_robot_controller/bus_servo/get_state"
SET = "/ros_robot_controller/bus_servo/set_state"
SET_TYPE = "ros_robot_controller_msgs/msg/SetBusServoState"
SURVEY = {1: 500, 2: 730, 3: 130, 4: 265, 5: 500}  # same as the console's survey pose (key P)
STEP_DEG = 240 / 1000


async def read(client, ids):
    cmd = [{"id": i, "get_id": 0, "get_position": 1, "get_offset": 1, "get_voltage": 0, "get_temperature": 0,
            "get_position_limit": 0, "get_voltage_limit": 0, "get_max_temperature_limit": 0, "get_torque_state": 0}
           for i in ids]
    r = await client.call_service(GET, {"cmd": cmd}, timeout=5.0)
    out = {}
    for i, s in zip(ids, r.get("state", [])):
        out[i] = {"position": (s.get("position") or [None])[-1], "offset": (s.get("offset") or [None])[-1]}
    return out


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["read", "survey", "set"])
    p.add_argument("servo", type=int, nargs="?")
    p.add_argument("offset", type=int, nargs="?")
    p.add_argument("--save", action="store_true", help="store the offset in the servo (survives power-off)")
    a = p.parse_args()
    async with RosbridgeClient(URL) as client:
        if a.action == "read":
            for i, s in (await read(client, [1, 2, 3, 4, 5])).items():
                off = s["offset"]
                print(f"servo {i}: position {s['position']}, stored offset {off}"
                      + (f" ({off * STEP_DEG:+.2f} deg)" if off is not None else ""))
        elif a.action == "survey":
            await client.publish("/servo_controller", "servo_controller_msgs/msg/ServosPosition", {
                "duration": 1.5, "position_unit": "pulse",
                "position": [{"id": i, "position": float(v)} for i, v in SURVEY.items()]})
            await asyncio.sleep(2.0)
            print("survey pose:", await read(client, list(SURVEY)))
        else:
            if a.servo is None or a.offset is None or not -125 <= a.offset <= 125:
                raise SystemExit("usage: set <servo id> <offset -125..125> [--save]")
            state = {"present_id": [1, a.servo], "offset": [1, a.offset]}
            if a.save:
                state["save_offset"] = [1, 1]
            await client.publish(SET, SET_TYPE, {"state": [state], "duration": 0.0})
            await asyncio.sleep(0.5)
            # the offset takes effect on the next move: command the current target again
            pos = SURVEY.get(a.servo, 500)
            await client.publish("/servo_controller", "servo_controller_msgs/msg/ServosPosition", {
                "duration": 0.5, "position_unit": "pulse", "position": [{"id": a.servo, "position": float(pos)}]})
            await asyncio.sleep(1.0)
            print(f"servo {a.servo}: offset {a.offset} ({a.offset * STEP_DEG:+.2f} deg)"
                  f"{' saved' if a.save else ' (not saved)'}; now", (await read(client, [a.servo]))[a.servo])


if __name__ == "__main__":
    asyncio.run(main())
