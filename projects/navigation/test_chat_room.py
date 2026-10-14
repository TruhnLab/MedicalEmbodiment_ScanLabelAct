"""Does the chat model understand the room? Prompts about places ("the wall behind you", "the wall with the two
doors") against the vLLM gateway (the chat key, .env), without the robot: the first tool call is checked, not
run. Variants of what the model gets (user, 2026-10-09):

    list     the object list only (as before)
    text     + the walls and no-go lines (room_view.walls_text) and walls_around in the robot status
    picture  + the room's top view once at the start (room_view.room_picture)

    python projects/navigation/test_chat_room.py [room] [--variants list text picture] [--repeat 2]

The robot stands at POSE (simulated status: localized). Reported per variant: prompts passed, seconds per answer.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assistant  # noqa: E402
import room_view  # noqa: E402
from landerpi import tools  # noqa: E402
from landerpi.room import Room  # noqa: E402

POSE = (1.0, 2.0, math.radians(90))  # room 20261007_101108: facing W5 (cabinets), W2 (doors, outlet) behind


class FakeNav:
    """What the assistant and get_robot_status read from the navigator."""
    def __init__(self, room):
        self.room, self.connected, self.status, self.live = room, True, "localized", None

    def current_pose(self):
        return np.array(POSE)

    def drive_state(self):
        return None

    def note(self, *a):
        pass


def near_wall(name, max_m=1.0):
    def check(room, call):
        if call["name"] != "drive_to_point":
            return False
        w = next(w for w in room_view.room_walls(room) if w["name"] == name)
        p = np.array([call["args"]["x"], call["args"]["y"]], float)
        return room_view._seg_dist(p, w["a"], w["b"])[0] <= max_m and inside(room, p)
    return check


def near_point(xy, max_m):
    def check(room, call):
        if call["name"] != "drive_to_point":
            return False
        p = np.array([call["args"]["x"], call["args"]["y"]], float)
        return float(np.hypot(*(p - xy))) <= max_m and inside(room, p)
    return check


def to_object(oid):
    return lambda room, call: call["name"] == "drive_to_object" and int(call["args"].get("object_id", -1)) == oid


def no_drive(room, call):
    return call["name"] not in ("drive_to_point", "drive_to_object")


def inside(room, p):
    import cv2
    poly = np.array([w["a"] for w in room_view.room_walls(room)], np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) >= 0.2


def centre(room):
    return np.mean([w["a"] for w in room_view.room_walls(room)], 0)


CASES = [  # (prompt, check, what is right)
    ("Drive to the wall behind you.", near_wall("W2"), "a point near W2"),
    ("Go to the wall with the two doors.", near_wall("W2"), "a point near W2"),
    ("Drive to the wall in front of you.", near_wall("W5"), "a point near W5"),
    ("Fahr zur Wand links von dir.", near_wall("W1"), "a point near W1"),
    ("Go to the radiator.", to_object(4), "drive_to_object 4"),
    ("Drive to the middle of the room.", None, "a point within 1 m of the room's centre"),
    ("Drive halfway to the column.", None, "a point between the robot and the column [28]"),
    ("Go to the wall with the radiator.", lambda r, c: near_wall("W1")(r, c) or to_object(4)(r, c),
     "a point near W1 or the radiator"),
    ("Drive to the other side of the no-go line on your right.", no_drive, "no drive (it can't cross)"),
]


def first_call(a, text):
    """The model's first answer to one message in a fresh conversation: (tool call or None, text, seconds)."""
    status = tools.run("get_robot_status", {}, tools.Context(services=a.services))
    msgs = [{"role": "system", "content": a.system_message()}] + a.picture_message() + [
        {"role": "user", "content": f"[robot status now: {json.dumps(status)}]\n"
                                    f"[answer in {assistant.language_of(text)}]\n" + text}]
    t0 = time.time()
    msg = a.ask_model(msgs)
    dt = time.time() - t0
    calls = msg.get("tool_calls") or []
    call = None
    if calls:
        f = calls[0]["function"]
        call = {"name": f["name"], "args": json.loads(f.get("arguments") or "{}")}
    return call, (msg.get("content") or "").strip(), dt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room", nargs="?", default="20261007_101108")
    p.add_argument("--variants", nargs="+", default=["list", "text", "picture"])
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()
    room = Room(args.room)
    nav = FakeNav(room)
    a = assistant.Assistant({"navigator": nav, "call": None})
    walls_text, around = room_view.walls_text, room_view.around
    c = centre(room)
    col = next(o for o in room.objects() if o["id"] == 28)["center"][:2]
    half = (np.array(POSE[:2]) + np.array(col)) / 2
    checks = {"Drive to the middle of the room.": near_point(c, 1.0),
              "Drive halfway to the column.": near_point(half, 0.5)}
    for variant in args.variants:
        room_view.walls_text = walls_text if variant != "list" else (lambda r: "(not given)")
        room_view.around = around if variant != "list" else (lambda r, p: None)
        assistant.MAP_PICTURE = variant == "picture"
        passed, times = 0, []
        print(f"--- {variant}", flush=True)
        for _ in range(args.repeat):
            for text, check, right in CASES:
                check = check or checks[text]
                try:
                    call, answer, dt = first_call(a, text)
                except Exception as e:
                    print(f"  ERROR {text}: {e}", flush=True)
                    continue
                ok = check(room, call) if call else check is no_drive
                passed += ok
                times.append(dt)
                got = f"{call['name']}({', '.join(f'{k}={v}' for k, v in call['args'].items())})" if call else \
                    f"answer: {answer[:80]}"
                print(f"  {'ok ' if ok else 'BAD'} {dt:4.1f} s  {text:58} {got}   (right: {right})", flush=True)
        print(f"{variant}: {passed}/{len(CASES) * args.repeat} right, median {np.median(times):.1f} s per answer",
              flush=True)


if __name__ == "__main__":
    main()
