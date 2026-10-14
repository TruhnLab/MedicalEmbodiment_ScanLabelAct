"""A stand-in for the robot's rosbridge, to test the navigator and the task console without the robot.

Replay: a recorded session's /scan and /odom in real time. Velocity commands are only logged (the
recording drives as it drove; its exploration started with a turn on the spot, so start the replay at 0
to test "localize with a turn"). Loops at the end.

    python projects/navigation/fake_robot.py 20261006_161813 [--start 0] [--speed 1]

Simulation (--sim): a robot in a room's 2D map that follows the velocity commands, with a simulated
lidar (test_goto.World) and odometry: for driving to objects from the task console.

    python projects/navigation/fake_robot.py --sim 20261007_101108 [--pose x y yaw]

Then start the helper with $env:LANDERPI_HOST = "127.0.0.1" and connect the console to
ws://127.0.0.1:9090 (the robot's port; nothing else on the laptop uses it).
"""

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.robot import TOPICS  # noqa: E402
from landerpi.room import SESSIONS  # noqa: E402


def load(session, start):
    msgs = []
    for name, topic in (("scan", TOPICS["scan"][0]), ("odom", TOPICS["odom"][0])):
        with open(SESSIONS / session / "streams" / f"{name}.jsonl") as f:
            for line in f:
                m = json.loads(line)["msg"]
                msgs.append((m["header"]["stamp"]["sec"] + m["header"]["stamp"]["nanosec"] * 1e-9, topic, m))
    msgs.sort(key=lambda x: x[0])
    t0 = msgs[0][0] + start
    return [(t - t0, topic, m) for t, topic, m in msgs if t >= t0]


def header(frame):
    t = time.time()
    return {"stamp": {"sec": int(t), "nanosec": int((t % 1) * 1e9)}, "frame_id": frame}


class Sim:
    """One simulated robot for all clients: follows the latest Twist (stops if none came for
    CMD_TIMEOUT_S), publishes /odom at 30 Hz and /scan at 10 Hz."""
    CMD_TIMEOUT_S = 1.0

    def __init__(self, room_name, pose=None):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from landerpi.room import Room
        from test_goto import World, start_spots
        room = Room(room_name)
        scene = room.session_path / "scene" / "scene.json"
        area = json.loads(scene.read_text()).get("area_polygon") if scene.exists() else None
        self.world = World(room)
        self.pose = np.array(pose if pose else start_spots(room, area, 1, seed=int(time.time()))[0], float)
        self.odom = np.zeros(3)   # odometry starts at 0 where the robot is put down
        self.cmd, self.cmd_time = (0.0, 0.0, 0.0), 0.0
        self.servo = {1: 500, 2: 730, 3: 130, 4: 100, 5: 500, 10: 500}  # the arm's start pose (app.js ARM_HOME)
        self.clients = {}         # websocket -> set of subscribed topics
        print(f"simulated robot in {room_name} at x {self.pose[0]:.2f} y {self.pose[1]:.2f} "
              f"yaw {math.degrees(self.pose[2]):.0f} deg", flush=True)

    async def send(self, topic, msg):
        for ws, subs in list(self.clients.items()):
            if topic in subs:
                try:
                    await ws.send(json.dumps({"op": "publish", "topic": topic, "msg": msg}))
                except websockets.ConnectionClosed:
                    pass

    async def run(self):
        dt, k = 1 / 30, 0
        while True:
            await asyncio.sleep(dt)
            vx, vy, wz = self.cmd if time.time() - self.cmd_time < self.CMD_TIMEOUT_S else (0.0, 0.0, 0.0)
            for pose in (self.pose, self.odom):
                c, s = math.cos(pose[2]), math.sin(pose[2])
                pose += [(c * vx - s * vy) * dt, (s * vx + c * vy) * dt, wz * dt]
                pose[2] = math.atan2(math.sin(pose[2]), math.cos(pose[2]))
            x, y, yaw = self.odom
            await self.send(TOPICS["odom"][0], {
                "header": header("odom"), "child_frame_id": "base_footprint",
                "pose": {"pose": {"position": {"x": x, "y": y, "z": 0.0},
                                  "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}}},
                "twist": {"twist": {"linear": {"x": vx, "y": vy, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": wz}}}})
            k += 1
            if k % 15 == 0:  # 2 Hz, the console subscribes with 200 ms throttle
                await self.send(TOPICS["servo_state"][0], {"servo_state": [{"id": i, "position": p} for i, p in self.servo.items()]})
            if k % 3 == 0:
                await self.send(TOPICS["scan"][0], {
                    "header": header("lidar_frame"), "angle_min": 0.0, "angle_max": 2 * math.pi * 503 / 504,
                    "angle_increment": 2 * math.pi / 504, "time_increment": 0.0, "scan_time": 0.1,
                    "range_min": 0.02, "range_max": 25.0, "ranges": self.world.ranges(self.pose), "intensities": []})


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", nargs="?", help="replay this recorded session")
    p.add_argument("--sim", metavar="ROOM", help="simulate a robot in this room instead")
    p.add_argument("--pose", type=float, nargs=3, metavar=("X", "Y", "YAW"), help="--sim: start pose (yaw in rad)")
    p.add_argument("--start", type=float, default=0.0, help="s into the recording")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--port", type=int, default=9090)
    args = p.parse_args()
    if args.sim:
        sim = Sim(args.sim, args.pose)
        asyncio.get_running_loop().create_task(sim.run())
        msgs = None
    else:
        sim = None
        msgs = load(args.session, args.start)
        print(f"{len(msgs)} messages, {msgs[-1][0]:.0f} s", flush=True)

    async def client(ws):
        subs = set()
        last_cmd = None

        async def replay():
            while True:
                t_start = time.monotonic()
                for t, topic, m in msgs:
                    delay = t / args.speed - (time.monotonic() - t_start)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if topic in subs:
                        await ws.send(json.dumps({"op": "publish", "topic": topic, "msg": m}))
                print("replay: from the start again", flush=True)

        if sim is not None:
            sim.clients[ws] = subs
            task = asyncio.create_task(asyncio.sleep(0))
        else:
            task = asyncio.create_task(replay())
        try:
            async for raw in ws:
                m = json.loads(raw)
                if m["op"] == "subscribe":
                    subs.add(m["topic"])
                    print(f"subscribe {m['topic']}", flush=True)
                elif m["op"] == "publish" and "angular" in m["msg"]:  # Twist
                    if sim is not None and m["topic"] == TOPICS["cmd_vel"][0]:
                        sim.cmd = (m["msg"]["linear"]["x"], m["msg"]["linear"]["y"], m["msg"]["angular"]["z"])
                        sim.cmd_time = time.time()
                    cmd = (m["topic"], m["msg"]["angular"]["z"], m["msg"]["linear"]["x"])
                    if cmd != last_cmd:
                        print(f"command {m['topic']}: wz {cmd[1]:+.2f} vx {cmd[2]:+.2f}", flush=True)
                        last_cmd = cmd
                elif m["op"] == "publish" and sim is not None and m["topic"] == TOPICS["servo_cmd"][0]:
                    for s in m["msg"].get("position", []):  # the arm gets there at once
                        sim.servo[s["id"]] = s["position"]
                    print(f"arm: {sim.servo}", flush=True)
                elif m["op"] == "publish":
                    print(f"command {m['topic']} (not replayed)", flush=True)
                elif m["op"] == "call_service":
                    # no rosapi here: the console falls back to its known LanderPi topics
                    await ws.send(json.dumps({"op": "service_response", "id": m.get("id"), "service": m["service"],
                                              "result": False, "values": "fake robot: no services"}))
        except websockets.ConnectionClosed:
            pass
        finally:
            task.cancel()
            if sim is not None:
                sim.clients.pop(ws, None)

    async with websockets.serve(client, "127.0.0.1", args.port, max_size=None):
        print(f"fake rosbridge on ws://127.0.0.1:{args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
