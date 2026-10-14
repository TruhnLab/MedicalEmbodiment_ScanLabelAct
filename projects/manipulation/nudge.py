"""Drive the base a few centimetres straight (for calibration set-ups; the robot is on a cable: at most ~10 cm).

    python projects/manipulation/nudge.py -0.08        # 8 cm back, at 5 cm/s
    python projects/manipulation/nudge.py 0.03 -0.04   # 3 cm ahead and 4 cm to the right (Mecanum: sideways)

Open loop (time x speed; 8 cm commanded went 6 cm, 2026-10-12), stop sent several times. Nothing else may drive meanwhile (no navigator, no console).
"""

import asyncio
import math
import sys

from landerpi.robot import ROSBRIDGE_URL, TOPICS
from landerpi.rosbridge import RosbridgeClient

SPEED = 0.05      # m/s
MAX_M = 0.10      # the cable (user, 2026-10-12)


def twist(vx, vy=0.0):
    return {"linear": {"x": vx, "y": vy, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}


async def drive(client, dx, dy=0.0):
    """dx ahead, dy to the left (m), straight, at SPEED."""
    dist = math.hypot(dx, dy)
    if dist > MAX_M:
        raise SystemExit(f"at most {MAX_M} m")
    if dist < 0.005:
        return
    vx, vy = SPEED * dx / dist, SPEED * dy / dist
    try:
        for _ in range(int(dist / SPEED / 0.1)):
            await client.publish(*TOPICS["cmd_vel"], twist(vx, vy))
            await asyncio.sleep(0.1)
    finally:
        for _ in range(5):
            await client.publish(*TOPICS["cmd_vel"], twist(0.0))
            await asyncio.sleep(0.05)


async def main(dx, dy):
    async with RosbridgeClient(ROSBRIDGE_URL) as c:
        await drive(c, dx, dy)
    print(f"drove {dx:+.3f} m ahead, {dy:+.3f} m left (open loop)")


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.0))
