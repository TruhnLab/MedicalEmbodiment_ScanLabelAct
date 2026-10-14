"""Each arm joint's real degrees per servo pulse, from single-joint sweeps (collect_views.py --sweeps).

    python projects/manipulation/solve_gains.py [views_dir]

Within a sweep only one joint moves, so the camera's rotation between two pictures (from the sheet's markers) is that
joint's angle, whatever the camera's mounting or the sheet's place. Fitted against the servos' measured positions.
"""

import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import arm
import solve_camera as sc
from check_arm import REPORTS


def main():
    views = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(REPORTS / "views_*")))[-1])
    K, D, data = sc.load(views)
    rows = []
    for name, q, obj, pix in data:
        meta = json.loads((views / f"{name}.json").read_text())
        ok, rv, tv = cv2.solvePnP(obj, pix, K, D)
        cam = cv2.Rodrigues(rv)[0].T   # camera orientation in the sheet frame
        rows.append((meta["command"], {int(k): v for k, v in meta["measured"].items()}, cam))
    result = {}
    for joint in arm.ARM_IDS[:4]:
        # the sweep of this joint: the largest group of pictures whose commands differ only in this joint
        sweeps = {}
        for r in rows:
            key = tuple(r[0][str(i)] for i in arm.ARM_IDS if i != joint)
            sweeps.setdefault(key, []).append(r)
        group = max(sweeps.values(), key=lambda g: len({x[0][str(joint)] for x in g}))
        if len(group) < 3:
            continue
        ref = group[0]
        dp, ang = [], []
        for r in group:
            d = r[1][joint] - ref[1][joint]
            R = ref[2].T @ r[2]
            a = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            dp.append(d)
            ang.append(math.copysign(a, d) if d else 0.0)
        dp, ang = np.array(dp, float), np.array(ang)
        slope = float(dp @ ang / (dp @ dp))
        rms = float(np.sqrt(np.mean((ang - slope * dp) ** 2)))
        result[joint] = slope
        print(f"joint {joint}: {len(group)} pictures over {int(dp.max() - dp.min())} pulses: {slope:.4f} deg/pulse "
              f"({slope / 0.24:.3f} x nominal), off the line by {rms:.2f} deg (rms)")
    return result


if __name__ == "__main__":
    main()
