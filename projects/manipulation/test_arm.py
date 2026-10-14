"""Checks arm.fk() against the robot's own /tf of a recorded session (no robot needed).

    python projects/manipulation/test_arm.py [session]      # default 20261007_140425

For every joint state message (sampled), the robot's TF chain base_footprint -> link5 / depth_camera_link at
that time is compared with fk() of the joint angles. 2026-10-09: 0.0 mm, 0.0 deg (see the end of the output).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "navigation"))
import arm  # noqa: E402
from depth_view import TfTable  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    session = REPO / "data" / "sessions" / (sys.argv[1] if len(sys.argv) > 1 else "20261007_140425")
    tf = TfTable()
    static = json.loads((session / "calib" / "tf_static.json").read_text())
    tf.add(static if isinstance(static, dict) else {"transforms": [t for m in static for t in m["transforms"]]})
    joints = [json.loads(line) for line in open(session / "streams" / "joints.jsonl")]
    tfs = (json.loads(line) for line in open(session / "streams" / "tf.jsonl"))
    errors = []
    pending = next(tfs, None)
    for j in joints[::50]:
        while pending is not None and pending["recv"] <= j["recv"] + 0.5:  # TF of the arm follows joint_states
            tf.add(pending["msg"])
            pending = next(tfs, None)
        names, pos = j["msg"]["name"], j["msg"]["position"]
        q = [pos[names.index(n)] for n in ["joint1", "joint2", "joint3", "joint4", "joint5"]]
        f = arm.fk(q)
        for frame, mine in (("link5", f["link5"]), ("depth_camera_link", f["depth_camera_link"])):
            ref = tf.chain(frame)
            if ref is None:
                continue
            d = np.linalg.norm(ref[:3, 3] - mine[:3, 3])
            ang = np.degrees(np.arccos(np.clip((np.trace(ref[:3, :3].T @ mine[:3, :3]) - 1) / 2, -1, 1)))
            errors.append((frame, d, ang, tuple(np.round(q, 2))))
    if not errors:
        raise SystemExit("no TF of the arm in this session")
    for frame in ("link5", "depth_camera_link"):
        e = [x for x in errors if x[0] == frame]
        print(f"{frame}: {len(e)} poses, position error max {max(x[1] for x in e) * 1000:.1f} mm, "
              f"angle max {max(x[2] for x in e):.2f} deg; joint angles seen: {sorted({x[3] for x in e})[:5]}")
    for name, pose in (("survey", {1: 500, 2: 730, 3: 130, 4: 265, 5: 500}),
                       ("start (H)", {1: 500, 2: 730, 3: 130, 4: 100, 5: 500})):
        f = arm.fk(arm.pulses_to_angles(pose))
        print(f"{name}: camera at {np.round(f['depth_camera_link'][:3, 3], 3)}, looking along {np.round(f['depth_camera_link'][:3, 2], 2)}; "
              f"gripper tip at {np.round(f['end_effector_link'][:3, 3], 3)}; collision: {arm.collision(arm.pulses_to_angles(pose))}")


if __name__ == "__main__":
    main()
