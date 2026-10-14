"""On-disk layout of a mapping session (documented in README.md next to this file).

data/sessions/<YYYYmmdd_HHMMSS>/
    meta.json               config, counts, clock offset, stop reason
    calib/                  rgb_camera_info.json, depth_camera_info.json, tf_static.json
    rgb/<stamp>.jpg         colour keyframes, the robot's JPEG saved as received (MJPEG, quality ~90)
    depth/<stamp>.png       depth keyframes, 16-bit PNG in millimetres, saved as received (lossless)
    keyframes.csv           one row per keyframe: stamps, files, odom pose at that time
    associations.txt        TUM RGB-D style: "t_rgb rgb/x.jpg t_depth depth/y.png"
    streams/<name>.jsonl    every scan/odom/imu/tf/joints/battery message: {"recv": laptop time, "msg": ...}

Every image keeps its own capture stamp (robot clock); rgb and depth of a keyframe may be a few
tens of milliseconds apart, and each one's camera pose is taken at its own stamp later.
"""

import json
import threading
from pathlib import Path

FORMAT_VERSION = 2
KEYFRAME_COLUMNS = ["index", "stamp", "rgb_stamp", "depth_stamp", "rgb", "depth", "odom_x", "odom_y", "odom_yaw"]


class SessionWriter:
    def __init__(self, root: Path):
        self.root = root
        for sub in ("rgb", "depth", "calib", "streams"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.bytes_written = 0
        self._lock = threading.Lock()
        self._streams = {}
        self._associations = []
        self._index = open(root / "keyframes.csv", "w", encoding="utf-8", newline="\n")
        self._index.write(",".join(KEYFRAME_COLUMNS) + "\n")

    def append_stream(self, name, msg, recv_time):
        """Small topics: one JSON line per message. Called from the event loop only."""
        f = self._streams.get(name)
        if f is None:
            f = self._streams[name] = open(self.root / "streams" / f"{name}.jsonl", "w", encoding="utf-8")
        line = json.dumps({"recv": round(recv_time, 6), "msg": msg}, separators=(",", ":")) + "\n"
        f.write(line)
        self._add_bytes(len(line))

    def write_json(self, relpath, obj):
        path = self.root / relpath
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        self._add_bytes(path.stat().st_size)

    def write_keyframe(self, index, t_rgb, jpeg, t_depth, png, pose):
        """Runs in a worker thread. Both images are stored exactly as they came from the robot."""
        rgb_file, depth_file = f"rgb/{t_rgb:.6f}.jpg", f"depth/{t_depth:.6f}.png"
        (self.root / rgb_file).write_bytes(jpeg)
        (self.root / depth_file).write_bytes(png)
        x, y, yaw = pose
        row = [index, f"{t_rgb:.6f}", f"{t_rgb:.6f}", f"{t_depth:.6f}", rgb_file, depth_file,
               f"{x:.4f}", f"{y:.4f}", f"{yaw:.5f}"]
        with self._lock:
            self._index.write(",".join(map(str, row)) + "\n")
            self._associations.append(f"{t_rgb:.6f} {rgb_file} {t_depth:.6f} {depth_file}")
            self.bytes_written += len(jpeg) + len(png)

    def close(self, meta):
        for f in self._streams.values():
            f.close()
        self._index.close()
        (self.root / "associations.txt").write_text("\n".join(sorted(self._associations)) + "\n", encoding="utf-8")
        self.write_json("meta.json", {"format_version": FORMAT_VERSION, **meta})

    def _add_bytes(self, n):
        with self._lock:
            self.bytes_written += n
