"""A room the robot works in: the 2D map, the labelled 3D scan and its objects, all in one frame.

    from landerpi.room import Room, list_rooms
    room = Room("20261006_154055")
    room.grid, room.resolution, room.origin       # occupancy (-1 unknown, 0..100), m/cell, (x, y) of cell [0, 0]
    room.objects()                                # scene.json's objects (boxes in the same frame)

A room is made from a labelled mapping session by projects/navigation/make_room.py and lives in
data/rooms/<name>/: room.json (which session, made when), map.pgm + map.yaml (ROS map_server format,
so standard tools read it too). The mesh and labels stay in the session (data/sessions/<s>/
reconstruction, scene/) and are not copied: relabelling the session updates the room.

Frames: the map is the session's last SLAM map (streams/slam_map.jsonl). reconstruct.py's floor
levelling only rotates about the map origin and shifts z; the tilt was 0.0-0.2 deg in all runs of
2026-10-06/07 (logs/reconstruct_*.log), i.e. < 1 cm in x/y at 1.5 m height. So scene.json's boxes are
in the 2D map's frame.
"""

import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOMS = REPO / "data" / "rooms"
SESSIONS = REPO / "data" / "sessions"

# map_server's conventions (map_saver defaults): free 254, occupied 0, unknown 205
PGM_FREE, PGM_OCC, PGM_UNKNOWN = 254, 0, 205


def session_map(session):
    """The last SLAM map a session recorded: (grid int8 [row = y, col = x], resolution, origin x, y)."""
    last = None
    with open(SESSIONS / session / "streams" / "slam_map.jsonl") as f:
        for line in f:
            last = line
    if last is None:
        raise FileNotFoundError(f"{session}: no SLAM map recorded")
    m = json.loads(last)["msg"]
    i = m["info"]
    grid = np.frombuffer(zlib.decompress(base64.b64decode(m["data_int8_zlib_b64"])), np.int8)
    o = i["origin"]["position"]
    return grid.reshape(i["height"], i["width"]).copy(), float(i["resolution"]), (o["x"], o["y"])


def write_map(folder, grid, resolution, origin):
    """map.pgm (top row = largest y, as map_server expects) + map.yaml."""
    img = np.full(grid.shape, PGM_UNKNOWN, np.uint8)
    img[grid == 0] = PGM_FREE
    img[grid >= 65] = PGM_OCC  # occupied_thresh 0.65 below
    cv2.imwrite(str(folder / "map.pgm"), img[::-1])
    (folder / "map.yaml").write_text(
        f"image: map.pgm\nmode: trinary\nresolution: {resolution}\norigin: [{origin[0]}, {origin[1]}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")


def read_map(folder):
    """map.pgm/.yaml -> (grid int8: -1 unknown, 0 free, 100 occupied; resolution; origin x, y)."""
    meta = {}
    for line in (folder / "map.yaml").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    img = cv2.imread(str(folder / meta["image"]), cv2.IMREAD_GRAYSCALE)[::-1]
    grid = np.full(img.shape, -1, np.int8)
    grid[img == PGM_FREE] = 0
    grid[img == PGM_OCC] = 100
    ox, oy = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])
    return grid, float(meta["resolution"]), (ox, oy)


class Room:
    def __init__(self, name):
        self.name = name
        self.path = ROOMS / name
        self.info = json.loads((self.path / "room.json").read_text())
        self.session = self.info["session"]
        self.session_path = SESSIONS / self.session
        self.grid, self.resolution, self.origin = read_map(self.path)

    def objects(self):
        p = self.session_path / "scene" / "scene.json"
        return json.loads(p.read_text())["objects"] if p.exists() else []

    def nogo_lines(self):
        """The exploration's no-go lines (x1, y1, x2, y2), e.g. doors to other rooms."""
        return self.info.get("nogo_lines", [])

    def world_to_cell(self, xy):
        xy = np.asarray(xy, float)
        return np.floor((xy - self.origin) / self.resolution).astype(int)  # (col, row)

    def __repr__(self):
        return f"Room({self.name!r}, session {self.session}, {self.grid.shape[1]}x{self.grid.shape[0]} cells)"


def list_rooms():
    """[{name, session, title, ...}] of every room, newest first."""
    out = []
    for p in sorted(ROOMS.glob("*/room.json"), reverse=True):
        out.append({"name": p.parent.name, **json.loads(p.read_text())})
    return out
