"""Makes a room the robot can be put into from a labelled mapping session (landerpi/room.py).

    python projects/navigation/make_room.py 20261006_154055                  # room named after the session
    python projects/navigation/make_room.py 20261007_101108 --title "Lab"    # title shown in the console

Writes data/rooms/<session>/: map.pgm + map.yaml (the session's last SLAM map) and room.json (session,
title, no-go lines of the run, object count). Running it again replaces the room, keeping its title
unless a new one is given (the labels are read from the session, so relabelling alone needs no rerun).
The consoles do the same: "Use as room" in the mapping console's 3D view, "×" in the task console removes one.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.room import ROOMS, SESSIONS, session_map, write_map  # noqa: E402


def last_nogo_lines(session):
    p = SESSIONS / session / "streams" / "nogo_lines.jsonl"
    lines = []
    if p.exists():
        for line in p.read_text().splitlines():
            lines = json.loads(line)["msg"]["lines"]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--title", help="name shown in the console (default: the session)")
    args = p.parse_args()

    s = SESSIONS / args.session
    scene = s / "scene" / "scene.json"
    if not (s / "reconstruction" / "mesh_view.ply").exists():
        sys.exit(f"{args.session}: not reconstructed yet")
    if not scene.exists():
        print(f"warning: {args.session} is not labelled yet (no scene/scene.json): the room has no objects")
    grid, res, origin = session_map(args.session)

    out = ROOMS / args.session
    old = json.loads((out / "room.json").read_text()) if (out / "room.json").exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    write_map(out, grid, res, origin)
    objects = json.loads(scene.read_text())["objects"] if scene.exists() else []
    info = {"session": args.session, "title": args.title or old.get("title") or args.session,
            "made": time.strftime("%Y-%m-%d %H:%M"), "objects": len(objects),
            "nogo_lines": last_nogo_lines(args.session)}
    (out / "room.json").write_text(json.dumps(info, indent=1))
    free = int((grid == 0).sum()) * res * res
    # one line: the mapping console shows it after "Use as room"
    print(f"\"{info['title']}\" from {args.session}: {len(objects)} objects, map {free:.0f} m2 free, "
          f"{len(info['nogo_lines'])} no-go lines (data/rooms/{args.session})")


if __name__ == "__main__":
    main()
