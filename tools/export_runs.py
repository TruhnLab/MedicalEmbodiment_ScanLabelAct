"""The exploration runs on the page (assets/js/site.js): the best run in each room we scanned, as its final
lidar map plus the path driven and a few numbers, and totals over all recordings.

    conda run -n landerpi python tools/export_runs.py

Reads ../LanderPi/data/sessions/*/ and rebuilds each map with the presentation video's own code
(projects/presentation_video/make_video.py: Recording, ScanMap), so the maps look like the videos'.
Writes assets/data/runs/<session>.png and assets/data/runs.json.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PAGE = Path(__file__).resolve().parents[1]
CODE = PAGE.parent / "LanderPi"
sys.path.insert(0, str(CODE / "projects" / "presentation_video"))
import make_video as mv  # noqa: E402

FREE = (230, 236, 247)
OCC = (27, 35, 51)
SCALE = 2          # map pixels per 5 cm cell in the PNG (nearest neighbour: crisp walls)
PATH_STEP_M = 0.04  # path points closer than this are dropped

# the best run of each room (user, 2026-10-14: a few good scans, not every recording); rooms told apart by
# their maps and labels: kitchen (bar stools), office (office chairs, standing desk), lounge (sofa, lounge chair)
RUNS = [("Kitchen", "20261007_101108"), ("Office", "20261007_093027"), ("Lounge", "20261007_140425")]


def last_status(session):
    f = session / "streams" / "explore_status.jsonl"
    if not f.exists():
        return None
    for line in reversed(f.read_text(encoding="utf-8").splitlines()):
        try:
            return json.loads(line)["msg"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def export(session, out):
    rec = mv.Recording(session)
    grid = mv.ScanMap(rec)
    img, _ = grid.at(rec.t_end + 2.0)
    free = img[..., 3] > 0
    occ = free & (img[..., 0] < 100)
    rgba = np.zeros_like(img)
    rgba[free] = (*FREE, 255)
    rgba[occ] = (*OCC, 255)
    rgba = np.ascontiguousarray(rgba[::-1])  # row 0 = the map's top
    h, w = rgba.shape[:2]
    rgba = cv2.resize(rgba, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / f"{session.name}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    sel = (rec.pose_t >= rec.t_start) & (rec.pose_t <= rec.t_end)
    xy = rec.pose_xy[sel]
    length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    pts, last = [], None
    for p in xy:
        if last is None or math.hypot(*(p - last)) >= PATH_STEP_M:
            pts.append(p)
            last = p
    pts = np.array(pts)
    px = (pts[:, 0] - grid.ox) / grid.RES * SCALE
    py = (h - (pts[:, 1] - grid.oy) / grid.RES) * SCALE

    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    scene = session / "scene" / "scene.json"
    status = last_status(session)
    t = datetime.strptime(session.name, "%Y%m%d_%H%M%S")
    return {
        "session": session.name,
        "date": t.strftime("%d %b"),
        "time": t.strftime("%H:%M"),
        "minutes": round(meta["duration_s"] / 60, 1),
        "keyframes": meta["counts"]["keyframes"],
        "path_m": round(length, 1),
        "area_m2": round(float(np.count_nonzero(free)) * grid.RES ** 2),
        "objects": len(json.loads(scene.read_text(encoding="utf-8"))["objects"]) if scene.exists() else None,
        "coverage": round(status["coverage"], 2) if status and "coverage" in status else None,
        "w": w * SCALE,
        "h": h * SCALE,
        "path": " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(px, py)),
    }


def path_length(session):
    poses = sorted(mv.read_stream(session, "slam_robot_pose"), key=lambda m: mv.stamp(m["header"]))
    xy = np.array([[m["pose"]["position"]["x"], m["pose"]["position"]["y"]] for m in poses])
    return float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 0.0


def main():
    sessions = CODE / "data" / "sessions"
    out = PAGE / "assets" / "data" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    runs = []
    for room, name in RUNS:
        r = {"room": room, **export(sessions / name, out)}
        runs.append(r)
        print(f"{room} {name}: {r['minutes']} min, {r['path_m']} m, {r['area_m2']} m2, {r['w']}x{r['h']} px")

    # every recording, in all rooms
    alls = [s for s in sorted(sessions.glob("2026*")) if (s / "meta.json").exists()]
    minutes = sum(json.loads((s / "meta.json").read_text(encoding="utf-8"))["duration_s"] for s in alls) / 60
    totals = {"recordings": len(alls), "rooms": len(RUNS), "days": len({s.name[:8] for s in alls}),
              "minutes": round(minutes), "path_m": round(sum(path_length(s) for s in alls))}
    print(totals)
    (PAGE / "assets" / "data" / "runs.json").write_text(json.dumps({"totals": totals, "runs": runs}, separators=(",", ":")),
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
