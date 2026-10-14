"""The exploration runs on the page (assets/js/site.js): the best run in each room we scanned, as its final
lidar map, the path driven and the surfaces the robot's camera had to see (its coverage over time), a few numbers,
and totals over all recordings.

    conda run -n landerpi python tools/export_runs.py

Reads ../LanderPi/data/sessions/*/ and rebuilds each map with the presentation video's own code
(projects/presentation_video/make_video.py: Recording, ScanMap), so the maps look like the videos'.
Writes assets/data/runs/<session>.png (map), <session>_cov.png (coverage over time, see coverage()) and
assets/data/runs.json.
"""

import base64
import json
import math
import sys
import zlib
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


def read_msgs(session, name):
    """A stream's messages in the order of their stamps (robot time, as the poses)."""
    msgs = mv.read_stream(session, name) if (session / "streams" / f"{name}.jsonl").exists() else []
    return sorted(msgs, key=lambda m: mv.stamp(m["header"])) if msgs and "header" in msgs[0] else msgs


def coverage(session, rec, grid, h, w):
    """When each surface cell (in the lidar map's cells, row 0 = top) became a surface to look at, was seen by
    the camera and was given up, from the explorer's /explore/coverage grids (100 seen, 50 not yet, 25 given up;
    offboard_ros/explorer.py). Only the cells that are surfaces at the end. As time codes 1-255 over the run
    (0 = never): R surface, G seen, B given up."""
    msgs = read_msgs(session, "explore_coverage")
    if not msgs:
        return None
    t0, t1 = rec.t_start, rec.t_end
    first = {k: np.full((h, w), np.inf) for k in ("surface", "seen", "given_up")}
    final = np.zeros((h, w), bool)
    for n, m in enumerate(msgs):
        info, t = m["info"], mv.stamp(m["header"])
        data = np.frombuffer(zlib.decompress(base64.b64decode(m["data_int8_zlib_b64"])), np.int8)
        data = data.reshape(info["height"], info["width"])
        rr, cc = np.nonzero(np.isin(data, (25, 50, 100)))
        x = info["origin"]["position"]["x"] + (cc + 0.5) * info["resolution"]
        y = info["origin"]["position"]["y"] + (rr + 0.5) * info["resolution"]
        col = np.floor((x - grid.ox) / grid.RES).astype(int)
        row = h - 1 - np.floor((y - grid.oy) / grid.RES).astype(int)
        ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        row, col, v = row[ok], col[ok], data[rr[ok], cc[ok]]
        for key, sel in (("surface", np.ones(len(v), bool)), ("seen", v == 100), ("given_up", v == 25)):
            a = first[key]
            a[row[sel], col[sel]] = np.minimum(a[row[sel], col[sel]], t)
        if n == len(msgs) - 1:
            final[row, col] = True

    def code(a):
        c = np.zeros(a.shape, np.uint8)
        f = np.isfinite(a) & final
        c[f] = 1 + np.round(254 * np.clip((a[f] - t0) / (t1 - t0), 0, 1)).astype(np.uint8)
        return c

    return np.dstack([code(first["surface"]), code(first["seen"]), code(first["given_up"])])


def status_series(session, offset, t0):
    """(seconds since the start, coverage) whenever the explorer's coverage changes; status messages carry no
    stamp, so their receive time (laptop) is turned into robot time with the recording's clock offset."""
    f = session / "streams" / "explore_status.jsonl"
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = j["msg"].get("coverage")
        if c is None or (out and round(c, 3) == out[-1][1]):
            continue
        out.append([round(j["recv"] - offset - t0, 1), round(c, 3)])
    return out


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
    xy, ts = rec.pose_xy[sel], rec.pose_t[sel]
    length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    path, last = [], None
    for p, t in zip(xy, ts):
        if last is None or math.hypot(*(p - last)) >= PATH_STEP_M:
            x = (p[0] - grid.ox) / grid.RES * SCALE
            y = (h - (p[1] - grid.oy) / grid.RES) * SCALE
            path.append([round(float(x), 1), round(float(y), 1), round(float(t - rec.t_start), 1)])
            last = p

    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    cov = coverage(session, rec, grid, h, w)
    series = None
    if cov is not None:
        # an opaque RGB PNG: the browser reads the codes back unchanged (no premultiplied alpha)
        cv2.imwrite(str(out / f"{session.name}_cov.png"), cv2.cvtColor(cov, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        series = status_series(session, meta.get("clock_offset_s", 0.0), rec.t_start)
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
        "scale": SCALE,
        "seconds": round(rec.t_end - rec.t_start, 1),
        "has_coverage": cov is not None,
        "coverage_series": series,
        "path": path,
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
