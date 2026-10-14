"""Presentation video of a task-console run (navigation's ● Record, data/task_runs/<time>/): the robot's camera with
the detector's boxes, the room's top view with the robot's track, the target from the room's object list and what
was found, the steps as they happen and the chat, sped up. Made afterwards from the recording; nothing extra is
recorded for it.

    python projects/presentation_video/task_run_video.py                       # the newest recorded run
    python projects/presentation_video/task_run_video.py data/task_runs/<time> --speed 2
    python projects/presentation_video/task_run_video.py --still 8            # one frame (s of video) as PNG

Inputs (navigation's recording, projects/navigation/taskrun.py): frames/*.jpg with frames.jsonl (time, the robot's
pose in the room, the detector's boxes), events.jsonl (chat, tools, drives, finds), meta.json (the room); the room's
map and objects (landerpi.room). Output: data/demo/<run>.mp4 (H.264, 1920 x 1080, 30 fps) and a PNG of its last frame.
"""

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from style import (ACCENT, BORDER, BOX, F_BOLD, F_SMALL, F_TITLE, FPS, H, MAP_FREE, MAP_OCC, MUTED, OK, ROBOT_FILL,
                   SURFACE, TEXT, USER_FILL, W, bubble, speed_badge, steps, write_video)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from landerpi.room import Room  # noqa: E402

RUNS = REPO / "data" / "task_runs"
OUT = REPO / "data" / "demo"

CAM = (40, 140, 1120, 700)          # the camera picture (640 x 400 scaled x1.75)
MAP = (1200, 140, 680, 520)         # the room's top view
LEAD_S, TAIL_S = 1.5, 4.0           # video seconds before the request and after the end
ROBOT_L, ROBOT_W = 0.31, 0.25       # m, the chassis (goto.py)
FOV_DEG, FOV_M = 68, 1.2
MARGIN_M = 1.8                      # of the room around the track on the map            # the camera's view on the map (the depth camera's 74 deg, minus the edges)
OBJ_LINE = (176, 182, 194)


# ---------- reading the run ----------

def tool_args(text):
    """The arguments of a "chat tool name {...} -> ..." line (a Python dict literal)."""
    m = re.match(r"chat tool (\w+) (\{.*?\}) ->", text)
    if not m:
        return None, {}
    try:
        return m[1], ast.literal_eval(m[2])
    except (ValueError, SyntaxError):
        return m[1], {}


def step_list(events, objects):
    """[(time, text)] of what the robot did, from the run's events, in words for an audience."""
    names = {o["id"]: o["label"] for o in objects}
    out = []
    for e in events:
        t, kind, text = e["time"], e["kind"], e["text"]
        if kind == "tool":
            name, args = tool_args(text)
            if name == "watch_for":
                out.append((t, f"Look out for the {args.get('item', 'item')}: the detector runs on the live camera"))
            elif name == "search_here":
                out.append((t, f"Search here for the {args.get('item', 'item')}: camera sweeps"))
            elif name == "approach_found" and '"arrived"' in text:
                out.append((t, "Arrived: about 20 cm in front of it, facing it"))
            elif name == "drive_to_object" and '"arrived"' in text:
                out.append((t, f"Arrived at the {names.get(args.get('object_id'), 'object')}"))
        elif kind == "drive":
            m = re.match(r"goto #(\d+) (.+)", text)
            if m:
                out.append((t, f"Drive towards the {m[2]}: a target from the room's object list"))
            m = re.match(r"goto found (.+?) at", text)
            if m:
                out.append((t, f"Drive up to the {m[1]}"))
        elif kind == "found":
            m = re.match(r"found (.+?) at", text)
            if m:
                out.append((t, f"{m[1].capitalize()} spotted on the way: the drive stops at once"))
    # a step's line is written when it ended (tool results): sorted by time, duplicates merged
    out.sort(key=lambda s: s[0])
    merged = []
    for s in out:
        if not merged or merged[-1][1] != s[1]:
            merged.append(s)
    return merged


def load(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    events = [json.loads(line) for line in open(run_dir / "events.jsonl", encoding="utf-8")]
    frames = [json.loads(line) for line in open(run_dir / "frames.jsonl", encoding="utf-8")]
    room = Room(meta["room"])
    objects = room.objects()
    req = next(e for e in events if e["kind"] == "user")
    answers = [e for e in events if e["kind"] == "assistant"]
    target = None
    for e in events:
        m = re.match(r"goto #(\d+) ", e["text"]) if e["kind"] == "drive" else None
        if m:
            target = next((o for o in objects if o["id"] == int(m[1])), None)
    finds = []
    for e in events:
        m = re.match(r"found (.+?) at \(([-\d.]+), ([-\d.]+)\)", e["text"]) if e["kind"] == "found" else None
        if m:
            finds.append((e["time"], m[1], float(m[2]), float(m[3])))
    st = step_list(events, objects)
    t_end = max(st[-1][0] if st else frames[-1]["time"], answers[-1]["time"] if answers else 0)
    return {"name": run_dir.name, "room": room, "objects": objects, "req": req, "answers": answers,
            "target": target, "finds": finds, "steps": st, "frames": frames, "dir": run_dir,
            "t_req": req["time"], "t_end": min(t_end, frames[-1]["time"] + 2)}


# ---------- the room's top view ----------

class TopView:
    """The room's map cropped to where the run happened (and the room's outline), scaled into MAP."""

    def __init__(self, run):
        room = run["room"]
        g, res, (ox, oy) = room.grid, room.resolution, room.origin
        # the track, the target and the finds, with MARGIN_M of the room around them (the whole kitchen made the
        # track a short strip in a corner)
        keep = [f["pose"][:2] for f in run["frames"]] + [(x, y) for _, _, x, y in run["finds"]]
        if run["target"] is not None:
            keep.append(run["target"]["center"][:2])
        keep = np.array(keep, float)
        lo, hi = keep.min(0) - MARGIN_M, keep.max(0) + MARGIN_M
        # world -> cell, cropped
        c0, r0 = np.floor((lo - (ox, oy)) / res).astype(int)
        c1, r1 = np.ceil((hi - (ox, oy)) / res).astype(int)
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, g.shape[1]), min(r1, g.shape[0])
        crop = g[r0:r1, c0:c1][::-1]  # top row = largest y
        img = np.full(crop.shape + (3,), SURFACE, np.uint8)
        img[crop == 0] = MAP_FREE
        img[crop >= 65] = MAP_OCC
        self.scale = min(MAP[2] / img.shape[1], MAP[3] / img.shape[0])
        size = (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale))
        self.base = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        self.ox, self.oy = MAP[0] + (MAP[2] - size[0]) // 2, MAP[1]
        self.x0, self.y1, self.res = ox + c0 * res, oy + r1 * res, res

    def px(self, x, y):
        """World -> pixel in the map's own picture (self.base)."""
        return ((x - self.x0) / self.res * self.scale, (self.y1 - y) / self.res * self.scale)

    def box(self, o):
        c, s = math.cos(math.radians(o.get("yaw_deg", 0))), math.sin(math.radians(o.get("yaw_deg", 0)))
        hx, hy = o["size"][0] / 2, o["size"][1] / 2
        return [self.px(o["center"][0] + c * dx - s * dy, o["center"][1] + s * dx + c * dy)
                for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


# ---------- the video ----------

def label_at(d, width, x, y, text, colour, centred=True):
    """A label centred at x (or starting at x) on a white backing, kept inside the map (width px)."""
    w = d.textlength(text, font=F_BOLD)
    x = min(max(x - w / 2 if centred else x, 4), width - w - 4)
    d.rounded_rectangle([x - 6, y, x + w + 6, y + 32], 8, fill=SURFACE)
    d.text((x, y + 1), text, font=F_BOLD, fill=colour)


def render(run, view, t, cache, speed):
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((CAM[0], 36), "LanderPi finds an object it was asked for and drives to it", font=F_TITLE, fill=TEXT)
    d.text((CAM[0], 92), "the robot's camera and its map of the room  ·  everything after the request runs on its "
                         "own", font=F_SMALL, fill=MUTED)
    el = max(0.0, t - run["t_req"])
    speed_badge(d, MAP[0], 60, speed, f"{int(el // 60)}:{int(el % 60):02d} since the request (real time)")
    # the camera
    fr = run["frames"]
    past = [f for f in fr if f["time"] <= t]
    f = past[-1] if past else fr[0]
    path = run["dir"] / "frames" / f"{f['n']:06d}.jpg"
    if path not in cache:
        cache.clear()
        cache[path] = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    pic = cv2.resize(cache[path], (CAM[2], CAM[3]), interpolation=cv2.INTER_LINEAR)
    if not past:
        pic = (pic * 0.45 + 255 * 0.55).astype(np.uint8)
    img.paste(Image.fromarray(pic), (CAM[0], CAM[1]))
    k = CAM[2] / 640
    for b in f["boxes"] if past else []:
        if not (b.get("found") or b.get("relevant")):
            continue
        x0, y0, x1, y1 = (CAM[0] + b["box"][0] * k, CAM[1] + b["box"][1] * k, CAM[0] + b["box"][2] * k,
                          CAM[1] + b["box"][3] * k)
        d.rectangle([x0, y0, x1, y1], outline=BOX, width=5 if b.get("found") else 2)
        if b.get("found"):  # the label only on the confirmed find, inside the picture
            label = b["label"]
            ly = y0 - 34 if y0 - 34 >= CAM[1] else y1
            d.rectangle([x0, ly, x0 + 16 + d.textlength(label, font=F_BOLD), ly + 34], fill=BOX)
            d.text((x0 + 8, ly + 2), label, font=F_BOLD, fill=TEXT)
    d.rectangle([CAM[0] - 1, CAM[1] - 1, CAM[0] + CAM[2], CAM[1] + CAM[3]], outline=BORDER, width=2)
    # the chat under the camera: the request, the robot's latest answer
    half = (CAM[2] - 20) // 2
    if t >= run["t_req"]:
        bubble(d, CAM[0], CAM[1] + CAM[3] + 22, half, "User (chat)", run["req"]["text"][len("chat: "):], ACCENT,
               USER_FILL, max_lines=3)
    said = [a for a in run["answers"] if a["time"] <= t]
    if said:
        bubble(d, CAM[0] + half + 20, CAM[1] + CAM[3] + 22, half, "Robot (chat)",
               said[-1]["text"][len("assistant: "):], OK, ROBOT_FILL, max_lines=4)
    # the room: drawn on its own picture, so nothing reaches past the map's edge
    mimg = Image.fromarray(view.base).convert("RGBA")
    dm = ImageDraw.Draw(mimg)
    layer = Image.new("RGBA", mimg.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    for o in run["objects"]:
        dm.polygon(view.box(o), outline=OBJ_LINE)
    tgt = run["target"]
    if tgt is not None:
        dm.polygon(view.box(tgt), outline=ACCENT, width=4)
    trail = [view.px(*g["pose"][:2]) for g in past]
    if len(trail) > 1:
        dm.line(trail, fill=ACCENT, width=4, joint="curve")
    if past:
        x, y, yaw = f["pose"]
        a0, a1 = yaw - math.radians(FOV_DEG / 2), yaw + math.radians(FOV_DEG / 2)
        cone = [view.px(x, y)] + [view.px(x + FOV_M * math.cos(a), y + FOV_M * math.sin(a))
                                  for a in np.linspace(a0, a1, 12)]
        dl.polygon(cone, fill=ACCENT + (45,))
        c, s_ = math.cos(yaw), math.sin(yaw)
        body = [view.px(x + c * dx - s_ * dy, y + s_ * dx + c * dy)
                for dx, dy in ((-ROBOT_L / 2, -ROBOT_W / 2), (ROBOT_L / 2, -ROBOT_W / 2), (ROBOT_L / 2, ROBOT_W / 2),
                               (-ROBOT_L / 2, ROBOT_W / 2))]
        dl.polygon(body, fill=(239, 68, 68, 255), outline=(255, 255, 255, 255))
        dl.line([view.px(x, y), view.px(x + 0.22 * c, y + 0.22 * s_)], fill=(255, 255, 255, 255), width=3)
    for tf, label, fx, fy in run["finds"]:
        if t >= tf:
            px, py = view.px(fx, fy)
            dl.ellipse([px - 9, py - 9, px + 9, py + 9], fill=BOX + (255,), outline=TEXT + (255,), width=2)
    mimg = Image.alpha_composite(mimg, layer)
    dm = ImageDraw.Draw(mimg)
    if tgt is not None:
        cx, cy = view.px(*tgt["center"][:2])
        label_at(dm, mimg.width, cx, cy + 16, tgt["label"], ACCENT)
    for tf, label, fx, fy in run["finds"]:
        if t >= tf:
            px, py = view.px(fx, fy)
            label_at(dm, mimg.width, px + 16, py - 17, label, TEXT, centred=False)  # beside it: the robot stops
                                                                                   # in front of it
    img.paste(mimg.convert("RGB"), (view.ox, view.oy))
    d.rectangle([view.ox - 1, view.oy - 1, view.ox + mimg.width, view.oy + mimg.height], outline=BORDER, width=2)
    # the steps
    st = run["steps"]
    items = []
    for i, (ts, text) in enumerate(st):
        last = i == len(st) - 1
        nxt = st[i + 1][0] if not last else ts
        items.append((text, "done" if t >= nxt and (not last or t >= ts) else "now" if t >= ts else "todo"))
    steps(d, MAP[0], view.oy + view.base.shape[0] + 30, items, width=MAP[2])
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path, help="data/task_runs/<time> (default: the newest)")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--still", type=float, help="one frame at this video time (s) as PNG instead")
    a = ap.parse_args()
    run_dir = a.run or sorted(p for p in RUNS.glob("2*") if (p / "frames.jsonl").exists())[-1]
    run = load(run_dir)
    view = TopView(run)
    t0, t1 = run["t_req"], run["t_end"]
    times = [t0] * int(LEAD_S * FPS) + list(np.arange(t0, t1, a.speed / FPS)) + [t1] * int(TAIL_S * FPS)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    if a.still is not None:
        out = OUT / f"{run['name']}_still_{a.still:g}.png"
        frame = render(run, view, times[min(len(times) - 1, int(a.still * FPS))], cache, a.speed)
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"wrote {out}")
        return
    out = OUT / f"{run['name']}_x{a.speed:g}.mp4"
    write_video(out, (render(run, view, t, cache, a.speed) for t in times))
    last = render(run, view, times[-1], cache, a.speed)
    cv2.imwrite(str(OUT / f"{run['name']}_end.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} ({len(times) / FPS:.0f} s, run {t1 - t0:.0f} s) and {run['name']}_end.png")


if __name__ == "__main__":
    sys.exit(main())
