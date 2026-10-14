"""Presentation video of a pick through the chat, from the robot's camera: the detector's box on the picture, the chat
request and answer, the steps as they happen, sped up. Made afterwards from what a run leaves behind; nothing extra
is recorded for it.

    python projects/presentation_video/pick_video.py                          # the newest pick (data/manipulation/picks/)
    python projects/presentation_video/pick_video.py --pick data/manipulation/picks/<time>
    python projects/presentation_video/pick_video.py --still 20               # one frame (s of video) as PNG, to check

Inputs: the pick's pictures and log (pick.PickLog), the search before it (navigation's data/task_runs/_last_search,
if it is the same run), the navigator's log of the run (logs/navigator_*.log: the chat and the progress notes with
their times). Output: data/manipulation/demo/pick_<HHMMSS>.mp4 (H.264, 1920 x 1080, 30 fps) and a PNG of its last
frame.

Speeds: SPEED until the gripper goes down, SPEED_GRASP for going down, closing and lifting (slow and careful on the
robot, little to see), real time for the end.
"""

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from style import (ACCENT, BOX, F_BOLD, F_SMALL, F_TITLE, FPS, MUTED, OK, ROBOT_FILL, SURFACE, TEXT, USER_FILL, wrap,
                   BORDER, H, W, bubble, speed_badge, steps, write_video)

REPO = Path(__file__).resolve().parents[2]
PICKS = REPO / "data" / "manipulation" / "picks"           # one folder per pick (pick.PickLog)
LAST_PICK = REPO / "data" / "manipulation" / "last_pick"   # before 2026-10-13: only the last pick, overwritten
SEARCH = REPO / "data" / "task_runs" / "_last_search"
OUT = REPO / "data" / "demo"                               # with the other task videos (task_run_video.py)

CAM = (40, 150, 1280, 800)          # x, y, w, h of the camera picture (640 x 400, scaled x2)
PANEL_X = 1370
SPEED, SPEED_GRASP = 2.0, 6.0
LEAD_S, TAIL_S = 1.5, 3.5           # video seconds before the request and after the answer

STEPS = [  # (name, the log event that starts it)
    ("Find it: detector + image check", "find"),
    ("Drive up to it", "approach"),
    ("Follow it: camera, base and arm together", "follow"),
    ("Centre the gripper over it", "centre"),
    ("Go down and close the gripper", "grasp"),
    ("Lift it", "lift"),
]


# ---------- reading the run ----------

def navigator_log(pick_end):
    """The navigator log covering the pick (its lines carry only the time of day; the file name the start)."""
    logs = sorted((REPO / "logs").glob("navigator_*.log"))
    best = None
    for f in logs:
        try:
            start = dt.datetime.strptime(f.stem.split("_", 1)[1], "%Y%m%d_%H%M%S")
        except ValueError:
            continue  # navigator_test.log and the like
        if start <= pick_end:
            best = (f, start)
    if best is None:
        raise SystemExit("no navigator log for this pick")
    return best


def parse_log(path, day):
    """[(datetime, text)] of the timed lines."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*)", line)
        if m:
            t = day.replace(hour=int(m[1]), minute=int(m[2]), second=int(m[3]))
            out.append((t + dt.timedelta(seconds=0.5), m[4]))  # the middle of its second
    return out


def plain(note):
    """A progress note in words for an audience: no picture counts, servo values or seconds."""
    note = re.sub(r";\s*\d+ pictures.*", "", note)
    m = re.match(r"holding it: contact at \d+ \(~(\d+) mm\)", note)
    if m:
        return f"holding it: about {m[1]} mm wide"
    note = re.sub(r"base there: object at .*", "close enough: the base stops", note)
    note = re.sub(r"over it \(.*\); going down", "over it: going down", note)
    return note


def load(pick_dir):
    L = [json.loads(line) for line in open(pick_dir / "log.jsonl", encoding="utf-8")]
    holding = next((x for x in L if x["kind"] == "holding"), None)
    if holding is None:
        raise SystemExit("this pick did not end holding the object")
    mtime = dt.datetime.fromtimestamp((pick_dir / "log.jsonl").stat().st_mtime)
    path, start = navigator_log(mtime)
    lines = parse_log(path, start.replace(hour=0, minute=0, second=0))
    # the pick log's clock (s since its start) on the day's: its "holding" and "there" against the notes of them
    pairs = []
    clock = next((x for x in L if x["kind"] == "clock"), None)
    if clock is not None:  # picks since 2026-10-13 write their start on the wall clock
        pairs.append(clock["wall"] - clock["t"])
    for kind, note in (("holding", "holding it:"), ("there", "base there:")):
        e = next((x for x in L if x["kind"] == kind), None)
        n = [t for t, text in lines if text.startswith(note) and abs((t - mtime).total_seconds()) < 300]
        if e is not None and n:
            pairs.append((n[-1] - dt.timedelta(seconds=e["t"])).timestamp())
    t0 = float(np.mean(pairs))
    t_hold = t0 + holding["t"]
    # the chat around it: the user's request before the pick, the answer after
    chat = [(t.timestamp(), text) for t, text in lines]
    req = max((x for x in chat if x[1].startswith("chat: ") and x[0] < t0), key=lambda x: x[0])
    ans = min((x for x in chat if x[1].startswith("assistant: ") and x[0] > t_hold), key=lambda x: x[0], default=None)
    events = {}
    for t, text in chat:
        if not req[0] <= t <= t_hold + 1:
            continue
        for key, pat in (("find", "chat tool watch_for"), ("find", "chat tool search_here"),
                         ("approach", "chat tool approach_found"), ("approach", "goto found"),
                         ("follow", "following the"), ("centre", "base there"), ("grasp", "over it"),
                         ("lift", "lifting it"), ("done", "holding it")):
            if text.startswith(pat) and key not in events:
                # a tool's line is written when it ended: approach_found started at its "goto found"
                events[key] = t if not pat.startswith("chat tool approach") else min(t, events.get(key, t))
    if "approach" in events and "find" in events:
        events["approach"] = max(events["approach"], events["find"])
    notes = [(t, plain(text)) for t, text in chat
             if req[0] <= t <= t_hold + 1 and re.match(r"(following|base there|centring|centred|over it|closing|"
                                                         r"lifting|holding it)", text)]
    # pictures: the search before (navigation's rolling recording) and the pick's own
    frames = []
    sf = SEARCH / "frames.jsonl"
    if sf.exists():
        for x in map(json.loads, open(sf, encoding="utf-8")):
            if req[0] - 5 <= x["time"] <= t0:
                jpg = SEARCH / "frames" / f"{x['n']:06d}.jpg"
                boxes = [b["box"] for b in x.get("boxes", []) if b.get("relevant") or b.get("found")]
                frames.append((x["time"], jpg, boxes[:1]))
    for x in L:
        if x["kind"] == "picture" and x.get("file"):
            frames.append((t0 + x["t"], pick_dir / x["file"], [x["box"]] if x["box"] else []))
    frames.sort(key=lambda f: f[0])
    return {"t0": t0, "t_hold": t_hold, "req": req, "ans": ans, "events": events, "notes": notes, "frames": frames}


# ---------- the video ----------

def timeline(run):
    """[(video s, run time)] at FPS: SPEED, SPEED_GRASP from "going down" to "holding", real time at the end."""
    t, t_end = run["req"][0], (run["ans"][0] if run["ans"] else run["t_hold"]) + 0.5
    t_grasp, t_hold = run["events"].get("grasp", run["t_hold"]), run["t_hold"]
    out = [run["req"][0]] * int(LEAD_S * FPS)
    while t < t_end:
        out.append(t)
        t += (SPEED_GRASP if t_grasp <= t < t_hold else SPEED if t < t_hold else 1.0) / FPS
    out += [t_end] * int(TAIL_S * FPS)
    return out


def speed_at(run, t):
    t_grasp, t_hold = run["events"].get("grasp", run["t_hold"]), run["t_hold"]
    return SPEED_GRASP if t_grasp <= t < t_hold else SPEED if t < t_hold else 1.0


def render(run, t, cache):
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((CAM[0], 40), "LanderPi picks up a cube it was asked for", font=F_TITLE, fill=TEXT)
    d.text((CAM[0], 96), "the robot's own camera  ·  everything after the request runs on its own", font=F_SMALL,
           fill=MUTED)
    # the camera
    past = [f for f in run["frames"] if f[0] <= t]
    f = past[-1] if past else run["frames"][0]
    if f[1] not in cache:
        cache.clear()
        cache[f[1]] = cv2.cvtColor(cv2.imread(str(f[1])), cv2.COLOR_BGR2RGB)
    pic = cv2.resize(cache[f[1]], (CAM[2], CAM[3]), interpolation=cv2.INTER_LINEAR)
    if not past:
        pic = (pic * 0.45 + 255 * 0.55).astype(np.uint8)
    img.paste(Image.fromarray(pic), (CAM[0], CAM[1]))
    k = CAM[2] / 640
    if past and t - f[0] < 1.5:
        for b in f[2]:
            x0, y0, x1, y1 = (CAM[0] + b[0] * k, CAM[1] + b[1] * k, CAM[0] + b[2] * k, CAM[1] + b[3] * k)
            d.rectangle([x0, y0, x1, y1], outline=BOX, width=5)
            label = "red cube · held" if t >= run["t_hold"] else "red cube"
            d.rectangle([x0, y0 - 34, x0 + 16 + d.textlength(label, font=F_BOLD), y0], fill=BOX)
            d.text((x0 + 8, y0 - 32), label, font=F_BOLD, fill=TEXT)
    d.rectangle([CAM[0] - 1, CAM[1] - 1, CAM[0] + CAM[2], CAM[1] + CAM[3]], outline=BORDER, width=2)
    # speed and time under the picture
    el = max(0.0, t - run["req"][0])
    speed_badge(d, CAM[0], CAM[1] + CAM[3] + 22, speed_at(run, t),
                f"{int(el // 60)}:{int(el % 60):02d} since the request (real time)")
    # the panel: request, steps, progress, answer
    x, w = PANEL_X, W - PANEL_X - 40
    y = bubble(d, x, CAM[1], w, "User (chat)", run["req"][1][len("chat: "):], ACCENT, USER_FILL)
    ev = run["events"]
    items = []
    for k_, (name, key) in enumerate(STEPS):
        start = ev.get(key)
        nxt = [ev[s_[1]] for s_ in STEPS[k_ + 1:] if s_[1] in ev] + [ev.get("done", run["t_hold"])]
        end = min(nxt) if nxt else None
        items.append((name, "done" if start is not None and end is not None and t >= end else
                      "now" if start is not None and t >= start else "todo"))
    y = steps(d, x, y, items, width=w)
    note = [n for n in run["notes"] if n[0] <= t]
    if note and t <= run["t_hold"] + 3:
        y += 10
        for line in wrap(d, note[-1][1], F_SMALL, w)[:2]:
            d.text((x, y), line, font=F_SMALL, fill=MUTED)
            y += 28
    y += 18
    if run["ans"] and t >= run["ans"][0]:
        bubble(d, x, y, w, "Robot (chat)", run["ans"][1][len("assistant: "):], OK, ROBOT_FILL)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", type=Path, help="a pick's folder (default: the newest; holding the object)")
    ap.add_argument("--still", type=float, help="write one frame at this video time (s) as PNG instead")
    a = ap.parse_args()
    if a.pick is None:
        picks = sorted(PICKS.glob("2*")) if PICKS.exists() else []
        done = [p for p in picks if (p / "log.jsonl").exists() and '"holding"' in (p / "log.jsonl").read_text()]
        a.pick = done[-1] if done else LAST_PICK
    run = load(a.pick)
    times = timeline(run)
    name = dt.datetime.fromtimestamp(run["t0"]).strftime("pick_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    if a.still is not None:
        out = OUT / f"{name}_still_{a.still:g}.png"
        frame = render(run, times[min(len(times) - 1, int(a.still * FPS))], cache)
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"wrote {out}")
        return
    out = OUT / f"{name}.mp4"
    write_video(out, (render(run, t, cache) for t in times))
    last = render(run, times[-1], cache)
    cv2.imwrite(str(OUT / f"{name}_end.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} ({len(times) / FPS:.0f} s, run {times[-1] - times[0]:.0f} s) and {name}_end.png")


if __name__ == "__main__":
    sys.exit(main())
