"""Recordings of task runs: what the task console's ● Record window saw (user, 2026-10-08: only manually started,
no recording of everything), and the rolling images of the last search for checking the detector.

    rec = TaskRecorder("data/task_runs/20261008_153000", room="20261007_101108")
    rec.frame(jpeg, stamp, pose, boxes)       # every camera image while recording (boxes: the detector's, or [])
    rec.event("drive", "to cabinet")          # drives, finds, stops, chat messages
    rec.stop()                                # writes meta.json, renders video.mp4 (boxes drawn) in a thread

A run's folder: frames/<n>.jpg as received from the camera (no re-encoding), frames.jsonl (one line per image:
time, robot pose, boxes), events.jsonl, meta.json (room, start, duration, what was asked, finds, size, keep),
video.mp4 (H.264, plays in the browser) and thumb.jpg. ~5 images/s x ~70 KB plus the video: ~25 MB per minute (sim test, 2026-10-08).

The last search (LAST_SEARCH): while something is looked for, its images go to data/task_runs/_last_search, reset
at every new search and capped at LAST_SEARCH_S, so a missed detection can be checked offline (test_detector)
without filling the disk.
"""

import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "task_runs"
LAST_SEARCH = RUNS / "_last_search"
LAST_SEARCH_S = 120.0
VIDEO_FPS = 5


class TaskRecorder:
    def __init__(self, path, room=None, rolling_s=None):
        """rolling_s: keep only the images of the last rolling_s seconds (the last search); no video."""
        self.path = Path(path)
        if rolling_s is not None and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        (self.path / "frames").mkdir(parents=True, exist_ok=True)
        self.rolling_s = rolling_s
        self.room, self.start_time, self.n = room, time.time(), 0
        self.kept = []          # (time, frame number) for the rolling cap
        self.requests, self.finds = [], []
        self.lock = threading.Lock()
        self.frames_log = open(self.path / "frames.jsonl", "a", encoding="utf-8")
        self.events_log = open(self.path / "events.jsonl", "a", encoding="utf-8")
        self.active = True

    def frame(self, jpeg, stamp, pose=None, boxes=()):
        with self.lock:
            if not self.active:
                return
            now = time.time()
            (self.path / "frames" / f"{self.n:06d}.jpg").write_bytes(jpeg)
            self.frames_log.write(json.dumps({"n": self.n, "time": round(now, 3), "stamp": stamp,
                                              "pose": None if pose is None else [round(float(v), 3) for v in pose],
                                              "boxes": list(boxes)}) + "\n")
            self.frames_log.flush()
            if self.rolling_s is not None:
                self.kept.append((now, self.n))
                while self.kept and now - self.kept[0][0] > self.rolling_s:
                    (self.path / "frames" / f"{self.kept.pop(0)[1]:06d}.jpg").unlink(missing_ok=True)
            self.n += 1

    def event(self, kind, text, **data):
        with self.lock:
            if not self.active:
                return
            if kind == "user":
                self.requests.append(text)
            if kind == "found":
                self.finds.append(text)
            self.events_log.write(json.dumps({"time": round(time.time(), 3), "kind": kind, "text": text, **data}) + "\n")
            self.events_log.flush()

    def stop(self, render=True):
        with self.lock:
            if not self.active:
                return
            self.active = False
            self.frames_log.close()
            self.events_log.close()
        self.write_meta()
        if render and self.rolling_s is None and self.n:
            threading.Thread(target=self.render, daemon=True, name="render").start()

    def write_meta(self, **extra):
        meta_file = self.path / "meta.json"
        old = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        size = sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
        meta = {"room": self.room, "start": self.start_time, "duration_s": round(time.time() - self.start_time, 1)
                if "duration_s" not in old else old["duration_s"], "frames": self.n, "requests": self.requests,
                "finds": self.finds, "size_mb": round(size / 1e6, 1), "keep": old.get("keep", False),
                "video": old.get("video", False), **extra}
        meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def render(self):
        """video.mp4: the images with their boxes (relevant thick yellow, found marked, others thin grey) and a
        line with the time and the latest event; thumb.jpg."""
        try:
            import imageio_ffmpeg
            frames = [json.loads(l) for l in (self.path / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l]
            events = [json.loads(l) for l in (self.path / "events.jsonl").read_text(encoding="utf-8").splitlines() if l]
            first = cv2.imread(str(self.path / "frames" / f"{frames[0]['n']:06d}.jpg"))
            h, w = first.shape[:2]
            # written under another name and renamed when complete: the runs list offers video.mp4 to play as soon
            # as it exists (half a file showed up while rendering, 2026-10-08)
            part = self.path / "video.part.mp4"
            writer = imageio_ffmpeg.write_frames(str(part), (w, h), fps=VIDEO_FPS, codec="libx264",
                                                 macro_block_size=1, ffmpeg_log_level="error",
                                                 output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            writer.send(None)
            for k, f in enumerate(frames):
                img = cv2.imread(str(self.path / "frames" / f"{f['n']:06d}.jpg"))
                if img is None:
                    continue
                draw_boxes(img, f["boxes"])
                last = [e for e in events if e["time"] <= f["time"] and e["kind"] != "user"]
                text = time.strftime("%H:%M:%S", time.localtime(f["time"])) + ("  " + last[-1]["text"] if last else "")
                cv2.rectangle(img, (0, h - 22), (w, h), (16, 12, 10), -1)
                cv2.putText(img, text[:95], (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (238, 233, 231), 1, cv2.LINE_AA)
                writer.send(np.ascontiguousarray(img[:, :, ::-1]))
                if k == len(frames) // 2:
                    cv2.imwrite(str(self.path / "thumb.jpg"), cv2.resize(img, (w // 2, h // 2)))
            writer.close()
            part.replace(self.path / "video.mp4")
            self.write_meta(video=True)
        except Exception as e:  # the images stay; the video can be made again
            self.write_meta(video=False, video_error=f"{type(e).__name__}: {e}")


def draw_boxes(img, boxes):
    for b in sorted(boxes, key=lambda b: b.get("relevant", False)):
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        rel = b.get("relevant", False)
        color = (21, 204, 250) if rel else (214, 206, 201)  # BGR: yellow / light grey
        if b.get("outline"):  # the mask, tinted (user, 2026-10-09), as in the console
            pts = [np.array(p, np.int32).reshape(-1, 1, 2) for p in b["outline"]]
            tint = img.copy()
            cv2.fillPoly(tint, pts, color)
            cv2.addWeighted(tint, 0.3 if rel else 0.15, img, 0.7 if rel else 0.85, 0, dst=img)
            cv2.polylines(img, pts, True, color, 2 if rel else 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3 if rel else 1)
        label = b["text"].replace("✓", "ok").replace("✗", "no").replace("…", "...") if "text" in b else \
            f"{'ok ' if b.get('found') else ''}{b['label']} {b.get('score', 0):.2f}"
        if not label:
            continue
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x0, max(0, y0 - th - 6)), (x0 + tw + 6, max(th + 6, y0)), color if rel else (16, 12, 10), -1)
        cv2.putText(img, label, (x0 + 3, max(th + 2, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (17, 17, 17) if rel else (238, 233, 231), 1, cv2.LINE_AA)


def list_runs():
    """[{name, ...meta}] of the recorded runs, newest first (not the rolling last search)."""
    out = []
    for m in sorted(RUNS.glob("*/meta.json"), reverse=True):
        if m.parent.name.startswith("_"):
            continue
        try:
            out.append({"name": m.parent.name, **json.loads(m.read_text(encoding="utf-8"))})
        except ValueError:
            continue
    return out
