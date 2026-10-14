"""Live detection in the navigator: while something is being looked for, the colour video (through the
helper's video relay: the same robot stream the console shows, no second one), the detector (detector.py),
3D positions with the latest depth image (perception.py) and the watcher that confirms a find. A confirmed
find stops a running drive or turn at once (navigator.on_found).

    live = LiveDetection(nav)            # nav: the Navigator
    live.start(["pencil"], area)         # area: polygon (map) or None
    live.state()                         # for /state: what is looked for, images/s, finds
    live.stop()

Without a camera (fake_robot.py --sim): LANDERPI_SIM_OBJECTS='[{"label": "pencil", "x": 1.0, "y": 2.0}]'
plants objects that count as seen whenever they are in the camera's view (field of view, range, not behind a
wall of the room map), with a few cm of noise: for the search logic, not for the detector.
"""

import json
import math
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from landerpi.mjpeg import MjpegReader
from landerpi.robot import TOPICS
from perception import SCORE_MIN, Watcher, in_area, inner, iou, locate, mask_points, nearest_group, place_dist
from scan_cells import ScanCells
from taskrun import LAST_SEARCH, LAST_SEARCH_S, TaskRecorder

HELPER = ("127.0.0.1", int(os.environ.get("LANDERPI_HELPER_PORT", "8765")))
VIDEO_QUALITY = 85         # the console's MJPEG quality: the helper serves both from one robot stream
MAX_RGB_DEPTH_S = 0.3      # colour and depth image this far apart at most for a 3D position
FLOOR_LOS_M = 2.0          # without usable depth, something whose line of sight meets the floor this close is
                           # placed there: the floor's height is known, the place is good
GONE_S = 1.5               # a remembered thing whose place was in view this long without it is gone
GONE_RANGE = (0.35, 1.2)   # m: where the detector reliably sees small things (test_far.py: further away it misses
                           # half of them, and "not seen" would not mean "not there")
GONE_MARGIN = 0.1          # ... and not at the picture's edge
STILL_DEPTH_S = 2.0        # standing still, a depth image up to this old still fits the colour image
TORCH_THREADS = 6          # of the laptop's 14 cores: detection at ~5 images/s, the navigator stays responsive
CAMERA_AHEAD = 0.09        # m, camera in front of the robot's centre in the survey pose (explore_core.py)
CLOSE_M = 0.8              # a found item closer than this and in view but not detected: a second, shrunk pass
CLOSE_EVERY_S = 0.6        # ... at most this often: on every image it halved the detection rate (2 images/s
                           # instead of ~5 during the final look, real run 2026-10-09)
CLOSE_SHRINK = 0.6         # (detector.detect: the mug at 20 cm scored 0.01-0.03 as is, 0.07-0.11 shrunk)
FREEZE = False             # hold the picture of a find (see freeze())
LOOK_S = 2.5               # the final look at a target after an approach (s at most)
LOOK_MIN_S = 1.0           # ... it ends this early once the target was seen
LOOK_SETTLE_S = 0.6        # ... and takes no image of its first 0.6 s (the camera settling after aiming, frames
                           # in the pipeline: a mask drawn off the item; real run 2026-10-09)
TARGET_BOX_M = 0.10        # a box within this of the target (3D) shows the target: the sighting's own size made
                           # it up to 30 cm, and the green block's box 23 cm away became the blue one's (same run)
LOGS = Path(__file__).resolve().parents[2] / "logs"
SIGHTING_LOG_DAYS = 7      # logs/sightings_<day>.jsonl: every sighting of what is looked for, ~0.4 KB each
SIGHTING_LOG_MB = 50       # per day at most (~5 sightings/s while searching: ~7 MB per hour)
UNFREEZE_M, UNFREEZE_DEG = 0.20, 10.0  # a frozen picture goes once the robot moved this much (it no longer fits)
SIM_FOV = math.radians(68)  # the depth camera's 74 deg, minus the image edges
SIM_RANGE = (0.4, 3.0)
SIM_K = (417.4, 418.1, 315.4, 191.8)  # colour camera intrinsics (calib/rgb_camera_info.json), for sim boxes
SIM_CAMERA_Z = 0.26                   # m, camera height in the survey pose (explore_core.py)
# The boxes are always those of one image (held boxes drifted off the object while the robot moved; user,
# 2026-10-08). At a find the overlay freezes on the image that confirmed it, until the next action starts
# (navigator.clear_overlay: a drive, turn, localization, room change) or the robot moved (manual driving);
# then live boxes again. Detection runs only while a find task is open (user, 2026-10-09: it kept detecting
# cups while the user drove around after the job): it stops after the final look at an approached target and
# at the end of a chat turn in which everything looked for was found; the picture stays frozen.


# The robot's own gripper, at the bottom corners of every image (camera and gripper move together), is boxed as
# "chair" (scores up to 0.59, 70 of 31 images with "chair" / "bar stool" prompts, run 20261009_090645; seen by
# scene-understanding too). Regions as shares of the 640 x 400 image.
GRIPPER = ((50 / 640, 345 / 400, 240 / 640, 1.0), (505 / 640, 345 / 400, 1.0, 1.0))
HELD_COVER = 0.5           # a box this much on the held item's mask is the held item
GRIPPER_COVER = 0.5        # a box covering one of them this much (overlap / union) is the gripper


def on_gripper(box, w, h):
    return any(iou(box, [x0 * w, y0 * h, x1 * w, y1 * h]) >= GRIPPER_COVER for x0, y0, x1, y1 in GRIPPER)


def verdict_text(e):
    v, what = e.get("verdict"), e.get("verdict_what") or ""
    return {"yes": f"✓ {what}", "no": f"✗ {what}", "pending": "checking…"}.get(v, "")


class LiveDetection:
    def __init__(self, nav):
        self.nav = nav
        self.watcher = Watcher()
        self.detector = None
        self.reader = None
        self.frame = None          # (jpeg, robot stamp) latest
        self.done_stamp = None
        self.rates = []            # times of processed images (images/s)
        self.error = None
        self.overlay = None        # what the console's camera picture shows: live boxes, or a frozen picture
        self.overlay_full = None   # the latest image's boxes with their masks (for freezing)
        self.frozen = False        # the overlay shows the image of a find until the next action
        self.recorder = None       # a ● Record window of the console (taskrun.py), set by the navigator
        self.last_search = None    # the rolling images of the last search (taskrun.LAST_SEARCH)
        self.last_boxes = []       # the latest processed image's boxes (also while the overlay is frozen)
        self.alias = {}            # detector word -> the item looked for (synonyms)
        self.last_jpeg = None
        self.recorded_stamp = None
        self.recorded_at = 0.0
        self.pending_overlay = None  # this image's boxes, drawn once the watcher has matched them
        self.last_close = 0.0        # time of the last close-up pass
        # another feeder of the overlay (projects/manipulation's pick) sets this a second ahead with each picture: the
        # console then polls the boxes fast (state: images_per_s > 0)
        self.fed_until = 0.0
        self.detector_lock = threading.Lock()  # one user of the detector at a time (also projects/manipulation)
        self.look = None           # the final look at a target: {"label", "position", "start", "best", "done"}
        self.prune_logs()
        self.frozen_pose = None    # the robot's pose when the overlay froze
        self.sim = json.loads(os.environ["LANDERPI_SIM_OBJECTS"]) if os.environ.get("LANDERPI_SIM_OBJECTS") else None
        # the second opinion on candidates (verifier.py); not in the simulation (no images)
        self.descriptions = {}     # item -> the user's description (what the verifier checks against)
        self.prev_stamp = None     # the previous detected image's stamp (how far the robot turned since)
        self.view = None           # (T_map_rgb, w, h) of the latest detected image (seen_empty)
        self.verifier = None
        self.adopted = 0           # earlier candidates being judged for the current search
        if self.sim is None:
            from verifier import Verifier
            self.verifier = Verifier(self.on_verdict)
            self.watcher.verify = True
        self.watcher.stood_still = lambda a, b: a is not None and b is not None and self.stood_still(a, b, 30.0)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self.run, daemon=True, name="detection").start()

    # ---------- control ----------

    def start(self, prompts, area=None, synonyms=None, description=None):
        """synonyms: {item: [other words]}: all go to the detector in one pass (it reacts to the exact word);
        a detection under any of them counts as the item (user, 2026-10-08). description: the user's words for
        the item (colour, size, ...), for the verifier."""
        with self.lock:
            self.watcher.start(prompts, area)
            for p in self.watcher.prompts:
                self.descriptions[p] = description or self.descriptions.get(p) or p
            self.alias = {w.strip().lower(): item for item, words in (synonyms or {}).items()
                          for w in [item, *words] if w and w.strip()}
            self.error = None
            adopted = self.watcher.adopt_earlier()
            checks = self.watcher.take_to_check()
        for e in checks:  # earlier candidates of other searches, judged against this description
            self.verifier.submit([e], self.descriptions.get(e["label"], e["label"]))
        self.adopted = len(adopted)
        if self.last_search is not None:
            self.last_search.stop(render=False)
        if self.sim is None:
            self.last_search = TaskRecorder(LAST_SEARCH, room=self.nav.room.name if self.nav.room else None,
                                            rolling_s=LAST_SEARCH_S)
            self.last_search.event("search", f"looking for {', '.join(self.watcher.prompts)}")
        self.video(True)
        self.wake.set()

    def stop(self):
        """The job is over (a new chat message, a new room, the chat cleared): no more detection."""
        self.look = None
        with self.lock:
            self.watcher.stop()
        if self.last_search is not None:
            self.last_search.stop(render=False)
            self.last_search = None
        self.video(self.recorder is not None or self.look is not None)

    def end_of_turn(self):
        """A chat turn ended: the boxes stay live until the next message (assistant.send stops the detection
        then; user, 2026-10-09: rather live boxes than a frozen last picture)."""

    def final_look(self, label, position, stop_after=True, size=None):
        """After an approach: look at the target for up to LOOK_S (with the close-up pass) and say whether it was
        seen. Afterwards the camera keeps showing the target's live boxes (its words only, no new finds) until
        the next chat message stops the detection. Called from a tool's thread."""
        done = threading.Event()
        self.look = {"label": label, "position": list(position), "size": size, "start": time.time(), "best": None,
                     "done": done}
        self.video(True)
        self.wake.set()
        done.wait(LOOK_S + 1.5)
        lk = self.look
        seen = lk is not None and lk["best"] is not None
        if lk is not None:
            lk["lingering"] = True
        if stop_after:  # the search is over; the boxes of the target stay live
            with self.lock:
                self.watcher.stop()
            if self.last_search is not None:
                self.last_search.stop(render=False)
                self.last_search = None
        return seen

    def freeze(self, overlay=None):
        """No frozen pictures any more (user, 2026-10-09): an earlier image's boxes and masks never quite fit the
        live video (the camera settling, the robot braking), and choosing which image to hold went wrong twice.
        The boxes stay live; FREEZE = True brings the old behaviour back."""
        if not FREEZE:
            return
        overlay = overlay or self.overlay_full
        if overlay is not None:  # an earlier image's: newer time, else the console keeps the newer live one
            self.overlay = {**overlay, "time": time.time()}
        self.frozen = True
        self.frozen_pose = self.nav.current_pose()

    def video(self, on):
        """The colour video is read while looking for something or recording."""
        if on and self.sim is None and self.reader is None:
            self.reader = MjpegReader(self.on_frame, topic=TOPICS["rgb"][0], quality=VIDEO_QUALITY,
                                      host=HELPER[0], port=HELPER[1])
            self.reader.path = f"/video?topic={quote(TOPICS['rgb'][0])}&quality={VIDEO_QUALITY}"
            self.reader.start()
        elif not on and self.reader is not None:
            self.reader.stop()
            self.reader = None

    def record(self, jpeg, stamp, boxes):
        """An image into the open recordings (only the boxes' fields worth keeping)."""
        if self.recorder is None and self.last_search is None:
            return
        pose = self.nav.current_pose()
        keep = [{k: b[k] for k in ("label", "score", "box", "outline", "relevant", "found", "text") if k in b}
                for b in boxes]
        for r in (self.recorder, self.last_search):
            if r is not None:
                r.frame(jpeg, stamp, pose, keep)
        self.recorded_stamp, self.recorded_at = stamp, time.time()

    def on_frame(self, jpeg, stamp, recv):
        self.frame = (jpeg, stamp)

    # ---------- the loop (own thread) ----------

    def run(self):
        while True:
            if not self.watcher.active and self.look is None:
                # recording without looking for anything: the images at ~5/s, no detection
                f = self.frame
                if self.recorder is not None and f is not None and f[1] != self.recorded_stamp \
                        and time.time() - self.recorded_at >= 0.2:
                    self.record(f[0], f[1], [])
                    continue
                self.wake.wait(0.1 if self.recorder is not None else 1.0)
                self.wake.clear()
                continue
            try:
                sightings = self.sim_sightings() if self.sim is not None else self.camera_sightings()
                if sightings is None:
                    time.sleep(0.05)
                    continue
                now = time.time()
                self.rates = [t for t in self.rates if now - t < 3.0] + [now]
                # the words followed: what is looked for, or during and after the final look the target's
                labels = [self.look["label"]] if self.look is not None else self.watcher.prompts
                stamp = self.pending_overlay[3] if self.pending_overlay and len(self.pending_overlay) > 3 else None
                shift = self.turn_px(self.prev_stamp, stamp)
                with self.lock:
                    new = self.watcher.add(sightings, now, labels, shift)
                    checks = self.watcher.take_to_check()
                self.prev_stamp = stamp
                self.seen_empty(now, labels)
                self.draw_pending()  # now each box knows its followed sighting and so its check result
                for e in checks:
                    if self.verifier is not None:
                        self.verifier.submit([e], self.descriptions.get(e["label"], e["label"]))
                if self.look is not None:
                    self.look_step(now)
                if self.last_jpeg is not None:
                    self.record(self.last_jpeg[0], self.last_jpeg[1], self.last_boxes)
                    self.last_jpeg = None
                for label in new:
                    f = self.watcher.found[label]
                    self.nav.note("found", f"found {label} at ({f['position'][0]:.2f}, {f['position'][1]:.2f}) "
                                           f"(score {f['score']:.2f}, {f['views']} views)")
                    self.nav.loop.call_soon_threadsafe(self.nav.on_found, label)
            except Exception as e:  # keep looking; the console shows it
                self.error = f"{type(e).__name__}: {e}"
                print(f"detection: {self.error}", flush=True)
                time.sleep(1.0)

    def on_verdict(self, entry, fits, what):
        """The verifier's answer (its thread). A fitting candidate in the search area becomes the find: a running
        drive stops (it went on meanwhile: user, 2026-10-09, no stop for checking)."""
        with self.lock:
            new = self.watcher.set_verdict(entry, fits, what)
        self.nav.note("check", f"image check of {entry['label']} #{entry['id']}: "
                               + {True: "fits", False: "does not fit", None: "not checked"}.get(fits, str(fits))
                               + f" ({what})")
        if new:
            p = entry["position"]
            self.nav.note("found", f"found {entry['label']} at ({p[0]:.2f}, {p[1]:.2f}) "
                                   f"(score {entry['score']:.2f}, {entry['views']} views, checked: {what})")
            self.nav.loop.call_soon_threadsafe(self.nav.on_found, entry["label"])

    def target_boxes(self, o, label, position, size=None):
        """The box of one image that shows the target itself: the nearest by place within TARGET_BOX_M, not one
        rejected by the image check; not just anything of its word."""
        near = [b for b in (o or {}).get("boxes", []) if b["label"] == label and b["score"] >= SCORE_MIN
                and b.get("pos") is not None and position is not None and b.get("relevant", True)
                and place_dist(b["pos"], position) < TARGET_BOX_M]
        return sorted(near, key=lambda b: place_dist(b["pos"], position))[:1]

    def look_step(self, now):
        lk = self.look
        if lk is None or lk.get("lingering"):
            return
        o = self.overlay_full
        settled = now - lk["start"] >= LOOK_SETTLE_S  # images from while the camera still moved show it elsewhere
        hit = self.target_boxes(o, lk["label"], lk["position"], lk.get("size")) if settled else []
        if hit and (lk["best"] is None or max(b["score"] for b in hit) > lk["best"][0]):
            for b in hit:
                b["found"] = True
            lk["best"] = (max(b["score"] for b in hit), o)
        if now - lk["start"] >= LOOK_S or (lk["best"] is not None and now - lk["start"] >= LOOK_MIN_S):
            self.freeze(lk["best"][1] if lk["best"] is not None else None)
            lk["done"].set()

    def words(self):
        """The detector's prompts: what is looked for and its synonyms; in a final look, the target's."""
        if self.look is not None:
            return [w for w, i in self.alias.items() if i == self.look["label"]] or [self.look["label"]]
        return list(dict.fromkeys([*self.watcher.prompts, *self.alias]))

    def close_targets(self, T_map_rgb, w, h):
        """Found items (or the final look's target) in view and closer than CLOSE_M: [label]."""
        if T_map_rgb is None or self.nav.rgb_K is None:
            return []
        pts = [(k, v["position"]) for k, v in self.watcher.found.items() if k in self.watcher.prompts]
        if self.look is not None:
            pts.append((self.look["label"], self.look["position"]))
        fx, fy, cx, cy = self.nav.rgb_K
        out = []
        for label, p in pts:
            q = np.array([p[0], p[1], p[2] + self.nav.depth.floor.z])
            c = (q - T_map_rgb[:3, 3]) @ T_map_rgb[:3, :3]
            if c[2] > 0.1 and np.linalg.norm(c) < CLOSE_M and 0 <= c[0] * fx / c[2] + cx < w \
                    and 0 <= c[1] * fy / c[2] + cy < h:
                out.append(label)
        return out

    def camera_sightings(self):
        frame = self.frame
        if frame is None or frame[1] == self.done_stamp:
            return None
        jpeg, stamp = frame
        self.done_stamp = stamp
        if self.detector is None:
            import torch
            torch.set_num_threads(TORCH_THREADS)
            from detector import Detector
            self.detector = Detector()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        with self.detector_lock:  # projects/manipulation's tools use the same detector (they take the lock too)
            self.detector.set_prompts(self.words())  # loads the text encoder only when they change
            dets = [d for d in self.detector.detect(img) if not on_gripper(d["box"], img.shape[1], img.shape[0])
                    and not self.on_held(d["box"])]
        self.last_jpeg = (jpeg, stamp)
        boxes = []
        nav = self.nav
        last = nav.depth.last
        T_base = nav.pose_matrix_at(stamp)
        B = nav.depth.camera_in_base("rgb_camera_link")
        seen = {self.alias.get(d["label"], d["label"]) for d in dets if d["score"] >= SCORE_MIN}
        missed = [t for t in self.close_targets(None if T_base is None or B is None else T_base @ B,
                                                img.shape[1], img.shape[0]) if t not in seen]
        if missed and time.time() - self.last_close >= CLOSE_EVERY_S:
            # up close the item is too large for the model: once more, shrunk (~+150 ms, only then)
            self.last_close = time.time()
            with self.detector_lock:
                self.detector.set_prompts(self.words())
                dets += [d for d in self.detector.detect(img, shrink=CLOSE_SHRINK)
                         if self.alias.get(d["label"], d["label"]) in missed and not self.on_held(d["box"])]
        out = []
        T = None if T_base is None or B is None else T_base @ B
        self.view = None if T is None or nav.rgb_K is None else (T, img.shape[1], img.shape[0])  # for seen_empty
        cells = ScanCells.ready(nav.room)
        for d in dets:
            s = {"label": self.alias.get(d["label"], d["label"]), "word": d["label"], "score": d["score"],
                 "box": d["box"], "outline": d.get("outline"), "stamp": stamp}
            boxes.append(s)
            loc, why = None, "no depth image"
            if last is not None and abs(last[2] - stamp) > MAX_RGB_DEPTH_S and not self.stood_still(last[2], stamp):
                why = f"no depth image within {MAX_RGB_DEPTH_S} s (nearest {abs(last[2] - stamp):.2f} s)"
            elif (d["mask"] is not None and last is not None and T is not None and nav.rgb_K is not None
                    and nav.depth.K is not None):
                pts = mask_points(inner(d["mask"]), last[0], nav.depth.K, last[1], nav.rgb_K, T)
                loc = locate(pts, nav.depth.floor.z)
                why = f"only {len(pts)} depth points in the mask"
                if loc is not None and cells is not None and not cells.supported(loc["position"], loc["size"]):
                    # floating: perhaps background mixed in; the nearest solid group only then (for stools it
                    # took the one in front: test_positions.py, 2026-10-09)
                    loc = locate(nearest_group(pts, T[:3, 3]), nav.depth.floor.z) or loc
                    why = f"depth placed it in the air at {loc['position']}"
            if loc is not None and (cells is None or cells.supported(loc["position"], loc["size"])):
                s.update(loc, quality="sure")
            elif T is not None and nav.rgb_K is not None:
                los = self.line_of_sight(d, T, cells)
                a = los.pop("floor")
                if a is not None and math.hypot(a[0] - T[0, 3], a[1] - T[1, 3]) <= FLOOR_LOS_M:
                    # near and below the camera's height: where the line of sight through the mask's bottom meets
                    # the floor is its place (a confirmed 3 cm block 40 cm away had no depth and stayed vague, so
                    # the search drove on: real run 2026-10-09 15:51). The floor plane, not the room's scan: that
                    # stopped the line at something 15 cm up that stood there when the room was scanned (17:19)
                    s.update(position=a, size=[0.05, 0.05, 0.05], quality="sure", placed="line of sight",
                             reason=why)
                else:
                    s.update(los, quality="vague", reason=why)
            else:
                self.log_sighting(s, T)
                continue
            ok, buf = cv2.imencode(".jpg", self.crop(img, d["box"]))
            s["crop"] = buf.tobytes() if ok else None
            out.append(s)
            self.log_sighting(s, T)
        self.pending_overlay = (boxes, img.shape[1], img.shape[0], stamp)  # drawn after the watcher matched them
        return out

    def line_of_sight(self, d, T, cells):
        """Where a sighting without a trustworthy depth place was seen from, the direction (map degrees) and the
        first scanned surface along the line through the mask's bottom (where it stands), as an estimate."""
        fx, fy, cx, cy = self.nav.rgb_K
        if d["mask"] is not None and d["mask"].any():
            rows = np.nonzero(d["mask"].any(1))[0]
            v = int(rows.max())
            u = float(np.nonzero(d["mask"][max(0, v - 3):v + 1].any(0))[0].mean())
        else:
            u, v = (d["box"][0] + d["box"][2]) / 2, d["box"][3]
        ray = T[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        origin = T[:3, 3] - np.array([0.0, 0.0, self.nav.depth.floor.z])
        hit = cells.ray_hit(origin, ray) if cells is not None else None
        # the floor plane: a line that goes down (the camera is 0.26 m high: whatever it meets on the floor is
        # lower than the camera, a thing on a table is above it and never meets the floor)
        floor = origin - (origin[2] / ray[2]) * ray if ray[2] < -1e-6 else None
        return {"from": [round(float(T[0, 3]), 2), round(float(T[1, 3]), 2)],
                "bearing_deg": round(math.degrees(math.atan2(ray[1], ray[0])), 1),
                "about": None if hit is None else [round(float(v), 2) for v in hit],
                "floor": None if floor is None else [round(float(floor[0]), 3), round(float(floor[1]), 3), 0.0]}

    def stood_still(self, t0, t1, longest=None):
        """Whether the robot stayed put between two stamps (then an older depth image fits the colour image:
        depth lagged 0.4-0.8 s behind while standing, real run 2026-10-09 17:19, and all sightings lost their
        place). Up to STILL_DEPTH_S apart."""
        if abs(t1 - t0) > (STILL_DEPTH_S if longest is None else longest):
            return False
        a, b = self.nav.pose_matrix_at(t0), self.nav.pose_matrix_at(t1)
        if a is None or b is None:
            return False
        turn = abs(math.atan2(b[1, 0], b[0, 0]) - math.atan2(a[1, 0], a[0, 0]))
        return float(np.hypot(*(b[:2, 3] - a[:2, 3]))) < 0.02 and math.degrees(min(turn, 2 * math.pi - turn)) < 1.0

    # ---------- the sightings log (for checking placements later; user, 2026-10-09) ----------

    def log_sighting(self, s, T):
        if s["label"] not in self.watcher.prompts and self.look is None:
            return
        path = LOGS / f"sightings_{time.strftime('%Y%m%d')}.jsonl"
        try:
            if path.exists() and path.stat().st_size > SIGHTING_LOG_MB * 1e6:
                return
            pose = self.nav.current_pose()
            rec = {"time": round(time.time(), 2), "stamp": s["stamp"], "room": self.nav.room.name if self.nav.room else None,
                   "label": s["label"], "word": s["word"], "score": round(s["score"], 3),
                   "box": [round(v) for v in s["box"]], "quality": s.get("quality"), "reason": s.get("reason"),
                   "position": s.get("position"), "size": s.get("size"), "points": s.get("points"),
                   "from": s.get("from"), "bearing_deg": s.get("bearing_deg"), "about": s.get("about"),
                   "robot": None if pose is None else [round(float(v), 3) for v in pose],
                   "camera": None if T is None else [round(float(v), 3) for v in T[:3, 3]],
                   "floor_z": round(float(self.nav.depth.floor.z), 3)}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    @staticmethod
    def prune_logs():
        cutoff = time.time() - SIGHTING_LOG_DAYS * 86400
        for p in LOGS.glob("sightings_*.jsonl"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass

    def draw_pending(self):
        p, self.pending_overlay = self.pending_overlay, None
        if p is not None:
            self.set_overlay(*p)

    def overlay_now(self):
        """The overlay for the console, with the boxes moved by how far the robot turned between their image and
        the newest one (what the console shows now: both come through the same relay). The detection takes
        ~0.2 s; while turning, the boxes trailed the picture (user, 2026-10-09). lag_ms: newest image - boxes'."""
        o, f = self.overlay, self.frame
        if o is None or o.get("stamp") is None or f is None:
            return o
        return {**o, "shift_px": round(self.turn_px(o["stamp"], f[1]), 1), "lag_ms": round((f[1] - o["stamp"]) * 1000)}

    def on_held(self, box):
        """Whether a box mostly shows what the gripper holds (nav.held's mask): no find of the next search."""
        h = getattr(self.nav, "held", None)
        m = None if not h else h.get("mask")
        if m is None:
            return False
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        part = m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        return part.size > 0 and float(part.mean()) >= HELD_COVER

    def seen_empty(self, now, labels):
        """Remembered sightings whose place is in view, near enough for the detector and not hidden, but which
        were not seen there for GONE_S: marked "not there any more" (moved, taken away). A moved block stayed
        at its old place and appeared twice (user, 2026-10-09). Only for the words the detector looks for now."""
        v, nav = self.view, self.nav
        if v is None:
            return
        T, w, h = v
        fx, fy, cx, cy = nav.rgb_K
        last = nav.depth.last
        for e in list(self.watcher.seen):
            p = e.get("position")
            if e["label"] not in labels or e.get("quality") != "sure" or e.get("missing_since") or p is None:
                continue
            q = np.array([p[0], p[1], p[2] + nav.depth.floor.z])
            c = (q - T[:3, 3]) @ T[:3, :3]
            dist = float(np.linalg.norm(c))
            inside = c[2] > 0.1 and GONE_RANGE[0] <= dist <= GONE_RANGE[1] and \
                w * GONE_MARGIN <= c[0] * fx / c[2] + cx <= w * (1 - GONE_MARGIN) and \
                h * GONE_MARGIN <= c[1] * fy / c[2] + cy <= h * (1 - GONE_MARGIN)
            if not inside or now - e["time"] < 0.3 or self.hidden(q, last):
                e.pop("unseen_since", None)  # out of view, seen just now, or something in front of it
                continue
            e.setdefault("unseen_since", now)
            if now - e["unseen_since"] >= GONE_S:
                self.watcher.mark_missing(e, now)
                self.nav.note("check", f"{e['label']} #{e['id']} not seen at its place any more: marked gone")

    def hidden(self, q, last):
        """Whether the depth camera sees something in front of the place q (map, z up): then not seeing the
        thing proves nothing."""
        if last is None or self.nav.depth.K is None:
            return False
        depth, T_cam = last[0], last[1]
        c = (q - T_cam[:3, 3]) @ T_cam[:3, :3]
        if c[2] <= 0.1:
            return False
        fx, fy, cx, cy = self.nav.depth.K
        u, v = int(round(c[0] * fx / c[2] + cx)), int(round(c[1] * fy / c[2] + cy))
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            return False
        patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3].astype(np.float32)
        patch = patch[patch > 0]
        return bool(len(patch)) and float(np.median(patch)) / 1000.0 < c[2] - 0.10

    def turn_px(self, t0, t1):
        """How far (px, sideways) the picture moved between two image stamps because the robot turned (odometry);
        turning left moves the scene right. 0 if unknown."""
        if t0 is None or t1 is None or self.nav.rgb_K is None:
            return 0.0
        a, b = self.nav.pose_matrix_at(t0), self.nav.pose_matrix_at(t1)
        if a is None or b is None:
            return 0.0
        turn = math.atan2(b[1, 0], b[0, 0]) - math.atan2(a[1, 0], a[0, 0])
        turn = (turn + math.pi) % (2 * math.pi) - math.pi
        return self.nav.rgb_K[0] * math.tan(turn) if abs(turn) < 0.6 else 0.0

    def set_overlay(self, sightings, w, h, stamp=None):
        """Boxes for the console (user, 2026-10-08: the detections over the live camera picture, the relevant ones
        highlighted). relevant: what is looked for, confident enough, inside the search area (as far as the
        3D position is known); found: at the place of a confirmed find."""
        boxes = []
        for s in sightings:
            if s.get("box") is None:
                continue
            pos = s.get("position") or s.get("about")
            e = s.get("_entry")  # the followed sighting this box belongs to (perception.Watcher.add), if any
            rejected = e is not None and e.get("verdict") == "no"
            relevant = s["score"] >= SCORE_MIN and (pos is None or in_area(pos[:2], self.watcher.area)) and not rejected
            found = e is not None and self.watcher.found.get(s["label"]) is e
            boxes.append({"label": s["label"], "score": round(s["score"], 2), "box": [round(v) for v in s["box"]],
                          "outline": s.get("outline"), "relevant": bool(relevant), "found": found,
                          # its own check result only (never a neighbour's or by place: user, 2026-10-09)
                          "text": verdict_text(e) if e is not None else "",
                          "pos": None if pos is None else [round(float(v), 3) for v in pos[:3]]})
        self.last_boxes = boxes
        # Live only the boxes: masks lag behind the picture visibly whenever anything moves (also driving by hand
        # or the camera tilting while aiming; user, 2026-10-09). The masks show on a frozen picture (freeze()).
        self.overlay_full = {"time": time.time(), "w": w, "h": h, "boxes": boxes, "stamp": stamp}
        if not self.frozen:
            self.overlay = {**self.overlay_full, "boxes": [{k: v for k, v in b.items() if k != "outline"}
                                                          for b in boxes]}

    def clear_overlay(self):
        """A new action started: the boxes of before (and a frozen find's image) go."""
        self.frozen = False
        self.frozen_pose = None
        self.overlay = self.overlay_full = None

    def unfreeze_if_moved(self):
        """A frozen picture no longer fits once the robot moved (e.g. driven by hand after the job)."""
        p0, p = self.frozen_pose, self.nav.current_pose()
        if not self.frozen or p0 is None or p is None:
            return
        turn = abs((math.degrees(p[2] - p0[2]) + 180) % 360 - 180)
        if math.hypot(p[0] - p0[0], p[1] - p0[1]) > UNFREEZE_M or turn > UNFREEZE_DEG:
            self.clear_overlay()

    @staticmethod
    def crop(img, box, margin=0.5, mark=True):
        """The box with some surroundings (the verifier judges "small" against them); mark: grey corner marks
        around the box itself, so in a pile the check judges the candidate, not its neighbour (neutral grey:
        a coloured frame could pass for the item's colour)."""
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * margin + 8, (y1 - y0) * margin + 8
        h, w = img.shape[:2]
        cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        out = img[cy0:min(h, int(y1 + my)), cx0:min(w, int(x1 + mx))].copy()
        if mark:
            a0, b0, a1, b1 = int(x0) - cx0 - 2, int(y0) - cy0 - 2, int(x1) - cx0 + 2, int(y1) - cy0 + 2
            L = max(3, int(0.25 * min(a1 - a0, b1 - b0)))
            for (px, py, dx, dy) in ((a0, b0, 1, 1), (a1, b0, -1, 1), (a0, b1, 1, -1), (a1, b1, -1, -1)):
                cv2.line(out, (px, py), (px + dx * L, py), (128, 128, 128), 1)
                cv2.line(out, (px, py), (px, py + dy * L), (128, 128, 128), 1)
        return out

    def sim_sightings(self):
        time.sleep(0.2)  # ~5 images/s, like the detector
        pose = self.nav.current_pose()
        if pose is None or self.nav.room is None:
            return []
        x, y, yaw = pose
        cx, cy = x + CAMERA_AHEAD * math.cos(yaw), y + CAMERA_AHEAD * math.sin(yaw)
        grid, res, origin = self.nav.room.grid, self.nav.room.resolution, self.nav.room.origin
        out = []
        for o in self.sim:
            if o["label"] not in self.watcher.prompts and (self.look is None or o["label"] != self.look["label"]):
                continue
            dx, dy = o["x"] - cx, o["y"] - cy
            d = math.hypot(dx, dy)
            if not SIM_RANGE[0] <= d <= SIM_RANGE[1] or abs(math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                                                                     math.cos(math.atan2(dy, dx) - yaw))) > SIM_FOV / 2:
                continue
            ts = np.arange(0.1, max(0.1, d - 0.15), 0.025)  # line of sight over the room map's walls
            c = ((cx + dx * ts / d - origin[0]) / res).astype(int)
            r = ((cy + dy * ts / d - origin[1]) / res).astype(int)
            inside = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
            if (grid[r[inside], c[inside]] >= 65).any():
                continue
            n = np.random.normal(0, 0.03, 2)
            # a rough box where the object would be in the image (for the console's overlay)
            fx, fy, cx, cy = SIM_K
            rel = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
            u, v, half = cx - fx * math.tan(rel), cy + fy * SIM_CAMERA_Z / d, 0.08 * fx / d
            out.append({"label": o["label"], "score": 0.6, "position": [o["x"] + n[0], o["y"] + n[1], 0.01],
                        "size": [0.15, 0.02, 0.01], "box": [u - half, v - half / 2, u + half, v + half / 2],
                        "stamp": time.time(), "crop": None})
        self.pending_overlay = (out, 640, 400)
        return out

    # ---------- for /state ----------

    def state(self):
        w = self.watcher
        now = time.time()
        self.unfreeze_if_moved()
        return {"looking_for": w.prompts, "area": None if w.area is None else np.round(w.area, 3).tolist(),
                "images_per_s": round(len([t for t in self.rates if now - t < 2.0]) / 2.0, 1)
                if w.active or self.look else (1.0 if now < self.fed_until else 0),
                "error": self.error, "simulated": self.sim is not None,
                "overlay": self.overlay_now(),
                "objective": w.objective,
                "found": [{"label": k, "position": v["position"], "score": round(v["score"], 2), "views": v["views"],
                           "age_s": round(now - v["time"]), "id": v.get("id")} for k, v in w.found.items()
                          if k in w.objective],
                "seen": [{"id": e["id"], "label": e["label"], "position": e["position"], "score": round(e["score"], 2),
                          "in_area": e["in_area"], "age_s": round(now - e["time"]), "verdict": e.get("verdict"),
                          "what": e.get("verdict_what")} for e in w.current()],
                "checks": None if self.verifier is None else {"calls": self.verifier.calls,
                                                               "last_ms": self.verifier.last_ms},
                "vague": sum(1 for e in w.seen if e["label"] in w.objective and e.get("quality") == "vague")}
