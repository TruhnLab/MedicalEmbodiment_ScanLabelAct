"""Open-vocabulary object detection in camera images: "find a pencil" with the item as a text prompt. Runs on
the laptop's CPU (no GPU; nothing heavy on the robot). The live search (search.py) and the offline
test (test_detector.py) use this.

    d = Detector()                            # default: MODEL; weights downloaded once to data/models/
    d.set_prompts(["pencil", "cup"])          # what to look for (text)
    for det in d.detect(bgr_image):           # [{"label", "score", "box": [x0, y0, x1, y1], "mask": HxW bool or None}]
        ...

Models (ultralytics): YOLOE (open vocabulary, text prompts, with masks in the "-seg" variants: the depth
pixels of the object itself, not of the background around a thin thing) and YOLO-World (boxes only).
Ultralytics is AGPL-3.0: using this module means using it under that licence (README, License).
"""

import os
import time
from pathlib import Path

import numpy as np

# Two OpenMP runtimes meet in one process: torch's libiomp5md.dll and libomp.dll of the environment's MKL numpy
# (conda-forge llvm-openmp); the second to start aborted the process ("OMP: Error #15", test_positions.py,
# 2026-10-08). Replacing numpy's MKL build would change the shared environment of the reconstruction and the
# labelling, so only processes that use the detector allow both (set before torch is imported).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# ... and torch must load before numpy's MKL first runs: afterwards Windows took MKL's libiomp5md.dll and torch
# failed ("WinError 127 ... shm.dll"; navigator, test_far.py, 2026-10-08). Importing this module loads it.
import torch  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "data" / "models"   # data/ is not in git
# YOLOE-11s-seg on the room's labelled views (test_detector.py, 20261007_101108, 2026-10-08): 68 % of the
# objects' views found (score >= 0.1), 200 ms per image (~5/s) on the laptop's CPU; YOLO-World v2 s: 44 %, 175 ms
MODEL = "yoloe-11s-seg.pt"


def use_models_dir():
    """ultralytics looks for helper files (YOLOE's 572 MB text encoder mobileclip_blt.ts, loaded anew at every
    set_classes) in the current directory and downloads them there if missing: the repo's root got it twice
    (2026-10-08). It also looks in its weights_dir setting: point that to data/models, in memory only (dict
    assignment without the settings file being saved)."""
    from ultralytics.utils import SETTINGS
    dict.__setitem__(SETTINGS, "weights_dir", MODELS)


class Detector:
    def __init__(self, model=MODEL, imgsz=640, conf=0.05):  # perception.SCORE_MIN
        from ultralytics import YOLO  # heavy import (torch): only where detection is used
        MODELS.mkdir(parents=True, exist_ok=True)
        self.name = model
        self.imgsz, self.conf = imgsz, conf
        self.prompts = []
        self.yoloe = "yoloe" in model.lower()
        use_models_dir()
        self.model = YOLO(str(MODELS / model))

    def set_prompts(self, prompts):
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if prompts == self.prompts:
            return
        if self.yoloe:
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
        else:
            self.model.set_classes(prompts)
        self.prompts = prompts

    def detect(self, bgr, conf=None, shrink=None):
        """bgr: HxWx3 uint8 (OpenCV). Returns detections, best first. shrink (0-1): detect in the image scaled
        down by this factor on a grey canvas of the same size, boxes and masks mapped back: up close an object
        can be too large for the model (a mug 20 cm in front: score 0.01-0.03 as is, 0.07-0.11 at 0.6, where
        it had 0.8 from 1 m; run 20261009_090645)."""
        if shrink:
            return self._detect_shrunk(bgr, conf, shrink)
        t0 = time.perf_counter()
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf, verbose=False)[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        out = []
        if r.boxes is None or not len(r.boxes):
            return out
        masks = None
        if getattr(r, "masks", None) is not None and r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5  # model resolution: scaled to the image below
        h, w = bgr.shape[:2]
        for i, (box, score, cls) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                  r.boxes.cls.cpu().numpy().astype(int))):
            m = None
            if masks is not None:
                import cv2
                # the masks come with the letterbox's padding (640 x 416 for our 640 x 400 images): crop it before
                # scaling, else rows near the top and bottom shift by up to 8 px (scene-understanding, 2026-10-09)
                mh, mw = masks.shape[1:]
                g = min(mh / h, mw / w)
                py, px = int(round((mh - h * g) / 2)), int(round((mw - w * g) / 2))
                core = masks[i][py:mh - py, px:mw - px]
                m = cv2.resize(core.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append({"label": self.prompts[cls] if cls < len(self.prompts) else str(cls), "score": float(score),
                        "box": [float(v) for v in box], "mask": m,
                        "outline": outline(m)})
        out.sort(key=lambda d: -d["score"])
        return out

    def _detect_shrunk(self, bgr, conf, k):
        import cv2
        h, w = bgr.shape[:2]
        sw, sh = int(w * k), int(h * k)
        x, y = (w - sw) // 2, (h - sh) // 2
        canvas = np.full_like(bgr, 114)  # ultralytics' letterbox grey
        canvas[y:y + sh, x:x + sw] = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        out = []
        for d in self.detect(canvas, conf):
            x0, y0, x1, y1 = d["box"]
            d["box"] = [min(max((x0 - x) / k, 0), w), min(max((y0 - y) / k, 0), h),
                        min(max((x1 - x) / k, 0), w), min(max((y1 - y) / k, 0), h)]
            if d["mask"] is not None:
                d["mask"] = cv2.resize(d["mask"][y:y + sh, x:x + sw].astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                d["outline"] = outline(d["mask"])
            if d["box"][2] - d["box"][0] > 1 and d["box"][3] - d["box"][1] > 1:
                out.append(d)
        return out


OUTLINE_PX = 1.5      # outline simplification: a few dozen points per piece, ~1 KB per object in /state
OUTLINE_MIN_PX = 30   # pieces of a mask smaller than this (pixels) are left out


def outline(mask):
    """The mask's outline for drawing: one polygon [[x, y], ...] per piece, or None. From the bitmap, not
    ultralytics' masks.xy: that joins a mask's pieces into one polygon, with straight lines across (stool legs,
    2026-10-09)."""
    import cv2
    if mask is None:
        return None
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [cv2.approxPolyDP(c, OUTLINE_PX, True).reshape(-1, 2).tolist() for c in cs
             if cv2.contourArea(c) >= OUTLINE_MIN_PX]
    return [p for p in polys if len(p) >= 3] or None
