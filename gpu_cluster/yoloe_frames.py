"""YOLOE (open vocabulary, masks) on a session's keyframes: the vocabulary's things in every keyframe, written as
masks/<index>.npz + frames_vlm.jsonl (scene understanding, step 1). Used by
projects/scene_understanding/detect_frames.py on the laptop (import) and in a Slurm job (yoloe_job.sbatch):

    python yoloe_frames.py --scene <scene dir> --models <weights dir>   # reads seg/vocabulary.json, seg/frames.jsonl, input/rgb/

Only numpy, OpenCV and ultralytics: no repo imports, so it runs in the cluster's landerpi-seg env.
Ultralytics is AGPL-3.0: using this script means using it under that licence (README, License).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_model(weights, vocab, weights_dir=None):
    """YOLOE with every vocabulary label and synonym as a class; returns (model, entry per class, prompts)."""
    from ultralytics import YOLO
    if weights_dir:
        # ultralytics looks for YOLOE's text encoder (mobileclip_blt.ts) in its weights_dir (and the current
        # directory), downloading it there otherwise; in memory only, the settings file is not written
        from ultralytics.utils import SETTINGS
        dict.__setitem__(SETTINGS, "weights_dir", Path(weights_dir))
    model = YOLO(str(weights))
    prompts, entry = [], []
    for e in vocab:
        for name in [e["label"], *e.get("synonyms", [])]:
            if name not in prompts:
                prompts.append(name)
                entry.append(e)
    model.set_classes(prompts, model.get_text_pe(prompts))
    return model, entry, prompts


def scale_masks(masks, shape):
    """YOLOE's masks are at the letterboxed input size (640 x 416 for a 640 x 400 image: 8 px of padding
    above and below); crop the padding, then scale to the image."""
    import cv2
    h, w = shape
    mh, mw = masks.shape[1:]
    g = min(mh / h, mw / w)
    py, px = int(round((mh - h * g) / 2)), int(round((mw - w * g) / 2))
    out = np.zeros((len(masks), h, w), bool)
    for k, m in enumerate(masks):
        out[k] = cv2.resize(m[py:mh - py, px:mw - px].astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
    return out


def detect_image(model, entry, prompts, img, conf, device=None):
    """img: BGR. Returns (objects for frames_vlm.jsonl, masks N x H x W bool, scores)."""
    # one label per thing (agnostic NMS): otherwise one stool can come as "bar stool" and "chair", two masks
    r = model.predict(img, imgsz=640, conf=conf, verbose=False, agnostic_nms=True, device=device)[0]
    if r.boxes is None or not len(r.boxes) or r.masks is None:
        return [], np.zeros((0, *img.shape[:2]), bool), np.zeros(0, np.float32)
    masks = scale_masks(r.masks.data.cpu().numpy() > 0.5, img.shape[:2])
    objs = []
    for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        e = entry[cls]
        objs.append({"label": e["label"], "prompt": prompts[cls], "box": [round(float(v), 1) for v in box],
                     "category": e["category"], "movable": e["movable"], "state": None})
    return objs, masks, r.boxes.conf.cpu().numpy().astype(np.float32)


def run(frames, rgb_dir, out_dir, vocab, weights, conf, weights_dir=None, device=None, on_progress=None):
    """frames: [{"index", "rgb", "stamp"}]. Writes out_dir/masks/<index>.npz and out_dir/frames_vlm.jsonl."""
    import cv2
    model, entry, prompts = load_model(weights, vocab, weights_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    t0, n_det = time.time(), 0
    with (out_dir / "frames_vlm.jsonl").open("w") as fh:
        for n, f in enumerate(frames, 1):
            img = cv2.imread(str(rgb_dir / f["rgb"]))
            objs, masks, scores = detect_image(model, entry, prompts, img, conf, device)
            np.savez_compressed(out_dir / "masks" / f"{f['index']}.npz", masks=np.packbits(masks, axis=-1),
                                width=img.shape[1], scores=scores, objects=np.arange(len(objs)))
            fh.write(json.dumps({"index": f["index"], "rgb": f["rgb"], "stamp": f["stamp"], "caption": None,
                                 "objects": objs, "model": Path(weights).name}) + "\n")
            n_det += len(objs)
            if n % 100 == 0 or n == len(frames):
                print(f"{n}/{len(frames)} frames, {n_det} detections, {time.time() - t0:.0f} s", flush=True)
                if on_progress:
                    on_progress(n, len(frames))
    return {"frames": len(frames), "detections": n_det, "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--models", required=True, help="YOLOE weights + text encoder (CLUSTER_MODELS_ON_MASTER in .env)")
    a = ap.parse_args()
    seg = Path(a.scene) / "seg"
    v = json.loads((seg / "vocabulary.json").read_text())
    frames = [json.loads(l) for l in (seg / "frames.jsonl").read_text().splitlines() if l.strip()]
    d = v["detect"]
    stats = run(frames, Path(a.scene) / "input" / "rgb", seg, v["vocabulary"], Path(a.models) / d["model"], d["conf"],
                weights_dir=a.models, device=0)
    (seg / "yoloe_done").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
