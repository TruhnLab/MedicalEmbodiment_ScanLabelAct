"""Step 1 of scene understanding: what is in each keyframe, with masks.

  a) views    a few keyframes that together see the whole room up close (greedy cover of the depth points)
  b) vocab    Qwen names the kinds of things in these views (no boxes), one text call merges the synonyms
  c) detect   YOLOE (open vocabulary, masks) finds them in every keyframe (gpu_cluster/yoloe_frames.py): as a
              Slurm job with the large model; on the laptop's CPU with the small one if the cluster is not
              reachable or no GPU is free within START_TIMEOUT

Writes scene/vocabulary.json, scene/frames_vlm.jsonl (detections per keyframe) and scene/masks/<index>.npz.
Replaced Qwen boxes in every keyframe + SAM 2 on 2026-10-09: 7-15 min instead of 40-50 for a run,
comparable objects (tests on four sessions of three rooms, compare_scenes.py).

    python projects/scene_understanding/detect_frames.py 20261006_161813
    python projects/scene_understanding/detect_frames.py 20261006_161813 --from detect --where laptop
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "gpu_cluster"))
sys.path.insert(0, str(REPO / "tools"))
from landerpi import detector  # noqa: E402  (first: torch must load before numpy's MKL, see detector.py)
import numpy as np  # noqa: E402
from landerpi.session import Session  # noqa: E402
import progress  # noqa: E402
import vlm  # noqa: E402
from schemas import VocabAnswer, VocabList  # noqa: E402
import yoloe_frames  # noqa: E402

VOXEL = 0.10      # m, coverage cells of the view choice
NEAR = 2.0        # m: a cell counts as seen by a view only this close (small things: outlets, items on tables)
COVER = 0.90      # share of all cells the chosen views must see
# yoloe-11l on the cluster: fewer fragments and better part links than 11s (four sessions, 2026-10-09), 15-30 s
# of GPU time per session, ~1.5-2.5 min with upload and queue; 11s on the laptop's CPU took 4-6 min
MODELS = {"cluster": "yoloe-11l-seg.pt", "laptop": "yoloe-11s-seg.pt"}
START_TIMEOUT = 600   # s waiting for a GPU before the laptop takes over

VOCAB_PROMPT = """This image is from a small ground robot's camera, 0.3 m above the floor, tilted slightly down.
The black shapes at the bottom edge are the robot's own gripper: ignore them.
List every kind of thing you can see that makes up the room, once per kind (not per instance), as JSON:
{"objects": [{"label": "<short common noun, e.g. bar stool, door, cabinet, power outlet>",
              "category": "furniture|door|appliance|fixture|container|floor_item|person|other",
              "movable": "fixed|movable|articulated"}]}
Rules:
- Things that make up the room: furniture, doors, built-in fixtures, larger appliances, and parts that are
  operated on their own. Typical examples, for orientation only (name what you see, also things not in
  this list): {examples}.
- Leave out people, and small portable things that will likely be elsewhere next time (cups, bottles, bags,
  shoes, clothes, papers, cables, boxes).
- movable: fixed = built in (kitchen counter, radiator, wall cabinet), articulated = doors, drawers,
  lids, cabinet doors, movable = everything else that can be moved (chairs, tables, bins).
- Do not list walls, skirting boards, floor, ceiling, windows, lamps, wall paint, or anything seen
  through glass."""

# Typical things that make up a room, as orientation in the prompt (not forced onto the list: another room
# may have none of them). Qwen's naming is not reproducible for things it names in 1-2 views (two runs on the
# same 131 views, 2026-10-09: "door closer", "handle" only in one of them); examples make the easily missed
# ones (handles, switches) more likely to be named where they are.
EXAMPLES = ["door", "door handle", "drawer", "cabinet", "cabinet door", "shelf", "table", "chair", "sofa",
            "bed", "sink", "kitchen counter", "radiator", "power outlet", "light switch", "bin"]

MERGE_PROMPT = """These object names were collected from several camera views of one room (count = in how many
views the name came up). They will be text prompts for an open-vocabulary object detector.
Merge names that mean the same kind of thing into one entry: pick the most common, plain English name
as label and list the others as synonyms. Keep different kinds apart (a stool is not a chair, a drawer is
not a cabinet, a cabinet door is not a cabinet). Keep every other kind, also rare ones.
Leave out (the first run, 2026-10-09, found these as objects: ceiling light, floor, skirting board, "fixture"):
- parts of the building: walls, floor, ceiling and anything on it (lights, panels, vents, pipes), skirting boards,
  windows;
- words that do not name a kind of thing (device, fixture, object, item);
- the robot's own gripper, and people;
- small portable things that will likely be elsewhere next time (cups, bottles, bags, shoes, clothes, papers,
  cables, boxes): the list is for what makes up the room.
Names (count, category, movable):
"""


def view_cells(s, f):
    pts, _, _ = s.depth_points(f)
    pts = pts[::5]
    cam = f.T_map_depth[:3, 3]
    pts = pts[(np.linalg.norm(pts - cam, axis=1) < NEAR) & (pts[:, 2] > 0.05)]  # not the empty floor
    c = np.floor(pts / VOXEL).astype(np.int64) + 10000
    return set((c[:, 0] * 20000 + c[:, 1]) * 20000 + c[:, 2])


def choose_views(s):
    """Greedy cover: the view that sees the most cells not seen yet, until COVER of all cells are seen."""
    with ThreadPoolExecutor(6) as pool:
        cells = list(pool.map(lambda f: view_cells(s, f), s.frames))
    total = set().union(*cells)
    seen, chosen = set(), []
    while len(seen) < COVER * len(total):
        i = max(range(len(cells)), key=lambda k: len(cells[k] - seen))
        chosen.append(s.frames[i].index)
        seen |= cells[i]
    print(f"  {len(chosen)} of {len(s.frames)} keyframes see {len(seen) / len(total):.0%} of {len(total)} cells "
          f"(up to {NEAR} m)", flush=True)
    return chosen


def vocabulary(s, views):
    by_index = {f.index: f for f in s.frames}

    def ask(i):
        text = vlm.chat([vlm.image_part(by_index[i].rgb_file.read_bytes()), {"type": "text", "text": VOCAB_PROMPT.replace("{examples}", ", ".join(EXAMPLES))}],
                        max_tokens=2000, schema=VocabAnswer)
        return VocabAnswer.model_validate_json(text).objects

    t0 = time.time()
    counts, kinds = Counter(), {}
    with ThreadPoolExecutor(3) as pool:  # 6 got HTTP 429 (2026-10-05)
        for n, objs in enumerate(pool.map(ask, views), 1):
            for o in objs:
                name = o.label.strip().lower()
                if o.category == "person" or not name:
                    continue
                counts[name] += 1
                kinds.setdefault(name, Counter())[(o.category, o.movable)] += 1
            progress.update(s.path / "scene", done=n, total=len(views))
    names = "\n".join(f"- {n} ({c}, {kinds[n].most_common(1)[0][0][0]}, {kinds[n].most_common(1)[0][0][1]})"
                      for n, c in counts.most_common())
    merged = VocabList.model_validate_json(vlm.chat([{"type": "text", "text": MERGE_PROMPT + names}],
                                                    max_tokens=8000, schema=VocabList)).objects
    vocab = [{"label": e.label.strip().lower(), "category": e.category, "movable": e.movable,
              "synonyms": sorted({x.strip().lower() for x in e.synonyms} - {e.label.strip().lower()})}
             for e in merged]
    print(f"  {len(counts)} names in {len(views)} views -> {len(vocab)} kinds, {time.time() - t0:.0f} s", flush=True)
    return {"views": views, "names": dict(counts.most_common()), "vocabulary": vocab, "model": vlm.MODEL,
            "seconds": round(time.time() - t0)}


def detect_laptop(s, v):
    """On the laptop's CPU: ~0.25 s per keyframe with yoloe-11s-seg (20261007_101108: 1361 keyframes, 340 s)."""
    d = v["detect"]
    frames = [{"index": f.index, "rgb": f.rgb_file.name, "stamp": f.stamp} for f in s.frames]
    sd = s.path / "scene"
    return yoloe_frames.run(frames, s.path / "rgb", sd, v["vocabulary"], detector.MODELS / d["model"], d["conf"],
                            weights_dir=detector.MODELS, on_progress=lambda n, t: progress.update(sd, done=n, total=t))


def detect_cluster(s, v):
    """As a Slurm job (gpu_cluster/yoloe_job.sbatch): upload the colour keyframes, the vocabulary and the frame
    list to the cluster storage, run, fetch masks/ and frames_vlm.jsonl. Raises if the job does not start
    within START_TIMEOUT (cancelled) or ends without results."""
    import gpu_cluster as gc
    sd = s.path / "scene"
    (sd / "frames.jsonl").write_text("".join(json.dumps({"index": f.index, "rgb": f.rgb_file.name, "stamp": f.stamp})
                                             + "\n" for f in s.frames))
    (sd / "vocabulary.json").write_text(json.dumps(v, indent=1))
    master, ocean, src = f"{gc.SCENES_ON_MASTER}/{s.name}", f"{gc.SCENES_ON_OCEAN}/{s.name}", gc.wsl_path(s.path)
    t0 = time.time()
    gc.sync_code()
    gc.ssh(gc.OCEAN, f"mkdir -p {ocean}/input {ocean}/seg && rm -rf {ocean}/seg/masks {ocean}/seg/yoloe_done")
    gc.rsync(f"{src}/rgb {gc.OCEAN}:{ocean}/input/")
    gc.rsync(f"{src}/scene/frames.jsonl {src}/scene/vocabulary.json {gc.OCEAN}:{ocean}/seg/")
    print(f"  uploaded in {time.time() - t0:.0f} s", flush=True)
    job = gc.ssh(gc.MASTER, f"{gc.SBATCH} --parsable --job-name=yoloe_{s.name} --output={master}/seg/slurm_yoloe.log "
                            f"{gc.REMOTE_CODE}/gpu_cluster/yoloe_job.sbatch {master} --models {gc.MODELS_ON_MASTER}"
                            ).strip().split(";")[0]
    print(f"  job {job} submitted", flush=True)
    # for the console's Stop (tools/robot_helper.py stop_job): it cancels this job
    (sd / "slurm_job").write_text(job)
    progress.update(sd, detail="waiting for a GPU")
    (sd / "yoloe_done").unlink(missing_ok=True)
    submitted = time.time()
    while True:
        time.sleep(10)
        try:
            state = gc.ssh(gc.MASTER, f"squeue -h -j {job} -o %T").strip()
            tail = gc.ssh(gc.MASTER, f"tail -n 1 {master}/seg/slurm_yoloe.log 2>/dev/null").strip()
        except Exception as e:  # network hiccup
            print("  (", str(e)[-80:], ")")
            continue
        print(f"  {time.strftime('%H:%M:%S')} {state or 'ended'}  {tail[-100:]}", flush=True)
        progress.update(sd, detail={"PENDING": "waiting for a GPU", "RUNNING": "running"}.get(state, "fetching"))
        if state == "PENDING" and time.time() - submitted > START_TIMEOUT:
            gc.ssh(gc.MASTER, f"scancel {job}")
            (sd / "slurm_job").unlink(missing_ok=True)
            raise RuntimeError(f"no GPU within {START_TIMEOUT} s (job {job} cancelled)")
        if not state:
            break
    (sd / "slurm_job").unlink(missing_ok=True)
    gc.rsync(f"{gc.OCEAN}:{ocean}/seg/masks {gc.OCEAN}:{ocean}/seg/frames_vlm.jsonl {gc.OCEAN}:{ocean}/seg/yoloe_done "
             f"{gc.OCEAN}:{ocean}/seg/slurm_yoloe.log {src}/scene/")
    if not (sd / "yoloe_done").exists():
        raise RuntimeError(f"job {job} ended without results (scene/slurm_yoloe.log)")
    stats = json.loads((sd / "yoloe_done").read_text())
    stats["total_seconds"] = round(time.time() - t0)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--from", dest="start", default="views", choices=["views", "detect"])
    ap.add_argument("--conf", type=float, default=0.10, help="detector score threshold")
    ap.add_argument("--where", default="auto", choices=["auto", "cluster", "laptop"],
                    help="auto: the cluster, the laptop if that fails")
    ap.add_argument("--model", help="YOLOE weights (default: MODELS by where it runs)")
    a = ap.parse_args()
    s = Session(a.session)
    sd = s.path / "scene"
    sd.mkdir(exist_ok=True)
    vf = sd / "vocabulary.json"
    if a.start == "views":
        t0 = time.time()
        progress.update(sd, stage="VLM: what is in the room")
        views = choose_views(s)
        print(f"  view choice {time.time() - t0:.0f} s", flush=True)
        vf.write_text(json.dumps(vocabulary(s, views), indent=1))
    v = json.loads(vf.read_text())
    print(f"{len(v['vocabulary'])} kinds: {', '.join(e['label'] for e in v['vocabulary'])}", flush=True)
    if (sd / "masks").exists():  # from an earlier run (possibly with another model): no mixing
        shutil.rmtree(sd / "masks")
    places = ["cluster", "laptop"] if a.where == "auto" else [a.where]
    for where in places:
        v["detect"] = {"model": a.model or MODELS[where], "conf": a.conf, "where": where}
        progress.update(sd, stage=f"YOLOE ({'GPU cluster' if where == 'cluster' else 'laptop'})", done=0, total=0)
        try:
            v["detect"].update(detect_cluster(s, v) if where == "cluster" else detect_laptop(s, v))
            break
        except Exception as e:
            if where == places[-1]:
                raise
            print(f"  cluster failed ({str(e)[-200:]}): on the laptop instead", flush=True)
    print(f"  {v['detect']}", flush=True)
    vf.write_text(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
