# Scene understanding: objects in 3D for an LLM / VLM

Turns a reconstructed mapping session into `scene.json`: the room's objects with label, description,
upright box in the map frame, state at scan time (doors), operable parts linked to their whole (`part_of`),
views and crops; plus `summary.md`, a short text for an LLM prompt.
Similar in spirit to HomeBody (tml.stanford.edu/homebody): the VLM decides what the objects are, a
detector finds and cuts them out.

```powershell
python projects/scene_understanding/run.py 20261006_161813              # all steps, 7-15 min
python projects/scene_understanding/run.py 20261006_161813 --from build  # redo from a step
```

| Step | Script | Where |
|---|---|---|
| 1 a few views that see the room up close; Qwen names what is in it (word list); YOLOE finds and segments it in every keyframe | `detect_frames.py` (+ `gpu_cluster/yoloe_*`) | laptop → vLLM gateway; Slurm job (yoloe-11l), laptop CPU (yoloe-11s) if the cluster fails |
| 2 masks → 3D, views merged into objects | `build_objects.py` | laptop |
| 3 small things on the floor, from depth | `floor_items.py` | laptop |
| 4 VLM: close pairs one object or two (fragments, duplicates, operable parts) | `judge_pairs.py` | laptop → vLLM |
| 5 VLM names each object from its best crops | `label_objects.py` → `scene.json` | laptop → vLLM |
| 6 text for an LLM prompt | `scene_summary.py` → `summary.md` | laptop |
| 7 data for the 3D view (object per mesh vertex, mask images) | `export_viewer.py` | laptop |
| compare two labellings of one room | `compare_scenes.py` | laptop |
| check images | `check_objects.py`, `check_session.py` | laptop |

Results go to `data/sessions/<session>/scene/`. In the console's **3D** view: **Label** (next to
Reconstruct, once a session is reconstructed) runs `run.py` through the helper, with its progress in
the info box; **Labels** (next to Mesh / Points) then shows each object's surface in its own colour,
upright boxes and tags, and a panel with the objects (filter field). Clicking an object shows its
description, state and the keyframes with its mask. Code: `console/labels3d.js`. A newer
reconstruction makes the labels outdated (an object id per mesh vertex): Label again.

Until 2026-10-09 step 1 was Qwen boxes in every keyframe + SAM 2.1 on the cluster: 40-50 min per run.
Tests on four sessions of three rooms (2026-10-09): comparable objects,
fewer duplicate seats, better part links with yoloe-11l; weaker on small room parts (switches, outlets).

## Setup

- vLLM gateway: `VLLM_API_KEY` (and optionally `VLLM_URL`, `VLLM_MODEL`) in the repo's `.env`
  (gitignored, not synced to the cluster). Model: Qwen3.8 Flash-Next (tests in `vlm.py`).
  Never a model of `VLLM_BLOCKED_MODELS` (`.env`).
- Cluster env, once on the master: `bash ~/code/LanderPi/gpu_cluster/setup_seg_env.sh` (env
  `landerpi-seg`: torch, ultralytics). YOLOE weights and its text encoder (`mobileclip_blt.ts`) are
  copied once from the laptop's `data/models/` to the cluster storage (`CLUSTER_MODELS_ON_MASTER` in `.env`).
- Laptop: the YOLOE weights in `data/models/` (downloaded by ultralytics on first use, see
  `landerpi/detector.py`, whose loading this step shares).

## How it works, and what is known to be weak

- Word list: views chosen greedily until 90 % of the room's surfaces within 2 m are seen (~100-170 of
  500-1400 keyframes). Qwen lists what makes up the room (furniture, doors, fixtures, operable parts;
  not people or small portable things), examples only as orientation. Things Qwen names in only 1-2 views
  come and go between runs (two runs on the same views, 2026-10-09).
- 3D points of a mask: only that keyframe's own measured depth (no filling from the mesh). A view needs
  measured depth on 30 % of its mask; objects need 3+ keyframes from 2+ viewpoints.
- Area of interest: floor reachable from the driven path without crossing walls or the run's no-go
  lines; older runs: the path's hull + 1.2 m; or `scene/area.json`.
- Merging: each keyframe's masks are matched one-to-one to the objects so far (2D IoU); masks of one
  frame are different objects. Labels are compared by their head noun ("door handle" is a handle, not a
  door): parts stay their own objects. Seating has an extra rule (one object per seat, by the density of
  points at seat height) for tight rows of identical stools.
- Pair check (`judge_pairs.py`): objects whose points touch or whose centres are < 40 cm apart are shown
  to Qwen (a keyframe with both, A green / B red, each one's best view and the facts). 3 votes: same,
  fragment (merged), operable part (linked by `part_of`), different. No merge if the detector found them
  as two objects in >= 40 % of the images showing both. Log: `scene/pairs.json`.
- Doors and other articulated things keep their state with the time seen; it may have changed since.
- Weak (2026-10-09): rows of identical stools (pairs as one object, or a stool split); cabinet doors often
  named "cabinet" (linked correctly); leg fragments sometimes stay; a wall piece can pass as a
  "partition"; small room parts missing when Qwen does not name them.
