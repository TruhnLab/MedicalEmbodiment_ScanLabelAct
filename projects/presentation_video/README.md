# Presentation video

A video of a mapping run for talks, sped up. It has two panels in sync:
- **Left:** the lidar map, with the path driven and the robot.
- **Right:** the 3D reconstruction building up.

It is made afterwards, on demand, from a finished recording. Nothing here runs during a run, and
nothing extra is recorded for it.

```
python projects/mapping/reconstruct.py data/sessions/<name>            # first, if not done yet
python projects/presentation_video/make_video.py data/sessions/<name>
```

Output: `data/sessions/<name>/video/<name>_x<speed>.mp4` (H.264, 1920 × 1080, 30 fps; plays in
PowerPoint). It takes a few minutes on the laptop; no GPU is needed.

## How it is made

- **Lidar map**: rebuilt from the recorded scans along the final SLAM path, every 2 s of the run
  (as SLAM publishes its map). Console colours, 1 m grid, the path driven, the robot and the
  camera's field of view (where the camera really looked, from the reconstruction's poses).
  No-go lines drawn in the console are shown in light red from the moment they were drawn
  (recordings from 2026-10-05 on; `streams/nogo_lines.jsonl`).
- **3D**: the final mesh, revealed at the moment the robot's camera first saw each part. Newly
  seen surfaces glow blue for a moment. It is the end result, built up in scan order: re-fusing a
  partial mesh for every frame would be slow and flicker.
  - The test for "when did the camera first see this vertex" is the reconstruction's own (on the
    depth image, not occluded).
  - It is computed once per reconstruction and cached in `video/first_seen.npz`.
- A small badge in the bottom left shows the time-lapse factor and the run time.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--speed N` | 8 | Time-lapse factor. `--target S` instead picks the speed so the video lasts about S seconds. |
| `--preview S --at F` | – | A quick test clip: only S seconds, starting at fraction F (0–1) of the run. |
| `--still F` | – | One frame at fraction F as PNG (e.g. for slides). |
| `--theme light\|dark` | light | White background (free space light blue-grey, walls dark) or the console's dark look. |
| `--glow S` | 1.2 | How long (video seconds) newly seen surfaces glow; 0 = off. |
| `--orbit S` | 0 | Append an S-second full-width orbit around the finished room. |

Inputs: `reconstruction/mesh_view.ply`, `cameras.json`, `streams/slam_robot_pose.jsonl`,
`scan.jsonl` (and `nogo_lines.jsonl` if present) and `calib/`.

## Task videos (picking up, finding things)

Two more videos for talks, in the same look (`style.py`), from what the task console's runs leave behind. Output in
`data/demo/` (H.264, 1920 × 1080, 30 fps; a PNG of the last frame next to it for a slide).

```
python projects/presentation_video/pick_video.py                         # the newest pick through the chat
python projects/presentation_video/task_run_video.py [data/task_runs/<time>] [--speed 2]
```

- **`pick_video.py`**: a pick from the robot's camera (the detector's box, "held" once in the gripper), the chat
  request and answer, the steps ticked off as they happen; ×2, ×6 for going down, closing and lifting. Reads the
  pick's folder (`data/manipulation/picks/<time>/`, one per pick since 2026-10-13), navigation's `_last_search` and the
  navigator's log of the run.
- **`task_run_video.py`**: a run recorded with the task console's **● Record** (`data/task_runs/<time>/`): the camera
  with the detector's boxes, the room's top view (the track, the camera's view, the target from the room's object list,
  what was found), the steps and the chat.
- `--still S` writes one frame (S seconds into the video) as PNG, to check the layout.

## Localization slide (a picture, no video)

```
python projects/presentation_video/localization_figure.py [--room <room> --run <session>]
```

Three panels from real data, minimal text: the start turn of a recorded run (its lidar scans stitched with odometry),
the search through the room's map (how well the turn fits at each position), the result (the points on the walls).
Navigation's own localization code does the work (`projects/navigation/localize.py`), on a run other than the one the
room was made from. Default: the kitchen (room 20261007_101108) and the run of 2026-10-06 16:40 (fit 0.65, the next best
place elsewhere 0.34). Output: `data/demo/localization.png` and the panels one by one (`localization_1/2/3.png`).
