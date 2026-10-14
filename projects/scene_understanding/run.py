"""Scene understanding of a reconstructed session, all steps in order (each can also run on its own).
The console's 3D view starts this with its Label button (via tools/robot_helper.py).

    python projects/scene_understanding/run.py 20261005_124116
    python projects/scene_understanding/run.py 20261005_124116 --from build     # redo from step 2

  1 detect    detect_frames.py   word list (VLM, ~130 views) + YOLOE masks   vLLM gateway + Slurm job, ~4 min
                                 in every keyframe (laptop CPU if the cluster fails)
  2 build     build_objects.py   masks -> 3D, merged over views              laptop, ~3-9 min
  3 floor     floor_items.py     small things on the floor, from depth      laptop, ~30 s
  4 pairs     judge_pairs.py     VLM: close pairs one object or two? merge  laptop -> vLLM gateway, ~1-5 min
  5 objects   label_objects.py   VLM names each object -> scene.json        laptop -> vLLM gateway, ~2 min
  6 summary   scene_summary.py   scene.json -> text for an LLM prompt
  7 viewer    export_viewer.py   data for the console's Labels view

7-15 min per run (2026-10-09). Until then step 1 was Qwen boxes in every keyframe + SAM 2 (40-50 min).

check_objects.py (top view + gallery images) is not part of it any more: the console's Labels view
shows the same; run it by hand if needed.

Progress goes to <session>/scene/progress.json (progress.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from landerpi.session import Session  # noqa: E402
import progress  # noqa: E402

STEPS = [("detect", "detect_frames.py", "word list and YOLOE"),
         ("build", "build_objects.py", "objects in 3D"),
         ("floor", "floor_items.py", "things on the floor"),
         ("pairs", "judge_pairs.py", "VLM: one object or two?"),
         ("objects", "label_objects.py", "VLM: naming the objects"),
         ("summary", "scene_summary.py", "summary"),
         ("viewer", "export_viewer.py", "data for the 3D view")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--from", dest="start", default="detect", choices=[s for s, _, _ in STEPS])
    a = ap.parse_args()
    scene_dir = Session(a.session).path / "scene"
    scene_dir.mkdir(exist_ok=True)
    (scene_dir / "progress.json").unlink(missing_ok=True)  # from an earlier run
    names = [s for s, _, _ in STEPS]
    todo = STEPS[names.index(a.start):]
    for n, (name, script, stage) in enumerate(todo, 1):
        print(f"=== {name}: {script}", flush=True)
        progress.update(scene_dir, step=n, steps=len(todo), stage=stage, done=0, total=0, detail=None)
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / script), a.session], check=True)


if __name__ == "__main__":
    main()
