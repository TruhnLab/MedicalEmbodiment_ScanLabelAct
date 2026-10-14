"""Progress of a labelling run in <session>/scene/progress.json, for the console's 3D view (the helper
passes it on, like reconstruct.py's progress.json): step, steps, stage, and done / total within a step."""

import json
import os
import time


def update(scene_dir, **fields):
    f = scene_dir / "progress.json"
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        p = {}
    p.update(fields, updated=time.time())
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(p), encoding="utf-8")
    os.replace(tmp, f)  # the helper never reads a half-written file
