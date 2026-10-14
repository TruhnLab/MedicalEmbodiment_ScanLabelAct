"""Step 6 of scene understanding: scene.json as a short text an LLM / VLM gets in its prompt.

    python projects/scene_understanding/scene_summary.py 20261005_124116     # prints it, writes scene/summary.md

One line per object (id, label, position, size, state at scan time, what it stands next to). Ids are
the keys into scene.json, where the agent finds the keyframes and crops of an object.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402

NEAR = 0.6  # m between box edges: "next to"
GROUPS = [("Doors and passages", {"door"}), ("Furniture", {"furniture"}),
          ("Appliances and fixtures", {"appliance", "fixture"}), ("Containers", {"container"}),
          ("On the floor", {"floor_item"}), ("Other", {"other"})]


def gap(a, b):
    """Distance between two upright boxes' footprints, approximated by their enclosing circles."""
    ra = np.hypot(*a["size"][:2]) / 2
    rb = np.hypot(*b["size"][:2]) / 2
    return max(0.0, float(np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])) - ra - rb)


def describe(o, objs):
    x, y, _ = o["center"]
    w, l, h = o["size"]
    parts = [f"[{o['id']}] {o['label']} at ({x:.1f}, {y:.1f}), {max(w, l):.2f} x {min(w, l):.2f} m, {h:.2f} m high"]
    if o.get("part_of") is not None:
        whole = next((p for p in objs if p["id"] == o["part_of"]), None)
        if whole is not None:
            parts.append(f"part of {whole['label']} [{whole['id']}]")
    if o.get("state"):
        parts.append(f"{o['state']} when scanned")
    if o["movable"] == "movable" and o["category"] != "floor_item":
        parts.append("movable")
    if o.get("robot_fits_below"):
        parts.append(f"robot can drive under it ({o['clearance_below_m']:.2f} m free)")
    if o.get("graspable_size"):
        parts.append("small enough to pick up")
    near = sorted(((gap(o, p), p) for p in objs if p is not o), key=lambda t: t[0])
    near = [f"{p['label']} [{p['id']}]" for g, p in near[:3] if g < NEAR]
    if near:
        parts.append("next to " + ", ".join(near))
    if o.get("confidence") == "low":
        parts.append("uncertain")
    return "; ".join(parts) + f". {o['description']}"


def summary(scene):
    objs = scene["objects"]
    area = np.array(scene["area_polygon"])
    lo, hi = area.min(0), area.max(0)
    lines = [f"# Scene {scene['session']} (scanned {scene['created']})", "",
             f"Map frame in metres, floor at z = 0. Area of interest about x {lo[0]:.1f}..{hi[0]:.1f}, "
             f"y {lo[1]:.1f}..{hi[1]:.1f}. {len(objs)} objects. Positions are box centres.",
             "The robot's camera is 0.3 m above the floor: table tops and anything above ~1.5 m were not seen. "
             "Doors, chairs and other movable things may have changed since the scan.", ""]
    for title, cats in GROUPS:
        sel = [o for o in objs if o["category"] in cats]
        if sel:
            lines.append(f"## {title}")
            lines += ["- " + describe(o, objs) for o in sorted(sel, key=lambda o: (o["label"], o["id"]))]
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    text = summary(json.loads((s.path / "scene" / "scene.json").read_text()))
    (s.path / "scene" / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
