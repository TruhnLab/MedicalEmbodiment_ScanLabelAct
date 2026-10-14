"""Offline checks of perception.Watcher (following boxes, image checks, finds, memory), with the situations of the
real runs of 2026-10-09: touching blocks whose 3D places wobble, a pile, a block swapped for another, the robot
turning, a box lost and seen again, re-judging earlier sightings for a new search. No robot, no gateway: the
image check answers by a "truth" carried along with each box.

    python projects/navigation/test_watcher.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception import Watcher  # noqa: E402

rng = np.random.default_rng(0)
failed = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failed.append(name)


def box(x0, y0, w=30, h=28):
    return [x0, y0, x0 + w, y0 + h]


def sighting(label, b, xy, truth, score=0.5, wobble=0.04, z=0.02):
    return {"label": label, "score": score, "box": list(b), "quality": "sure", "size": [0.04, 0.04, 0.03],
            "position": [xy[0] + rng.uniform(-wobble, wobble), xy[1] + rng.uniform(-wobble, wobble), z],
            "crop": truth.encode()}


def run(w, frames, answer, t0=0.0, dt=0.2, labels=None):
    """frames: [(sightings, shift_px)]; answer(crop) -> (fits, what) for each check. Returns the box texts."""
    texts = []
    for k, (sights, shift) in enumerate(frames):
        w.add(sights, t0 + k * dt, labels, shift)
        for e in w.take_to_check():
            w.set_verdict(e, *answer(e["crop"].decode()))
        texts.append([(s["crop"].decode(), (s.get("_entry") or {}).get("verdict_what")) for s in sights])
    return texts


def by_colour(want):
    return lambda truth: (truth == want, f"{truth} block")


# 1. touching red and blue blocks, camera still, 3D places wobbling +-4 cm (real run 16:43)
w = Watcher()
w.verify = True
w.start(["blue block"])
frames = [([sighting("blue block", box(300, 200), (0.90, 0.20), "red"),
            sighting("blue block", box(334, 200), (0.90, 0.25), "blue")], 0.0) for _ in range(40)]
texts = run(w, frames, by_colour("blue"))
red = {t[0][1] for t in texts[2:]}
blue = {t[1][1] for t in texts[2:]}
check("touching blocks keep their own labels", red == {"red block"} and blue == {"blue block"}, f"red box {red}, blue box {blue}")
check("the blue one is the find", w.found.get("blue block", {}).get("verdict_what") == "blue block")
check("two sightings, two checks", len(w.seen) == 2, f"{len(w.seen)} sightings")

# 2. a pile of four touching blocks: each its own sighting and crop
w = Watcher()
w.verify = True
w.start(["red block"])
pile = [("red", box(300, 200), (1.00, 1.00)), ("green", box(330, 200), (1.045, 1.0)),
        ("blue", box(300, 228), (1.0, 1.045)), ("yellow", box(330, 228), (1.045, 1.045))]
frames = [([sighting("red block", b, xy, c, score=0.3 + 0.1 * (c == "red"), wobble=0.02) for c, b, xy in pile], 0.0)
          for _ in range(8)]
run(w, frames, by_colour("red"))
crops = sorted(e["crop"].decode() for e in w.seen)
check("pile: four sightings, each with its own crop", crops == ["blue", "green", "red", "yellow"], str(crops))
check("pile: the red one is the find", w.found.get("red block", {}).get("crop") == b"red")

# 3. the blue block swapped for a red one (out of view for 2 s while the hand is there; the robot stands)
w = Watcher()
w.verify = True
w.stood_still = lambda a, b: True
w.start(["blue block"])
before = [([sighting("blue block", box(334, 200), (0.90, 0.25), "blue")], 0.0) for _ in range(10)]
run(w, before, by_colour("blue"))
first = w.found.get("blue block")
after = [([sighting("blue block", box(334, 200), (0.90, 0.25), "red")], 0.0) for _ in range(10)]
texts = run(w, after, by_colour("blue"), t0=4.0)
check("swap: the new thing gets its own check", texts[-1][0][1] == "red block", str(texts[-1]))
check("swap: the old find is no find any more", "blue block" not in w.found and first is not None)

# 4. the robot turns 10 px per image: the boxes move, followed through the shift
w = Watcher()
w.verify = True
w.start(["cup"])
frames = [([sighting("cup", box(300 + 25 * k, 150), (2.0, 1.0), "cup")], 25.0) for k in range(10)]
run(w, frames, by_colour("cup"))
check("turning: one sighting while the box moves 25 px per image", len(w.seen) == 1, f"{len(w.seen)} sightings")
frames = [([sighting("cup", box(300 + 25 * k, 150), (2.0, 1.0), "cup")], 0.0) for k in range(10)]
w2 = Watcher()
w2.verify = True
w2.start(["cup"])
run(w2, frames, by_colour("cup"))
check("(without the turn's shift it is never followed: no sighting)", len(w2.seen) != 1, f"{len(w2.seen)} sightings")

# 5. lost and found again while the robot drives: the same sighting, no new check (real run 2026-10-12: the
#    list grew by dozens per thing, the fire extinguisher was checked 15 times)
w = Watcher()
w.verify = True
w.stood_still = lambda a, b: False
w.start(["cup"])
asked = []


def count(truth):
    asked.append(truth)
    return True, "a cup"


for k in range(6):  # seen, lost for 1.5 s, seen again (boxes elsewhere in the picture: the robot drove)
    run(w, [([sighting("cup", box(100 + 80 * k, 150), (2.0, 1.0), "cup", wobble=0.02)], 0.0)] * 3, count, t0=k * 2.0)
check("lost and found while driving: one sighting, one check", len(w.seen) == 1 and len(asked) == 1,
      f"{len(w.seen)} sightings, {len(asked)} checks")
w.stood_still = lambda a, b: True
run(w, [([sighting("cup", box(100, 150), (2.0, 1.0), "cup", wobble=0.02)], 0.0)] * 3, count, t0=20.0)
check("lost and found while standing still: checked again (something happened)", len(w.seen) == 1 and len(asked) == 2,
      f"{len(w.seen)} sightings, {len(asked)} checks")
w2 = Watcher()
w2.verify = True
w2.start(["block"])
run(w2, [([sighting("block", box(300, 200), (1.0, 1.0), "red", wobble=0.01),
           sighting("block", box(334, 200), (1.0, 1.05), "blue", wobble=0.01)], 0.0)] * 3, by_colour("red"))
run(w2, [([sighting("block", box(300, 200), (1.0, 1.0), "red", wobble=0.01)], 0.0)] * 3, by_colour("red"), t0=5.0)
check("touching blocks: no re-link (two within reach), a new sighting", len(w2.seen) == 3, f"{len(w2.seen)} sightings")

# 6. the image check: tiny boxes wait; a "does not fit" from far is checked again when twice as large
w = Watcher()
w.verify = True
w.start(["cup"])
asked = []


def unsure_then_yes(truth):
    asked.append(truth)
    return (False, "a blur") if len(asked) == 1 else (True, "a cup")


small = [([sighting("cup", box(300, 150, 10, 10), (2.5, 1.0), "cup")], 0.0)] * 3
run(w, small, unsure_then_yes)
check("a 10 px box is not sent", not asked and w.seen[0].get("verdict") is None)
grow = [([sighting("cup", box(300, 150, 16 + 2 * k, 16 + 2 * k), (2.5, 1.0), "cup")], 0.0) for k in range(14)]
run(w, grow, unsure_then_yes, t0=0.6)
check("sent at 18 px ('a blur': no), sent again at twice the size, then found", len(asked) == 2 and "cup" in w.found,
      f"{len(asked)} checks")

# 6b. approaching: a "fits" from far is looked at once more when the box is twice as large; still: never
w = Watcher()
w.verify = True
w.start(["cup"])
asked = []


def far_yes_near_no(truth):
    asked.append(truth)
    return (True, "a cup") if len(asked) == 1 else (False, "a bowl")


still = [([sighting("cup", box(300, 150, 20, 20), (2.0, 1.0), "cup")], 0.0)] * 20
run(w, still, far_yes_near_no)
check("still picture: one check only", len(asked) == 1 and "cup" in w.found, f"{len(asked)} checks")
closer = [([sighting("cup", box(300, 150, 20 + 3 * k, 20 + 3 * k), (2.0, 1.0), "cup")], 0.0) for k in range(10)]
run(w, closer, far_yes_near_no, t0=4.0)
check("approaching: checked again at twice the size; 'bowl' is no find", len(asked) == 2 and "cup" not in w.found,
      f"{len(asked)} checks, found {list(w.found)}")

# 7. a new search re-judges earlier sightings: the green search's rejected blue block is the blue search's find
w = Watcher()
w.verify = True
w.start(["green block"])
run(w, [([sighting("green block", box(300, 200), (0.7, 0.0), "blue")], 0.0)] * 3, by_colour("green"))
w.stop()
w.start(["blue block"])
adopted = w.adopt_earlier()
for e in w.take_to_check():
    w.set_verdict(e, *by_colour("blue")(e["crop"].decode()))
check("new search: the earlier blue block is found at once", len(adopted) == 1 and "blue block" in w.found)

# 8. the simulation (no image check): a box followed twice is the find
w = Watcher()
w.start(["cup"])
run(w, [([sighting("cup", box(300, 150), (2.0, 1.0), "cup")], 0.0)] * 2, by_colour("cup"))
check("without a check: found after two images", "cup" in w.found)

print("all ok" if not failed else f"{len(failed)} failed: {failed}")
sys.exit(1 if failed else 0)
