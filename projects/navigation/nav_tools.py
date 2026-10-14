"""The chat's navigation tools (landerpi/tools.py rules): driving to an object or a point, the survey pose,
the robot's status. They run in the navigator (navigator.py) and use it through ctx.services:
"navigator" (the Navigator) and "call" (runs a navigator method on its event loop and returns the result).
"""

import base64
import math
import time

import cv2
import numpy as np

from landerpi.robot import SURVEY_POSE
from landerpi.tools import ToolError, tool
import room_view
from perception import in_area, search_area

DRIVE_TIMEOUT_S = 180.0
SURVEY_TOLERANCE = 25      # servo pulses (0-1000): closer than this to the survey pose counts as there
SURVEY_MOVE_S = 1.5        # the console moves the arm to a pose in 1.5 s (app.js armGoTo)
# camera tilted down for the last close look at the floor: the survey pose's tilt (servo 4) is 265 and looks
# slightly up; 100 looks at the floor right in front (app.js ARM_HOME). In between: to be checked on the robot
LOOK_DOWN_POSE = {**SURVEY_POSE, 4: 170}
VIEW_M = (0.6, 1.4)        # stops for a search: this far from the search area's edge (the camera sees the floor
                           # from ~0.7 m in the survey pose, explore_core.py; small things need closeness)
MAX_STOPS = 4
QUICK_LOOK_M, CLOSER_LOOK_M = 0.5, 1.5  # a match this close outside the area: 1 / 2 search stops (_effort)
# aiming the camera (look_at): Hiwonder bus servos turn 240 deg over their 0-1000 range (nominal; _aim checks
# the result on /tf and corrects); the tilt stays within TILT_RANGE (100 looks at the floor right in front)
TILT_SERVO = 4
RAD_PER_PULSE = math.radians(240) / 1000
TILT_RANGE = (90, 340)
AIM_STEPS = 4
AIM_PITCH_TOL = math.radians(3)
AIM_YAW_TOL = math.radians(5)
AIM_MOVE_S = 0.8
LOOK_AROUND_STEPS = 4      # search_here "here": a full turn on the spot in this many steps first
SWEEP_DEG = 30             # left and right of the area at each stop (the camera sees 74 deg)


def _nav(ctx):
    return ctx.services["navigator"], ctx.services["call"]


def _where(nav):
    """The robot's position after a tool acted: the model claimed "the robot stayed where it is" after a search
    had moved it (simulation, 2026-10-08)."""
    p = nav.current_pose()
    return None if p is None else {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
                                   "heading_deg": round(math.degrees(p[2]))}


def _ready(nav):
    if nav.goto is None:
        raise ToolError("no room is loaded: the user has to choose the room in the task console")
    if not nav.connected:
        raise ToolError("the robot is not connected: the user has to press Connect")
    if nav.status != "localized":
        raise ToolError(f"the robot is not localized ({nav.status}): the user has to press ⟳ Localize (one turn "
                        "on the spot) first")


def _survey(ctx):
    """Arm into the survey pose unless it is there: the camera looks ahead, the depth camera's obstacle
    detection is calibrated for it (config/camera_correction.json)."""
    nav, call = _nav(ctx)
    pos = nav.arm_positions()
    if pos and all(abs(pos.get(i, -999) - p) <= SURVEY_TOLERANCE for i, p in SURVEY_POSE.items()):
        return False
    call(nav.arm_to, SURVEY_POSE, SURVEY_MOVE_S)
    time.sleep(SURVEY_MOVE_S + 0.3)
    return True


def _drive(ctx, start, survey=True, new_finds=True, look=False):
    """Starts a drive or turn (start: a navigator call) and waits until it ends; STOP in the console ends it.
    survey: arm into the survey pose first (driving: the depth camera's obstacles need it; turning on the spot
    doesn't)."""
    nav, call = _nav(ctx)
    _ready(nav)
    moved = _survey(ctx) if survey else False
    found_before = {k: e.get("id") for k, e in nav.live.watcher.found.items()}
    call(*start)
    t0 = time.time()
    g = nav.goto
    while g.state in ("planning", "driving", "docking", "turning"):
        if ctx.stopped() or time.time() - t0 > DRIVE_TIMEOUT_S:
            call(nav.stop, "stopped during this request" if ctx.stopped() else f"no end after {DRIVE_TIMEOUT_S:.0f} s")
            break
        time.sleep(0.2)
    if g.state == "arrived" and new_finds and nav.live.watcher.active:
        if look:  # a turn to look: if a box of the item is in front but not yet a sighting, the next image or two
            t_end = time.time() + LOOK_AFTER_TURN_S
            while time.time() < t_end and _forming(nav):
                time.sleep(0.1)
        _await_checks(nav)  # a candidate seen on the way may still be confirmed (the robot does not stop for it)
        w = nav.live.watcher
        if any(p in w.found and w.found[p].get("id") != found_before.get(p) for p in w.prompts):
            g.state, g.message = "found", "a find was confirmed by the image check as the robot arrived"
    result = {"result": g.state, "message": g.message, "seconds": round(time.time() - t0, 1),
              "robot_now": _where(nav)}
    if g.state == "found":  # live detection saw what is looked for (watch_for / search_here): not an error
        result["found"] = nav.live.state()["found"]
    elsewhere = [e for p in nav.live.watcher.prompts for e in _elsewhere(nav, p)]
    if elsewhere:  # seen on the way, outside the search area: maybe the one the user means
        result["also_seen_outside_the_area"] = elsewhere
    if moved:
        result["arm"] = "moved to the survey pose first"
    if g.state == "failed":
        raise ToolError(g.message)
    if g.state == "stopped":
        raise ToolError(g.message)  # names the reason (navigator.stop)
    return result


@tool("drive_to_object",
      "Drive the robot to an object of the room (by its [id] in the object list) and stop about 20 cm in front "
      "of it, facing it. Waits until the robot is there (typically 10-30 s). Needs a localized robot. Puts the "
      "arm into the survey pose first if needed.",
      {"object_id": {"type": "integer", "description": "the object's [id] in the room's object list"}},
      moves_robot=True)
def drive_to_object(ctx, object_id):
    nav, _ = _nav(ctx)
    objs = {o["id"]: o for o in nav.room.objects()} if nav.room else {}
    if int(object_id) not in objs:
        raise ToolError(f"there is no object [{object_id}] in this room")
    r = _drive(ctx, (nav.start_goto, int(object_id)))
    return _aim_quietly(ctx, _target_point(nav, object_id=int(object_id)), r) if r["result"] == "arrived" else r


@tool("drive_to_point",
      "Drive the robot to a point on the floor of the room (map coordinates in metres, like the positions in the "
      "object list) and stop there, facing the way it came. For places that are not an object, e.g. 'between "
      "the door and the cabinet'. Waits until the robot is there. Needs a localized robot.",
      {"x": {"type": "number", "description": "map x in metres"}, "y": {"type": "number", "description": "map y in metres"}},
      moves_robot=True)
def drive_to_point(ctx, x, y):
    nav, _ = _nav(ctx)
    return _drive(ctx, (nav.start_goto, None, (float(x), float(y))))


@tool("turn_to_object",
      "Turn the robot on the spot until it faces an object of the room (by its [id]), e.g. after driving somewhere, "
      "to look at something across the room. It does not move away from where it stands. Needs a localized robot.",
      {"object_id": {"type": "integer", "description": "the object's [id] in the room's object list"}},
      moves_robot=True)
def turn_to_object(ctx, object_id):
    nav, _ = _nav(ctx)
    objs = {o["id"]: o for o in nav.room.objects()} if nav.room else {}
    if int(object_id) not in objs:
        raise ToolError(f"there is no object [{object_id}] in this room")
    return _drive(ctx, (nav.start_turn, int(object_id)), survey=False, look=True)


# A direction word and a positive angle, not a signed angle: told "the block is on your left", the model called
# turn_by_degrees(-90) and the robot turned right (real run 2026-10-09 16:09; the description said positive = left)
@tool("turn_by_degrees",
      "Turn the robot on the spot, to its own left or right (as seen from the robot), or around. Needs a localized "
      "robot.",
      {"direction": {"type": "string", "enum": ["left", "right", "around"],
                     "description": "the robot's left or right; around = 180 degrees"},
       "degrees": {"type": "number", "description": "how far, 0-180 (e.g. 90 to face what is beside the robot); "
                                                     "not needed for around"}},
      required=["direction"], moves_robot=True)
def turn_by_degrees(ctx, direction, degrees=90):
    nav, _ = _nav(ctx)
    d = 180.0 if direction == "around" else abs(float(degrees))
    if d > 180:
        raise ToolError("the angle must be between 0 and 180 degrees")
    if direction == "right":
        d = -d
    elif direction not in ("left", "around"):
        raise ToolError("direction must be left, right or around")
    return _drive(ctx, (nav.start_turn, None, d), survey=False, look=True)


@tool("arm_to_survey_pose",
      "Move the arm into the survey pose (camera straight ahead, as for driving). Driving does it by itself; use "
      "this when the user asks for it.",
      moves_robot=True)
def arm_to_survey_pose(ctx):
    nav, _ = _nav(ctx)
    if not nav.connected:
        raise ToolError("the robot is not connected: the user has to press Connect")
    return {"result": "moved to the survey pose" if _survey(ctx) else "the arm was already in the survey pose"}


@tool("get_robot_status",
      "Where the robot is (map x, y in metres, heading in degrees), whether it is localized, and the last drive.")
def get_robot_status(ctx):
    nav, _ = _nav(ctx)
    pose = nav.current_pose()
    d = nav.drive_state()
    hint = {"idle": "not localized yet: Armed, then ⟳ Localize (one turn on the spot)",
            "ambiguous": "not sure where it is: wrong room? otherwise drive a bit and ⟳ Localize again",
            "uncertain": "the scans stopped fitting the room: ⟳ Localize again",
            "failed": "localizing failed: see the console's message"}.get(nav.status)
    return {"connected": nav.connected, "room": nav.room.name if nav.room else None, "localization": nav.status,
            **({"what_to_do": hint} if hint else {}),
            "pose": None if pose is None else {"x": round(float(pose[0]), 2), "y": round(float(pose[1]), 2),
                                               "heading_deg": round(math.degrees(pose[2]))},
            # which wall is ahead / behind / left / right ("the wall behind you"; room_view.py)
            "walls_around": None if pose is None or nav.room is None else room_view.around(nav.room, pose),
            "holding": (getattr(nav, "held", None) or {}).get("label"),
            "last_drive": None if d is None else {"target": d["target"]["label"], "state": d["state"],
                                                  "message": d["message"]}}


# ---------- finding things (live detection, live.py) ----------

def _found(nav, item):
    f = nav.live.watcher.found.get(item)
    return None if f is None else {"item": item, "seen_id": f.get("id"), "x": round(f["position"][0], 2),
                                   "y": round(f["position"][1], 2), "score": round(f["score"], 2), "views": f["views"],
                                   "image_check": {"yes": f"fits: {f.get('verdict_what')}",
                                                   "unchecked": "not checked"}.get(f.get("verdict"), "not checked")}


def _elsewhere(nav, item):
    """Matches of the item outside the search area, nearest to it first: they don't stop the robot, but may be
    the one meant (the user decides). Not those the image check rejected, nor those seen gone."""
    w = nav.live.watcher
    ok = ("yes", "unchecked") if w.verify else ("yes", "unchecked", None)
    out = []
    for e in w.seen:
        if e["label"] == item and not e["in_area"] and e.get("quality", "sure") == "sure" \
                and not e.get("missing_since") and e.get("verdict") in ok and e.get("position") is not None:
            d = 0.0 if w.area is None else -cv2.pointPolygonTest(
                np.asarray(w.area, np.float32).reshape(-1, 1, 2), (float(e["position"][0]), float(e["position"][1])), True)
            out.append({"seen_id": e["id"], "x": round(e["position"][0], 2), "y": round(e["position"][1], 2),
                        "distance_to_area_m": round(max(0.0, d), 2), **_verdict(e)})
    return sorted(out, key=lambda r: r["distance_to_area_m"])


def _effort(nav, item):
    """How many stops the search in the area gets: a match just outside it makes the right one there less likely
    and the user can decide sooner (user, 2026-10-12): 1 stop if one lies within QUICK_LOOK_M of the area, 2
    within CLOSER_LOOK_M, else all. Returns (stops, the nearest match outside or None)."""
    out = _elsewhere(nav, item)
    if not out:
        return MAX_STOPS, None
    d = out[0]["distance_to_area_m"]
    return (1 if d <= QUICK_LOOK_M else 2 if d <= CLOSER_LOOK_M else MAX_STOPS), out[0]


def _sighting(nav, item=None, seen_id=None):
    e = nav.live.watcher.lookup(item, seen_id)
    if e is None:
        raise ToolError(f"no sighting #{seen_id}" if seen_id is not None else
                        f"{item} has not been seen: search for it first")
    return e


SYNONYMS_PARAM = {"type": "array", "items": {"type": "string"},
                  "description": "2-3 other English words for the same thing (the detector reacts to the exact word), "
                                 "e.g. for cup: mug, coffee cup"}


DESCRIPTION_PARAM = {"type": "string",
                     "description": "what the item looks like, with every attribute the user gave (colour, size, "
                                    "shape, material, what it is like), e.g. 'small green cube, like a dice'; NOT where "
                                    "it is (an image check judges each candidate's picture against it, and cannot see "
                                    "'next to the outlet'; the place goes into relation / near_object_ids)"}
# After turning to look (watching for something): the green block stood in front when "turn around" ended, its
# check came 3 s later, and the chat had already turned on by 45 deg (real run 2026-10-09 17:19). The turn waits
# only while a box of the item is about to become a sighting (at most this long), then for its check; with
# nothing in sight it returns at once (user: no pause for nothing)
LOOK_AFTER_TURN_S = 1.0
CHECK_WAIT_S = 4.0  # a drive or search that ends while candidates are being checked waits this long for the verdict


def _forming(nav):
    """A box of what is looked for in the latest images that is not yet a sighting (seen in one image so far)."""
    w = nav.live.watcher
    return any(t["entry"] is None and t["label"] in w.prompts and time.time() - t["t"] < 0.5 for t in list(w.tracks))


def _await_checks(nav, item=None):
    """Wait (briefly) for running image checks of what is looked for: a find may still come."""
    t0 = time.time()
    while time.time() - t0 < CHECK_WAIT_S and any(
            e.get("verdict") == "pending" and (item is None or e["label"] == item)
            and e["label"] in nav.live.watcher.prompts
            for e in nav.live.watcher.seen):
        time.sleep(0.2)


def _synonyms(item, synonyms):
    return {item: [w.strip().lower() for w in (synonyms or []) if w and w.strip()][:4]}


@tool("watch_for",
      "Start looking for an item with the camera while the robot moves (open vocabulary: e.g. pencil, red cup, "
      "keys). Any following drive or turn stops as soon as the item is seen in the given place, and reports where; "
      "sightings elsewhere are only reported. Use it before driving to where the user says the item roughly is. "
      "If the user says where it is, give that place (relation + near_object_ids). Does not move the robot.",
      {"item": {"type": "string", "description": "what to look for, in a few English words (e.g. pencil)"},
       "synonyms": SYNONYMS_PARAM,
       "description": DESCRIPTION_PARAM,
       "relation": {"type": "string", "enum": ["between", "near", "in_front_of"],
                    "description": "where the user says it is, relative to near_object_ids (leave out if unknown)"},
       "near_object_ids": {"type": "array", "items": {"type": "integer"},
                           "description": "[ids] of the objects that describe the place"}},
      required=["item"])
def watch_for(ctx, item, synonyms=None, relation=None, near_object_ids=(), description=None):
    nav, _ = _nav(ctx)
    _ready(nav)
    item = item.strip().lower()
    area = None
    if relation and near_object_ids:
        # without the place any cup in the room stopped the drive: one on a cabinet across the room, while the
        # user had said "left of the outlet" (real run 2026-10-09)
        try:
            area = search_area(nav.room.objects(), list(near_object_ids), relation, robot_xy=nav.current_pose()[:2])
        except ValueError as e:
            raise ToolError(str(e))
    nav.live.start([item], area, synonyms=_synonyms(item, synonyms), description=description)
    return {"result": f"looking for {item}" + (f" {relation.replace('_', ' ')} {list(near_object_ids)}" if area is not None
                                                else " anywhere"),
            "camera": "simulated" if nav.live.sim is not None else "live video", **_seen_before(nav, item),
            **({"earlier_candidates_being_checked": nav.live.adopted} if nav.live.adopted else {})}


def _verdict(e):
    """The image check's result for the chat (a rejected sighting is not the item; real run 2026-10-09: the chat
    approached a candidate rejected as "white electrical outlet", the list did not say so)."""
    v = e.get("verdict")
    if v is None or v == "unchecked":
        return {}
    return {"image_check": {"yes": f"fits: {e.get('verdict_what')}", "no": f"REJECTED: {e.get('verdict_what')}",
                            "pending": "running"}.get(v, v)}


def _seen_before(nav, item):
    """Earlier sightings of the item (a search for it before, in this conversation): one in the new place can
    be approached at once (approach_found with its seen_id)."""
    e = [{"seen_id": s["id"], "x": round(s["position"][0], 2), "y": round(s["position"][1], 2),
          "in_this_area": s["in_area"], "minutes_ago": round((time.time() - s["time"]) / 60, 1),
          **({"not_there_when_checked": True} if s.get("missing_since") else {}), **_verdict(s)}
         for s in nav.live.watcher.seen if s["label"] == item and s.get("quality", "sure") == "sure"]
    return {"seen_before": e} if e else {}


@tool("search_here",
      "Search for an item in an area and stop as soon as it is seen. The area is given by objects of the room "
      "(between two objects, near or in_front_of one) or is around the robot (here). The robot looks at the area "
      "from up to 4 stops around it, turning the camera across it, and finally looks down. Drive near the area "
      "first if it is far away (drive_to_point / drive_to_object).",
      {"item": {"type": "string", "description": "what to look for, in a few English words"},
       "synonyms": SYNONYMS_PARAM,
       "description": DESCRIPTION_PARAM,
       "near_object_ids": {"type": "array", "items": {"type": "integer"},
                           "description": "[ids] of the objects that describe the area (empty for here)"},
       "relation": {"type": "string", "enum": ["between", "near", "in_front_of", "here"]}},
      required=["item", "relation"], moves_robot=True)
def search_here(ctx, item, relation, near_object_ids=(), synonyms=None, description=None):
    nav, call = _nav(ctx)
    _ready(nav)
    item = item.strip().lower()
    pose = nav.current_pose()
    try:
        area = search_area(nav.room.objects(), list(near_object_ids or []), relation, robot_xy=pose[:2])
    except ValueError as e:
        raise ToolError(str(e))
    syn = _synonyms(item, synonyms)
    if not synonyms and nav.live.watcher.prompts == [item]:
        syn = {item: [w for w, i in nav.live.alias.items() if i == item and w != item]}  # keep watch_for's words
    nav.live.start([item], area, synonyms=syn, description=description)
    t0 = time.time()
    if _found(nav, item):  # confirmed before, in this area: no need to drive
        return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, 0))
    if relation == "here":
        # around the robot: first a look around on the spot. The stops outside the area and "facing its middle"
        # are for areas at objects; for "here" the middle is the robot itself: a turn in a random direction, then a
        # drive 1.8 m away to the first stop (real run 2026-10-12 14:05)
        for _ in range(LOOK_AROUND_STEPS):
            r = _drive(ctx, (nav.start_turn, None, 360.0 / LOOK_AROUND_STEPS), survey=False, look=True)
            if r["result"] == "found" or _found(nav, item):
                return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, 1))
    stops = _search_stops(nav, area, pose)
    centre = np.asarray(area).mean(0)
    looked = 0
    for i, (x, y) in enumerate(stops):
        limit, outside = _effort(nav, item)  # again before every stop: a match outside may come up meanwhile
        if i >= limit:
            break
        if i > 0 or np.hypot(x - pose[0], y - pose[1]) > 0.3:
            try:
                r = _drive(ctx, (nav.start_goto, None, (float(x), float(y))))
            except ToolError as e:
                _await_checks(nav, item)
                if _found(nav, item):  # seen on the way, confirmed just as the drive gave up
                    return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, looked))
                if "no way" in str(e):  # an unreachable stop: on to the next one (it ended the search before)
                    continue
                raise
            if r["result"] == "found":
                return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, looked))
        here = nav.current_pose()
        face = math.degrees(math.atan2(centre[1] - here[1], centre[0] - here[0]) - here[2])
        for deg in (face, SWEEP_DEG, -2 * SWEEP_DEG):  # face the area, then sweep across it
            deg = (deg + 180) % 360 - 180
            if abs(deg) > 3:
                r = _drive(ctx, (nav.start_turn, None, deg), survey=False)
                if r["result"] == "found":
                    return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, looked))
            time.sleep(0.6)  # a few images at this heading (~5 per second)
            if _found(nav, item):
                return _searched(nav, item, relation, near_object_ids, _report(nav, item, t0, looked))
        looked += 1
    limit, outside = _effort(nav, item)
    if outside is None or limit == MAX_STOPS:
        # a last look down at the floor from where the robot stands, then back to the survey pose for driving
        call(nav.arm_to, LOOK_DOWN_POSE, SURVEY_MOVE_S)
        time.sleep(SURVEY_MOVE_S + 1.5)
        _survey(ctx)
    r = _report(nav, item, t0, looked)
    if r["result"] != "found" and outside is not None:
        where = f"{outside['distance_to_area_m']:.1f} m outside it (#{outside['seen_id']})"
        if limit < MAX_STOPS:
            how = "a quick look (1 stop)" if limit == 1 else f"a shorter look ({looked} stops)"
            r["searched"] = f"{how} in the area, because a match was seen {where}"
        else:
            r["searched"] = f"the whole area ({looked} stops); a match was also seen {where}"
        r = _aim_quietly(ctx, _target_point(nav, seen_id=outside["seen_id"]), r)  # the camera shows the user that one
        if str(r.get("camera", "")).startswith(("looking at it", "as close as")):
            r["camera_shows"] = f"#{outside['seen_id']} outside the area"
    return _searched(nav, item, relation, near_object_ids, r)


def _report(nav, item, t0, looked):
    _await_checks(nav, item)
    f = _found(nav, item)
    if f is None:
        r = {"result": "not found", "looked_from_stops": looked, "seconds": round(time.time() - t0),
             "robot_now": _where(nav), "hint": "ask the user whether to search a wider area or elsewhere"}
    else:
        r = {"result": "found", **f, "seconds": round(time.time() - t0), "robot_now": _where(nav)}
    elsewhere = _elsewhere(nav, item)
    if elsewhere:
        r["also_seen_outside_the_area"] = elsewhere
    return r


def _search_stops(nav, area, pose):
    """Up to MAX_STOPS free spots VIEW_M from the area's edge, spread around it (different sides), the one
    nearest to the robot first. The robot's own spot counts if it is already that close."""
    room = nav.room
    grid, res, origin = room.grid, room.resolution, np.array(room.origin)
    free = (grid == 0).astype(np.uint8)
    clear = cv2.distanceTransform(free, cv2.DIST_L2, 3) * res
    rows, cols = np.nonzero(clear > 0.30)  # the robot fits (half diagonal 0.20 m + margin)
    xy = (np.stack([cols, rows], 1) * res + origin + res / 2)[::3]
    poly = np.asarray(area, np.float32).reshape(-1, 1, 2)
    d = np.array([-cv2.pointPolygonTest(poly, (float(px), float(py)), True) for px, py in xy])
    cand = xy[(d >= VIEW_M[0]) & (d <= VIEW_M[1])]
    stops = []
    if in_area(pose[:2], area, margin=VIEW_M[1]):
        stops.append((float(pose[0]), float(pose[1])))
    if len(cand):
        centre = np.asarray(area).mean(0)
        ang = np.arctan2(cand[:, 1] - centre[1], cand[:, 0] - centre[0])
        order = np.argsort(np.hypot(cand[:, 0] - pose[0], cand[:, 1] - pose[1]))
        used = [math.atan2(s[1] - centre[1], s[0] - centre[0]) for s in stops]
        for i in order:
            if len(stops) >= MAX_STOPS:
                break
            if all(abs(math.atan2(math.sin(ang[i] - a), math.cos(ang[i] - a))) > math.radians(70) for a in used):
                stops.append((float(cand[i, 0]), float(cand[i, 1])))
                used.append(float(ang[i]))
    return stops


@tool("approach_found",
      "Drive to an item the camera found (watch_for / search_here) or to a sighting by its number (seen_id, also "
      "those outside the search area, see list_seen) and stop about 20 cm in front of it, facing it.",
      {"item": {"type": "string", "description": "the item as it was looked for"},
       "seen_id": {"type": "integer", "description": "or: the sighting's number"}},
      required=[], moves_robot=True)
def approach_found(ctx, item=None, seen_id=None):
    nav, _ = _nav(ctx)
    f = _sighting(nav, item, seen_id)
    if f.get("quality") == "vague":
        raise ToolError(f"#{f['id']} is a vague sighting (only a direction"
                        + (f", roughly at ({f['about'][0]:.1f}, {f['about'][1]:.1f})" if f.get("about") else "")
                        + "): watch_for the item and drive near that place to look, then approach what is found")
    size = [max(0.05, float(v)) for v in (f.get("size") or [0.1, 0.1, 0.1])]
    obj = {"id": None, "label": f["label"], "center": f["position"], "size": size, "yaw_deg": 0.0}
    # its own target confirmed on the way is no new find: the approach arrives, aims and looks (real run
    # 2026-10-09: reported as a find, the final look was skipped)
    r = _drive(ctx, (nav.start_goto_target, obj), new_finds=False)
    if f.get("verdict") == "no":
        r["note"] = f"#{f['id']} had been rejected by the image check: {f.get('verdict_what')}"
    if r["result"] == "arrived" and "blocked" in str(r.get("message", "")):
        # other things lie around it (seen blocks are obstacles): this is as close as it gets; the chat took
        # it for a failure and searched again until it ran out of steps (real run 2026-10-09 16:00)
        r["message"] = (str(r["message"]).split(" (")[0] + ": other things lie around it, so this is as close as "
                        "the robot gets without touching them; the approach is done")
    if r["result"] != "arrived":
        return r
    r = _aim_quietly(ctx, _target_point(nav, seen_id=f["id"]), r)
    return _final_look(nav, f, r)


def _final_look(nav, f, result):
    """The job is done: a last look shows the item in the camera picture (frozen, with its mask), then the
    detection stops (user, 2026-10-09: it kept detecting while the user drove around afterwards). A memory is a
    hint: if the camera looks at the place and the item is not there, the sighting is marked (it may have moved)."""
    seen = nav.live.final_look(f["label"], f["position"], size=f.get("size"))
    result["camera_sees_it"] = seen
    aimed = str(result.get("camera", "")).startswith(("looking at it", "as close as"))
    if not seen and aimed and nav.live.sim is None:
        nav.live.watcher.mark_missing(f)
        result["note"] = f"the {f['label']} was not seen at its remembered place: it may have been moved"
    return result


def _searched(nav, item, relation, near_object_ids, result):
    """Remember the search also without a find (the chat's memory: "searched near [22]: nothing")."""
    where = relation.replace("_", " ") + (f" {list(near_object_ids)}" if near_object_ids else "")
    found = result.get("seen_id") if result.get("result") == "found" else None
    nav.live.watcher.searched(item.strip().lower(), where, f"found #{found}" if found else "nothing found")
    return result


@tool("list_seen",
      "Everything the camera confirmed while looking for things (inside and outside the search areas), numbered: "
      "for 'go back to the cup you saw before'. Positions in the room's map.")
def list_seen(ctx):
    nav, _ = _nav(ctx)
    now = time.time()
    out = []
    for e in nav.live.watcher.seen:
        r = {"seen_id": e["id"], "item": e["label"], "quality": e.get("quality", "sure"), "score": round(e["score"], 2),
             "minutes_ago": round((now - e["time"]) / 60, 1)}
        if e.get("quality") == "vague":
            r.update({"seen_from": e["from"], "direction_deg": e["bearing_deg"],
                      "roughly_at": None if e.get("about") is None else e["about"][:2]})
        else:
            r.update({"x": round(e["position"][0], 2), "y": round(e["position"][1], 2), "in_search_area": e["in_area"],
                      **({"not_there_when_checked": True} if e.get("missing_since") else {})})
        r.update(_verdict(e))
        out.append(r)
    return {"seen": out, "searches": [{"item": x["item"], "where": x["where"], "result": x["result"],
                                       "minutes_ago": round((now - x["time"]) / 60, 1)} for x in nav.live.watcher.searches]}


@tool("confirm_found",
      "Ask the vision-language model whether the camera's best image of a found item (or a sighting by number) really "
      "shows it (slow: a few seconds). Use it only when it matters, e.g. before something acts on the item, or if "
      "the user doubts it.",
      {"item": {"type": "string", "description": "the item as it was looked for"},
       "seen_id": {"type": "integer", "description": "or: the sighting's number"}}, required=[])
def confirm_found(ctx, item=None, seen_id=None):
    nav, _ = _nav(ctx)
    f = _sighting(nav, item, seen_id)
    # the item asked about, not the label of the search that saw it (real run 2026-10-09: "is this a small green
    # block?" about the blue one: "not confirmed: a small blue block")
    item = (item or "").strip().lower() or f["label"]
    item = nav.live.descriptions.get(item, item)
    if not f.get("crop"):
        raise ToolError("there is no camera image of it (simulated robot)")
    import requests
    import vlm
    from assistant import chat_key
    question = (f"Does this image show a {item}? Answer as JSON: "
                '{"is_it": true or false, "what_it_shows": "a few words"}')
    body = {"model": vlm.MODEL, "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f["crop"]).decode()}}]}]}
    r = requests.post(vlm.URL, json=body, headers={"Authorization": "Bearer " + chat_key()}, verify=False, timeout=60)
    r.raise_for_status()
    answer = vlm.parse_json(r.json()["choices"][0]["message"]["content"] or "{}")
    return {"result": "confirmed" if answer.get("is_it") else "not confirmed", "what_it_shows": answer.get("what_it_shows")}


# ---------- looking at a target: camera tilt (servo 4) and the base's heading ----------

def _camera_errors(nav, target):
    """Angles from the colour camera's axis to a map point (from /tf: the camera sits on the arm): (down, right) in
    radians, positive = the target is below / right of the image centre. None while the pose is unknown."""
    pose = nav.current_pose()
    B = nav.depth.camera_in_base("rgb_camera_link")
    if pose is None or B is None:
        return None
    from depth_view import planar
    T = planar(pose) @ B
    p = (np.asarray(target, float) - T[:3, 3]) @ T[:3, :3]  # map -> optical frame (x right, y down, z forward)
    if p[2] <= 0.05:
        return None
    return math.atan2(p[1], p[2]), math.atan2(p[0], p[2])


def _aim(ctx, target):
    """Turn the base and tilt the camera until the camera's axis points at target (map x, y, z), user 2026-10-08:
    in front of a cup the camera looked over it. Closed loop on /tf: the servo's scale is only nominal."""
    nav, call = _nav(ctx)
    sign = 1.0
    err = _camera_errors(nav, target)
    if err is None:
        raise ToolError("the camera's position is unknown (no /tf from the arm)")
    for _ in range(AIM_STEPS):
        down, right = err
        if abs(right) > AIM_YAW_TOL:
            _drive(ctx, (nav.start_turn, None, -math.degrees(right)), survey=False)
            err = _camera_errors(nav, target) or err
            down, right = err
        if abs(down) < AIM_PITCH_TOL:
            break
        pos = nav.arm_positions().get(TILT_SERVO)
        if pos is None:
            raise ToolError("the arm's servo positions are unknown")
        # lower tilt values look further down (survey 265 looks slightly up, the start pose's 100 at the floor)
        new = int(min(TILT_RANGE[1], max(TILT_RANGE[0], pos - sign * down / RAD_PER_PULSE)))
        if new == int(pos):
            break  # at the end of the range
        call(nav.arm_to, {TILT_SERVO: new}, AIM_MOVE_S)
        time.sleep(AIM_MOVE_S + 0.6)  # the servo and the /tf of its joint
        after = _camera_errors(nav, target)
        if after is None:
            break
        if abs(after[0]) > abs(down):
            sign = -sign  # the servo turns the other way than assumed: correct it next step
        err = after
    down, right = err
    return {"camera": "looking at it" if abs(down) < 2 * AIM_PITCH_TOL else "as close as the tilt range allows",
            "off_centre_deg": {"down": round(math.degrees(down), 1), "right": round(math.degrees(right), 1)}}


def _aim_quietly(ctx, target, result):
    """After an approach: look at the target; a failure to aim doesn't make the approach fail."""
    try:
        result.update(_aim(ctx, target))
    except ToolError as e:
        result["camera"] = f"could not aim the camera: {e}"
    return result


@tool("look_at",
      "Point the camera at an object of the room (object_id) or at a found item (item): turns the robot on the spot "
      "and tilts the camera to the object's middle. Driving tools do it by themselves at the end; driving again puts "
      "the camera back into the survey pose.",
      {"object_id": {"type": "integer", "description": "an object's [id] in the room list"},
       "item": {"type": "string", "description": "or: a found item as it was looked for"},
       "seen_id": {"type": "integer", "description": "or: a sighting's number (list_seen)"}},
      required=[], moves_robot=True)
def look_at(ctx, object_id=None, item=None, seen_id=None):
    nav, _ = _nav(ctx)
    _ready(nav)
    r = _aim(ctx, _target_point(nav, object_id, item, seen_id))
    if item or seen_id is not None:
        r = _final_look(nav, _sighting(nav, item, seen_id), r)
    return r


def _target_point(nav, object_id=None, item=None, seen_id=None):
    if item or seen_id is not None:
        f = _sighting(nav, item, seen_id)
        if f.get("position") is None:
            raise ToolError(f"#{f['id']} is only a direction, without a place: turn towards it with turn_by_degrees")
        x, y, z = f["position"]
        return [x, y, z + nav.depth.floor.z]  # found heights are above the floor; the map's floor is at floor.z
    objs = {o["id"]: o for o in nav.room.objects()} if nav.room else {}
    if object_id is None or int(object_id) not in objs:
        raise ToolError(f"there is no object [{object_id}] in this room")
    o = objs[int(object_id)]
    z0, z1 = o.get("z_range", (0.0, o["size"][2]))
    return [o["center"][0], o["center"][1], (z0 + z1) / 2 + nav.depth.floor.z]
