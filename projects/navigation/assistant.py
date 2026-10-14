"""The task console's chat: a language model on the vLLM gateway set in .env (projects/scene_understanding/vlm.py's
settings, but its own key: VLLM_CHAT_API_KEY in the repo's .env; never another model server, never a model
of VLLM_BLOCKED_MODELS) that answers and acts through tools (landerpi/tools.py). Runs in the navigator
(navigator.py: POST /chat).

    a = Assistant(services)                    # services for the tools ({"navigator": ..., "call": ...})
    new = a.send("drive to the cabinet near the sink", armed=True, stopped=lambda: False)
    # -> [{"role": "user"|"assistant"|"tool", "text": ...}, ...]: this turn's messages for the console

With every message the model gets the room's object list (scene/summary.md, written by scene_understanding
from the same scene.json the console shows, so the [ids] match) and the robot's status. It calls tools until
it has an answer (at most MAX_ROUNDS). The conversation is kept for the robot (all console tabs share it)
and cleared when the room changes (ids are only valid within one room's list).

Tools: TOOL_MODULES are imported and register themselves; other projects add their module there.
"""

import base64
import importlib
import json
import re
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_understanding"))
import vlm  # noqa: E402  the gateway's settings (URL, model, key from .env); stopgap import, see README
import room_view  # noqa: E402
from landerpi import tools  # noqa: E402

def chat_key():
    """The chat has its own key on the gateway (VLLM_CHAT_API_KEY in the repo's .env; user, 2026-10-07);
    VLLM_API_KEY stays the labelling's. No fallback to it: a missing key is an error the console shows."""
    return vlm.setting("VLLM_CHAT_API_KEY")


GERMAN = {"und", "der", "die", "das", "ist", "nicht", "ich", "wo", "bitte", "noch", "mal", "zum", "zur", "liegt",
           "fahre", "fahr", "zu", "dem", "den", "ein", "eine", "mir", "kannst", "du", "hier", "nach", "suche", "finde",
           "wie", "viele", "gibt", "es", "im", "sind", "was", "welche", "welcher", "mach", "dann", "ja", "nein", "auf",
           "neben", "vor", "hinter", "zwischen", "steht", "stehen", "sollte", "wieder", "schau", "dreh", "geh", "gehe"}


def language_of(text):
    """English or German (the users' languages); anything else: the model decides."""
    words = set(re.findall(r"[a-zäöüß]+", text.lower()))
    if re.search(r"[äöüß]", text.lower()) or len(words & GERMAN) >= 2:
        return "German"
    if re.fullmatch(r"[\x00-\x7f]*", text):
        return "English"
    return "the language of the user's message"


TOOL_MODULES = ["nav_tools", "manipulation_tools"]  # projects/manipulation (its folder on sys.path: navigator.py)
NUDGE = ("[system] You announced an action but did not call a tool, so nothing happened. If the user's request "
         "needs an action, call the tool now. If no action is needed, answer exactly NO_ACTION.")  # projects/navigation/nav_tools.py; other projects add theirs
MAX_ROUNDS = 10    # find + search + approach + answer take 4-6 rounds; 6 cut one short (2026-10-09)
MAP_PICTURE = True    # the room's top view at the start of each request (room_view.py; see test_chat_room.py)
HISTORY = 30          # messages kept (user, assistant, tool calls and results)
TIMEOUT_S = 60

SYSTEM = """You are the assistant of LanderPi, a small mobile robot (mecanum base, 5-servo arm with a camera and \
gripper) in a room that was scanned before. You talk to the person operating it in the task console and act \
through the tools.

Rules:
- Use the tools to act; don't describe what you would do. Refer to objects by their [id] in the object list.
- With watch_for and search_here always give 2-3 synonyms (other English words for the same thing, e.g. cup: \
mug, coffee cup): the camera's detector reacts to the exact word. Also give the description: everything the \
user said about how the item looks (colour, size, shape, what it is like), never where it is (the image check \
sees only a crop: "between the radiator and the cabinet" made it answer "unsure" to a red cube). The detector finds things by kind and hardly \
by colour; an image check judges each candidate against the description before it counts as found, and \
rejected candidates are reported (e.g. "REJECTED by the image check: a red cube"). A rejected sighting is not \
the item: don't approach it unless the user says so. A new search re-checks earlier candidates of other searches \
against its description (earlier_candidates_being_checked): one that fits becomes the find at once.
- If the user says the robot should not move, use only tools that don't move it (watch_for, list_seen, \
get_robot_status): "look for a chair but don't move" is watch_for, not search_here.
- Do exactly what the user asked, then stop and answer. Don't add steps on your own: after a find, report it \
and ask; don't approach, confirm or search further unless asked. If you ask the user something, wait for \
the answer (no tool calls after the question).
- If a request fits several objects (e.g. "the cabinet" with three cabinets), ask which one and name the \
candidates with what tells them apart (position, what they are next to). Don't guess.
- If something is not in the list, say so (the list is from the scan: things may have moved, small things and \
table tops may be missing).
- After acting, say briefly what happened (one or two sentences). If a tool reports an error, explain it simply \
and say what the user can do.
- Always answer in the language of the user's last message (usually English or German), also in the short \
notes before a tool call; never switch to another language because of the tool results. Be brief.

How the robot works (tell the user when it matters):
- Before driving the robot must be localized in the chosen room: the user switches on Armed and presses ⟳ Localize; the robot turns once on the spot and finds its position. Driving does not localize it.
- Anything that moves the robot needs Armed (top right). STOP or the space bar stops it at once.
- It stops about 20 cm in front of an object, facing it. The object list is from a scan and can be off.
- Finding things that are not in the list (or moved), e.g. "I dropped a pencil between the radiator and the \
chair": ALWAYS call watch_for(item) first, before any drive (the camera only looks while watching, and things \
seen on the way are remembered), with the described place (relation and near_object_ids) if the user gave one, \
so that only a find there stops the drive; then drive near the described place \
(the drive stops by itself if the item is seen on the way), then, if not seen yet, search_here with the \
objects that describe the area. Once found, say where; approach_found only if the user wants the robot there. \
When search_here found nothing in the area but a match outside it (also_seen_outside_the_area, \
"searched": "a quick look ... because a match was seen 0.3 m outside it"), say both: what was searched in the area, \
and where the match outside is (the camera shows it now); ask whether that is the one, and stop there: no approach_found before the user answers (the user decides whether a match outside the described place is it). \
Never drive_to_point onto a found item's position: the robot would drive over it. Finds are already checked \
against the description (image_check); confirm_found only if the user doubts a find. A sighting away from the place the user described is not the one \
meant: mention it, but don't approach it unless the user says so; keep looking at the described place.
- watch_for returns seen_before: earlier sightings of the item in this conversation. If one is in the \
described place (in_this_area), tell the user and approach_found it if asked to go there, instead of searching.
- Each message may bring [memory of this conversation]: what the camera saw while looking for things, and \
searches without a find. It is a hint, not a fact: things get moved, and detections can be wrong. Sure \
sightings (#n at (x, y)) can be approached with approach_found; if the item is not there anymore the result \
says so. Vague ones (only a direction, maybe a rough place) cannot be approached: watch_for the item and drive \
near the rough place or in that direction to look. Say how sure a memory is when you use it. A search without a find means it is probably not there: look elsewhere first, unless the user says to look there again.
- Something beside or behind the robot ("behind you", "on your left"): watch_for it, then turn towards it \
(turn_by_degrees left / right / around); don't drive to a point for it: the camera has to see it first. The \
user's direction relative to the robot (left, right, in front, behind, close to you) is what counts: don't work \
out the place from the object list instead ("very close to the left in front of you" was taken as near a desk \
on the right, and the robot turned right). "Left in front" is only a small turn left (the camera sees ~35 deg \
to each side). For things close to the robot use relation "here" (watch_for without objects, search_here \
"here"), not objects nearby.
- When approach_found arrives, the approach is done, also if it stopped farther away because other things lie \
around the item: say so, don't search or approach again.
- Picking things up (manipulation_tools): find it (watch_for / search_here), approach_found (stops ~20 cm in \
front), then pick_up_object (drives the last centimetres itself); to bring it somewhere drive_to_object / \
drive_to_point, then put_down_object. Only items lying on the floor, 15-45 mm wide, about block height, roughly \
square to the robot; putting down needs ~10 cm of free floor in front of the robot. The robot holds one item at a \
time (robot status: holding).
- The camera looks only while a search is open: after approach_found the robot takes a last look and stops \
watching; for a new search call watch_for again.
- Everything the camera confirmed while looking (also outside the search area) is numbered (seen_id; results \
list those outside as also_seen_outside_the_area; list_seen shows all). Mention such sightings; if the user \
says one of them is the one meant, use its seen_id with approach_found / look_at.
- After drive_to_object and approach_found the camera looks at the target by itself (look_at does it on \
request); the next drive puts it back into the survey pose.
- Tool results give robot_now: where the robot is after acting (searching moves it). Say where it is from that.
- Places that are not an object ("the wall behind you", "the corner by the door", "the middle of the room"): \
work out a point from the walls, the objects and the robot's walls_around, and use drive_to_point. Keep at least \
0.4 m from walls and objects. A top view of the room comes at the start of the conversation (walls W1.., \
objects by [id], a 1 m grid in map coordinates).

Each user message starts with the robot's status at that moment ([robot status now: ...]); it is always more recent than anything said before.

The room's objects (positions in metres in the room's map; the robot's pose uses the same frame):
{room}

The room's sides (the scan's outline, as in the top view) and the no-go lines:
{walls}"""


class Assistant:
    def __init__(self, services):
        for m in TOOL_MODULES:
            importlib.import_module(m)
        self.services = services
        self.history = []     # OpenAI messages without the system message
        self.log = []         # for the console: {"role", "text", "time"}
        self.room = None
        self.lock = threading.Lock()
        assert not vlm.blocked(vlm.MODEL)

    def reset(self, why=""):
        self.history, self.log = [], []
        if why:
            self.log.append({"role": "note", "text": why, "time": time.time()})

    def system_message(self):
        nav = self.services["navigator"]
        room = nav.room
        if room is None:
            room_text = "(no room loaded: the user has to choose one in the task console)"
        else:
            summary = room.session_path / "scene" / "summary.md"
            scene = room.session_path / "scene" / "scene.json"
            if summary.exists() and scene.exists() and summary.stat().st_mtime >= scene.stat().st_mtime - 60:
                room_text = summary.read_text(encoding="utf-8")
            else:  # no summary for this labelling: a plain list from scene.json
                room_text = "\n".join(f"- [{o['id']}] {o['label']} at ({o['center'][0]:.1f}, {o['center'][1]:.1f})"
                                      for o in room.objects())
        return SYSTEM.format(room=room_text, walls=room_view.walls_text(room) if room is not None else "(no room)")

    def picture_message(self):
        """The room's top view, once per conversation: right after the system message, so the gateway's prefix
        cache keeps it (the robot is not drawn: its pose changes, it comes as text with every message)."""
        room = self.services["navigator"].room
        if not MAP_PICTURE or room is None:
            return []
        url = "data:image/png;base64," + base64.b64encode(room_view.room_picture(room)).decode()
        return [{"role": "user", "content": [
            {"type": "text", "text": "[top view of the room: walls W1.., objects by [id], 1 m grid in map "
                                     "coordinates; the robot's position comes as text with each message]"},
            {"type": "image_url", "image_url": {"url": url}}]}]

    def ask_model(self, messages):
        body = {"model": vlm.MODEL, "temperature": 0, "max_tokens": 800,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": messages, "tools": tools.schemas(), "tool_choice": "auto"}
        for attempt in range(3):
            try:
                r = requests.post(vlm.URL, json=body, headers={"Authorization": "Bearer " + chat_key()},
                                  verify=False, timeout=TIMEOUT_S)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))  # the gateway rate-limits (429) when other projects use it

    def send(self, text, armed=False, stopped=lambda: False):
        """One user message; returns this turn's messages for the console."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("the assistant is still working on the last message")
        try:
            nav = self.services["navigator"]
            room = nav.room.name if nav.room else None
            if room != self.room:
                if self.room is not None:
                    self.reset(f"New room ({room}): the conversation starts over.")
                self.room = room
            start = len(self.log)
            if getattr(nav, "live", None) is not None:
                # a new message ends the last job's detection (its boxes stayed live until now); the new request
                # starts its own if it looks for something
                nav.live.stop()
                nav.live.clear_overlay()
            # the robot's status goes with every message: in the system message only, the model answered "not
            # localized" from an earlier turn although it was localized by then (2026-10-07 test)
            status = tools.run("get_robot_status", {}, tools.Context(services=self.services))
            # ... and the language to answer in, decided here: left to the model it answered English messages in
            # German after a German turn, or all in German when asked "the language of the message", and drifted
            # into Chinese with long tool results (Qwen, 2026-10-08)
            live = getattr(nav, "live", None)
            memory = live.watcher.memory() if live is not None else ""
            self.history.append({"role": "user", "content": f"[robot status now: {json.dumps(status)}]\n"
                                 + (f"[memory of this conversation: {memory}]\n" if memory else "")
                                 + f"[answer in {language_of(text)}]\n" + text})
            self.log.append({"role": "user", "text": text, "time": time.time()})
            nav.note("user", f"chat: {text}")
            ctx = tools.Context(armed=armed, stopped=stopped, services=self.services)
            acted = nudged = False
            for _ in range(MAX_ROUNDS):
                try:
                    msg = self.ask_model([{"role": "system", "content": self.system_message()}]
                                         + self.picture_message() + self.history)
                except Exception as e:
                    self.log.append({"role": "assistant", "text": f"(The language model is not reachable: {e})",
                                     "time": time.time()})
                    break
                calls = msg.get("tool_calls") or []
                self.history.append({k: v for k, v in (("role", "assistant"), ("content", msg.get("content") or ""),
                                                        ("tool_calls", calls or None)) if v is not None})
                text_out = (msg.get("content") or "").strip()
                if nudged and "NO_ACTION" in text_out:
                    self.history.pop()  # nothing to do after all: the answer before stands
                    break
                if text_out:
                    self.log.append({"role": "assistant", "text": text_out, "time": time.time()})
                    nav.note("assistant", f"assistant: {text_out[:200]}")
                if not calls:
                    # The model sometimes only announces an action ("Ich fahre jetzt zu den Schubladen") without
                    # calling a tool, and nothing happens (real run 2026-10-08, twice in a row). Once per message:
                    # remind it, unless it asked the user something.
                    if not acted and not nudged and not text_out.endswith("?"):
                        nudged = True
                        self.history.append({"role": "user", "content": NUDGE})
                        continue
                    break
                acted = True
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    self.log.append({"role": "tool", "text": f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})",
                                     "time": time.time()})
                    result = tools.run(name, args, ctx)
                    self.log[-1]["result"] = result
                    nav.note("tool", f"chat tool {name} {args} -> {json.dumps(result)[:300]}")
                    self.history.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(result)})
            else:
                self.log.append({"role": "assistant", "text": "(Stopped after too many steps.)", "time": time.time()})
            if getattr(nav, "live", None) is not None:
                nav.live.end_of_turn()  # found what was looked for: the detection stops (no job, no compute)
            del self.history[:-HISTORY]
            while self.history and self.history[0]["role"] == "tool":
                del self.history[0]  # a tool result without its call confuses the model
            return self.log[start:]
        finally:
            self.lock.release()
