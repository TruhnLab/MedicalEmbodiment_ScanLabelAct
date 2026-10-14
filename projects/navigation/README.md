# Navigation in a known room

Goal: put the robot into a room that was mapped, reconstructed and labelled before
(`projects/mapping`, `projects/scene_understanding`), pick the room in a console, and the robot finds
out by itself where it is. Then click a labelled object, or say what to do in a chat, and the robot drives
there; things that are not in the scan (a cup somebody put down) it finds with its camera.
In the spirit of [HomeBody](https://tml.stanford.edu/homebody/): the scanned room is the robot's
map of the world, and skills (navigate, find, pick, put down) take their targets from it.
Picking and putting down are `projects/manipulation`'s, as tools in this project's chat.

## Milestones

| | What | Status |
|---|---|---|
| T1 | Rooms from labelled sessions; localization tested offline on other runs of the same room | ✅ |
| T2 | Task console: room dropdown, labelled 3D view + live robot; automatic localization | ✅ on the robot |
| T3 | Click an object or a spot → drive there, stop 0.2 m in front | ✅ on the robot |
| T4 | Chat: text prompt → the model acts through tools (gateway with an OpenAI-compatible API) | ✅ on the robot |
| T5 | Finding things that are not in the scan: live detection, image check, search, approach | ✅ on the robot (small blocks, cups) |
| – | Picking up and putting down | `projects/manipulation` (its tools run in the chat), early |

Weak spots: driving between many small things (approach paths, docking) needs another look; localization
can be lost when the robot is lifted and put down elsewhere (⟳ Localize again).

## Task console (T2)

Double-click `start_task_console.bat` (opens http://127.0.0.1:8765/tasks.html with the same helper as
the mapping console; a running helper is reused unless its code changed).

The top bar, camera, drive strip, lidar and arm cards and the ⚙ drawer are the mapping console's own
(`tasks_boot.js` takes them from `index.html` and runs the same `app.js`): same **Connect**, **Armed**,
**STOP**, keys (W A S D, Q E, arrows, R F, H home, P survey, Space) and settings. **M** (or a click on the
small camera) swaps the room and the camera.

1. Choose the room the robot stands in (room toolbar; the bin removes a room from the list). The page
   shows its mesh with the labels (as in the mapping console's 3D view).
2. **Connect** (starts the navigator; **Disconnect** stops it), **Armed**, **⟳ Localize**: one slow turn on
   the spot (~15 s), then the search. The robot appears in red with its lidar scan in light blue.
   STOP / Space / disarming / a driving key stops the turn. Driving to objects is locked until
   "Localized". (Localizing standing still was dropped, 2026-10-07: the turn sees the whole room, so a
   confidently wrong result is much less likely.)
3. "Localization not sure": the wrong room, or a room whose parts look alike: drive a bit, localize again.
4. From then on every scan is tracked; "Localization uncertain" if the scans stop fitting the room.
5. Click an object (3D, tag or list): its details and **Go to …** (needs Connect, Armed and "Localized";
   otherwise the note says what is missing). The robot drives there (green dashed path, green stop) and
   stops 0.2 m in front of it, facing it. The button is **Cancel** meanwhile; STOP / Space / disarming /
   a driving key stop it too.
   **📍 Pin & go** instead: click a spot on the floor (orange pin), **Go to the pin**; the robot arrives
   facing the way it came.
   Every Go starts without remembered depth obstacles (a blocked way that is clear again is found on the
   next try); during a drive they are also forgotten when the camera sees their place empty (3 images).
6. Or type into the **Assistant** card (see Chat below). While it works, a box under the message shows its
   notes, tool calls and what the robot does; it folds to "n steps" when the answer is there.
7. **● Record** / **Runs**: see Recording runs below.

## Driving to an object (T3, `goto.py`)

- **Where to stop:** in front of one face of the object's box, facing it, GAP = 0.2 m from the robot's
  front; of all faces and spots along them the one with the shortest path. Elongated objects (cabinets,
  drawers, wall fixtures) are approached from a long side. Spots where the robot doesn't fit are skipped.
  If no side is reachable (a stool among stools, a shoe under a table), the robot goes as close as it
  fits and says how far it got.
- **The box gives the direction, the sensors the distance** (boxes from the scan can be 10-20 cm off):
  at the stop, if the lidar or depth camera sees something straight ahead further than 0.2 m (up to
  0.8 m), the robot creeps on until it is 0.2 m away; closer than planned, it stops earlier.
- **Path and safety: the explorer's** (`offboard_ros/explore_core.py`, imported unchanged): Dijkstra
  keeping 26 cm (preferably 55 cm) from obstacles, holonomic pure pursuit, stop when something is close
  in the travel direction (distance grows with speed), never driving towards the lidar's blind rear.
  Obstacles: the room map, the no-go lines, the live lidar scan, and the depth camera's obstacles 10-45 cm
  above the floor (`depth_view.py`, the explorer's `depth_obstacles.py` rules), re-planned every second.
- **Low things** (2026-10-09: the robot nearly drove into a 3 cm block): within 1 m, a 5 cm cell whose top
  stands 2 cm above the floor right around it is an obstacle too (a local step, so a camera tilted by 1.5 deg
  doesn't turn the floor into one; synthetic 3 cm cube found at 0.4-0.8 m, room recording: no false cells).
  Things the detector placed on the floor are obstacles as well, except the drive's own target. What the
  gripper holds is neither.
- **A find is approached from the robot's side:** its only stop faces the robot; if blocked (other things
  lying around it), the robot goes as close as it fits instead of driving round to another side.
- **The arm:** poses are reached from above (12 pulses past the pose first): the servos stop short by their
  deadband, from below the camera was up to 7 deg off from what /tf says (measured by `projects/manipulation`).
- **Speed:** up to 0.4 m/s (slower near obstacles and the goal), turning 0.8 rad/s; facing the travel
  direction while driving (the lidar sees ahead), turning towards the object over the last metre.
- **Stops** when the scans or odometry are older than 0.5 s or localization is no longer "localized".
- **Remembered obstacles never box the robot in:** within 0.3 m of the robot live and remembered points are
  ignored for planning (the collision check still sees them). A real run (2026-10-07 14:51) ended close to a
  cabinet front; the depth obstacles remembered there for good blocked every later plan until a restart.
  Since then every new goal starts without remembered obstacles, and a plan that finds the robot boxed in
  says so.

**Tried and dropped (2026-10-07): correcting the stop from depth points near the box.** On the room's own
recording (keyframes 0.6-1.8 m from each object, box shifted 15 cm on purpose), the depth points within
30 cm of the box mix the object with its neighbours, the wall and door frames: sideways the corrected stop
was still 13 cm off (median), with correct boxes it moved stops at a door and a wall fixture by 30 cm, and
along the approach the visible surfaces differ from the boxes by a median of 14 cm anyway (cabinet boxes
reach 20-30 cm beyond their fronts). Docking's sensor rule covers the distance; for the side the object's
own pixels are needed: a detector (label → image box) or the VLM, then its depth points.

Simulation (`test_goto.py`, room's map as the world, simulated lidar, no depth camera), 2026-10-07:

| Room | Drives | Arrived | Time (median) | Gap the sensors see at the end | Closest to anything |
|---|---|---|---|---|---|
| 20261007_101108 (40 objects x 2 starts) | 80 | 80 | 14 s | 0.17-0.21 m | 0.32 m |
| ... with every box shifted 15 cm | 80 | 80 | 15 s | median 0.21 m (one 0.49) | 0.29 m |
| 20261007_093027 (43 x 2) | 86 | 85 | 8 s | median 0.23 m | 0.25 m |

The one failure: an office chair (no progress, 0.7 m from it). With the console against a simulated robot
(`fake_robot.py --sim <room>`): localized, drove to a door (0.21 m in front of it), Space stopped a drive.

Behind it: `navigator.py` (started through the helper, `logs/navigator_*.log`) talks to the robot's
rosbridge directly (scan + odometry in, `/controller/cmd_vel` out only while turning) and serves its state
on 127.0.0.1:8767. Don't run SLAM / an exploration at the same time. The navigator's room is shared by
all open task consoles (one robot, one room): choosing a room in one tab switches the others.

Without the robot: `python projects/navigation/fake_robot.py 20261006_161813 [--start 60]` replays a
recorded run as the robot's rosbridge on 127.0.0.1:9090, then start the helper with
`$env:LANDERPI_HOST='127.0.0.1'`. The replay drives as it was recorded; the turn button only "turns"
while the recording does. Replayed in room `20261006_154055` (2026-10-07): standing still localized at
3 starting points (scans fit 0.86-0.98), the turn at 2 (0.93-0.97); tracking fit 0.9-0.99 per scan.
A jump in the replay (it loops) made it "Uncertain", as it should.

## Chat (T4, `assistant.py`, `nav_tools.py`)

The **Assistant** card in the task console: "drive to the cabinet near the sink", "now to the drawer on the
right end of the long cabinet", "there is a red cube left of the outlet, pick it up", in any language. A
language model on an OpenAI-compatible gateway (vLLM; `VLLM_URL`, `VLLM_MODEL` and the chat's own key
`VLLM_CHAT_API_KEY` in `.env`, see `.env.example`) gets the room's object list (`scene/summary.md`), the room's
walls (below) and, with every message, the robot's status and the conversation's memory of what it saw; it acts
through tools and says what happened. If several objects fit it asks which one. One conversation per robot
(all tabs), started over when the room changes. No thinking mode: ~1-3 s per step.

Tools (`landerpi/tools.py`: the registry and the rules all projects follow):
- driving: `drive_to_object`, `drive_to_point`, `turn_to_object`, `turn_by_degrees` (left / right / around and
  an angle: a signed angle got the side wrong), `arm_to_survey_pose`, `get_robot_status`;
- finding: `watch_for`, `search_here`, `approach_found`, `list_seen`, `look_at`, `confirm_found` (below);
- `projects/manipulation`: `pick_up_object`, `put_down_object` (`manipulation_tools.py`).

Tools that move the robot are refused unless the console is Armed; drives wait until they end; STOP / Space end
them. Other projects add their tools in their own module, listed in `assistant.TOOL_MODULES`. The user's
direction relative to the robot ("left in front of you", "behind you") counts, not a place worked out from the
object list.

Simulated robot (`fake_robot.py --sim`), 2026-10-07: "drive to the door with the small rectangular window"
→ door [5], arrived 0.21 m in front; "now go to the bar stool" → asked which of the nearest five; "the one
closest to the radiator" → stool [13]; "fahre zur Steckdose" → the power outlet, answer in German; not armed
or not localized → it says what to do. The model answers in ~1-3 s per step.

**The room for the model** (`room_view.py`): besides the object list, the room's sides (the scan's outline,
simplified; a side is a wall where the lidar map has obstacles along it, else "open side") with what is along
them and a point to stand in front of each, the no-go lines as such (not walls), a top view once per
conversation (`chat_map.png` in the room's folder), and with every message `walls_around`: the wall ahead,
behind, left and right with a point in front of it. `test_chat_room.py` checks place prompts against the
gateway without the robot (2026-10-09, room 20261007_101108, 9 prompts × 2): object list only 12/18; with the
walls as text 18/18; with the picture as well 18/18; 0.6 s per answer either way. Without the ready points the
model picked points as if the turned room's walls ran along x and y (6/9).

## Finding things (`landerpi/detector.py`, `perception.py`, `live.py`, `verifier.py`)

"I dropped a pencil between the radiator and the column": the chat calls `watch_for("pencil", relation="between",
near=[4, 28])` and drives there; the drive stops as soon as a pencil in that place is seen and confirmed. If
needed, `search_here(...)`: up to 4 stops around the area, a camera sweep at each, finally a look down;
"here" (around the robot) first turns on the spot in four steps. `approach_found` stops ~20 cm in front of
where it was seen, aims the camera at it and takes a last look. The room view shows the search area (orange)
and finds (yellow); the camera picture shows the detector's boxes.

The detector runs only while a search is open, and until the next chat message (its boxes stay live meanwhile).

- **Detector:** YOLOE-11s-seg (open vocabulary, text prompt, masks) on the laptop's CPU, ~5 images/s, models
  in `data/models/` (`test_detector.py`: 68 % of the room's labelled object views found; YOLO-World: 44 %).
  Up close a missed find gets a second pass on the picture shrunk to 0.6 (at most every 0.6 s): a mug 20 cm
  away scored 0.01-0.03 as is. Boxes on the robot's own gripper (bottom corners, "chair") are dropped.
- **Prompts:** the item plus 2-3 synonyms from the chat (cup: mug, coffee cup), all in one detector pass. The
  place goes into `relation` / `near_object_ids` (the search area); the description for the image check says
  how the item looks.
- **3D position:** the mask without its edge, its depth points, each camera at its own time
  (`test_positions.py` on recorded keyframes: median 0 cm, 84 % within 10 cm of the scanned box). Standing
  still, a depth image up to 2 s old still fits (depth lagged 0.4-0.8 s at times).
- **Sure or vague:** a position must rest on something the room's scan knows (`scan_cells.py`: the
  reconstruction as 5 cm cells, cached in the room's folder; cups had been placed 1.05 m up in the air). Without
  usable depth, something whose line of sight meets the floor within 2 m is placed there. Otherwise the
  sighting is **vague**: where it was seen from and the direction; vague sightings never stop a drive, are not
  on the map and cannot be approached; the chat goes and looks.
- **Following boxes:** each thing in the picture is followed by its box from image to image (the most
  overlapping box in the next image, moved by how far the robot turned meanwhile). Followed in 2 images, it is a
  numbered sighting with its 3D place (median of its latest views), gets the image check and keeps its label
  while followed. A newly followed box continues the one remembered sighting within 10 cm, if there is only one
  (with touching blocks there are several: a new sighting); it keeps its check result if the robot moved
  meanwhile, and is checked again if the robot stood still (something happened in the scene: a hand, a swap).
  Nothing is merged or deleted by place. `test_watcher.py`: touching blocks with 3D places wobbling 4 cm keep
  their labels; a pile of four gives four sightings with their own crops; a swapped block is checked anew; lost
  and found while driving stays one sighting with one check.
- **Image check** (`verifier.py`; a red, a blue and a green cube were all "small green block": YOLOE weighs the
  kind of thing, hardly its colour, and its label is only the prompt): the chat model on the gateway (its key,
  no conversation) judges each candidate's crop (the candidate marked by grey corners) against the description:
  fits or does not fit. Once per sighting; again when its box has grown to twice the size of its last check
  (approaching it; at most 3 checks), so a judgement from far away is corrected up close and a still picture is
  never checked again. Boxes under 18 px wait; candidates arriving together go in one request (~1-2 s). The
  robot does not stop for it. No answer from the gateway: the candidate counts ("not checked").
- **Found** = a sure sighting in the search area, confirmed by the image check. A turn while watching waits
  briefly if a box of the item is about to become a sighting or its check is running (and returns as soon as
  it answers). A new search re-judges earlier checked sightings of other searches (their crops) against its
  description: the green search's "rejected: a blue block" can be the blue search's find at once.
- **Outside the search area:** confirmed matches outside it are kept and numbered (`list_seen`, "the other
  cup") and never approached without the user. One just outside shortens the search inside: 1 stop if within
  0.5 m of the area, 2 within 1.5 m, else all; then the camera turns to it and the chat asks whether it is the
  one.
- **Seen empty:** a remembered sighting whose place is in view (0.35-1.2 m, where the detector reliably sees
  small things), not hidden (the depth camera sees nothing in front of it), but which is not seen there for
  1.5 s is marked "not there any more": off the map, no find. Only for what the detector looks for at the time.
- **What the gripper holds** (`projects/manipulation` sets `navigator.held` with its mask) is no find and no
  obstacle: boxes mostly on its mask are dropped, depth pixels that show it are blanked.
- **Memory of the conversation:** sightings (sure, vague, rejected, not there any more) and searches without a
  find come with every chat message as short text. The chat treats it as a hint: sure ones are approached (and
  checked there), vague ones looked for, searched places avoided. It lasts as long as the chat (cleared with
  it, and at a room change). The scanned objects are not part of it: they are always in the object list.
- **Camera picture:** live boxes only, labelled by their own sighting's check ("✓ red block", "✗ blue block",
  "checking…"); they follow the robot's turns (odometry). Every sighting of what is looked for goes to
  `logs/sightings_<day>.jsonl` (~0.4 KB each, 7 days, at most 50 MB a day).
- **Far objects** (`test_far.py`, room 20261007_101108, small objects by distance from the camera, 2026-10-08):

  | setting | 0.3-0.7 m | 0.7-1.0 m | 1.0-1.5 m | 1.5-2.5 m | ms/image |
  |---|---|---|---|---|---|
  | 11s, score 0.10 | 75 % | 17 % | 30 % | 40 % | 208 |
  | 11s, 0.05 + synonyms (used) | 75 % | 33 % | 33 % | 42 % | 172 |
  | 11m, 0.05 + synonyms | 75 % | 38 % | 40 % | 40 % | 586 |
  | 11l, 0.05 + synonyms | 75 % | 33 % | 37 % | 48 % | 608 |

  The larger models gain a few percent at a third of the speed (under 2 images/s): not worth it. Larger input
  (960/1280 px) was worse than 640. False alarms (prompts for things not in the room, 60 frames): any box in
  1.4 % of frames at 0.05, 0.6 % at 0.10.
- Simulation: `LANDERPI_SIM_OBJECTS='[{"label": "pencil", "x": -0.5, "y": 1.45}]'` for the navigator
  (or the helper) with `fake_robot.py --sim`: planted objects count as seen when in the camera's view.
- `LOOK_DOWN_POSE` (camera tilt 170, the last look of a full search) is still a guess between the survey pose's
  265 and the start pose's 100.

## Recording runs (`taskrun.py`)

**● Record** in the task console records until it is pressed again: the camera images with the detector's
boxes, drives, finds and the chat, to `data/task_runs/<time>/` (images, `frames.jsonl`, `events.jsonl`,
`meta.json`), then makes `video.mp4` (H.264, boxes and a status line drawn) and `thumb.jpg`. ~25 MB per minute
(images + video). **Runs** lists them: play, ☆ keep (protects from deleting), 🗑 delete. Nothing is recorded
otherwise, except `_last_search`: the last 120 s of each search, overwritten by the next (for checking what
the detector saw). Without the robot: `fake_video.py <session>` serves a recording's images as the video
(`$env:LANDERPI_HELPER_PORT='8790'` for the navigator). `projects/presentation_video/task_run_video.py` renders a
recorded run for talks (camera, top view with the track, the chat's steps).

## Rooms

In the mapping console's 3D view: choose a labelled session, **Use as room** (asks for the name shown in
the task console; **Update room** renews it, e.g. after a new reconstruction). In the task console, the bin
next to the room list removes one (the session and its labels stay). From a terminal:

```powershell
python projects/navigation/make_room.py 20261007_093027 [--title "Lab"]
```

A room is `data/rooms/<session>/`: `map.pgm` + `map.yaml` (the session's last SLAM map, ROS
map_server format) and `room.json` (session, title, the run's no-go lines). The mesh and labels are
read from the session, not copied (`landerpi/room.py`). The 2D map and the labels share one frame:
the reconstruction's floor levelling tilts by 0.0–0.2 deg only (all runs of 2026-10-06/07).

## Localization (`localize.py`)

Lidar scans against the room's 2D map. Numpy only, no ROS, so the same code runs live and offline.

1. **Global** (once, when the robot was put down; ⟳ Localize in the console): the robot turns once on the spot (the lidar's rear
   ~155 deg are filtered out of `/scan`); the scans are stitched with odometry and matched against the
   map everywhere the robot can stand (the scene's area polygon, 15 cm from walls): every 20 cm and
   3 deg against a blurred map, then the 40 best candidates refined to 1 cm / 0.25 deg. If a different
   pose scores almost as well (> 85 %), the result is flagged ambiguous.
2. **Tracking** (every scan): odometry predicts, a search ±15 cm / ±6 deg corrects. 4–5 ms per scan. Points
   within 45 cm in front (±70 deg) are left out: the arm reaches there (a pick's arm movement made the
   tracking lose the pose); the collision check still gets them.

Scoring is a likelihood field (each scan point counts by its distance to the nearest wall of the map),
so people and moved chairs just don't count instead of pulling the pose.

### Offline test

```powershell
python projects/navigation/test_localize.py 20261006_154055 20261006_161813   # room, another run of it
```

Each run has its own SLAM frame, so the test checks that all localizations of a run agree on one
room ← run transform, and writes `data/rooms/<room>/test_<run>.png` (the run's scans placed by
tracking on the room's map). Results 2026-10-07, room `20261006_154055`, global test every 30 s:

| Run | Global within 30 cm / 5 deg | Tracking fits the map (vs. the run's own SLAM pose) |
|---|---|---|
| 20261006_154055 (itself) | 10/10 | 0.96 (0.71) |
| 20261006_161813 | 12/12 | 0.95 (0.55) |
| 20261006_164000 | 10/11 (one 8 deg off, 5 cm) | 0.94 (0.81) |

The global search takes ~4 s. Its results are within ~15 cm of the reference; the reference itself
(the run's live SLAM pose, before slam_toolbox's later corrections) fits the room worse than the
tracked pose does, so tracking from the global result is what counts.

Without an area polygon (room `20261007_101108` before labelling: all 108 m2 of free cells, including
lidar rays through doors) the global search takes ~12 s; 19/21 tests within 30 cm / 5 deg, the
other two at 31 cm and 10 deg, i.e. near the truth, none in a wrong place.

Known limits: the search only covers the area polygon of the scan (the area the exploration
covered, within its no-go lines); a robot put down outside it is not found. Furniture that moved a lot
lowers the score but walls dominate.
