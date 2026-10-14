# Navigation in a known room

The robot is put into a room that was scanned and labelled before ([mapping](../mapping/README.md),
[scene understanding](../scene_understanding/README.md)). It finds out where it is with its lidar, drives to a
labelled object or a spot on the floor, and finds things that are not in the scan (a cup somebody put down) with
its camera. A chat with a language model takes tasks in plain language and acts through tools, among them
[manipulation](../manipulation/README.md)'s pick and put down. As in [HomeBody](https://tml.stanford.edu/homebody/),
the scanned room is the robot's model of the world, and the skills take their targets from it.

## Use

- **Make a room:** in the mapping console's 3D view choose a labelled session, **Use as room** (**Update room**
  after a new reconstruction), or `python projects/navigation/make_room.py <session> [--title "Lab"]`.
- **Task console:** `start_task_console.bat` opens http://127.0.0.1:8765/tasks.html (the mapping console's helper,
  camera, drive keys and **STOP**). Don't run SLAM or an exploration at the same time.
  1. Choose the room the robot stands in: its mesh with the labels.
  2. **Connect**, **Armed**, **⟳ Localize**: one slow turn on the spot (~15 s); if "not sure", drive a bit.
  3. Click an object and **Go to …**, or **📍 Pin & go** for a spot on the floor. The robot stops 0.2 m in front of
     the object, facing it. **Cancel**, STOP, Space, disarming or a driving key end the drive.
  4. Or write to the **Assistant**: "drive to the cabinet near the sink", "there is a red cube left of the outlet,
     pick it up", in any language. A box under the message shows its steps.
  5. **● Record** until pressed again: camera with the detector's boxes, drives, chat, as `video.mp4` in
     `data/task_runs/<time>/`; **Runs** plays, keeps or deletes them.
- **Settings** (`.env`, see `.env.example`): `VLLM_URL`, `VLLM_MODEL` and `VLLM_CHAT_API_KEY` for the chat and the
  image check.
- **Without the robot:** `fake_robot.py <session>` replays a recorded run as the robot's rosbridge on
  127.0.0.1:9090, `--sim <room>` simulates a robot that follows the commands; start the helper with
  `$env:LANDERPI_HOST='127.0.0.1'`. `fake_video.py <session>` serves a recording's images as the camera,
  `LANDERPI_SIM_OBJECTS='[{"label": "pencil", "x": -0.5, "y": 1.45}]'` plants things to find. Offline:
  `test_localize.py <room> <run>`, `test_goto.py <room>`, `test_watcher.py`, `test_chat_room.py [room]`.

## How it works

- **Room** (`landerpi/room.py`): `data/rooms/<session>/` holds the session's last SLAM map and `room.json`
  (title, no-go lines); mesh and labels come from the session, in the same frame as the map.
- **Localization** (`localize.py`, numpy only): the turn's scans, stitched with odometry, are matched against the
  map wherever the robot can stand, every 20 cm and 3° on a blurred map, the 40 best refined to 1 cm / 0.25°; a
  second pose scoring almost as well makes it "not sure". Then every scan is tracked (odometry predicts, ±15 cm /
  ±6° correct, 4-5 ms), without the arm's reach in front. The score is a likelihood field: people don't pull it.
- **Where to stop** (`goto.py`): in front of a face of the object's box, facing it, the spot with the shortest
  path; long objects from a long side, a find from the robot's side, as close as it fits if no side is free. At
  the stop the lidar or depth camera sets the last few centimetres (scanned boxes can be 10-20 cm off).
- **Path and safety:** the explorer's planner, pure pursuit and collision check (`offboard_ros/explore_core.py`),
  re-planned every second against the map, no-go lines, live scan and depth obstacles (`depth_view.py`: 10-45 cm
  above the floor; within 1 m anything 2 cm above the floor around it). Detected things on the floor count too,
  except the target and what the gripper holds. Up to 0.4 m/s; stops if scans or odometry are older than 0.5 s or
  localization is lost.
- **Chat** (`assistant.py`, `nav_tools.py`): the model gets the object list, the walls (`room_view.py`: each side
  with what is along it, a top view, the walls ahead, behind, left and right), the robot's status and the
  conversation's sightings. Its tools (`landerpi/tools.py` rules): drive to an object or a point, turn, survey
  pose, status; `watch_for`, `search_here`, `approach_found`, `list_seen`, `look_at`, `confirm_found`;
  `pick_up_object`, `put_down_object`. Tools that move the robot need Armed.
- **Detection** (`landerpi/detector.py`, `live.py`): YOLOE-11s-seg on the laptop's CPU (~5 images/s), prompted with
  the item and 2-3 synonyms, while a search is open and until the next chat message. Close up, a missed item gets
  a second pass on the shrunk picture; boxes on the robot's gripper are dropped.
- **Sightings** (`perception.py`, `scan_cells.py`): each thing is followed by its box from image to image (overlap,
  shifted by the robot's turn); seen twice, it is a numbered sighting placed by its mask's depth points. It is
  sure if it rests on the room's scan (5 cm cells) or, without depth, where its line of sight meets the floor;
  otherwise vague (seen from where, in which direction): never approached, the chat goes and looks.
- **Image check** (`verifier.py`): the detector weighs the kind of thing, hardly the colour, so the chat model
  judges each sighting's crop against the item's appearance: fits or not. Again when the box has doubled (closer),
  at most 3 times; the robot doesn't stop for it.
- **Search:** found = a sure sighting in the search area that passed the check; a drive or turn stops for it.
  `search_here` visits up to 4 stops around the area with a camera sweep each. Matches outside the area are
  reported, never approached without the user, and shorten the search. A remembered sighting whose place is in
  view (0.35-1.2 m) but empty for 1.5 s is marked gone.
- **Memory:** sightings and searches without a find go with every chat message, for as long as the chat lasts.

## Results and limits

| Test (date) | Result |
|---|---|
| `test_localize.py`, 3 runs of one room, global every 30 s (2026-10-07) | 32/33 within 30 cm / 5° in ~4 s; tracking fits the map 0.94-0.96 |
| `test_goto.py`, simulated drives in 2 rooms, also with boxes 15 cm off (2026-10-07) | 245/246 arrived, about 0.2 m from the object at the end |
| `test_chat_room.py`, place prompts, 9 × 2 (2026-10-09) | 18/18 with the walls as text, 12/18 with the object list only |
| `test_positions.py`, detections + depth on recorded keyframes (2026-10-08) | median 0 cm, 84 % within 10 cm of the scanned box |
| `test_far.py`, small objects by distance from the camera (2026-10-08) | found 75 % at 0.3-0.7 m, 33-42 % at 0.7-2.5 m |

Localization, driving, the chat and finding small blocks and cups run on the robot. Weak spots:
- Driving between many small things (approach paths, docking) needs another look.
- Localization searches only the scanned area; a robot lifted and put down elsewhere needs ⟳ Localize again.
- Small things further than 0.7 m are often missed; vague sightings (no usable depth) can't be approached.
- The image check needs the gateway; without an answer a candidate counts as "not checked".

## Files

| File | What |
|---|---|
| `navigator.py` | the task console's backend (127.0.0.1:8767, started by the helper): localization, drives, detection |
| `localize.py`, `goto.py` | localization against the room's 2D map; where to stop, the path, following it |
| `depth_view.py` | obstacles above the lidar's slice from the depth camera |
| `assistant.py`, `nav_tools.py` | the chat (model, rules, tool loop) and its navigation and finding tools |
| `room_view.py` | the room for the model: sides, top view, walls around the robot |
| `live.py`, `perception.py` | live detection, 3D places, following boxes, finds, memory |
| `verifier.py`, `scan_cells.py` | the image check; the room's scan as 5 cm cells |
| `taskrun.py` | recordings of task runs (● Record) |
| `make_room.py` | a room from a labelled session |
| `detector.py` | re-export of `landerpi/detector.py` |
| `fake_robot.py`, `fake_video.py` | stand-ins for the robot's rosbridge and the camera |
| `test_*.py` | offline tests (above); `test_detector.py`: detector speed and recall on a room's objects |
| `console/tasks.*` | the task console page |
