# The web consoles

Two static web pages, served by the local helper (`tools/robot_helper.py`, http://127.0.0.1:8765/), that talk
to the robot through rosbridge (WebSocket, port 9090):

| Console | Start | For |
|---|---|---|
| Mapping console (`index.html`, `app.js`) | `start_mapping_console.bat` | driving by hand, SLAM, exploration, recording, 3D reconstruction and labelling (`projects/mapping`, `projects/scene_understanding`) |
| Task console (`tasks.html`, `tasks.js`) | `start_task_console.bat` | the robot in a mapped room: localization, driving to objects, the chat (`projects/navigation`, `projects/manipulation`) |

Both use the same helper (a running one is reused unless its code changed, `tools/start_console.ps1`) and the
same top bar, camera, drive and arm controls (`tasks_boot.js` takes them from `index.html`). Don't run SLAM or an
exploration while the task console moves the robot.

Open the pages through the helper, not as files: browsers block web pages from loading anything from
local-network devices ("Local Network Access"), so the robot video only works through the helper, which serves
the page and relays the video. The helper also runs what a web page can't: SSH tasks on the robot (reboot,
shutdown, video server restart), `docker compose` for the offboard ROS container, the recorder, the
reconstruction and the labelling.

## Mapping console

- **Camera:** MJPEG from the robot's `web_video_server`, like the factory app (~14 fps, a few ms latency). If the
  video server stops sending, the console has the helper restart it; raw frames over rosbridge remain as a slow
  fallback (~1.7 s latency).
- **Drive** and **Arm & gripper:** keyboard/button control with speed limits; only sends while *Motion armed* is
  checked and a key is held; stops on key release, window blur and Space.
- **Lidar:** top-down view of a `LaserScan` topic (forward = up). **Topics / Nodes:** everything the robot
  publishes, with a live message viewer.
- **Map** view (key M): Start SLAM (the `offboard_ros` container), ▶ Explore, ● Record, no-go lines, save the map
  (`projects/mapping/README.md`).
- **3D** view: Reconstruct, Label (each turns into a Stop button while it runs), Mesh / Points / Labels,
  Use as room (`projects/mapping`, `projects/scene_understanding`).
- **Battery:** voltage in the header (2S pack: 8.4 V full, charge below ~7.2 V).

Topics are detected automatically (falling back to the known LanderPi topics if the robot's `rosapi` is
unavailable); the choices are remembered in the browser.

## Task console

Choose the room, **Connect**, **Armed**, **⟳ Localize**; then click an object and **Go to …**, or **📍 Pin & go**
to a spot on the floor, or ask in the **Assistant** card ("find the red cube near the sofa and pick it up").
**● Record** saves a run as video, **Runs** lists them. Details: `projects/navigation/README.md`.

## Keyboard

| Keys | Action |
|---|---|
| W / S, A / D | drive forward/back, rotate |
| Q / E | strafe (mecanum) |
| ← / → | arm base pan (servo 1) |
| ↑ / ↓ | camera tilt / wrist pitch (servo 4) |
| Shift + ↑ / ↓ | shoulder (servo 2) |
| Ctrl + ↑ / ↓ | elbow (servo 3) |
| Shift + ← / → | wrist rotate (servo 5) |
| R / F | gripper close / open (servo 10) |
| H | arm to the start pose |
| P | arm to the survey pose (camera ahead, as for mapping); Shift + P saves the current pose as survey pose |
| M | mapping console: camera ↔ map view; task console: swap room and camera |
| Space | stop everything |

If a joint moves the opposite way to its key, tick **rev** in its row (remembered).

## Battery warnings (instead of the robot's loud buzzer)

The controller board's firmware beeps loudly below ~7.0 V; no ROS setting turns that off. The console
therefore warns earlier and quietly (voltage averaged over 10 s):

| Level | Voltage | Console | Robot LEDs / display |
|---|---|---|---|
| OK | ≥ 7.4 V | – | dim green, "Battery OK" |
| Low | < 7.4 V | amber banner, one soft laptop chime | amber, "LOW: charge soon" |
| Critical | < 7.1 V | red banner, two chimes, **motion disarmed** | blinking red, "CHARGE NOW!" |

Chime and robot indicators can be switched off in ⚙. The LEDs/display are only driven while the console is
connected; on disconnect they go dark / show "console offline".

## Third-party libraries (`vendor/`, downloaded)

The repository contains no third-party code. On its first start the helper downloads these files from
unpkg.com into `console/vendor/` (gitignored), at fixed versions, and checks each file's sha256
(`tools/console_vendor.py`; without the helper: `python tools/console_vendor.py`). This needs internet once;
if a download fails, the helper window and the console say so.

| Library | Version | Files | License |
|---|---|---|---|
| [three.js](https://threejs.org) | 0.180.0 | `three.module.min.js`, `three.core.min.js`, OrbitControls, PLYLoader | MIT |
| [roslibjs](https://github.com/RobotWebTools/roslibjs) | 1.4.1 | `roslib.min.js` | BSD-3-Clause |
| [pako](https://github.com/nodeca/pako) | 2.1.0 | `pako.min.js` | MIT and Zlib |
| [UPNG.js](https://github.com/photopea/UPNG.js) | 2.1.0 | `UPNG.js` | MIT |
