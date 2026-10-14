# Room mapping

Goal: the robot explores a room and we build a map of it — 2D floor plan, 3D geometry and
(later) a semantic layer in one coordinate frame — as input for real2sim and in-room tasks.
Inspired by [HomeBody](https://tml.stanford.edu/homebody/).

**Split of work:** the robot publishes its sensors (plus light compression) and takes drive
commands. Recording, SLAM and reconstruction run on the laptop.

## Milestones

| | What | Status |
|---|---|---|
| M1 | Recorder: stream sensors to the laptop, save keyframes | ✅ |
| M2 | Live 2D SLAM on the laptop (slam_toolbox in Docker) + mapping UI | ✅ first version |
| M3 | Offline 3D reconstruction (Open3D TSDF + ICP) | done |
| M4 | Autonomous exploration: frontiers + camera coverage, continuous driving | first version |
| M5 | Semantic labels: objects named and placed in 3D | in `projects/scene_understanding` |

## Live mapping (M2)

1. Double-click `start_mapping_console.bat`, press **Connect**, then switch to the **Map** view (top bar,
   or key **M**). The camera becomes a picture-in-picture; driving keys keep working.
2. In the map view, **Start SLAM**. This starts the `landerpi-offboard-ros` container
   ([offboard_ros/](../../offboard_ros/)): Docker Desktop must be running. The first start builds
   the image (a few minutes).
3. Put the arm in the survey pose (**P** in the console) and drive slowly through the room.
   The map grows live; **New map** starts over, **Save map** writes `data/maps/map_<time>.pgm/.yaml`.
4. To record for 3D reconstruction, press **● Record** in the map view (with SLAM running): the
   helper starts `record.py` in the background (log: `logs/recorder_*.log`), the button shows time,
   and **■ Stop recording** finishes it cleanly. With SLAM running, the session also gets SLAM's
   drift correction (`streams/slam_map_odom.jsonl`, 20 Hz) and the robot pose in the map
   (`streams/slam_robot_pose.jsonl`, 5 Hz), so every image can be placed in the map.
   During an exploration it also records the explorer's status (`explore_status.jsonl`, on change) and
   every 5 s the SLAM map and the coverage grid (`slam_map.jsonl`, `explore_coverage.jsonl`), to replay
   its decisions afterwards.
5. **Save map** only saves if the map changed since the last save.

How it works: `offboard_ros/relay.py` subscribes to the robot's rosbridge (scan, odom, TF,
joints: small, full rate) and republishes them in the container, with the robot's timestamps as
`/clock`. slam_toolbox publishes `/map` and `map → odom`; `robot_pose.py` publishes `/robot_pose`
and `/robot_path`. The page reads these from the container's rosbridge on `ws://127.0.0.1:9091`.

The robot's `/scan` has its rear sector filtered out (its own body/arm), so the map fills in
behind the robot only after it turns.

## 3D reconstruction (M3)

In the console's **3D** view: pick a recorded session, press **Reconstruct** (runs
`reconstruct.py` on the laptop, log in `logs/reconstruct_<session>.log`), then look at the result
as **Mesh** or **Points** with the camera path (drag to rotate, right-drag to pan, wheel to zoom).
From a terminal: `python projects/mapping/reconstruct.py [data/sessions/<name>]`.

Per keyframe, the camera pose in the map is SLAM's `map → odom` × odometry × the robot/arm
transforms to the depth (and colour) camera, each at the image's own capture time. Depth becomes
points, coloured by projecting them into the colour camera (points at depth edges are dropped:
they smear between foreground and background). Then:

1. **Pose refinement (ICP)**: each keyframe is aligned point-to-plane to the model built from the
   keyframes before it (within 4 m). A correction is kept only if it is small (≤ 6 cm, ≤ 4°) and
   the fit is good; corrections do not accumulate, so a bad match can't drag later frames off.
   `--no-icp` switches it off.
2. **Floor levelling**: the largest horizontal plane is made z = 0 and level (`--no-level`).
3. **Fusion**: all keyframes go into a TSDF volume (Open3D `VoxelBlockGrid`, 1 cm voxels by
   default, `--voxel 0.015` for big rooms; surfaces seen ≥ 2×).

Keyframes are thinned first: one per 10 cm of travel or 10° of turning (`--keyframe-step`,
`--keyframe-turn`; `--keyframe-step 0` uses all). Fast runs record about 3.5 keyframes per second,
mostly near-duplicates. The log prints the time each stage took (`[time] …`). A 14-minute
exploration run (1,458 keyframes, 11 × 9 m) takes about 4.5 min instead of 54 min, with the
same result.

Output in `<session>/reconstruction/`: `mesh.ply`, `cloud.ply`, `mesh_view.ply` (≤ 1 M triangles
for the browser), `cameras.json`, `poses.csv`, `preview.png`. The
3D view converts the (sRGB) vertex colours to linear and shows them unlit, so they match the
camera images.

For best results: survey pose (**P**), ≤ 0.1 m/s, and stop and turn slowly in place every metre
or so; walls seen from several viewpoints get sharper and better coloured.

## Autonomous exploration (M4)

In the **Map** view (SLAM running, "Motion armed" ticked): press **▶ Explore**. The robot drives by
itself and records the room (the recording starts with it and stops when the room is done).

There is one exploration mode (since 2026-10-06; before, four modes differed in speed, camera range
and viewpoints per surface).
- **The lidar discovers:** frontiers (openings, what lies around a corner) and walls the camera has
  not seen at all yet.
- **RGB + depth map:** whatever the camera has seen at least twice is a surface to map, done once
  80 % of its height (up to 1.2 m) was seen well. That is floor, walls, furniture and obstacles alike,
  with no categories. Once seen, it stays seen unless it gets clearly worse (later far looks only add
  a voxel or two that were never seen well).
  - "Seen well": 2 good views, within 2 m, at most 60° from head-on (the floor: within 1.3 m, up to
    81°; the shiny floor gives hardly any depth from further away at the camera's 26 cm height).
  - The 3D coverage is computed live from the depth images (`offboard_ros/coverage3d.py`, validated
    against the reconstruction in `projects/coverage3d`), with heights measured from the floor, which
    it estimates from the same images.
- **Without depth images** the exploration pauses (after 15 s; the console says why).
- **Driving:**
  - 0.3 m/s, turning 0.6 rad/s;
  - at a viewpoint the robot faces the surfaces for 4 s, turning at most 0.3 rad/s (sharper images);
  - a surface gets 2 tries before it is given up: a visit that had it in view and left it unseen is one
    failed try;
  - gaps from 25 cm are visited.
  - areas larger than 1.5 m are split into 1.5 m pieces with a viewpoint each (the camera sees
    ~2 m), 1.2 m away; floor areas into 0.6 m pieces, with a stop that puts the whole piece 0.7-1.3 m
    from the camera (where the floor gives depth; in a narrow corridor the robot looks along it, not
    across). A spot already looked from is not used again for the same piece (it may be for others).

The robot explores in **one pass**:
- it visits the viewpoints of everything not yet seen well, in the order of the shortest route
  through them;
- it finishes each area before moving on, instead of mapping the room first and then making a
  second round.

An earlier version first mapped the whole room and then made a second round for the camera. In the
simulator it took about twice as long and re-drove 40 % of its path (one pass: 16 %), for similar
coverage (2026-10-02).

A real run in a crowded room (20261007_101108) took 17 min and saw 85 % of all surfaces well; most of
the rest is hidden (under tables, behind chair legs) and given up after two tries.

**No-go lines** (✎ No-go line, two clicks on the map; also during exploration; drag an end to move
it, right-click a line to remove it): the robot doesn't
cross them and doesn't explore behind them. The exploration area is always the start point's side:
if a line is drawn between the robot and its start, the robot comes back across it (crossing as few
lines as possible) and stays on that side. They belong to the map: **New map** removes them.

- **STOP**, **Space**, a driving key, or un-arming **pauses** it; **▶ Resume** continues where it
  stopped. **New map** or **Stop** (SLAM) ends it for good.
- Map overlay (legend in the map view): surfaces green = seen well, red = still to see, orange = given
  up, grey = furniture only the depth camera sees; dashed = planned path, no-go lines in red.
- How it works (`offboard_ros/explorer.py` + `explore_core.py`, in the offboard container):
  1. one slow turn on the spot at the start (the lidar is blind to the rear ~155 deg);
  2. drives to the viewpoints of surfaces the camera hasn't seen yet, in the order of the shortest
     route through them (re-planned every second as the map grows; it keeps heading for a
     viewpoint until it is reached, its surfaces were seen on the way, or it is given up); paths
     keep 26+ cm (preferably 55 cm) from obstacles; a goal area is given up after 30 s without new
     map or camera coverage;
  3. meanwhile the mecanum base turns the camera towards surfaces it hasn't seen yet (up to 70 deg
     away from the travel direction: never towards the blind rear);
  4. when no known surface is left: on to the nearest frontier of the lidar map (free space next
     to unknown);
  5. done when no frontier and no surface is left to visit: each one seen, given up after two tries,
     smaller than 25 cm or unreachable (or 98 % seen). Before that, one more look around.
- Safety: stops when the lidar sees something closer than 32 cm in the travel direction (people,
  chairs not in the map yet) and re-plans; obstacles the depth camera sees above the lidar's slice
  (10-45 cm above the floor) count too; zero velocity if the lidar data are older than 0.5 s or the
  SLAM pose older than 1 s; the relay sends a stop if the exploration's commands stop arriving.
- Tests without moving the robot: `python offboard_ros/test_explore.py
  [--wall x1 y1 x2 y2 --wall-at 90]` (simulated robot on a saved map); dry run with the real sensors: start the container with
  `$env:LANDERPI_CMD_OUT='/landerpi/cmd_vel_dryrun'` (commands go to an unused topic).

## Camera calibration (2026-10-02)

The camera pose comes from the robot model and the *commanded* arm servo positions. Measured:

- **Pan:** servo 1 (arm base) had no zero offset; the camera looked 7.15 deg right. Fixed in the
  servo itself (offset +38, saved): `python tools/arm_calibration.py read|survey|set <id> <offset> [--save]`.
  Check: record ~20 s standing still in the survey pose in front of a corner
  (`record.py --all-frames --max-minutes 0.35 --live-port 0`), then `camera_pan_check.py`
  (depth points vs lidar scan; now +0.06 deg).
- **Tilt/roll:** under the arm's weight the servos sit a few steps low; the real camera looks
  2.8 deg up and is rolled -2.3 deg. Not fixable in the servos (load dependent), so
  `camera_calibration.py --save` (floor + walls of a corner) writes `config/camera_correction.json`,
  which `reconstruct.py` applies to keyframes in the survey pose; for recordings before the servo
  fix it also corrects the old 7.15 deg pan. Repeat both if the arm or camera mount is changed.

## Recording (M1)

```powershell
conda activate landerpi
python projects/mapping/record.py                     # Ctrl+C to stop
python projects/mapping/record.py --max-minutes 15 --max-gb 5
python projects/mapping/report.py                     # check the newest session -> report.png
```

Drive slowly with the console meanwhile (≤ 0.1–0.15 m/s, turns ≤ 0.3 rad/s). A keyframe is kept
every 5 cm / 5° of motion or when the arm moves (`--min-move`, `--min-turn`). The status line
shows rates, data rate, lag and hints ("slow down", "battery low").

A live view of what the recorder sees is served locally at `http://127.0.0.1:8766/stream.mjpg`
(status: `/status.json`), for the mapping UI.

### How the data travels (measured 2026-09-30)

| Stream | Path | Rate | Latency | Data |
|---|---|---|---|---|
| Colour | MJPEG from the robot's `web_video_server` (JPEG q90 on the robot) | ~14–15 fps | ~40 ms | ~1.4 MB/s |
| Depth | 16-bit PNG via rosbridge, compressed on the robot by our `depth_compressor` | ~10 fps | ~120 ms | ~53 KB/frame |
| Scan, odom, IMU, TF, joints, battery | rosbridge | full rate (10 / 30 / 45 / 43 / 20 / 1 Hz) | – | small |

Total ~2.5–3 MB/s of a ~8 MB/s Wi-Fi link. Raw images over rosbridge (the first approach) gave
~1.7 s latency, starved the small topics and filled rosbridge's memory until it hung.

The depth compressor is a plain `image_transport republish` process in the factory ROS
container, started over SSH by `record.py` when missing (gone after reboot) — see
`landerpi/robot_ssh.py`, which can also restart a hung rosbridge or `web_video_server` and
report their memory/CPU (`health()`).

### Session format

See [session.py](session.py): `rgb/*.jpg` and `depth/*.png` exactly as received, `keyframes.csv`,
TUM-style `associations.txt`, calibration in `calib/`, every small message in `streams/*.jsonl`.
Every image keeps its own robot timestamp; camera poses are computed per image later from odom + TF
(the camera sits on the arm: `odom → base_footprint → … → link4 → … → rgb_camera_link`).

Sizes: with `--all-frames` ~0.12 GB/min; with keyframes it depends on how much the robot moves.
