# Room mapping

The robot explores a room by itself and records it; the laptop builds a 2D map live (SLAM) and a 3D mesh
afterwards, in one coordinate frame. The recordings feed the scene labelling (`projects/scene_understanding`),
which turns them into a room with objects for navigation (`projects/navigation`). The robot only streams its
sensors (with light compression) and takes drive commands; SLAM, exploration, recording and reconstruction run on
the laptop, SLAM and exploration in a ROS 2 container (`offboard_ros/`).

## Use

In the mapping console (`start_mapping_console.bat`, see `console/README.md`):

1. **Connect**, switch to the **Map** view (key **M**) and press **Start SLAM**. This starts the
   `landerpi-offboard-ros` container; Docker Desktop must be running, and the first start builds the image.
2. Put the arm in the survey pose (**P**), tick **Motion armed** and press **▶ Explore**. The robot explores and
   records the room and stops when it is done. **STOP**, Space, a driving key or un-arming pauses it,
   **▶ Resume** continues. To drive by hand instead, use **● Record** / **■ Stop recording**.
3. Optional **✎ No-go lines** (two clicks; drag an end to move one, right-click to remove it): the robot neither
   crosses them nor explores behind them. **New map** starts over and removes them.
4. In the **3D** view, pick the session and press **Reconstruct**; view the result as Mesh or Points. **Label**
   continues in `projects/scene_understanding`.

From a terminal (conda env `landerpi`):

```powershell
python projects/mapping/record.py --max-minutes 15      # record while driving by hand; Ctrl+C stops
python projects/mapping/report.py                       # check the newest session -> report.png
python projects/mapping/reconstruct.py data/sessions/<name>
python offboard_ros/test_explore.py --minutes 5         # simulated exploration on a saved map, no robot
```

Without moving the robot: set `$env:LANDERPI_CMD_OUT='/landerpi/cmd_vel_dryrun'` before starting the container
(its drive commands go to an unused topic).

## How it works

- **Data path:** `offboard_ros/relay.py` brings the robot's lidar, odometry, TF and joint states from its rosbridge
  into the container, with the robot's time stamps as `/clock`. Colour comes as MJPEG from the robot's
  web_video_server (~14 fps), depth as 16-bit PNG compressed on the robot (~10 fps), together ~2.5-3 MB/s.
- **SLAM:** slam_toolbox builds the 2D map and the robot's drift correction (`map -> odom`). Scans are held back
  until the robot's pose at their time stamp is known (`scan_gate.py`); without this, slam_toolbox froze mid-run.
- **Recording:** a keyframe (colour + depth, each with its own robot time stamp) every 5 cm or 5° of motion, and
  every small message (scan, odometry, TF, joints, SLAM correction) in `streams/*.jsonl`, so every image's camera
  pose can be computed later. During an exploration also the explorer's status, map and coverage, to replay its
  decisions.
- **Exploration** (`explorer.py`, `explore_core.py`): the lidar discovers (frontiers, walls the camera has not
  seen at all), the depth camera decides what is done. A live 3D coverage (`coverage3d.py`, `projects/coverage3d`)
  counts every surface (floor, walls, furniture alike) as seen once 80 % of it up to 1.2 m has 2 good views: within
  2 m and at most 60° from head-on, the floor within 1.3 m.
- **Viewpoints and route:** unseen walls and furniture are split into 1.5 m pieces with a viewpoint ~1.2 m away,
  unseen floor into 0.6 m pieces with a stop that puts the whole piece 0.7-1.3 m from the camera. The robot visits
  them in one pass along the shortest route (re-planned every second), faces each for 4 s, turns the camera
  towards unseen surfaces while driving (up to 35° off its path), and gives a surface up after two visits that
  left it unseen. It is done when no frontier and no surface to visit is left.
- **Safety:** 0.3 m/s; paths keep at least 26 cm (preferably 55 cm) from obstacles. Obstacles the lidar or the
  depth camera sees (10-45 cm above the floor) in the travel direction halve the speed below 55 cm and stop the
  robot below 27 cm (from its centre). Zero velocity if lidar data are older than 0.5 s or the SLAM pose older than
  1 s; the relay stops the robot if commands stop arriving; without depth images the exploration pauses.
- **Reconstruction** (`reconstruct.py`): keyframes thinned to one per 10 cm or 10°; each camera pose from SLAM,
  odometry and the arm's transforms at the image's own time (colour stamps corrected by -0.1 s) plus the calibrated
  camera tilt; ICP alignment to the growing model (corrections up to 6 cm / 4°, not accumulated); the floor
  levelled to z = 0; TSDF fusion with Open3D (1 cm voxels, surfaces seen at least twice); colours from the best
  views with exposure compensation.
- **Camera calibration:** the camera pose comes from the robot model and the commanded arm servos.
  `camera_pan_check.py` compares depth with lidar to check the heading; `camera_calibration.py --save` measures
  tilt and roll from a room corner into `config/camera_correction.json`, used by the reconstruction and the
  explorer. Repeat after changing the arm or the camera mount.

## Results and limits

| Test (date) | Result |
|---|---|
| Exploration of a crowded room, run 20261007_101108 (2026-10-07) | 17 min; 85 % of all surfaces seen well, the rest mostly hidden and given up |
| Reconstruction of a 14-min run, 1,458 keyframes, 11 x 9 m (2026-10-02) | ~4.5 min on the laptop's CPU |
| Live coverage vs. the mesh's floor gaps, two runs (2026-10-05) | 96-97 % of the gaps predicted (`projects/coverage3d`) |

- The camera sits 26 cm high: the floor under tables and the backs of chairs stay holes, and the shiny floor gives
  depth only up close. Crowded rooms take longer (many small floor pieces).
- People who walk through the room end up in the mesh; there is no filter for moving objects.
- The camera pose relies on the commanded servo positions; the arm sags a few servo steps under its weight, hence
  the tilt calibration.

## Files

| File | What |
|---|---|
| `record.py`, `recorder.py` | recording (command line; the console starts the same recorder) |
| `live_view.py` | local view of what the recorder sees, for the console |
| `session.py` | the session format on disk (see below) |
| `report.py` | checks a recording: rates, gaps, sync, overview picture |
| `reconstruct.py` | 3D reconstruction -> `<session>/reconstruction/` |
| `camera_pan_check.py`, `camera_calibration.py` | camera heading check, tilt/roll calibration |
| `offboard_ros/` | the container: `relay.py`, `scan_gate.py`, `scan_fix.py`, `slam.yaml`, `robot_pose.py`, `explorer.py`, `explore_core.py`, `coverage3d.py`, `depth_obstacles.py`, `slam_watchdog.py`, `launch.py` |
| `offboard_ros/test_explore.py`, `replay_slam.py` | exploration simulator, SLAM replay of a recording |

A session (`data/sessions/<time>/`): `rgb/*.jpg` and `depth/*.png` as received, `keyframes.csv`, TUM-style
`associations.txt`, calibration in `calib/`, every small message in `streams/*.jsonl`. The reconstruction writes
`mesh.ply`, `cloud.ply`, `mesh_view.ply` (for the browser), `cameras.json` (every keyframe's camera poses and
intrinsics) and `preview.png`.
