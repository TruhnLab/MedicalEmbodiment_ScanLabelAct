# Manipulation

The LanderPi's 5-servo arm picks up small things from the floor and puts them down, steered by the camera that
sits on the gripper. In the task console's chat the robot finds an item (navigation's `watch_for` /
`search_here`), approaches it (`approach_found`), picks it up (`pick_up_object`), drives somewhere else and puts it
down (`put_down_object`).

Status (2026-10-12): picking works for block-sized items lying roughly square to the robot. In the last chat run the
robot picked up a 30 mm cube, put it down and picked it up again (52 s and 49 s per pick).

## Picking up (`pick.py`)

The camera and the gripper are rigidly connected (only the wrist roll is between them), so "gripper straight above
the object" always looks the same in the picture: the object at one spot. That spot is taught once, with an object
centred between the jaws by hand (`grasp.py down`, then `grasp.py teach` -> `config/grasp_reference.json`). A pick
then:

1. **Finds the object** with the open-vocabulary detector (YOLOE, `landerpi/detector.py`), running in the
   background at ~4 pictures/s. It looks from the arm's current pose first (after `approach_found` the camera
   already looks at the item), then with the gripper over a point ~11 cm in front of the bumper, closer in front,
   and further up.
2. **Follows it in:** the base drives by the camera (3-6 cm/s, at most 50 cm, the lidar checks the way) while
   the arm keeps the gripper over the object, until the object is ~7 cm in front of the bumper. Every detection
   carries the time its picture was taken; between pictures the base subtracts what it has driven since, and it
   drives only on pictures less than 1.5 s old.
3. **Centres the gripper** on pictures taken with the arm and the base standing still, until the object sits within
   8 px (~4 mm) of the taught spot.
4. **Grasps:** down along the gripper's axis (60 deg below horizontal), closes until the gripper stalls, squeezes a
   little, lifts and carries the object in the survey pose (the pose the navigation drives with).

`put_down_object` puts it down ~7 cm in front of the bumper, opens and returns the arm to the survey pose.

Every pick is recorded in `data/manipulation/picks/<time>/`: each picture, detection, control step and the robot's
localization (the last 30 picks are kept). `projects/presentation_video/pick_video.py` makes a video of one.

**In the chat** (`manipulation_tools.py`, listed in navigation's `assistant.TOOL_MODULES`): the tools run inside the
navigator and use its robot connection, video stream, detector (under `live.detector_lock`) and lidar scan. They need
the console to be Armed; STOP ends them with the arm where it is. While the robot holds something, `nav.held` carries
its label and its mask in the camera picture, which navigation leaves out of the depth obstacles and of searches.

**Limits:**

- Items lying on the floor, 15-45 mm wide, about block height (the spot was taught with a 30 mm block).
- Roughly square to the robot: the wrist does not turn yet (a cube turned ~25 deg was gripped across its corners;
  it held).
- The detector needs words that name the item: "red cube" found the cube at 0.8-0.9, "block" alone a sheet of paper.
- Put down only right in front of the robot, without checking that the floor there is free.
- ~50 s per pick, mostly the slow, guarded descent and lift.

**Lessons from the live runs (2026-10-12):** the wheel odometry reported 10 cm where the camera saw 3 cm at these
low speeds, so the base is driven by the camera, not by odometry. Driving on to an old detection pushed the cube out of
view, hence the time stamps and the age limit. YOLOE takes ~2 s to switch its text prompts, so a pick keeps the
navigator's prompts when they include its words, and then also accepts the navigator's synonyms ("red cube toy").

`test_pick_sim.py` tests the approach against a simulated robot: a base slower than commanded, sagging servos, a
camera with latency and noise. All cases within reach end with the gripper within 1.2 cm of the object.

## Arm model and calibration

- **`arm.py`:** kinematics from the robot's own URDF, servo pulses <-> joint angles (0.24 deg per pulse, 500 =
  0), inverse kinematics, and a collision check against the floor and the robot's front. The URDF is Hiwonder's and
  not in this repository: export it from the robot into `config/landerpi.urdf`, e.g. with
  `docker exec -u 1000:1000 MentorPi bash -lc "ros2 param get /robot_state_publisher robot_description"` on the
  robot (remove the leading `String value is: `). Without it the chat runs and the two arm tools say that the arm's
  model is missing. `test_arm.py` checks the model against a recording's /tf: 0.0 mm, 0.06 deg.
- **Jaw tips:** they reach 13 mm beyond the URDF's `end_effector_link` (measured on the grid sheet; one clean
  reading).
- **Gripper** (`check_arm.py gripper`, `gripper-to`, `grip`; `config/gripper.json`): servo 10 from 201 (open,
  ~48 mm between the jaws) to 545 (closed, 12-13 mm; commands beyond only strain the servo). A 29.5 mm block makes
  contact at 399-403 (the jaws give way ~3.5 mm) and holds firmly with a 14-pulse lag.
- **Joints** (`check_arm.py joints`): the servos stop 5-13 pulses short of a pose approached from below (gripper
  tip 11-25 mm off), 0-5 from above (2-5 mm). Poses are therefore approached from one side and read back.
- **Camera** (`grid_sheet.py` prints an A4 sheet with targets and ArUco markers; `look_at_sheet.py`,
  `collect_views.py`, `solve_camera.py`): the colour camera sits ~19 mm to the right of where the URDF has it and
  is rotated by -1.5 deg about its axis (`config/camera_mount.json`, fitted against the sheet's position as the
  arm found it).
- **Open:** single-joint sweeps seen by the camera (`collect_views.py --sweeps`, `solve_gains.py`) suggest 0.232-0.265
  deg per pulse, but with those values the gripper missed the sheet's targets by 30 mm, with 0.24 by ~3 mm. The
  sweep values are kept in `config/arm_joints.json` and not used. Touching targets by "a joint stops following"
  (`touch.py`) was unreliable without someone watching.

## About the robot's servos

- `/controller_manager/servo_states` and `/joint_states` (and so the arm's /tf) report the last *commands*, not
  measured positions. Real positions come only from `/ros_robot_controller/bus_servo/get_state`.
- Ask `get_state` for positions only (~50 ms for six servos). Asking for voltage, temperature, limits etc. as well
  got no answer and hung `ros_robot_controller` until a reboot. Without that node (no `/ros_robot_controller/battery`
  messages) there are no servo reads: reboot the robot.
- After every boot the robot commands the gripper to 150, past its open stop, so the servo strains until something
  else moves it (probably a Hiwonder start-up action group; not changed). A pick opens it to 201.

## Files

| File | What |
|---|---|
| `pick.py` | the pick and the put-down (`pick`, `put_down`), behind a small robot interface |
| `manipulation_tools.py` | the chat tools `pick_up_object`, `put_down_object` (inside the navigator) |
| `grasp.py` | the same from a script: `pick [--prompt ...]`, `put`, `up`, and `down` / `teach` for the reference |
| `test_pick_sim.py` | the approach against a simulated robot |
| `arm.py`, `test_arm.py` | kinematics and its check |
| `check_arm.py` | servo checks on the robot: `health`, `gripper`, `gripper-to`, `grip`, `joints` |
| `grid_sheet.py`, `look_at_sheet.py`, `collect_views.py`, `solve_camera.py`, `solve_gains.py`, `touch.py` | calibration with the printed sheet |
| `nudge.py` | drive a few cm straight (calibration set-ups) |

Scripts that move the arm or the base need someone watching the robot. `check_arm.py gripper`, `grip` and `joints`
ask before they move (`--yes` skips that, e.g. when started without a terminal for input).
