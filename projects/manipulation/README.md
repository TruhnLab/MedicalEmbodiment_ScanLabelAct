# Manipulation

The LanderPi's 5-servo arm picks up small things from the floor and puts them down, steered by the camera that sits
on the gripper. In the task console's chat it is one step of a task: the navigation finds an item and approaches it
(`projects/navigation`: `watch_for`, `search_here`, `approach_found`), this project picks it up
(`pick_up_object`), the navigation drives somewhere else, and this project puts it down (`put_down_object`). It
uses the shared detector (`landerpi/detector.py`) and needs the robot's URDF.

## Use

**Once per robot:**

1. Export the robot's URDF (Hiwonder's, not in this repository) into `config/landerpi.urdf`: on the robot,
   `docker exec -u 1000:1000 MentorPi bash -lc "ros2 param get /robot_state_publisher robot_description"`, and
   remove the leading `String value is: `. Without it everything else runs; the arm tools say the model is missing.
2. Check the gripper's range and opening (`check_arm.py gripper`, `gripper-to <pulse>` with a ruler) against
   `config/gripper.json`.
3. Teach the grasp spot: `grasp.py down` puts the open gripper at grasp height, centre a block between the jaws by
   hand, then `grasp.py teach` (writes `config/grasp_reference.json`).

**In the task console** (Connect, Armed, Localized): "find the red cube and pick it up", then "bring it to the
door and put it down". STOP or Space ends a pick with the arm where it is.

**From a script:** `grasp.py pick --prompt "red cube,red block"`, `grasp.py put`, `grasp.py up` (back to the
survey pose). Everything that moves the arm or the base needs someone watching the robot.

## How it works

- **A taught spot instead of a precise model:** the camera and the gripper are rigidly connected (only the wrist
  roll between them), so "gripper straight above the object" always looks the same in the picture. The pick steers
  until the object sits on the taught spot; errors of the arm model and the camera mounting cancel out.
- **Finding:** the detector runs in the background (~4 pictures/s) and every box goes to the console. The pick
  looks from the arm's current pose first (after `approach_found` the camera already looks at the item), then from
  lookout poses in front of the robot.
- **Following in:** the base drives by the camera (3-6 cm/s, at most 50 cm, the lidar checks the way) while the arm
  keeps the gripper over the object, until the object is ~7 cm in front of the bumper. The wheel odometry is not
  used for this: it over-reports at these speeds. Each detection carries the time its picture was taken; between
  pictures the base subtracts what it has driven since and drives only on pictures under 1.5 s old.
- **Centring:** on pictures taken with arm and base standing still, until the object is within 8 px (~4 mm) of the
  taught spot.
- **Grasping:** down along the gripper's axis (60 deg below horizontal) in guarded steps, close until the gripper
  stalls, squeeze a little, lift and carry the object in the survey pose. Putting down: ~7 cm in front of the
  bumper, open, back to the survey pose.
- **Guarded moves:** every arm step is checked for collisions with the floor and the robot's front first, and a
  servo that lags far behind its command stops the move.
- **Arm model** (`arm.py`): the URDF, 0.24 deg per servo pulse (500 = 0), the jaw tips 13 mm beyond
  `end_effector_link`, and a mounting correction of the colour camera (`config/camera_mount.json`, fitted with the
  printed calibration sheet). Under load the servos stop up to 13 pulses short of a command, so the pick
  computes the camera's pose from the measured positions.
- **Servo positions** are read with `/ros_robot_controller/bus_servo/get_state` (positions only): the robot's
  `/joint_states` and `/controller_manager/servo_states` report the last commands, not measurements.
- **In the chat** (`manipulation_tools.py`): the tools run inside the navigator and share its robot connection,
  video, detector and lidar. While the robot holds something, `nav.held` carries its mask in the camera picture,
  which the navigation leaves out of its depth obstacles and searches.
- **Recording:** every pick goes to `data/manipulation/picks/<time>/` (pictures, detections, control steps,
  localization; the last 30 are kept); `projects/presentation_video/pick_video.py` makes a video of one.

## Results and limits

| Test | Result |
|---|---|
| Chat run, 30 mm cube (2026-10-12) | picked up (52 s), put down, picked up again (49 s) |
| Script picks, cube at grasp distance up to ~25 cm beyond (2026-10-12) | 51-55 s; contact at its width (29-30 mm) |
| `test_pick_sim.py` (simulated base, servos, camera) | gripper within 1.2 cm of the object in all cases within reach |
| `test_arm.py` (model against the robot's /tf) | 0.0 mm, 0.06 deg |

- Items lying on the floor, 15-45 mm wide (the gripper opens 12-48 mm), about block height: the spot was taught
  with a 30 mm block.
- Roughly square to the robot: the wrist does not turn yet (a cube turned ~25 deg was gripped across its corners).
- The detector needs words that name the item ("red cube" found the cube at 0.8-0.9, "block" alone a sheet of paper).
- Puts down only right in front of the robot, without checking that the floor there is free.
- ~50 s per pick, most of it the slow, guarded descent and lift.
- Asking `get_state` for more than positions (voltage, temperature) hung the robot's `ros_robot_controller` until a
  reboot. After every boot the robot commands the gripper past its open stop (pulse 150), so the servo strains until
  a pick opens it.

## Files

| File | What |
|---|---|
| `pick.py` | the pick and the put-down, behind a small robot interface |
| `manipulation_tools.py` | the chat tools `pick_up_object`, `put_down_object` (inside the navigator) |
| `grasp.py` | the same from a script: `pick`, `put`, `up`; `down` / `teach` for the grasp spot |
| `test_pick_sim.py` | the approach against a simulated robot |
| `arm.py`, `test_arm.py` | kinematics, collision check, and the model's check against a recording |
| `check_arm.py` | servo checks on the robot: `health`, `gripper`, `gripper-to`, `grip`, `joints` |
| `grid_sheet.py`, `look_at_sheet.py`, `collect_views.py`, `solve_camera.py` | camera calibration with a printed A4 sheet |
| `solve_gains.py`, `touch.py`, `nudge.py` | calibration experiments: joint gains from the camera, touching targets, short drives |
