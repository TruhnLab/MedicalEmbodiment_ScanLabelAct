# The robot: network, power, what the factory image runs

Notes on operating the Hiwonder LanderPi with this repository. Everything here was checked on our
robot (Raspberry Pi 5, ROS 2 Humble in the Docker container `MentorPi`, factory image).

## Network

```
[model gateway, GPU cluster] <-- VPN / LAN --> [laptop] -- dock LAN --> institute network
                                                   \---- built-in Wi-Fi --> robot hotspot "HW..." (192.168.149.1)
```

The robot stays on its own hotspot and never needs the institute network, internet or VPN. The laptop
is the hub: its built-in Wi-Fi joins the robot's hotspot, the docking station's LAN gives it everything
else.

| Robot interface | Address |
|---|---|
| rosbridge (all ROS topics) | `ws://192.168.149.1:9090` |
| MJPEG video (web_video_server) | `http://192.168.149.1:8080` (keep `/` in topic URLs literal, not `%2F`) |
| SSH | `pi@192.168.149.1` (`landerpi/robot_ssh.py`; ROS commands through `docker exec -u 1000:1000 MentorPi ...`) |

## First connection

1. Power on the robot and wait for the beep.
2. Connect the laptop's Wi-Fi to the robot hotspot (name starts with `HW`; the password is in the robot's manual).
3. Check the link: double-click `tools\check_connection.bat` (the window stays open). From a terminal,
   optionally also testing a model server:
   ```powershell
   .\tools\check_connection.ps1
   .\tools\check_connection.ps1 -VllmUrl http://<vllm-host>:8000
   ```
   Every run is also saved to `logs\`.
4. Double-click `start_mapping_console.bat` and press **Connect** (see `console/README.md`).

## Power (reboot / shutdown)

A web page can't run commands on the robot, so `tools/robot_helper.py` does it over SSH (nothing is
installed on the robot). It listens only on `127.0.0.1:8765` and accepts requests only from the console.
The buttons are in the console's ⚙ → Robot power.

- **Reboot rejoins the Wi-Fi automatically.** While the robot restarts, its hotspot disappears and Windows
  falls back to another network. If the laptop was on the robot hotspot when you pressed Reboot, the helper
  switches back to it via `netsh wlan connect` once it reappears (up to 4 min), and the console reconnects
  by itself. It never switches Wi-Fi in any other situation.
- **Shutdown stops Linux safely but does not cut power.** The controller board, lidar etc. stay on battery
  until you flip the power switch. Wait ~20 s after shutdown, then switch off.
- **Robot does not boot or crashes right after power-on:** check the battery first (2026-10-13 it was an
  empty pack). The controller board must be on before the Pi boots.

The SSH login comes from `.env`: `LANDERPI_PASSWORD` (the robot's password, see its manual) and
`LANDERPI_USER` (default `pi`); the environment variable `LANDERPI_HOST` points the tools at another address.

## What the factory image runs

rosbridge and web_video_server are already running after boot.

| What | Topic / port |
|---|---|
| Drive (Twist) | `/controller/cmd_vel` (used by the factory apps; `/cmd_vel` also works) |
| RGB camera | `/ascamera/camera_publisher/rgb0/image` (raw only, so video goes via MJPEG on port 8080) |
| Depth camera | `/ascamera/camera_publisher/depth0/image_raw`, `.../depth0/points` |
| Lidar (LD19) | `/scan` (filtered: the rear ~155° are cut out), `/scan_raw` |
| Odometry / IMU | `/odom` (EKF), `/odom_raw`, `/imu` |
| Arm command | `/servo_controller` (`servo_controller_msgs/ServosPosition`: `duration`, `position_unit: "pulse"`, `position: [{id, position}]`) |
| Arm state | `/controller_manager/servo_states` (the last *commands*); servos 1–5 = arm, 10 = gripper, range 0–1000 (500 = centre). Measured positions only via `ros_robot_controller/bus_servo/get_state` |
| Battery | `/ros_robot_controller/battery` (UInt16, mV) |

Topic names and robot facts used by the code: `landerpi/robot.py`.

Two things to avoid: calling `/rosapi/message_details` (it kills rosapi), and asking `get_state` for more
than positions (voltage etc. hung `ros_robot_controller` until a reboot).

## Starting rosbridge on the robot

If the check script reports port 9090 as closed, rosbridge is not running. SSH in (user `pi`, password from the
robot's manual):

```bash
ssh pi@192.168.149.1
docker ps                        # ROS 2 runs in a container; note its name
docker exec -it <container> bash

ros2 pkg list | grep -E "rosbridge|web_video"
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Leave that running. If no camera/lidar/velocity topics show up in the console, the robot's drivers aren't
running either (e.g. after `~/.stop_ros.sh`). Reboot the robot to restore the factory services, or start
the bringup launch file from the Hiwonder docs.

If `rosbridge_server` is not installed, the robot needs temporary internet access to run
`sudo apt install ros-humble-rosbridge-suite` (switch it to LAN mode in `wifi_conf.py`, e.g. with a phone
hotspot).

## Safety

- Test driving with the robot on the floor, with space around it and low speed limits.
- Everything that moves the robot needs **Armed** in the console; **STOP** or the space bar stops it.
- The autonomous parts stop when the lidar or odometry data are older than 0.5 s, and the relay sends a
  stop when the exploration's commands stop arriving. If the Wi-Fi drops mid-drive, the last command may
  still persist until the robot's controller times out: keep a hand near the power switch.
