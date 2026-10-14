# Offboard ROS container: ROS 2 on the laptop ("offboard" = off the robot) for everything too heavy
# for the robot: SLAM and autonomous exploration. Fed from the robot's rosbridge by relay.py.
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-slam-toolbox \
        ros-humble-rosbridge-server \
        ros-humble-tf2-ros \
        ros-humble-rosidl-runtime-py \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-opencv \
        gdb \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "websockets>=12"

WORKDIR /offboard_ros
COPY relay.py robot_pose.py slam_watchdog.py scan_fix.py scan_gate.py depth_obstacles.py coverage3d.py explorer.py explore_core.py replay_slam.py slam.yaml launch.py ./

ENV ROBOT_URL=ws://192.168.149.1:9090
EXPOSE 9091
# exec: ros2 launch becomes the main process and gets the stop signal, so every node shuts down
# cleanly (the relay closes its robot connection instead of leaving it half-open)
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && exec ros2 launch /offboard_ros/launch.py"]
