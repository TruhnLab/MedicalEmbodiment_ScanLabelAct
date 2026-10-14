"""Offboard ROS container: robot relay + slam_toolbox + robot pose + exploration + a local rosbridge for the UI."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

SIM_TIME = {"use_sim_time": True}


def generate_launch_description():
    return LaunchDescription([
        # robot -> local ROS graph (uses wall time itself; it *publishes* /clock)
        ExecuteProcess(cmd=["python3", "/offboard_ros/relay.py"], output="screen"),
        Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox",
             parameters=["/offboard_ros/slam.yaml", SIM_TIME], output="screen"),
        ExecuteProcess(cmd=["python3", "/offboard_ros/robot_pose.py", "--ros-args", "-p", "use_sim_time:=true"],
                       output="screen"),
        # stack traces if slam_toolbox freezes (diagnostics only)
        ExecuteProcess(cmd=["python3", "/offboard_ros/slam_watchdog.py"], output="screen"),
        # autonomous exploration (idle until the Map view sends "start" on /explore/command)
        ExecuteProcess(cmd=["python3", "/offboard_ros/explorer.py", "--ros-args", "-p", "use_sim_time:=true"],
                       cwd="/offboard_ros", output="screen"),
        # local rosbridge for the mapping UI (port 9091 on the laptop)
        Node(package="rosbridge_server", executable="rosbridge_websocket", name="rosbridge_offboard",
             parameters=[{"port": 9091, "use_sim_time": True}], output="screen"),
        Node(package="rosapi", executable="rosapi_node", name="rosapi", parameters=[SIM_TIME]),
    ])
