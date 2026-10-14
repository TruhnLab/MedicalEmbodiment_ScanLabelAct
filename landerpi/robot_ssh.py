"""The few light robot-side tasks we do over SSH (nothing is installed or configured permanently).

* depth compressor: republishes raw depth as lossless 16-bit PNG (~53 KB instead of 500 KB per frame)
* web_video_server restart: safety net in case it stops sending frames (not observed so far;
  an earlier "hang" was a URL-encoded topic name, which it does not decode)
* rosbridge restart: its send buffers grow without limit when data comes faster than a client
  reads it (seen at 2.3 GB after raw-image tests); then it stops answering
* health: memory/CPU of those processes (read-only)

Both are plain processes in the factory ROS container and are gone after a reboot, so callers use
the idempotent ensure_*/restart_* functions whenever they need them.
"""

import time

import paramiko

from .robot import ROBOT_HOST, TOPICS
from .settings import setting

# the login from the environment or .env; no password in the code, the repository is public (2026-10-14)
USER = setting("LANDERPI_USER", "pi")
PASSWORD = setting("LANDERPI_PASSWORD", "")

CONTAINER = "MentorPi"
# ROS processes we start must match the factory ones: same environment AND same user (uid 1000).
# Started as root, ROS 2's shared-memory transport silently delivers no data between them.
ROS_USER = "1000:1000"
ENV_FILE = "/tmp/landerpi_ros_env"
WEB_VIDEO_BIN = "/home/ubuntu/third_party_ros2/third_party_ws/install/web_video_server/lib/web_video_server/web_video_server"

DEPTH_COMPRESSOR = (
    "/opt/ros/humble/lib/image_transport/republish raw compressed --ros-args -r __node:=depth_compressor "
    f"-r in:={TOPICS['depth'][0]} -r out/compressed:={TOPICS['depth_png'][0]} "
    "-p out.format:=png -p out.png_level:=1"
)

# copy the environment of a running factory ROS process (web_video_server, else rosbridge)
_PREPARE_ENV = (
    "P=$(pgrep -f lib/web_video_server/web_video_server | head -1); "
    "[ -n \"$P\" ] || P=$(pgrep -f rosbridge_websocket | head -1); "
    f"tr '\\0' '\\n' < /proc/$P/environ | grep -v '^$' > {ENV_FILE}"
)


def run(command, sudo=False, timeout=15):
    """Runs a shell command on the robot. Returns (exit code, stdout, stderr).
    With sudo, the password goes to stdin, not the command line."""
    if not PASSWORD:
        raise RuntimeError("LANDERPI_PASSWORD is not set: the robot's SSH password, in .env (see .env.example)")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ROBOT_HOST, username=USER, password=PASSWORD, timeout=timeout,
                   allow_agent=False, look_for_keys=False)
    try:
        if sudo:
            command = f"sudo -S -p '' {command}"
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        if sudo:
            stdin.write(PASSWORD + "\n")
            stdin.flush()
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        return stdout.channel.recv_exit_status(), out, err
    finally:
        client.close()


def _start_ros_process(command, log):
    return (f"docker exec -d -u {ROS_USER} --env-file {ENV_FILE} -w /home/ubuntu {CONTAINER} "
            f"bash -c \"exec {command} > {log} 2>&1\"")


def _running(pattern):
    code, _, _ = run(f"docker exec {CONTAINER} pgrep -f '{pattern}'")
    return code == 0


def ensure_depth_compressor(wait_s=8):
    """Starts the PNG depth compressor unless it already runs. Returns 'running' or 'started'."""
    if _running("republish raw compressed"):
        return "running"
    code, _, err = run(_PREPARE_ENV + " && " + _start_ros_process(DEPTH_COMPRESSOR, "/tmp/landerpi_depth_compressor.log"))
    if code != 0:
        raise RuntimeError(f"could not start the depth compressor: {err}")
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _running("republish raw compressed"):
            return "started"
        time.sleep(1)
    raise RuntimeError("the depth compressor did not start")


ROSBRIDGE_PATTERN = "rosbridge_websocket_launch|lib/rosbridge_server/rosbridge_websocket|lib/rosapi/rosapi_node"


def _pid(pattern):
    _, out, _ = run(f"pgrep -f '{pattern}' | head -1")
    return out or None


# rosbridge pings its clients and drops those that stop answering. Without this (factory default:
# off), a client that disappears without closing its connection (killed container, sleeping laptop,
# Wi-Fi drop) keeps receiving into ever-growing buffers until rosbridge hangs (seen at 2.8 and 4 GB).
# Also: service calls in their own thread with a timeout. By default (Humble) rosbridge runs them on
# its main loop and waits forever; one slow/unanswered call (e.g. rosapi restarting) blocks all
# traffic, messages pile up in memory and rosbridge hangs (seen at 2.4 GB with 4 connections stuck
# in CLOSE-WAIT while pings were on).
ROSBRIDGE_CMD = ("/opt/ros/humble/lib/rosbridge_server/rosbridge_websocket --ros-args -r __node:=rosbridge_websocket "
                 "-p port:=9090 -p websocket_ping_interval:=5 -p websocket_ping_timeout:=15 "
                 "-p call_services_in_new_thread:=true -p default_call_service_timeout:=5.0")
ROSAPI_CMD = "/opt/ros/humble/lib/rosapi/rosapi_node --ros-args -r __node:=rosapi"


def rosbridge_has_pings():
    code, _, _ = run("pgrep -f 'call_services_in_new_thread:=true'")
    return code == 0


def restart_rosbridge(wait_s=25):
    """Restarts rosbridge (+ rosapi), with client pings enabled (see ROSBRIDGE_CMD).
    A hung rosbridge ignores SIGTERM (busy in C code), so it gets SIGKILL after a grace period."""
    old = _pid("lib/rosbridge_server/rosbridge_websocket")
    code, _, err = run(
        _PREPARE_ENV
        + f" && docker exec {CONTAINER} pkill -f '{ROSBRIDGE_PATTERN}'; sleep 3; "
        + f"docker exec {CONTAINER} pkill -9 -f '{ROSBRIDGE_PATTERN}'; sleep 1; "
        + _start_ros_process(ROSBRIDGE_CMD, "/tmp/landerpi_rosbridge.log") + "; "
        + _start_ros_process(ROSAPI_CMD, "/tmp/landerpi_rosapi.log"))
    if code != 0:
        raise RuntimeError(f"could not restart rosbridge: {err}")
    deadline = time.time() + wait_s
    while time.time() < deadline:
        new = _pid("lib/rosbridge_server/rosbridge_websocket")
        port_open, _, _ = run("timeout 2 bash -c 'echo > /dev/tcp/127.0.0.1/9090'")
        if new and new != old and port_open == 0:
            return "restarted"
        time.sleep(1)
    raise RuntimeError("rosbridge did not come back")


# Factory nodes that cost CPU without being useful to us. joystick_control (wireless gamepad) spins
# at ~100% of one core when no gamepad is connected (measured 2026-09-30); it also publishes to
# /controller/cmd_vel. Stopping it is not permanent: bringup starts it again after a reboot.
IDLE_FACTORY_NODES = {"joystick_control": "joystick_control"}


def stop_idle_factory_nodes():
    """Stops the idle factory nodes above. Returns the names that were running."""
    stopped = []
    for name, pattern in IDLE_FACTORY_NODES.items():
        if _running(pattern):
            # a node busy-looping ignores SIGTERM; try SIGINT (ROS's normal shutdown), then SIGKILL
            run(f"docker exec {CONTAINER} pkill -INT -f '{pattern}'; sleep 2; "
                f"docker exec {CONTAINER} pkill -9 -f '{pattern}'")
            if _running(pattern):
                raise RuntimeError(f"could not stop {name}")
            stopped.append(name)
    return stopped


# Read-only: the robot's fixed transforms. They are published once at startup ("latched",
# transient-local), so a rosbridge restarted later never forwards them; this subscribes the way
# latched topics need, inside the robot's ROS container, and prints them as JSON.
_STATIC_TF_SCRIPT = r'''
import json, time, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
rclpy.init(); node = rclpy.create_node("landerpi_static_tf_dump"); got = {}
def cb(msg):
    for t in msg.transforms:
        got[t.child_frame_id] = {"header": {"stamp": {"sec": t.header.stamp.sec, "nanosec": t.header.stamp.nanosec},
            "frame_id": t.header.frame_id}, "child_frame_id": t.child_frame_id, "transform": {
            "translation": {k: getattr(t.transform.translation, k) for k in "xyz"},
            "rotation": {k: getattr(t.transform.rotation, k) for k in "xyzw"}}}
qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
node.create_subscription(TFMessage, "/tf_static", cb, qos)
# each publisher is discovered separately (sometimes after >3 s): wait for both cameras, max 12 s
need = {"depth_camera_link", "rgb_camera_link"}
start = time.time()
while time.time() - start < 12 and not (need <= set(got) and time.time() - start > 3):
    rclpy.spin_once(node, timeout_sec=0.2)
print(json.dumps(list(got.values())))
'''


def static_transforms():
    """All static transforms of the robot as rosbridge-style dicts (read-only)."""
    import base64
    import json
    script = base64.b64encode(_STATIC_TF_SCRIPT.encode()).decode()
    code, out, err = run(
        _PREPARE_ENV + f" && echo {script} | base64 -d > /tmp/landerpi_static_tf.py && "
        f"docker cp /tmp/landerpi_static_tf.py {CONTAINER}:/tmp/landerpi_static_tf.py && "
        f"docker exec -u {ROS_USER} --env-file {ENV_FILE} {CONTAINER} python3 /tmp/landerpi_static_tf.py", timeout=30)
    if code != 0:
        raise RuntimeError(f"could not read static transforms: {err[-300:]}")
    return json.loads(out.splitlines()[-1])


def health():
    """Read-only snapshot: {name: {"rss_mb", "cpu"}} for our key processes, plus the load average."""
    patterns = {"rosbridge": "lib/rosbridge_server/rosbridge_websocket",
                "web_video_server": "lib/web_video_server/web_video_server",
                "depth_compressor": "republish raw compressed"}
    script = "; ".join(
        f"P=$(pgrep -f '{pat}' | head -1); [ -n \"$P\" ] && echo {name} $(ps -o rss=,%cpu= -p $P)"
        for name, pat in patterns.items()) + "; echo load $(cut -d' ' -f1 /proc/loadavg)"
    _, out, _ = run(script)
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "load":
            result["load"] = float(parts[1])
        elif len(parts) == 3:
            result[parts[0]] = {"rss_mb": round(int(parts[1]) / 1024), "cpu": float(parts[2])}
    return result


def restart_web_video_server():
    """Kills the (possibly hung) web_video_server and starts a fresh one."""
    code, _, err = run(
        _PREPARE_ENV
        + f" && docker exec {CONTAINER} pkill -f lib/web_video_server/web_video_server; sleep 1; "
        + _start_ros_process(WEB_VIDEO_BIN, "/tmp/landerpi_web_video.log"))
    if code != 0:
        raise RuntimeError(f"could not restart web_video_server: {err}")
    return "restarted"
