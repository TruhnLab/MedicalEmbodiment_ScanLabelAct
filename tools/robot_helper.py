"""Local helper for the LanderPi consoles: serves the console pages, relays the robot's video, and
reboots / shuts down the robot or restarts its video server if it stops sending.

Mapping console: http://127.0.0.1:8765/ (start_mapping_console.bat opens it).
Task console (projects/navigation): http://127.0.0.1:8765/tasks.html (start_task_console.bat).
Both use this one helper; tools/start_console.ps1 starts it only if it isn't running with the current code.

A web page cannot run commands on the robot, so the console asks this helper
(running on the laptop), which runs them over SSH. Nothing is installed on the robot.

    conda activate landerpi
    python tools/robot_helper.py            # listens on http://127.0.0.1:8765

Robot login: LANDERPI_USER (default pi) and LANDERPI_PASSWORD from the environment or .env;
LANDERPI_HOST (environment) for another address.
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import paramiko

import console_vendor
from landerpi.mjpeg import MjpegReader
from landerpi.robot import ROBOT_HOST as HOST
from landerpi.robot import TOPICS
from landerpi.robot_ssh import (USER, ensure_depth_compressor, restart_rosbridge, restart_web_video_server, rosbridge_has_pings,
                                stop_idle_factory_nodes)
from landerpi.robot_ssh import run as run_on_robot

# Only the console opened from disk (origin "null") or localhost may call us, and every call must
# carry this header. That forces a CORS preflight, so other websites can't trigger a reboot.
ALLOWED_ORIGINS = {"null", "http://localhost", "http://127.0.0.1"}
REQUIRED_HEADER = "X-LanderPi-Console"

CONSOLE_DIR = (Path(__file__).resolve().parents[1] / "console").resolve()
# the consoles' JS libraries are downloaded at start if missing (console_vendor.py); requests for them wait
# for that, so a page opened right away doesn't get a 404 on the first start
VENDOR_READY = threading.Event()
vendor_missing = []
OFFBOARD_COMPOSE = Path(__file__).resolve().parents[1] / "offboard_ros" / "docker-compose.yml"

# offboard ROS container (offboard_ros/) controls; "reset" throws the current map away
SLAM_COMMANDS = {
    "start": ["up", "-d"],
    "stop": ["stop"],
    "reset": ["up", "-d", "--force-recreate"],
}


# ---------- recorder (projects/mapping/record.py) as a background process ----------

REPO = Path(__file__).resolve().parents[1]
RECORDER_STOP_URL = "http://127.0.0.1:8766/stop"
_recorder = None  # subprocess.Popen while a recording runs


def recorder_action(action):
    """start: launches record.py (log in logs/). stop: asks it to finish cleanly (it writes its
    remaining keyframes and meta.json). Returns (ok, message)."""
    global _recorder
    running = _recorder is not None and _recorder.poll() is None
    if action == "start":
        if running:
            return False, "a recording is already running"
        (REPO / "logs").mkdir(exist_ok=True)
        log = open(REPO / "logs" / f"recorder_{time.strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8")
        _recorder = subprocess.Popen([sys.executable, "-u", str(REPO / "projects" / "mapping" / "record.py")],
                                     cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True, f"recorder started (pid {_recorder.pid}, log {log.name})"
    if action == "stop":
        import urllib.request
        try:
            req = urllib.request.Request(RECORDER_STOP_URL, method="POST", headers={"X-LanderPi-Console": "1"})
            urllib.request.urlopen(req, timeout=5).close()
            return True, "recorder is finishing"
        except OSError as e:
            return False, f"no recording to stop ({e})"
    return False, "unknown action"


# ---------- recorded sessions and their 3D reconstruction (projects/mapping/reconstruct.py) ----------

SESSIONS_DIR = REPO / "data" / "sessions"
SESSION_FILES = {".ply", ".png", ".csv"}  # what the 3D view may download from a session
SCENE_FILES = {".json", ".bin", ".jpg"}  # ... and from its scene/ folder (projects/scene_understanding)
_reconstructions = {}  # session name -> Popen
_labelings = {}  # session name -> Popen (projects/scene_understanding/run.py)


def list_sessions():
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions
    for d in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        meta_file = d / "meta.json"
        if not d.is_dir() or not meta_file.exists():
            continue  # still recording (meta.json is written at the end)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        rec = d / "reconstruction"
        job = _reconstructions.get(d.name)
        running = job is not None and job.poll() is None
        failed = job is not None and job.poll() not in (None, 0)
        scene = d / "scene"
        lab = _labelings.get(d.name)
        labeling = lab is not None and lab.poll() is None
        label_failed = lab is not None and lab.poll() not in (None, 0)
        labels = (scene / "scene.json").exists() and (scene / "vertex_objects.bin").exists()
        labels_time = (scene / "vertex_objects.bin").stat().st_mtime if labels else None
        sessions.append({
            "name": d.name,
            "duration_s": meta.get("duration_s"),
            "keyframes": meta.get("counts", {}).get("keyframes"),
            "slam": (d / "streams" / "slam_map_odom.jsonl").exists(),
            "mesh": (rec / "mesh.ply").exists(),
            "mesh_time": (rec / "mesh.ply").stat().st_mtime if (rec / "mesh.ply").exists() else None,
            "mesh_view": (rec / "mesh_view.ply").exists(),  # lighter copy for the browser
            "cloud": (rec / "cloud.ply").exists(),
            "reconstructing": running,
            "progress": _read_json(rec / "progress.json") if running else None,
            "failed": failed,
            "error": _last_error(d.name) if failed else None,
            "labels": labels,
            "labels_time": labels_time,
            # labels belong to one mesh (an object id per vertex): a newer reconstruction needs new labels
            "labels_stale": bool(labels and (rec / "mesh_view.ply").exists()
                                 and (rec / "mesh_view.ply").stat().st_mtime > labels_time),
            "labeling": labeling,
            "label_progress": _read_json(scene / "progress.json") if labeling else None,
            "label_failed": label_failed,
            "label_error": _last_error(d.name, "label") if label_failed else None,
            # a room of the task console (projects/navigation/make_room.py) is made from this session
            "in_rooms": (REPO / "data" / "rooms" / d.name / "room.json").exists(),
        })
    return sessions


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _last_error(name, job="reconstruct"):
    """The last line of a failed reconstruction's (or labelling's) log (the exception message)."""
    try:
        lines = (REPO / "logs" / f"{job}_{name}.log").read_text(encoding="utf-8").strip().splitlines()
        return lines[-1][-300:] if lines else None
    except OSError:
        return None


def delete_session(name):
    """Deletes a recorded session with its reconstruction (the console asks for confirmation)."""
    session = SESSIONS_DIR / name
    if "/" in name or "\\" in name or ".." in name or not session.is_dir():
        return False, "unknown session"
    if not (session / "meta.json").exists():
        return False, "still recording (or an unfinished recording): stop it first"
    job = _reconstructions.get(name)
    if job is not None and job.poll() is None:
        return False, "being reconstructed right now"
    lab = _labelings.get(name)
    if lab is not None and lab.poll() is None:
        return False, "being labelled right now"
    try:
        shutil.rmtree(session)
    except OSError as e:  # e.g. a file is open in another program
        return False, f"could not delete everything: {e}"
    _reconstructions.pop(name, None)
    _labelings.pop(name, None)
    (REPO / "logs" / f"reconstruct_{name}.log").unlink(missing_ok=True)
    (REPO / "logs" / f"label_{name}.log").unlink(missing_ok=True)
    return True, f"deleted {name}"


def stop_job(jobs, name, what):
    """Ends a running reconstruction or labelling with all its child processes (run.py starts one per step).
    A labelling's Slurm job, if one is waiting or running (scene/slurm_job, detect_frames.py), is cancelled
    too. Files written so far stay; the next run overwrites them."""
    job = jobs.get(name)
    if job is None or job.poll() is not None:
        return False, f"no {what} running"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(job.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        job.kill()
    job.wait(timeout=10)
    jobs.pop(name, None)  # stopped on purpose: not shown as "failed"
    slurm = SESSIONS_DIR / name / "scene" / "slurm_job"
    if what == "labelling" and slurm.exists():
        job_id = slurm.read_text().strip()
        slurm.unlink(missing_ok=True)

        def cancel():
            try:
                sys.path.insert(0, str(REPO / "tools"))
                import gpu_cluster as gc
                gc.ssh(gc.MASTER, f"scancel {job_id}")
                print(f"label: cancel sent for Slurm job {job_id}", flush=True)
            except Exception as e:
                print(f"label: could not cancel Slurm job {job_id}: {e}", flush=True)
        threading.Thread(target=cancel, daemon=True).start()
    return True, f"{what} of {name} stopped"


def start_reconstruction(name):
    session = SESSIONS_DIR / name
    if "/" in name or "\\" in name or not (session / "meta.json").exists():
        return False, "unknown session"
    job = _reconstructions.get(name)
    if job is not None and job.poll() is None:
        return False, "already reconstructing"
    (REPO / "logs").mkdir(exist_ok=True)
    try:
        (session / "reconstruction" / "progress.json").unlink(missing_ok=True)  # from an earlier run
    except OSError:
        pass  # locked by a reader for a moment: the new run overwrites it anyway
    log = open(REPO / "logs" / f"reconstruct_{name}.log", "w", encoding="utf-8")
    _reconstructions[name] = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "projects" / "mapping" / "reconstruct.py"), str(session)],
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return True, f"reconstructing {name} (log {log.name})"


def start_labeling(name):
    """Objects of a reconstructed session (projects/scene_understanding/run.py): the VLM runs from the
    laptop (the vLLM gateway is not reachable from the cluster nodes), YOLOE as a Slurm job (on the laptop
    if the cluster fails)."""
    session = SESSIONS_DIR / name
    if "/" in name or "\\" in name or not (session / "meta.json").exists():
        return False, "unknown session"
    if not (session / "reconstruction" / "mesh_view.ply").exists():
        return False, "reconstruct it first"
    job = _reconstructions.get(name)
    if job is not None and job.poll() is None:
        return False, "being reconstructed right now"
    lab = _labelings.get(name)
    if lab is not None and lab.poll() is None:
        return False, "already labelling"
    (REPO / "logs").mkdir(exist_ok=True)
    log = open(REPO / "logs" / f"label_{name}.log", "w", encoding="utf-8")
    _labelings[name] = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "projects" / "scene_understanding" / "run.py"), name],
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        # the log is UTF-8: printing a VLM answer with a character outside cp1252 ended a run (2026-10-09)
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return True, f"labelling {name} (log {log.name})"


# ---------- task console (projects/navigation): rooms and the navigator process ----------

ROOMS_DIR = REPO / "data" / "rooms"
NAVIGATOR_URL = "http://127.0.0.1:8767"
_navigator = None  # subprocess.Popen started by this helper
_navigator_started = 0.0
# a started navigator answers after ~3-5 s (imports, room); a start request meanwhile must not start a second
# one. Later, only an answer counts: a kept process handle said "running" after its python had been killed,
# and the console never got its navigator back (2026-10-07)
NAVIGATOR_STARTUP_S = 15.0
HELPER_PORT = 8765  # set in main()


def list_rooms():
    """Rooms made by projects/navigation/make_room.py, newest first, with what their session offers."""
    rooms = []
    for f in sorted(ROOMS_DIR.glob("*/room.json"), reverse=True):
        info = _read_json(f) or {}
        s = SESSIONS_DIR / info.get("session", "")
        mesh, ids = s / "reconstruction" / "mesh_view.ply", s / "scene" / "vertex_objects.bin"
        labels = (s / "scene" / "scene.json").exists() and ids.exists()
        # counted now: the session's labels may have been rebuilt since make_room.py
        objects = len((_read_json(s / "scene" / "scene.json") or {}).get("objects", [])) if labels else 0
        rooms.append({**info, "name": f.parent.name, "objects": objects, "mesh": mesh.exists(), "labels": labels,
                      "labels_time": ids.stat().st_mtime if labels else None,
                      "labels_stale": bool(labels and mesh.exists() and mesh.stat().st_mtime > ids.stat().st_mtime)})
    return rooms


def make_room(session, title):
    """projects/navigation/make_room.py for a labelled session (replaces its room; ~1 s)."""
    if "/" in session or "\\" in session or ".." in session or not (SESSIONS_DIR / session / "meta.json").exists():
        return False, "unknown session"
    cmd = [sys.executable, str(REPO / "projects" / "navigation" / "make_room.py"), session]
    if title:
        cmd += ["--title", title]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    out = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, out[-1][-300:] if out else ""


def delete_room(name):
    """Removes data/rooms/<name> (the session and its labels stay)."""
    room = ROOMS_DIR / name
    if "/" in name or "\\" in name or ".." in name or not (room / "room.json").exists():
        return False, "unknown room"
    try:
        shutil.rmtree(room)
    except OSError as e:
        return False, f"could not delete: {e}"
    return True, f"room {name} removed"


# ---------- task runs: the task console's ● Record recordings (projects/navigation/taskrun.py) ----------

TASK_RUNS_DIR = REPO / "data" / "task_runs"
TASK_RUN_FILES = {".mp4", ".jpg"}  # what the browser may get from a run (the video and its thumbnail)


def list_task_runs():
    runs = []
    for m in sorted(TASK_RUNS_DIR.glob("*/meta.json"), reverse=True):
        if m.parent.name.startswith("_"):
            continue  # the rolling last search
        meta = _read_json(m) or {}
        runs.append({**meta, "name": m.parent.name, "video": (m.parent / "video.mp4").exists(),
                     "thumb": (m.parent / "thumb.jpg").exists()})
    return runs


def task_run_action(op, name, on=True):
    run = TASK_RUNS_DIR / name
    if "/" in name or "\\" in name or ".." in name or name.startswith("_") or not (run / "meta.json").exists():
        return False, "unknown run"
    if op == "delete":
        meta = _read_json(run / "meta.json") or {}
        if meta.get("keep"):
            return False, "this run is starred (keep): unstar it first"
        try:
            shutil.rmtree(run)
        except OSError as e:
            return False, f"could not delete everything: {e}"
        return True, f"deleted {name}"
    if op == "keep":
        meta = _read_json(run / "meta.json") or {}
        meta["keep"] = bool(on)
        (run / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return True, f"{name} {'starred' if on else 'unstarred'}"
    return False, "unknown action"


def navigator_running():
    """The running navigator's code version (navigator.code_version; 0 for one too old to say), or None."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{NAVIGATOR_URL}/state", timeout=2) as r:
            return int(json.load(r).get("code") or 0)
    except (OSError, ValueError):
        return None


def navigator_code_version():
    """As navigator.code_version(), without importing the navigator (torch)."""
    nav = REPO / "projects" / "navigation"
    files = [*nav.glob("*.py"), *(REPO / "landerpi").glob("*.py"), REPO / "offboard_ros" / "explore_core.py",
             REPO / "offboard_ros" / "depth_obstacles.py", REPO / "offboard_ros" / "coverage3d.py",
             *(REPO / "projects" / "manipulation").glob("*.py")]
    return int(max(f.stat().st_mtime for f in files if f.exists()))


def navigator_action(action):
    """start: launches navigator.py unless one answers already (e.g. started by an earlier helper).
    stop: asks it to stop the robot and exit."""
    global _navigator, _navigator_started
    if action == "start":
        running = navigator_running()
        if running is not None and running >= navigator_code_version():
            return True, "navigator already running"
        if running is not None:  # older code: stop it (it stops the robot), start the current one
            navigator_action("stop")
            time.sleep(1.5)
        alive = _navigator is not None and _navigator.poll() is None
        if alive and time.time() - _navigator_started < NAVIGATOR_STARTUP_S:
            return True, "navigator starting"
        if alive:  # a process that doesn't answer (hung, or its python child was killed): replace it
            _navigator.kill()
        (REPO / "logs").mkdir(exist_ok=True)
        log = open(REPO / "logs" / f"navigator_{time.strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8")
        _navigator = subprocess.Popen([sys.executable, "-u", str(REPO / "projects" / "navigation" / "navigator.py")],
                                      cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                      # its live detection reads the video through this helper's relay (live.py)
                                      env={**os.environ, "LANDERPI_HELPER_PORT": str(HELPER_PORT)},
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _navigator_started = time.time()
        return True, f"navigator started (pid {_navigator.pid}, log {log.name})"
    if action == "stop":
        import urllib.request
        try:
            req = urllib.request.Request(f"{NAVIGATOR_URL}/quit", method="POST", headers={REQUIRED_HEADER: "1"})
            urllib.request.urlopen(req, timeout=5).close()
        except OSError:
            pass  # exits while answering
        return True, "navigator stopped"
    return False, "unknown action"


def code_version():
    """Newest change of the helper's code: start_console.ps1 reuses a running helper only if it matches."""
    files = [Path(__file__), *(REPO / "landerpi").glob("*.py")]
    return int(max(f.stat().st_mtime for f in files))


CODE_VERSION = code_version()


def start_depth_compressor():
    try:
        print(f"depth compressor: {ensure_depth_compressor()}", flush=True)
    except Exception as e:  # the recorder tries again when it starts
        print(f"depth compressor: {e}", flush=True)


def slam_action(action):
    """Runs docker compose for the offboard ROS container. Returns (ok, output)."""
    try:
        p = subprocess.run(["docker", "compose", "-f", str(OFFBOARD_COMPOSE), *SLAM_COMMANDS[action]],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"docker not available: {e}"
    return p.returncode == 0, (p.stderr or p.stdout).strip()[-300:]
# Windows' registry can map .js to text/plain; be explicit
for ext, mime in {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.items():
    mimetypes.add_type(mime, ext)

COMMANDS = {
    "reboot": "systemctl reboot",
    "shutdown": "systemctl poweroff",
}


def check_robot():
    """Proves SSH login and sudo work, without changing anything."""
    # "sudo true" checks the password works; hostname/uptime run as the normal user
    code, out, err = run_on_robot("true && hostname && uptime -p", sudo=True)
    return code == 0, out or err


def power_action(action):
    """Sends reboot/poweroff. The SSH session usually drops as the robot goes down."""
    try:
        code, _, err = run_on_robot(COMMANDS[action], sudo=True)
        return code == 0, err
    except (paramiko.SSHException, EOFError, OSError):
        # connection closed by the shutting-down robot: that's success
        return True, ""


# ---------- Wi-Fi: rejoin the robot hotspot after a reboot ----------
# While the robot reboots its hotspot disappears and Windows falls back to another network
# (e.g. eduroam). If the laptop was on the robot's hotspot before, switch back once it reappears.

def current_ssid():
    """SSID of the connected Wi-Fi, or None. Parses `netsh wlan show interfaces` (any UI language)."""
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                             text=True, errors="replace", timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"^\s*SSID\s*:\s*(.+?)\s*$", out, re.MULTILINE)
    return match.group(1) if match else None


def rejoin_wifi(ssid, timeout_s=240, interval_s=5):
    """Keeps asking Windows to join `ssid` until it is connected again (or we give up)."""
    time.sleep(20)  # give the robot time to actually go down
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if current_ssid() == ssid:
            print(f"Wi-Fi: back on {ssid}")
            return
        # fails harmlessly while the hotspot isn't visible yet; uses the saved Windows profile
        subprocess.run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"],
                       capture_output=True, timeout=15)
        time.sleep(interval_s)
    print(f"Wi-Fi: could not rejoin {ssid} within {timeout_s} s")


# ---------- video: one robot stream per (topic, quality), newest frame to every viewer ----------
# web_video_server's MJPEG has no congestion control: if a reader is slower than the camera, frames
# pile up in TCP buffers and latency grows over time. So we read the robot stream as fast as it comes
# (MjpegReader also drops the duplicate frames web_video_server sends) and hand each browser only
# the newest frame whenever it is ready for one.

BOUNDARY = "frame"


class VideoHub:
    _hubs = {}
    _lock = threading.Lock()

    def __init__(self, topic, quality):
        self.key = (topic, quality)
        self.viewers = 0
        self._cond = threading.Condition()
        self._frame, self._stamp, self._id = None, None, 0
        self.reader = MjpegReader(self._on_frame, topic=topic, quality=quality).start()

    def _on_frame(self, jpeg, stamp, recv):
        with self._cond:
            self._frame, self._stamp = jpeg, stamp
            self._id += 1
            self._cond.notify_all()

    def wait_newer(self, last_id, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self._id != last_id, timeout=timeout)
            return self._id, self._frame, self._stamp

    @classmethod
    def acquire(cls, topic, quality):
        with cls._lock:
            hub = cls._hubs.get((topic, quality)) or cls._hubs.setdefault((topic, quality), cls(topic, quality))
            hub.viewers += 1
            return hub

    @classmethod
    def release(cls, hub):
        with cls._lock:
            hub.viewers -= 1
        # keep the robot stream a few seconds, e.g. across a page reload
        threading.Timer(5.0, cls._close_if_unused, args=(hub,)).start()

    @classmethod
    def _close_if_unused(cls, hub):
        with cls._lock:
            if hub.viewers > 0 or cls._hubs.get(hub.key) is not hub:
                return
            del cls._hubs[hub.key]
        hub.reader.stop()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS or origin.startswith(("http://localhost:", "http://127.0.0.1:")):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", REQUIRED_HEADER)
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def _reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/video":
            return self._proxy_video()
        if path.startswith("/data/task_runs/"):
            return self._task_run_file(path)
        if path == "/api/task_runs":
            return self._reply(200, {"runs": list_task_runs()})
        if path.startswith("/data/sessions/"):
            return self._session_file(path)
        if path == "/api/sessions":
            return self._reply(200, {"sessions": list_sessions()})
        if path == "/api/rooms":
            return self._reply(200, {"rooms": list_rooms()})
        if path == "/api/version":
            return self._reply(200, {"version": CODE_VERSION})
        if path == "/api/vendor":
            return self._reply(200, {"ready": VENDOR_READY.is_set(), "missing": vendor_missing})
        if path != "/api/status":
            return self._static(path)
        if REQUIRED_HEADER not in self.headers:
            return self._reply(403, {"error": "missing header"})
        try:
            ok, info = check_robot()
            self._reply(200, {"helper": True, "robot": ok, "info": info, "host": HOST, "wifi": current_ssid()})
        except Exception as e:  # robot off, wrong network, wrong password, ...
            self._reply(200, {"helper": True, "robot": False, "info": str(e), "host": HOST, "wifi": current_ssid()})

    def do_POST(self):
        action = self.path.removeprefix("/api/")
        slam = action.removeprefix("slam/") if action.startswith("slam/") else None
        rec = action.removeprefix("recorder/") if action.startswith("recorder/") else None
        recon = action.removeprefix("reconstruct/") if action.startswith("reconstruct/") else None
        delete = action.removeprefix("delete-session/") if action.startswith("delete-session/") else None
        label = action.removeprefix("label/") if action.startswith("label/") else None
        nav = action.removeprefix("navigator/") if action.startswith("navigator/") else None
        # rooms (projects/navigation): room/make/<session>?title=..., room/delete/<room>
        run_op, _, run_arg = urlsplit(action).path.removeprefix("task_runs/").partition("/") \
            if action.startswith("task_runs/") else (None, "", "")
        room_op, _, room_arg = urlsplit(action).path.removeprefix("room/").partition("/") \
            if action.startswith("room/") else (None, "", "")
        if (action != "restart-video" and action not in COMMANDS and slam not in SLAM_COMMANDS
                and rec not in ("start", "stop") and not recon and not delete and not label
                and nav not in ("start", "stop") and not (room_op in ("make", "delete") and room_arg)
                and not (run_op in ("delete", "keep") and run_arg)):
            return self._reply(404, {"error": "not found"})
        if REQUIRED_HEADER not in self.headers:
            return self._reply(403, {"error": "missing header"})
        if run_op:
            on = parse_qs(urlsplit(action).query).get("on", ["1"])[0] != "0"
            ok, message = task_run_action(run_op, run_arg, on)
            print(f"task run {run_op}: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message})
        if room_op:
            if room_op == "make":
                title = parse_qs(urlsplit(action).query).get("title", [""])[0].strip()[:80]
                ok, message = make_room(room_arg, title)
            else:
                ok, message = delete_room(room_arg)
            print(f"room {room_op}: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message, "message": message})
        if nav:
            ok, message = navigator_action(nav)
            print(f"navigator {nav}: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message})
        if delete:
            ok, message = delete_session(delete)
            print(f"delete-session: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message})
        if label:
            name, _, op = label.partition("/")
            ok, message = stop_job(_labelings, name, "labelling") if op == "stop" else start_labeling(label)
            print(f"label: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message})
        if recon:
            name, _, op = recon.partition("/")
            ok, message = stop_job(_reconstructions, name, "reconstruction") if op == "stop" else start_reconstruction(recon)
            print(f"reconstruct: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message})
        if rec:
            ok, message = recorder_action(rec)
            print(f"recorder {rec}: {message}", flush=True)
            return self._reply(200 if ok else 409, {"ok": ok, "error": None if ok else message, "message": message})
        if slam:
            ok, out = slam_action(slam)
            print(f"slam {slam}: {'ok' if ok else 'failed ' + out}")
            if ok and slam in ("start", "reset"):
                # depth images from the start: the exploration uses them for obstacles and coverage, and
                # started without them for ~10 s while the recorder started the compressor (2026-10-06)
                threading.Thread(target=start_depth_compressor, daemon=True).start()
            return self._reply(200 if ok else 502, {"ok": ok, "error": None if ok else out})
        if action == "restart-video":
            # safety net in case the robot video server stops sending frames
            try:
                restart_web_video_server()
                print("restart-video: done")
                return self._reply(200, {"ok": True})
            except Exception as e:
                return self._reply(502, {"ok": False, "error": str(e)})
        ssid = current_ssid()
        try:
            ok, err = power_action(action)
        except Exception as e:
            return self._reply(502, {"ok": False, "error": str(e)})
        print(f"{action}: {'sent' if ok else 'failed ' + err}")
        # only switch Wi-Fi back if we were on the robot's own hotspot (factory SSIDs start with "HW")
        rejoin = ok and action == "reboot" and bool(ssid) and ssid.startswith("HW")
        if rejoin:
            print(f"Wi-Fi: will rejoin {ssid} when the robot is back")
            threading.Thread(target=rejoin_wifi, args=(ssid,), daemon=True).start()
        self._reply(200 if ok else 502, {"ok": ok, "error": err, "rejoinWifi": ssid if rejoin else None})

    def _static(self, path):
        """Serves the console. Loaded from here (not as a file), the page may show the robot's video:
        browsers block requests from pages to local-network devices ("Local Network Access")."""
        if path.startswith("/vendor/"):
            VENDOR_READY.wait(120)
        target = (CONSOLE_DIR / (path.lstrip("/") or "index.html")).resolve()
        if CONSOLE_DIR not in target.parents or not target.is_file():
            return self._reply(404, {"error": "not found"})
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")  # always the current version, no Ctrl+F5 needed
        self.end_headers()
        self.wfile.write(data)

    def _task_run_file(self, path):
        """A run's video or thumbnail, with HTTP ranges (the browser's video player jumps around with them)."""
        target = (REPO / path.lstrip("/")).resolve()
        runs = TASK_RUNS_DIR.resolve()
        if runs not in target.parents or target.suffix not in TASK_RUN_FILES or not target.is_file():
            return self._reply(404, {"error": "not found"})
        size = target.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            start = int(m[1]) if m[1] else max(0, size - int(m[2]))
            end = min(size - 1, int(m[2])) if m[1] and m[2] else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4" if target.suffix == ".mp4" else "image/jpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with open(target, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (OSError, ConnectionError):
            pass  # the player closed the connection (normal when it jumps)

    def _session_file(self, path):
        """Serves reconstruction results (ply/png/csv) of recorded sessions to the 3D view, and the
        labels (scene/: json, per-vertex object ids, keyframe images with masks)."""
        target = (REPO / path.lstrip("/")).resolve()
        sessions = SESSIONS_DIR.resolve()
        in_scene = sessions in target.parents and target.relative_to(sessions).parts[1:2] == ("scene",)
        allowed = target.suffix in SESSION_FILES or (in_scene and target.suffix in SCENE_FILES)
        if sessions not in target.parents or not allowed or not target.is_file():
            return self._reply(404, {"error": "not found"})
        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream" if target.suffix == ".ply"
                         else mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(target, "rb") as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def _proxy_video(self):
        """MJPEG to the browser, always the newest frame: never a backlog, so latency can't build up.
        Query: topic=..., quality=... (like web_video_server)."""
        params = parse_qs(urlsplit(self.path).query)
        topic = params.get("topic", [TOPICS["rgb"][0]])[0]
        quality = max(10, min(100, int(params.get("quality", ["80"])[0])))
        hub = VideoHub.acquire(topic, quality)
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            while True:
                frame_id, frame, stamp = hub.wait_newer(last, timeout=5)
                if frame is None:
                    continue  # no frame yet (robot starting up); keep the connection
                last = frame_id
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nX-Timestamp: {stamp:.6f}\r\n"
                                 f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame + b"\r\n")
        except (OSError, ConnectionError):
            pass  # browser closed the image
        finally:
            VideoHub.release(hub)

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; actions are printed above


class HelperServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process listen on the same port, and requests then land
    # on either helper, e.g. an old version ("Recorder start failed: not found"). Never share the port.
    allow_reuse_address = False


def pids_listening_on(port):
    """PIDs listening on 127.0.0.1:<port> (Windows netstat)."""
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, errors="replace").stdout
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        # listening sockets have no remote address (the state name depends on the Windows language)
        if len(parts) == 5 and parts[1].endswith(f":{port}") and parts[2] in ("0.0.0.0:0", "[::]:0"):
            pids.add(int(parts[4]))
    return pids


def bind_replacing_old_helper(port):
    """Starts listening on the port; an already running helper (possibly older code) is stopped first,
    so starting a console .bat always gives you the current version (start_console.ps1 only starts
    the helper if the running one has older code)."""
    for attempt in range(2):
        try:
            return HelperServer(("127.0.0.1", port), Handler)
        except OSError:
            if attempt:
                raise
            old = pids_listening_on(port) - {os.getpid()}
            print(f"Another helper is running on port {port} (pid {', '.join(map(str, old))}): replacing it.")
            for pid in old:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            time.sleep(1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    global HELPER_PORT
    HELPER_PORT = args.port
    server = bind_replacing_old_helper(args.port)
    print(f"LanderPi consoles: mapping http://127.0.0.1:{args.port}/   tasks http://127.0.0.1:{args.port}/tasks.html"
          f"   (robot {USER}@{HOST}). Ctrl+C to stop.")
    threading.Thread(target=lambda: print("Robot check:", check_robot_safe(), flush=True), daemon=True).start()
    threading.Thread(target=rosbridge_watchdog, daemon=True).start()
    threading.Thread(target=check_vendor, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def check_vendor():
    global vendor_missing
    try:
        vendor_missing = console_vendor.ensure(log=lambda m: print(m, flush=True))
    except Exception as e:  # never keep the pages waiting
        vendor_missing = [f"check failed: {e}"]
    if vendor_missing:
        bar = "!" * 72
        print(f"{bar}\nThe consoles' JavaScript libraries could not be downloaded (no internet?): "
              f"{', '.join(vendor_missing)}\nThe console can't connect to the robot or show 3D without them. "
              f"Restart the helper with internet access.\n{bar}", flush=True)
    VENDOR_READY.set()


def check_robot_safe():
    try:
        ok, info = check_robot()
        notes = []
        if ok:
            # free the CPU core the idle gamepad node burns (comes back after every reboot)
            stopped = stop_idle_factory_nodes()
            if stopped:
                notes.append(f"stopped idle: {', '.join(stopped)}")
            # rosbridge must drop dead clients, or it buffers for them until it hangs
            if not rosbridge_has_pings():
                restart_rosbridge()
                notes.append("rosbridge restarted with client pings")
        return f"{'OK' if ok else 'FAILED'} {info}" + "".join(f" | {n}" for n in notes)
    except Exception as e:
        return f"not reachable ({e})"


def rosbridge_answers(timeout=5):
    """True if the robot's rosbridge completes a WebSocket handshake (a hung one accepts the TCP
    connection but never answers)."""
    import socket
    try:
        with socket.create_connection((HOST, 9090), timeout=timeout) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: robot\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n")
            return b" 101 " in s.recv(64)
    except OSError:
        return False


def rosbridge_watchdog(interval_s=20, settings_check_every=3):
    """Safety net: restart rosbridge if it stops answering twice in a row. Every minute it also
    checks that rosbridge runs with our settings: after a robot reboot the factory one is back."""
    failures = 0
    rounds = 0
    while True:
        time.sleep(interval_s)
        rounds += 1
        if rosbridge_answers():
            failures = 0
            if rounds % settings_check_every == 0:
                try:
                    if not rosbridge_has_pings():
                        print("rosbridge runs without our settings (robot rebooted?): restarting it ...", flush=True)
                        restart_rosbridge()
                        stop_idle_factory_nodes()
                        print("rosbridge restarted with our settings", flush=True)
                except Exception as e:
                    print(f"rosbridge settings check failed: {e}", flush=True)
            continue
        failures += 1
        if failures >= 2 and check_robot_safe_reachable():
            print("rosbridge not answering: restarting it ...", flush=True)
            try:
                restart_rosbridge()
                print("rosbridge restarted", flush=True)
            except Exception as e:
                print(f"rosbridge restart failed: {e}", flush=True)
            failures = 0


def check_robot_safe_reachable():
    try:
        return check_robot()[0]
    except Exception:
        return False  # robot off / other network: nothing to restart


if __name__ == "__main__":
    main()
