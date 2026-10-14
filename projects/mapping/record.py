"""Record a mapping session: stream the robot's sensors to the laptop and save keyframes.

    conda activate landerpi
    python projects/mapping/record.py                  # Ctrl+C to stop
    python projects/mapping/record.py --max-minutes 15 --max-gb 5
    python projects/mapping/record.py --all-frames --max-minutes 0.5   # throughput test

Drive with the console meanwhile. On start it makes sure the robot's depth compressor runs
(a light process in the factory ROS container, gone after reboot; see landerpi/robot_ssh.py).
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from landerpi import RosbridgeClient
from landerpi.mjpeg import MjpegReader
from landerpi.robot import ROSBRIDGE_URL
from landerpi.robot_ssh import ensure_depth_compressor, static_transforms, stop_idle_factory_nodes

from live_view import LiveView
from recorder import KeyframePolicy, Limits, Recorder
from session import SessionWriter

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=ROSBRIDGE_URL, help="rosbridge WebSocket URL")
    p.add_argument("--slam-url", default="ws://127.0.0.1:9091", help="rosbridge of the offboard SLAM container")
    p.add_argument("--out", type=Path, default=REPO / "data" / "sessions", help="parent folder for sessions")
    p.add_argument("--max-gb", type=float, default=20.0, help="stop when this much data is written")
    p.add_argument("--max-minutes", type=float, default=60.0, help="stop after this long")
    p.add_argument("--min-move", type=float, default=0.05, help="metres between keyframes")
    p.add_argument("--min-turn", type=float, default=5.0, help="degrees between keyframes")
    p.add_argument("--all-frames", action="store_true", help="keep every rgb/depth pair (no keyframe filter)")
    p.add_argument("--rgb-quality", type=int, default=90, help="JPEG quality requested from the robot")
    p.add_argument("--depth-hz", type=float, default=10.0, help="depth rate requested from the robot")
    p.add_argument("--live-port", type=int, default=8766, help="local port for the UI's live view (0 = off)")
    return p.parse_args()


async def main(args):
    print("Preparing the robot (depth compressor, idle factory nodes) ...", end=" ", flush=True)
    loop = asyncio.get_running_loop()
    try:
        compressor = await loop.run_in_executor(None, ensure_depth_compressor)
        stopped = await loop.run_in_executor(None, stop_idle_factory_nodes)
        print(f"compressor {compressor}" + (f", stopped {', '.join(stopped)}" if stopped else ""))
    except Exception as e:
        sys.exit(f"\nCannot prepare the robot: {e}")

    root = args.out / time.strftime("%Y%m%d_%H%M%S")
    writer = SessionWriter(root)
    # the robot's fixed transforms are published once at its startup, so rosbridge rarely forwards
    # them; read them directly (read-only) so the camera's position on the arm is known
    static = []
    try:
        static = await loop.run_in_executor(None, static_transforms)
        writer.write_json("calib/tf_static.json", {"transforms": static})
        print(f"Static transforms: {len(static)}")
        missing = {"depth_camera_link", "rgb_camera_link"} - {t["child_frame_id"] for t in static}
        if missing:
            print(f"Warning: no static transform for {', '.join(sorted(missing))} yet; reconstruct.py "
                  "takes it from an earlier session if the recording doesn't catch it")
    except Exception as e:
        print(f"Warning: could not read the robot's static transforms ({e}); 3D poses will need them later")
    live = LiveView(args.live_port).start() if args.live_port else None
    policy = KeyframePolicy(args.min_move, args.min_turn, all_frames=args.all_frames)
    limits = Limits(args.max_gb, args.max_minutes)

    print(f"Connecting to {args.url} ...")
    # separate connections, so depth frames never delay the small sensor messages
    clients = {name: await RosbridgeClient(args.url).connect() for name in ("sensors", "depth")}
    # SLAM poses from the offboard container, if it runs (optional)
    try:
        clients["slam"] = await RosbridgeClient(args.slam_url).connect(timeout=3)
        print(f"SLAM: recording map poses from {args.slam_url}")
    except (OSError, asyncio.TimeoutError):
        print("SLAM not running: images will only have odometry poses (start SLAM in the Map view first)")
    recorder = Recorder(clients, writer, policy, limits, live, depth_hz=args.depth_hz)
    # start from the directly read static transforms; rosbridge may add (never remove) some
    recorder._static_tf.update({t["child_frame_id"]: t for t in static})
    if static:
        recorder._calib_done.add("tf_static")
    if live:
        # the console's "Stop recording" button ends the recording cleanly (via the helper)
        live.on_stop = lambda: loop.call_soon_threadsafe(recorder.stop, "stopped from the console")
    print(f"Recording to {root}")
    if live:
        print(f"Live view: http://127.0.0.1:{live.port}/stream.mjpg   status: http://127.0.0.1:{live.port}/status.json")
    print("Ctrl+C to stop.\n")
    mjpeg = None
    try:
        await recorder.start()
        mjpeg = recorder.mjpeg = MjpegReader(recorder.on_rgb_frame, quality=args.rgb_quality).start()
        status = asyncio.create_task(recorder.run_status_loop())
        await recorder.stopped.wait()
        status.cancel()
    except asyncio.CancelledError:
        recorder.stop("stopped by user")
    finally:
        if mjpeg:
            mjpeg.stop()
        for client in clients.values():
            await client.close()
        print(f"\n\nStopping: {recorder.stop_reason}. Writing remaining keyframes ...")
        recorder.finish()
        if live:
            live.stop()
        r = recorder
        print(f"Done: {r.keyframes} keyframes ({r.pairs} pairs seen, {r.skipped_still} skipped while still, "
              f"{r.dropped_backlog} dropped, {r.gaps} gaps), {writer.bytes_written / 1e9:.3f} GB")
        print(f"Session: {root}")
        print(f"Check it with:  python projects/mapping/report.py \"{root}\"")


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        sys.exit(f"Cannot reach the robot: {e}")
