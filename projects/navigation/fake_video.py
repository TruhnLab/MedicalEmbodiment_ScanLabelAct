"""A stand-in for the helper's video relay without the robot: a recorded session's colour keyframes as an MJPEG
stream with X-Timestamp headers, looping, so the navigator's live detection and recording can be tested.

    python projects/navigation/fake_video.py 20261007_101108 [--port 8790] [--fps 8]
    $env:LANDERPI_HELPER_PORT = "8790"   # then the navigator reads its video here (live.py)

Serves GET /video?... (any query) on 127.0.0.1. The timestamps are the laptop's clock (the fake robot's odometry
uses it too), not the recording's.
"""

import argparse
import csv
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.room import SESSIONS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--start", type=int, default=0, help="first keyframe")
    args = p.parse_args()
    s = SESSIONS / args.session
    with open(s / "keyframes.csv") as f:
        files = [s / r["rgb"] for r in csv.DictReader(f)][args.start:]
    print(f"{len(files)} images from {args.session} at {args.fps} fps on http://127.0.0.1:{args.port}/video", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not self.path.startswith("/video"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            i = 0
            try:
                while True:
                    jpeg = files[i % len(files)].read_bytes()
                    self.wfile.write(f"--frame\r\nContent-Type: image/jpeg\r\nX-Timestamp: {time.time():.6f}\r\n"
                                     f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                    i += 1
                    time.sleep(1.0 / args.fps)
            except (OSError, ConnectionError):
                pass

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
