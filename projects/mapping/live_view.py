"""Local HTTP server that lets a UI show what the recorder sees, without a second stream over Wi-Fi.

    http://127.0.0.1:8766/stream.mjpg   the recorder's colour stream (every frame, ~14 fps)
    http://127.0.0.1:8766/status.json   recorder status (rates, lag, keyframes, size, hints)
    POST /stop (header X-LanderPi-Console)  end the recording cleanly
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = "frame"


class LiveView:
    def __init__(self, port=8766):
        self._frame = None
        self._frame_id = 0
        self._status = {}
        self._cond = threading.Condition()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True
        self.port = port
        self.on_stop = None  # set by the recorder: called on POST /stop

    def start(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        with self._cond:
            self._frame_id = -1  # wakes streaming clients so they exit
            self._cond.notify_all()
        self._server.shutdown()

    def update_frame(self, jpeg):
        """jpeg: encoded bytes (passed through as received from the robot)."""
        with self._cond:
            self._frame = jpeg
            self._frame_id += 1
            self._cond.notify_all()

    def update_status(self, status):
        self._status = status

    @property
    def status(self):
        return self._status

    def _handler(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/status.json":
                    body = json.dumps(view._status).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")  # read-only status, harmless
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream.mjpg":
                    self._stream()
                else:
                    self.send_error(404)

            def do_POST(self):
                # stop the recording cleanly. The custom header forces a CORS preflight, which we
                # don't answer, so other websites can't trigger this; the helper calls it directly.
                if self.path != "/stop" or "X-LanderPi-Console" not in self.headers or view.on_stop is None:
                    return self.send_error(403)
                view.on_stop()
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = 0
                try:
                    while True:
                        with view._cond:
                            view._cond.wait_for(lambda: view._frame_id != last, timeout=5)
                            if view._frame_id < 0:
                                return
                            if view._frame_id == last or view._frame is None:
                                continue
                            last, frame = view._frame_id, view._frame
                        self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame + b"\r\n")
                except (ConnectionError, OSError):
                    pass  # client closed the stream

            def log_message(self, *args):
                pass

        return Handler
