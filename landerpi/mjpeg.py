"""Reads the colour camera as an MJPEG stream from the robot's web_video_server.

This is what the factory app does: JPEG is encoded on the robot (~60-130 KB per frame instead of
~770 KB raw), which gives ~14 fps with only milliseconds of latency. Every part carries an
X-Timestamp header with the image's ROS stamp, so frames stay exactly timed.
"""

import socket
import threading
import time
from urllib.parse import quote

from .robot import ROBOT_HOST, TOPICS

VIDEO_PORT = 8080


class MjpegReader:
    """Calls on_frame(jpeg_bytes, stamp, recv_time) from a background thread for every new frame.
    Reconnects when the stream stalls; `stalls` counts how often that happened."""

    def __init__(self, on_frame, topic=TOPICS["rgb"][0], quality=90, host=ROBOT_HOST, port=VIDEO_PORT,
                 stall_s=3.0):
        self.on_frame = on_frame
        self.path = f"/stream?topic={quote(topic)}&type=mjpeg&quality={quality}"
        self.host, self.port, self.stall_s = host, port, stall_s
        self.frames = 0
        self.bytes_received = 0
        self.stalls = 0
        self.last_frame_time = None
        self._stop = threading.Event()
        self._sock = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="mjpeg")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # clean close
            except OSError:
                pass
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._read_stream()
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                self.stalls += 1
                time.sleep(1.0)

    def _read_stream(self):
        with socket.create_connection((self.host, self.port), timeout=self.stall_s) as sock:
            self._sock = sock
            sock.sendall(f"GET {self.path} HTTP/1.0\r\nHost: {self.host}\r\n\r\n".encode())
            f = sock.makefile("rb")
            last_stamp = None
            while not self._stop.is_set():
                headers = self._read_headers(f)
                length = headers.get("content-length")
                if length is None:
                    continue  # HTTP response header or boundary line
                jpeg = f.read(int(length))
                recv = time.time()
                self.bytes_received += len(jpeg)
                stamp = float(headers.get("x-timestamp", recv))
                if stamp == last_stamp:
                    continue  # web_video_server repeats frames; keep each capture once
                last_stamp = stamp
                self.frames += 1
                self.last_frame_time = recv
                self.on_frame(jpeg, stamp, recv)

    @staticmethod
    def _read_headers(f):
        headers = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("stream closed")
            line = line.strip()
            if not line:
                if headers:
                    return headers
                continue
            if b":" in line:
                key, value = line.split(b":", 1)
                headers[key.strip().lower().decode()] = value.strip().decode()
