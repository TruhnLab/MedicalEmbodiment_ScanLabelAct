"""Minimal asyncio client for rosbridge (ROS 2).

Just what the laptop tools need: subscribe, publish and call_service. Messages arrive as
plain dicts (rosbridge JSON; uint8[] fields such as image data are base64 strings).
Subscription callbacks run in the event loop and must return quickly; hand heavy work
(decoding, disk I/O) to a thread.
"""

import asyncio
import itertools
import json
import logging
import time

import websockets

from .robot import ROSBRIDGE_URL

log = logging.getLogger(__name__)


class RosbridgeClient:
    def __init__(self, url=ROSBRIDGE_URL):
        self.url = url
        self.bytes_received = 0
        self.closed = asyncio.Event()
        self._ws = None
        self._reader = None
        self._subs = {}         # topic -> [callback(msg, recv_time)]
        self._advertised = set()
        self._pending = {}      # service call id -> future
        self._ids = itertools.count()

    async def connect(self, timeout=5.0):
        # images are ~1 MB per message, so no size limit
        self._ws = await asyncio.wait_for(
            websockets.connect(self.url, max_size=None, ping_interval=10, ping_timeout=20), timeout)
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
        if self._reader is not None:
            await asyncio.gather(self._reader, return_exceptions=True)

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *exc):
        await self.close()

    async def _send(self, msg):
        await self._ws.send(json.dumps(msg))

    async def subscribe(self, topic, msg_type, callback, throttle_ms=0, queue_length=1):
        """queue_length=1 makes rosbridge drop old messages when we (or the Wi-Fi) are too slow."""
        first = topic not in self._subs
        self._subs.setdefault(topic, []).append(callback)
        if first:
            await self._send({"op": "subscribe", "id": f"sub:{topic}", "topic": topic, "type": msg_type,
                              "throttle_rate": throttle_ms, "queue_length": queue_length})

    async def unsubscribe(self, topic):
        if self._subs.pop(topic, None) is not None:
            await self._send({"op": "unsubscribe", "id": f"sub:{topic}", "topic": topic})

    async def publish(self, topic, msg_type, msg):
        if topic not in self._advertised:
            await self._send({"op": "advertise", "topic": topic, "type": msg_type})
            self._advertised.add(topic)
        await self._send({"op": "publish", "topic": topic, "msg": msg})

    async def call_service(self, service, args=None, timeout=5.0):
        call_id = f"call:{next(self._ids)}"
        future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        await self._send({"op": "call_service", "id": call_id, "service": service, "args": args or {}})
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(call_id, None)

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                recv_time = time.time()
                self.bytes_received += len(raw)
                msg = json.loads(raw)
                op = msg.get("op")
                if op == "publish":
                    for callback in self._subs.get(msg["topic"], ()):
                        try:
                            callback(msg["msg"], recv_time)
                        except Exception:
                            log.exception("callback for %s failed", msg["topic"])
                elif op == "service_response":
                    future = self._pending.get(msg.get("id"))
                    if future and not future.done():
                        if msg.get("result", True):
                            future.set_result(msg.get("values"))
                        else:
                            future.set_exception(RuntimeError(f"{msg.get('service')}: {msg.get('values')}"))
                elif op == "status":
                    log.warning("rosbridge: %s", msg.get("msg"))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.closed.set()
