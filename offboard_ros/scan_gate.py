"""Holds lidar scans back until the robot's pose (odom -> base_footprint) for their time stamp is in TF.

slam_toolbox froze for good in every live run on 2026-10-05: a lock-order deadlock in tf2_ros
(Humble). Its scan callback waits for the scan's pose (tf2_ros::Buffer::waitForTransform, holding
the buffer's lock, wanting the transform-request lock) while its TF listener thread, on receiving that
pose, holds the request lock and wants the buffer's lock (stack traces: data/maps/slam_freeze_*.txt).
The waiting path is only taken when a scan arrives before its pose, which over Wi-Fi happens all the
time (poses arrive 30-330 ms late). Reproduced offline by replaying a recording with delayed poses
(replay_slam.py --tf-delay 0.33: frozen within a minute); never without the delay.
So scans are passed on only once a pose MARGIN_S newer than them has been passed on. Not just "at or
after": SLAM receives poses and scans on separate threads, and with the bare minimum a scan was still
sometimes processed before its pose had been stored (froze in the replay too). MARGIN_S later, the
pose went out several TF messages (44 Hz) earlier. The added scan delay doesn't slow the robot's own
pose: that is SLAM's slowly changing map -> odom correction times the live odometry.
"""

import time

MARGIN_S = 0.15
MAX_HOLD_S = 0.6  # a scan whose pose doesn't arrive within this is passed on anyway (SLAM drops it)


def stamp_s(header):
    return header["stamp"]["sec"] + header["stamp"]["nanosec"] * 1e-9


class ScanGate:
    def __init__(self, odom_frame="odom", base_frame="base_footprint"):
        self.odom_frame, self.base_frame = odom_frame, base_frame
        self.pose_time = None  # newest stamp of odom -> base_footprint passed on
        self.held = []         # [(scan stamp, arrival time, item)] in arrival order

    def tf(self, msg):
        """A /tf message was passed on: returns the scans that may follow now."""
        for t in msg.get("transforms", []):
            if t["header"]["frame_id"] == self.odom_frame and t["child_frame_id"] == self.base_frame:
                s = stamp_s(t["header"])
                if self.pose_time is None or s > self.pose_time:
                    self.pose_time = s
        return self._release(lambda st, arrived: self.pose_time is not None and st + MARGIN_S <= self.pose_time)

    def scan(self, stamp, item):
        """A scan arrived: returns the scans that may be passed on now (maybe this one)."""
        self.held.append((stamp, time.monotonic(), item))
        return self._release(lambda st, arrived: self.pose_time is not None and st + MARGIN_S <= self.pose_time)

    def poll(self):
        """Scans held longer than MAX_HOLD_S (the pose stream stopped): passed on anyway."""
        now = time.monotonic()
        return self._release(lambda st, arrived: now - arrived > MAX_HOLD_S)

    def _release(self, ready):
        out = []
        # in order: a later scan never overtakes an earlier one
        while self.held and ready(self.held[0][0], self.held[0][1]):
            out.append(self.held.pop(0)[2])
        return out
