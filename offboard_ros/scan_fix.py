"""Fixed-size lidar scans for slam_toolbox.

The LanderPi's lidar sends 500-505 rays per 360-degree scan (the count changes from scan to scan).
slam_toolbox (Karto) registers the sensor with the first scan's count and silently drops every
scan with a different count ("LaserRangeScan contains 502 range readings, expected 504"): only
17-61 % of the scans were used, the scan matching had large gaps and SLAM's pose jumped by tens of
cm (2026-10-02). Every scan is therefore resampled to the same RAYS angle bins over the full circle;
where two rays fall into one bin the nearer one wins, empty bins are "no return" (inf).
"""

import math

RAYS = 504


def fixed_scan(msg):
    """rosbridge-style LaserScan dict -> the same scan with exactly RAYS rays from 0 to 2*pi."""
    inc = 2 * math.pi / RAYS
    out = [math.inf] * RAYS
    out_i = [0.0] * RAYS
    intensities = msg.get("intensities") or []
    a0, da = msg["angle_min"], msg["angle_increment"]
    for j, r in enumerate(msg["ranges"]):
        if r is None or not math.isfinite(r):
            continue
        k = int(round(((a0 + j * da) % (2 * math.pi)) / inc)) % RAYS
        if r < out[k]:
            out[k] = r
            out_i[k] = intensities[j] if j < len(intensities) and intensities[j] is not None else 0.0
    msg = dict(msg)
    msg["angle_min"] = 0.0
    msg["angle_increment"] = inc
    msg["angle_max"] = inc * (RAYS - 1)
    msg["ranges"] = out
    msg["intensities"] = out_i
    if msg.get("time_increment"):
        msg["time_increment"] = msg.get("scan_time", 0.1) / RAYS
    return msg
