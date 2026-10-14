"""Check a recorded mapping session: completeness, rates, gaps, sync and a visual overview.

    python projects/mapping/report.py data/sessions/20260930_101500
    python projects/mapping/report.py            # newest session

Prints a summary and writes report.png into the session folder.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STREAMS = ["scan", "odom", "imu", "tf", "joints", "battery"]


def stamps_of(path):
    """Robot timestamps of a .jsonl stream (tf uses its first transform's header). Messages without
    a header (e.g. battery) fall back to the laptop's receive time."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            msg = record["msg"]
            header = msg.get("header") or (msg.get("transforms") or [{}])[0].get("header")
            if header:
                out.append(header["stamp"]["sec"] + header["stamp"]["nanosec"] * 1e-9)
            else:
                out.append(record["recv"])
    return np.array(sorted(out))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["msg"] for line in f]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, nargs="?", help="session folder (default: newest)")
    args = p.parse_args()
    session = args.session or max((REPO / "data" / "sessions").iterdir(), key=lambda d: d.name)
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    with open(session / "keyframes.csv", encoding="utf-8") as f:
        kfs = list(csv.DictReader(f))

    print(f"Session {session.name}: {meta['duration_s']} s, stopped: {meta['stop_reason']}")
    print(f"  data written     {meta['bytes_written'] / 1e9:.2f} GB")
    c = meta["counts"]
    print(f"  keyframes        {c['keyframes']} (of {c['pairs']} rgb/depth pairs; {c['skipped_still']} skipped while "
          f"still, {c['dropped_backlog']} dropped by backlog, {c['gaps']} coverage gaps, {c['unpaired']} unpaired)")
    print(f"  calibration      {', '.join(meta['calibration_received']) or 'MISSING'}")

    print("\n  stream        msgs    rate    max gap")
    for name in STREAMS:
        path = session / "streams" / f"{name}.jsonl"
        if not path.exists():
            print(f"  {name:10s}  missing")
            continue
        t = stamps_of(path)
        if len(t) < 2:
            print(f"  {name:10s}  {len(t):6d}")
            continue
        gaps = np.diff(t)
        print(f"  {name:10s}  {len(t):6d}  {len(t) / (t[-1] - t[0]):5.1f} Hz  {gaps.max():6.2f} s")

    if kfs:
        dt = np.array([float(k["rgb_stamp"]) - float(k["depth_stamp"]) for k in kfs]) * 1000
        print(f"\n  rgb-depth offset per keyframe: mean {dt.mean():.1f} ms, max {np.abs(dt).max():.1f} ms")
        missing = [k for k in kfs if not (session / k["rgb"]).exists() or not (session / k["depth"]).exists()]
        print(f"  image files missing: {len(missing)}")

    make_figure(session, kfs)
    print(f"\n  overview: {session / 'report.png'}")


def make_figure(session, kfs):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Mapping session {session.name}")

    # trajectory + keyframes
    odom_path = session / "streams" / "odom.jsonl"
    a = ax[0, 0]
    if odom_path.exists():
        poses = [(m["pose"]["pose"]["position"]["x"], m["pose"]["pose"]["position"]["y"]) for m in load_jsonl(odom_path)]
        if poses:
            xy = np.array(poses)
            a.plot(xy[:, 0], xy[:, 1], lw=1, color="0.6", label="odometry")
    if kfs:
        kx = np.array([float(k["odom_x"]) for k in kfs])
        ky = np.array([float(k["odom_y"]) for k in kfs])
        kyaw = np.array([float(k["odom_yaw"]) for k in kfs])
        a.quiver(kx, ky, np.cos(kyaw), np.sin(kyaw), color="tab:blue", scale=25, width=0.004, label="keyframes")
    a.set_aspect("equal", adjustable="datalim")
    a.set_title("Trajectory (odom frame, m)")
    a.legend(loc="best", fontsize=8)
    a.grid(alpha=0.3)

    # a keyframe from the middle: colour and depth
    if kfs:
        k = kfs[len(kfs) // 2]
        rgb = cv2.imread(str(session / k["rgb"]))
        depth = cv2.imread(str(session / k["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb is not None:
            ax[0, 1].imshow(rgb[:, :, ::-1])
            ax[0, 1].set_title(f"Keyframe {k['index']} (colour)")
        if depth is not None:
            valid = depth[depth > 0]
            vmax = np.percentile(valid, 98) if valid.size else 1
            im = ax[1, 1].imshow(np.where(depth > 0, depth, np.nan), cmap="turbo", vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax[1, 1], fraction=0.03, label="mm")
            cover = 100 * valid.size / depth.size
            ax[1, 1].set_title(f"Keyframe {k['index']} (depth, {cover:.0f}% valid)")
        for a2 in (ax[0, 1], ax[1, 1]):
            a2.axis("off")

    # lidar scan closest to that keyframe, robot at the origin, forward = +x
    scan_path = session / "streams" / "scan.jsonl"
    a = ax[1, 0]
    if scan_path.exists():
        scans = load_jsonl(scan_path)
        if scans:
            t_kf = float(kfs[len(kfs) // 2]["stamp"]) if kfs else None
            def t_of(m):
                return m["header"]["stamp"]["sec"] + m["header"]["stamp"]["nanosec"] * 1e-9
            s = min(scans, key=lambda m: abs(t_of(m) - t_kf)) if t_kf else scans[-1]
            r = np.array([np.nan if v is None else v for v in s["ranges"]], dtype=float)
            ang = s["angle_min"] + np.arange(len(r)) * s["angle_increment"]
            ok = np.isfinite(r) & (r > s["range_min"]) & (r < s["range_max"])
            a.scatter(r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok]), s=2, color="tab:blue")
            a.plot(0, 0, "r^")
            a.set_aspect("equal")
            a.set_title(f"Lidar scan near that keyframe ({ok.sum()}/{len(r)} valid beams)")
            a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(session / "report.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
