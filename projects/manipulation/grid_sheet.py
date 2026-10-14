"""The printed sheet for the arm and camera calibration: a mm grid, numbered targets for the gripper tip, ArUco
markers for the camera. A4 landscape, laid on the floor with its long bottom edge under the robot's front.

    python projects/manipulation/grid_sheet.py      # -> data/calibration/grid_sheet.pdf (print at 100 %, "actual size")

Sheet frame (mm): x = ahead of the bottom edge (the robot's front), y = left of the centre line (as base_footprint,
seen from above with the robot at the bottom). The sheet need not lie exactly: the evaluation fits its offset and
turn from all targets (or the markers) and reports what remains.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType: matplotlib's default Type 3 fonts are refused by some printer drivers
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "calibration" / "grid_sheet.pdf"
PAGE = (297.0, 210.0)  # A4 landscape: width (y) x depth (x), mm
MARGIN = 7.0           # printers leave ~5 mm blank at the edges

# gripper tip targets: the floor reach ends ~50 mm ahead of the chassis pointing down, ~120 mm tilted
# (arm.py, URDF only); the 140 mm row shows where it really ends
TARGET_X = [20, 50, 80, 110, 140]
TARGET_Y = [60, 0, -60]
TARGETS = {i + 1: (x, y) for i, (x, y) in enumerate((x, y) for x in TARGET_X for y in TARGET_Y)}

# ArUco markers (DICT_4X4_50) for the camera: id -> centre (x, y) in mm; SIZE without the white border
MARKER_SIZE = 36.0
MARKERS = {0: (170, 105), 1: (170, 35), 2: (170, -35), 3: (170, -105), 4: (75, 115), 5: (75, -115)}


def sheet(path=OUT):
    W, D = PAGE
    fig = plt.figure(figsize=(W / 25.4, D / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    # page coordinates: horizontal = -y (left of the robot on the left), vertical = x
    ax.set_xlim(-W / 2, W / 2)
    ax.set_ylim(0, D)
    ax.set_aspect("equal")
    ax.axis("off")
    top, side = D - MARGIN, W / 2 - MARGIN
    for v in range(-140, 141, 10):
        ax.plot([v, v], [MARGIN, top], color="0.55" if v % 50 == 0 else "0.85", lw=0.6 if v % 50 == 0 else 0.3, zorder=0)
    for v in range(10, int(top) + 1, 10):
        ax.plot([-side, side], [v, v], color="0.55" if v % 50 == 0 else "0.85", lw=0.6 if v % 50 == 0 else 0.3, zorder=0)
    for v in range(50, int(top), 50):
        ax.text(-side + 1, v + 1, f"x {v}", fontsize=5, color="0.4")
    for v in range(-100, 101, 50):
        ax.text(-v + 1, MARGIN + 1, f"y {v}", fontsize=5, color="0.4")
    ax.plot([0, 0], [MARGIN, top], color="0.3", lw=0.8, zorder=1)  # centre line
    ax.fill([-6, 6, 0], [0, 0, 9], color="black")                  # centre mark on the bottom edge
    ax.text(10, 2, "bottom edge under the robot's front, centre mark under its middle", fontsize=6)

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (x, y) in MARKERS.items():
        img = cv2.aruco.generateImageMarker(d, mid, 600)
        h = MARKER_SIZE / 2
        ax.add_patch(plt.Rectangle((-y - h - 4, x - h - 4), MARKER_SIZE + 8, MARKER_SIZE + 8, color="white", zorder=2))
        ax.imshow(img, cmap="gray", extent=(-y - h, -y + h, x - h, x + h), interpolation="nearest", zorder=3)
        ax.text(-y, x - h - 3.5, f"M{mid}  x {x} y {y}", fontsize=5, ha="center", zorder=4)

    for n, (x, y) in TARGETS.items():
        ax.add_patch(plt.Circle((-y, x), 4, fill=False, color="red", lw=0.8, zorder=4))
        ax.plot([-y - 7, -y + 7], [x, x], color="red", lw=0.5, zorder=4)
        ax.plot([-y, -y], [x - 7, x + 7], color="red", lw=0.5, zorder=4)
        ax.text(-y + 5, x + 4, f"T{n}", fontsize=7, color="red", weight="bold", zorder=4)

    # print check: the bar must measure 100 mm
    ax.plot([-side + 30, -side + 130], [top - 4, top - 4], color="black", lw=1.2)
    for e in (-side + 30, -side + 130):
        ax.plot([e, e], [top - 6, top - 2], color="black", lw=1.2)
    ax.text(-side + 80, top - 3, "must measure 100 mm (print at actual size)", fontsize=6, ha="center", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"{path}: {len(TARGETS)} targets, {len(MARKERS)} markers")


if __name__ == "__main__":
    sheet()
