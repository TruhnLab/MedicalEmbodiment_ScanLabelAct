"""Phase 1 check: does the 3D coverage predict the holes of the 3D reconstruction?

    python projects/coverage3d/evaluate.py data/sessions/<name>     (reconstructed session)

Expected surfaces come from the lidar map: the floor over the area the robot explored (its side of the
no-go lines), and walls/furniture 0.1-0.5 m high at the lidar's obstacles next to it. The depth images
(with the reconstruction's poses) say which of them the camera saw well enough (coverage3d.py). The
reconstruction's mesh says which of them actually got built. Output: the agreement as numbers and
<session>/coverage3d/check.png (top view; legend in the console output).
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offboard_ros"))  # coverage3d.py is shared with the explorer (container)
# stopgap (repo rule): the lidar map from the recording, as the presentation video builds it
sys.path.insert(0, str(REPO / "projects" / "presentation_video"))
import coverage3d as c3  # noqa: E402
import make_video as mv  # noqa: E402

WALL_Z = (0.10, 0.50)     # wall/furniture heights checked (the camera sits ~0.3 m high)
FLOOR_Z = (-0.10, 0.08)


def main():
    session = Path(sys.argv[1]).resolve()
    cams = json.loads((session / "reconstruction" / "cameras.json").read_text())
    K = cams["depth_intrinsics"][2:6]

    # --- 3D coverage from all keyframes ---
    cov = c3.VoxelCoverage()
    if "--floor-cos" in sys.argv:
        cov.min_cos_floor = float(sys.argv[sys.argv.index("--floor-cos") + 1])
    t0 = time.perf_counter()
    for f in cams["frames"]:
        d = cv2.imread(str(session / f["depth"]), cv2.IMREAD_UNCHANGED)
        if d is not None:
            cov.add(d, K, np.array(f["T_map_depth"]))
    ms = 1000 * (time.perf_counter() - t0) / len(cams["frames"])
    print(f"{len(cams['frames'])} depth images, {ms:.0f} ms each (decode + coverage update, one core)")

    # --- expected surfaces from the lidar map ---
    rec = mv.Recording(session)
    sm = mv.ScanMap(rec)
    sm.at(rec.t_end + 10)
    occ = (sm.hits >= 2) & (sm.hits >= 0.3 * sm.free)
    free = (sm.free > 0) & ~occ
    R, ox, oy = sm.RES, sm.ox, sm.oy
    H, W = occ.shape
    blocked = np.zeros((H, W), np.uint8)
    nogo = session / "streams" / "nogo_lines.jsonl"
    lines = json.loads(nogo.read_text().splitlines()[-1])["msg"]["lines"] if nogo.exists() else []
    for x1, y1, x2, y2 in lines:  # extended 0.6 m: drawn roughly to the wall
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        p = lambda x, y: (int((x - ox) / R), int((y - oy) / R))  # noqa: E731
        cv2.line(blocked, p(x1 - 0.6 * ux, y1 - 0.6 * uy), p(x2 + 0.6 * ux, y2 + 0.6 * uy), 1, 5)
    # the area the robot can drive in (0.26 m clearance, as the explorer), plus the free space up to
    # its walls: a narrow gap at a line's end doesn't let the next room count (as in explore_core)
    open_ = free & (blocked == 0)
    drivable = open_ & (ndimage.distance_transform_edt(open_) * R >= 0.26)
    labels, _ = ndimage.label(drivable)
    pr = ((rec.pose_xy[:, 1] - oy) / R).astype(int).clip(0, H - 1)
    pc = ((rec.pose_xy[:, 0] - ox) / R).astype(int).clip(0, W - 1)
    ids = np.unique(labels[pr, pc])
    area = ndimage.binary_dilation(np.isin(labels, ids[ids > 0]), iterations=7, mask=open_)
    # only what the robot could have looked at: within 3 m (camera range + margin) of its path (rooms can
    # be open where no line was drawn; what the robot never approached is no coverage gap)
    near_path = np.zeros((H, W), bool)
    near_path[pr, pc] = True
    near_path = ndimage.distance_transform_edt(~near_path) * R <= 3.0
    area &= near_path
    walls = occ & ndimage.binary_dilation(area, iterations=2) & (blocked == 0)

    # map cells -> voxel columns (both 5 cm)
    rr, cc = np.mgrid[:H, :W]
    vx, vy = cov.column_index(ox + (cc + 0.5) * R, oy + (rr + 0.5) * R)
    ok = (vx >= 0) & (vx < cov.shape[0]) & (vy >= 0) & (vy < cov.shape[1])
    vx, vy = vx.clip(0, cov.shape[0] - 1), vy.clip(0, cov.shape[1] - 1)
    covered = cov.covered()

    def column(z_range, grid):
        z0, z1 = cov.z_index(z_range[0]), cov.z_index(z_range[1]) + 1
        return grid[vx, vy, z0:z1].any(axis=2) & ok

    floor_pred = column(FLOOR_Z, covered)
    # walls: covered in a 3x3 neighbourhood (lidar and depth cells differ by a cell or so)
    wall_pred = ndimage.maximum_filter(column(WALL_Z, covered), size=3)

    # --- what the reconstruction actually built ---
    mesh = o3d.io.read_triangle_mesh(str(session / "reconstruction" / "mesh.ply"))
    V = np.asarray(mesh.vertices)
    mr = ((V[:, 1] - oy) / R).astype(int)
    mc = ((V[:, 0] - ox) / R).astype(int)
    inside = (mr >= 0) & (mr < H) & (mc >= 0) & (mc < W)
    mesh_floor = np.zeros((H, W), bool)
    mesh_wall = np.zeros((H, W), bool)
    fl = inside & (V[:, 2] > FLOOR_Z[0]) & (V[:, 2] < FLOOR_Z[1])
    wl = inside & (V[:, 2] > WALL_Z[0]) & (V[:, 2] < WALL_Z[1])
    mesh_floor[mr[fl], mc[fl]] = True
    mesh_wall[mr[wl], mc[wl]] = True
    mesh_floor = ndimage.binary_closing(mesh_floor, iterations=1)
    mesh_wall = ndimage.maximum_filter(mesh_wall, size=3)

    def report(name, expected, pred, built):
        e = expected.sum()
        both = (expected & pred & built).sum()
        gap_ok = (expected & ~pred & ~built).sum()        # predicted gap, really missing
        false_gap = (expected & ~pred & built).sum()      # predicted gap, but built
        missed = (expected & pred & ~built).sum()         # predicted covered, but missing
        print(f"{name}: {e * R * R:.1f} m2 expected | predicted covered {100 * (expected & pred).sum() / e:.0f} %, "
              f"built {100 * (expected & built).sum() / e:.0f} % | agreement {100 * (both + gap_ok) / e:.0f} % "
              f"(gaps found {100 * gap_ok / max(1, gap_ok + missed):.0f} % of the real gaps; "
              f"false gaps {false_gap * R * R:.2f} m2)")

    report("floor", area, floor_pred, mesh_floor)
    report("walls 0.1-0.5 m", walls, wall_pred, mesh_wall)

    # --- picture ---
    out = np.full((H, W, 3), 255, np.uint8)
    out[free] = (238, 238, 238)
    colours = {(True, True): (170, 225, 170), (False, False): (60, 60, 230), (False, True): (60, 170, 250),
               (True, False): (200, 80, 200)}
    for (p, b), col in colours.items():
        out[area & (floor_pred == p) & (mesh_floor == b)] = col
    dark = {k: tuple(int(0.6 * v) for v in col) for k, col in colours.items()}
    for (p, b), col in dark.items():
        out[walls & (wall_pred == p) & (mesh_wall == b)] = col
    pts = np.stack([pc, pr], 1).astype(np.int32)
    cv2.polylines(out, [pts], False, (255, 140, 0), 1)
    out = np.flipud(out)
    ys, xs = np.nonzero((out != 255).any(-1))
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    (session / "coverage3d").mkdir(exist_ok=True)
    path = session / "coverage3d" / "check.png"
    cv2.imwrite(str(path), cv2.resize(out, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print(f"picture: {path}\n  green: predicted covered and built | red: predicted gap, really missing | "
          f"orange: predicted gap but built | purple: predicted covered but missing | darker: walls | "
          f"blue line: path")


if __name__ == "__main__":
    main()
