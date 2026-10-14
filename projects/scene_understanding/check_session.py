"""Checks a session's data before segmentation: do lifted depth points land on the mesh, and do the
colour and depth images line up (a mask drawn in the colour image is lifted through the depth image)?

    python projects/scene_understanding/check_session.py 20261005_124116

Prints the distance of the depth points to the mesh and writes <session>/scene/check_session.jpg:
a few colour keyframes with the depth points projected into them, coloured by distance, and the
depth image's object edges in white. The edges should sit on the object outlines in the photo.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.session import Session  # noqa: E402


def mesh_distances(session, frames):
    mesh = o3d.t.io.read_triangle_mesh(str(session.mesh_file()))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    out = []
    for f in frames:
        pts = session.depth_points(f)[0][::20]
        d = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
        out.append(d)
    return out


def overlay(session, f):
    """Depth points drawn into the colour image (colour = distance), depth edges in white."""
    img = f.rgb()
    pts, v, u = session.depth_points(f, frame_out="map")
    uv, ok = session.project(f, pts, "rgb")
    z = np.linalg.norm(pts - f.T_map_rgb[:3, 3], axis=1)
    col = cv2.applyColorMap(np.clip((z - 0.25) / 2.75 * 255, 0, 255).astype(np.uint8)[:, None],
                            cv2.COLORMAP_TURBO)[:, 0]
    dots = img.copy()
    for (x, y), c in zip(uv[ok][::7].astype(int), col[ok][::7]):
        cv2.circle(dots, (int(x), int(y)), 1, tuple(int(a) for a in c), -1)
    img = cv2.addWeighted(img, 0.45, dots, 0.55, 0)
    # outlines from depth: edge pixels of the depth image, moved into the colour image via 3D
    z_img = f.depth()
    edges = cv2.Canny(np.clip(z_img / 4.0 * 255, 0, 255).astype(np.uint8), 20, 60)
    ev, eu = np.nonzero(edges & (z_img > 0.25).astype(np.uint8) * 255)
    w, h, fx, fy, cx, cy = session.K["depth"]
    ez = z_img[ev, eu]
    ep = np.stack([(eu - cx) * ez / fx, (ev - cy) * ez / fy, ez], axis=1) @ f.T_map_depth[:3, :3].T + f.T_map_depth[:3, 3]
    euv, eok = session.project(f, ep, "rgb")
    for x, y in euv[eok].astype(int):
        img[y, x] = (255, 255, 255)
    cv2.putText(img, f"#{f.index}", (6, 20), 0, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--frames", type=int, nargs="*", help="frame positions in cameras.json for the image")
    a = ap.parse_args()
    s = Session(a.session)
    print(f"{s.name}: {len(s.frames)} keyframes, rgb {s.K['rgb']}, depth {s.K['depth']}")

    d = mesh_distances(s, s.frames[::10])
    alld = np.concatenate(d)
    print(f"depth points vs mesh: median {np.median(alld) * 100:.1f} cm, 90% {np.percentile(alld, 90) * 100:.1f} cm, "
          f"> 10 cm {np.mean(alld > 0.10) * 100:.1f} %")
    worst = sorted(range(len(d)), key=lambda i: -np.median(d[i]))[:3]
    print("worst frames:", ", ".join(f"#{s.frames[::10][i].index} ({np.median(d[i]) * 100:.1f} cm)" for i in worst))

    pick = a.frames or list(np.linspace(0, len(s.frames) - 1, 6).astype(int))
    tiles = [overlay(s, s.frames[i]) for i in pick]
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    out = s.path / "scene" / "check_session.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
