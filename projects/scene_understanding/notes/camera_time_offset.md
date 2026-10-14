# Note for the mapping side: colour images are about 0.1 s late against depth

Found 2026-10-06 while placing labelled objects in 3D (projects/scene_understanding). Affects
`projects/mapping/reconstruct.py` (colouring, `T_map_rgb` in `cameras.json`) and everything that maps
colour pixels to depth, e.g. the scene labels. Nothing here was changed on the mapping side.

## Symptom

In `20261005_160227` one wall outlet became two objects, 19 cm apart along the wall. All views had
full measured depth. The position depended on which way the robot was turning:

| Keyframes | Turning | Outlet position along the wall (x) |
|---|---|---|
| 360-373 | left, +12 to +30 deg/s | 2.97-3.10 m |
| 375-379 | right, about -27 deg/s | 2.72-2.82 m |

0.2 m at 1.7 m distance is about 7 degrees, and it flips with the turning direction: a timing error,
not missing depth. The robot turns a lot while exploring (median 16 deg/s between keyframes, p90 32).

## Measurements

- **Depth and poses agree.** Depth points vs. the mesh, by turn rate (`20261005_160227`): 0.82 cm
  (0-5 deg/s), 0.87 (5-15), 0.83 (15-25), 0.83 (25-40). A depth timing error would grow with the turn
  rate.
- **Colour is off against depth.** `projects/scene_understanding/check_camera_timing.py` turns the colour
  camera by turn rate x dt and scores how well depth edges fall on colour edges, over the keyframes
  taken while turning:

  | Session | Turning keyframes | Best dt |
  |---|---|---|
  | 20261005_160227 | 127 | -0.12 s (-0.09 almost equal) |
  | 20261005_142856 | 216 | -0.09 s (-0.12 almost equal) |

  Negative: a colour image shows the scene about 0.1 s **before** its timestamp. At 30 deg/s that is
  3 degrees, about 20 px in a 640 px image.

## Likely cause

The colour timestamps come from the camera driver (`X-Timestamp` header of web_video_server's MJPEG,
`landerpi/mjpeg.py`). The driver probably stamps when it publishes, not when the image was exposed.
The depth images come through a separate path and fit the poses.

## Suggested fix

1. Subtract the offset from the colour stamps before looking up the colour camera pose, e.g. a
   constant `RGB_TIME_OFFSET = -0.1` in `reconstruct.py` (or in the recorder), with this note as the
   reason. Then re-run `reconstruct.py` and the labelling.
2. Check: `check_camera_timing.py <session>` should then report a best dt near 0.
3. Optionally measure it per session (the offset may depend on load or exposure time).

Until that fix, scene understanding corrected the colour poses itself: `landerpi.session.Session(path,
rgb_time_offset=-0.1)` turns each colour pose by turn rate x offset (yaw only). With it,
`check_camera_timing.py --offset -0.1` found the best dt at 0.00 s (20261005_160227), and the wall outlet
there was one object again.

**Update 2026-10-06:** fixed at the source in `projects/mapping/reconstruct.py`
(`RGB_TIME_OFFSET = -0.10`); new `cameras.json` files record it as `rgb_time_offset_s`. `Session` now
treats `rgb_time_offset` as the total correction and only adds what a file is missing, so sessions
reconstructed before the fix are corrected here and newer ones are not corrected twice.
