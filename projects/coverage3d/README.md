# 3D coverage from depth

Which surfaces has the depth camera seen well enough for the 3D reconstruction? The exploration steers by
this: it drives to viewpoints of surfaces not yet seen well (`projects/mapping/README.md`, M4). The model is
`offboard_ros/coverage3d.py`, shared by the explorer (live, in the offboard container) and this folder's
offline check (`evaluate.py`).

## The model

- Every depth image is counted into a 5 cm voxel grid. A view of a voxel is **good** if it is within the
  camera range (2 m in the exploration; 2.5 m for the table below) and at most 60° from head-on.
- **The floor** has its own limits: up to 81° from head-on (from the camera's 26 cm height it is always
  seen at a flat angle, and the reconstruction builds it fine), but only within 1.3 m. The shiny floor
  gives hardly any depth from further away: share of valid floor pixels by distance in run
  20261006_142432: 0.7-1.3 m 72-85 %, 1.5-1.7 m 24 %, beyond 1.7 m < 5 %.
- **Heights are measured from the floor**, estimated live from the depth images (the most common height
  near the ground). In the map frame the floor sat 3.6-5.4 cm above z = 0 in all runs of 2026-10-06; the
  reconstruction levels it afterwards, the live model has to do it itself.
- A voxel is **covered** after 2 good views.

### Checked against the reconstruction (`evaluate.py`)

`evaluate.py <session>` (a reconstructed session) compares the model with what was actually built.
- **Expected surfaces** come from the lidar map:
  - the floor over the area the robot explored (its side of the no-go lines, within 3 m of its path);
  - walls and furniture 0.1-0.5 m high at the lidar's obstacles.
- **Output:** the numbers below, plus `<session>/coverage3d/check.png`, a top view:
  - green: covered and built;
  - red: predicted gap, really missing;
  - orange: predicted gap but built;
  - purple: predicted covered but missing.

| Run | Floor: prediction = reconstruction | Real floor gaps found | Walls: agreement |
|---|---|---|---|
| 20261005_124116 | 98 % | 97 % | 98 % |
| 20261005_142856 | 97 % | 96 % | 95 % |

The model predicts the reconstruction's holes reliably. It costs about 13 ms per depth image (one core),
about 5 % of a core at the 4 images/s the explorer receives.

## What the exploration uses: one rule for every surface

`coverage3d.grid_status` turns the voxels into targets on the 2D map, the same rule for floor, walls,
furniture and obstacles:

- A map cell (a 5 cm column up to 1.2 m) **holds a surface** if the camera saw a voxel in it at least
  twice, even poorly (a real surface, not a single flying pixel).
- It is **seen** once 80 % of those voxels have 2 good views. The floor counts as one surface, even when
  its noise spreads it over two voxel layers.
- **Once seen, it stays seen** unless it gets clearly worse: below 60 % and more than 2 voxels missing.
  Later far or grazing looks add voxels that are only "observed"; without this, cells flipped back to
  unseen at 0.1-0.4 m² per minute, each a new viewpoint (run 20261007_101108: coverage flat at 85 % for
  the last 7.5 of 16.8 minutes). Larger changes still count: the voxels never seen well were mostly
  0.6-1.25 m high, real parts of furniture.

The lidar is for discovery only: frontiers, and its walls the camera has not seen at all.

## History

- **2026-10-05, offline model** (above), validated with `evaluate.py`.
- **2026-10-05, "Quick 3D":** an exploration mode next to the 2D ones, with the floor in a 0.6 m band along
  the walls as an extra target.
- **2026-10-06, one mode, one rule:** Quick 3D became the only exploration mode, and the special cases
  were replaced by the rule above.
- **2026-10-06/07, from real runs:** the floor only within 1.3 m, the floor as one surface, heights from the
  estimated floor, "once seen stays seen".
