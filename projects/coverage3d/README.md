# 3D coverage from depth

Which surfaces has the depth camera seen well enough for the 3D reconstruction? The exploration steers by this:
it drives to viewpoints of surfaces not yet seen well (`projects/mapping`). The model is
`offboard_ros/coverage3d.py`, used live by the explorer in the offboard container; `evaluate.py` here checks it
against what a reconstruction actually built.

## Use

```powershell
python projects/coverage3d/evaluate.py data/sessions/<name>     # a reconstructed session
```

It prints the agreement between the model and the mesh and writes `<session>/coverage3d/check.png`, a top view:
green covered and built, red predicted gap and really missing, orange predicted gap but built, purple predicted
covered but missing.

## How it works

- Every depth image is counted into a 5 cm voxel grid. A view of a voxel is **good** if it is within the camera
  range (2 m in the exploration; 2.5 m in `evaluate.py`) and at most 60° from head-on.
- **The floor** has its own limits: up to 81° from head-on (from the camera's 26 cm height it is always seen at a
  flat angle, and the reconstruction builds it fine), but only within 1.3 m. The shiny floor gives hardly any depth
  from further away: share of valid floor pixels by distance in run 20261006_142432: 0.7-1.3 m 72-85 %,
  1.5-1.7 m 24 %, beyond 1.7 m < 5 %.
- **Heights are measured from the floor**, estimated live from the depth images (the most common height near the
  ground). In the map frame the floor sat 3.6-5.4 cm above z = 0 in all runs of 2026-10-06; the reconstruction
  levels it afterwards, the live model has to do it itself.
- A voxel is **covered** after 2 good views.
- **The exploration's targets** (`coverage3d.grid_status`), one rule for floor, walls, furniture and obstacles:
  - a map cell (a 5 cm column up to 1.2 m) **holds a surface** if the camera saw a voxel in it at least twice,
    even poorly (a real surface, not a single flying pixel);
  - it is **seen** once 80 % of those voxels have 2 good views; the floor counts as one surface, even when its
    noise spreads it over two voxel layers;
  - **once seen, it stays seen** unless it gets clearly worse (below 60 % and more than 2 voxels missing): later
    far or grazing looks add a voxel or two that were never seen well, and would otherwise turn it unseen again.
- The lidar is for discovery only: frontiers, and its walls the camera has not seen at all.

## Results and limits

`evaluate.py` takes the expected surfaces from the lidar map: the floor over the area the robot explored (its side
of the no-go lines, within 3 m of its path), and walls and furniture 0.1-0.5 m high at the lidar's obstacles.

| Run (2026-10-05) | Floor: prediction = reconstruction | Real floor gaps found | Walls: agreement |
|---|---|---|---|
| 20261005_124116 | 98 % | 97 % | 98 % |
| 20261005_142856 | 97 % | 96 % | 95 % |

The model predicts the reconstruction's holes reliably. It costs about 13 ms per depth image (one core), about 5 %
of a core at the 4 images/s the explorer receives. Limits: the check covers walls only 0.1-0.5 m high, and both
runs were recorded with the earlier exploration.

## Files

| File | What |
|---|---|
| `offboard_ros/coverage3d.py` | the model (voxel grid, floor height, `grid_status`), shared with the explorer |
| `evaluate.py` | the check against a reconstruction |
