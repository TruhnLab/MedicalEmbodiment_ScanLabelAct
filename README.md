<div align="center">

# Scan, Label, Act: A first step towards robots that assist medical staff

**Dennis Eschweiler** &nbsp;·&nbsp; **Daniel Truhn**

Lab for Artificial Intelligence in Medicine,
Department of Diagnostic and Interventional Radiology,
University Hospital RWTH Aachen

### [Project page](https://truhnlab.github.io/MedicalEmbodiment_ScanLabelAct/)

</div>

A [Hiwonder LanderPi](https://www.hiwonder.com/) explores a room by itself. From the recording, a laptop builds a
2D map, a 3D mesh and a list of the objects in the room. That labelled room then serves as the robot's model of the
world: the robot finds its position in it, drives to objects, looks for things with its camera, picks up small
items, and takes tasks in plain language through a chat with a vision-language model. The idea follows
[HomeBody](https://tml.stanford.edu/homebody/): scan the room once, then let the skills take their targets from it.

The robot only streams its sensors and takes commands. Everything else runs on a laptop without a GPU, plus an
optional model gateway and an optional GPU cluster.

```
 ROBOT (Raspberry Pi 5, ROS 2)          LAPTOP (Windows, CPU only)                 OPTIONAL
 ┌─────────────────────────┐            ┌─────────────────────────────────┐
 │ lidar, odometry         │──sensors──>│ mapping console: SLAM,          │──> model gateway (vLLM):
 │ depth + RGB camera      │            │   exploration, reconstruction,  │    labelling, chat
 │ mecanum base,           │<─commands──│   labelling                     │──> GPU cluster (Slurm):
 │ 5-servo arm, gripper    │            │ task console: localization,     │    labelling's detector
 └─────────────────────────┘            │   driving, picking, chat        │
                                        └─────────────────────────────────┘
```

## How it works

1. **Explore and record:** the robot maps the room with its lidar and drives to viewpoints until its depth
   camera has seen every surface well or given up on it after two tries, typically the hidden parts under
   tables (slam_toolbox, own route planner).
2. **Reconstruct:** the recorded keyframes are fused into a 3D mesh (Open3D).
3. **Label:** a vision-language model names what the room contains, an open-vocabulary detector (YOLOE) finds
   it in the images, and the results become 3D objects with labels.
4. **Navigate:** the robot localizes itself in the scanned room with its lidar and drives to objects or points;
   live detection finds things that are not in the scan.
5. **Manipulate:** the arm picks up small items from the floor and puts them down, steered by the camera on the
   gripper.
6. **Chat:** a language model gets the room's objects and the robot's status and acts through tools (drive,
   look, search, pick, put down).

Mapping, reconstruction and labelling work; navigation, live detection and the chat run on the robot; picking
works for small blocks and is still early.

## Repository layout

```
landerpi/            shared Python package (robot connection, settings, rooms, detector, chat tools)
console/             the web consoles for mapping and for tasks
tools/               local helper that serves the consoles and runs jobs; connection check; cluster CLI
offboard_ros/        ROS 2 container for the laptop: SLAM and exploration
projects/            mapping, scene_understanding, navigation, manipulation, coverage3d, presentation_video
gpu_cluster/         Slurm jobs
config/              calibration
docs/                operating the robot
```

Each folder in `projects/` has a README: how to use it, how it works, results and known limits.

## Getting started

You need the robot with its factory image, a Windows laptop with conda and Docker Desktop, and for labelling
and the chat an OpenAI-compatible model gateway.

```powershell
conda env create -f environment.yml
conda run -n landerpi pip install -e .
copy .env.example .env        # fill in the model gateway (and a cluster, if any)
```

Connect the laptop to the robot's Wi-Fi (`docs/robot.md`), then start `start_mapping_console.bat` to map a
room, or `start_task_console.bat` to work in a mapped one (`console/README.md`). The arm needs the robot's own
URDF, which is not included (`projects/manipulation/README.md`).

## License

The code in this repository is MIT (`LICENSE`). The consoles' JavaScript libraries (three.js, roslibjs, pako,
UPNG.js) are not part of it: the local helper downloads them on its first start, at fixed versions with checked
hashes, and they keep their own permissive licenses (list in `console/README.md`).

The object detector, [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOE, is not part of this
repository: it is installed as a dependency and its weights are downloaded at run time. It is licensed under
AGPL-3.0. The parts that use it (live detection and picking in the task console, the scene labelling;
`landerpi/detector.py`, `gpu_cluster/yoloe_frames.py`) are therefore used under AGPL-3.0 together with it;
anyone who distributes them with Ultralytics, or offers them as a network service, has to comply with
AGPL-3.0. Mapping, reconstruction and localization do not need it.

## Acknowledgements

[HomeBody](https://tml.stanford.edu/homebody/) for the idea; Hiwonder for the robot;
[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox), [Open3D](https://www.open3d.org),
[Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) and [Qwen](https://github.com/QwenLM).
