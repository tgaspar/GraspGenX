<!-- <img src="fig/cover.png" width="1000" height="250" title="readme1">  -->

<div align="center">
  <img src="assets/cover2.png" alt="GraspGenX logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <h1>GraspGenX: Cross-Embodiment Foundation Model for Grasp Generation</h1>
</div>
<p align="center">
  <a href="https://graspgenx.github.io">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2606.00998">
    <img alt="Arxiv paper link" src="https://img.shields.io/badge/arxiv-2606.00998-blue">
  </a>
  <a href="https://huggingface.co/adithyamurali/GraspGenXModel">
    <img alt="Model Checkpoints link" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://www.youtube.com/watch?v=a2sv9EVQJXE&feature=youtu.be">
    <img alt="Video link" src="https://img.shields.io/badge/video-red">
  </a>
  <a href="LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg">
  </a>
</p>

GraspGenX is a cross-embodiment grasp generation framework that produces high-quality 6-DOF grasps for **any** robot gripper — including novel grippers not seen during training. Unlike [GraspGen](https://github.com/NVlabs/GraspGen), which trains a separate model per gripper, GraspGenX trains a **single** model conditioned on a gripper representation derived from the gripper's swept volume. This enables generalization across grippers with different morphologies, kinematics, and degrees of freedom. We release pretrained model checkpoints trained with a large-scale simulated grasp dataset, spanning **over 2 billion grasps** computed across **32 procedurally generated grippers** in 6 kinematic families and **8000+ objects**.

<!-- <img src="assets/gifs/clutter_removal.gif" width="180" height="135" title="readme1"> <img src="assets/gifs/real_g1.gif" width="90" height="135" title="readme2"> <img src="assets/gifs/real_piper.gif" width="171" height="135" title="readme3"> <img src="assets/gifs/surge.gif" width="153" height="135" title="readme4"> <img src="assets/gifs/battery_removal.gif" width="180" height="135" title="readme5"> -->

<table>
  <tr>
    <td><img src="assets/gifs/clutter_removal.gif" height="135" alt="Clutter removal"></td>
    <td><img src="assets/gifs/real_g1.gif" height="135" alt="Unitree G1"></td>
    <td><img src="assets/gifs/real_piper.gif" height="135" alt="Real Piper"></td>
    <td><img src="assets/gifs/surge.gif" height="135" alt="Surge hand"></td>
    <td><img src="assets/gifs/battery_removal.gif" height="135" alt="Battery removal"></td>
  </tr>
</table>



## Contents

1. [Release News](#release-news)
2. [Upcoming Features](#upcoming-features)
3. [Installation](#installation)
   - [Docker](#installation-with-docker)
   - [uv Installation](#installation-with-uv)
4. [Inference Demos](#inference-demos)
   - [Object Point Clouds](#predicting-grasps-for-segmented-object-point-clouds)
   - [Object Meshes](#predicting-grasps-for-object-meshes)
   - [Scene Point Clouds](#advanced-predicting-grasps-for-objects-from-scene-point-clouds)
5. [End-to-End Demo And VLA Data Generation](#end-to-end-demos)
6. [Integrating a New Gripper](#integrating-a-new-gripper)
7. [REST API Server](#rest-api-server)
8. [Agentic Workflows](#agentic-workflows)
9. [FAQ](#faq)
10. [License](#license)
11. [Citation](#citation)
12. [Contact](#contact)

## Release News

- \[06/01/2026\] Initial code and model release!

## Upcoming Features

- Support for YAM gripper (feel free to contribute and make a PR!)
- DROID VLA pick-and-place data generation (feel free to contribute and make a PR!)
- Grasp Data Generation in Simulation (Newton)
- Full Training Dataset release

## Installation

Choose your preferred installation method. For training, we recommend **Docker**. For inference, **uv** is sufficient.

| Method | Use Case | Complexity |
|--------|----------|------------|
| **Docker** | Training + Inference | ⭐⭐⭐ |
| **uv** | Inference | ⭐ |

### Installation with uv
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:
```bash
git clone https://github.com/NVlabs/GraspGenX.git && cd GraspGenX && uv sync
```
> **Note:** `pip install -e .` also works inside a conda or Python virtual environment with python=3.10/3.11 .

### Installation with Docker
Training has only been tested inside the Docker.
```bash
git clone https://github.com/NVlabs/GraspGenX.git && cd GraspGenX && bash docker/build.sh
```

## Setup Checkpoints and Gripper Assets

### Checkpoints

The GraspGenX model checkpoints are downloaded automatically on the first import from [`https://huggingface.co/adithyamurali/GraspGenXModel`](https://huggingface.co/adithyamurali/GraspGenXModel) and cloned to `<repo_root>/ext/graspgenx_checkpoints`. If you have already cloned the checkpoints elsewhere, specify the path by setting the env variable:
```bash
export GRASPGENX_CHECKPOINT_DIR=/path/to/your/graspgenx_checkpoints
```

### Gripper Config and Assets

Each gripper needs a corresponding config file (with sweep volume information etc.) and URDF as an input to GraspGenX. We have already curated such gripper meta data for a host of popular grippers (see [gripper_descriptions](https://huggingface.co/datasets/adithyamurali/gripper_descriptions)). This package is needed at runtime for running inference on supported grippers. This package is automatically cloned on the first import to `<repo_root>/ext/gripper_descriptions`, but if you have downloaded it elsewhere you'll have to specify the following ENV variable:
```bash
export GRASPGENX_GRIPPER_CFG_DIR=/path/to/your/gripper_descriptions
```

Verify the assets can be imported properly:
```bash
uv run python -c "from graspgenx import get_gripper_descriptions_root; print(get_gripper_descriptions_root())"
```

To list all the available grippers:
```bash
uv run python scripts/list_grippers.py
```

## Inference Demos

GraspGenX supports two input modalities:
1. **Partial point cloud observations** — e.g. from a depth camera
2. **Object meshes** — `.obj`, `.stl`, `.ply`, or USD formats




### Predicting grasps for segmented object point clouds

The demo script generates grasp poses for segmented object point clouds and visualizes the results using [viser](https://viser.studio/). You can access the GUI by visiting [http://localhost:8080](http://localhost:8080) on a browser. Note that exact port may be different sometimes (it will be specified in the log output). Click Next to cycle through the objects. The gripper mesh for the top scoring grasp is visualized in blue.

```bash
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85 \
    --plot_top_mesh
```

<table>
  <tr>
    <td><img src="assets/figs/obj_pc_robotiq1.png" width="300" height="250" title="obj_pc_1"></td>
    <td><img src="assets/figs/obj_pc_robotiq2.png" width="268" height="250" title="obj_pc_2"></td>
    <td><img src="assets/figs/obj_pc_surge1.png" width="258" height="250" title="obj_pc_2"></td>
    <td><img src="assets/figs/obj_pc_unitree_g11.png" width="258" height="250" title="obj_pc_3"></td>
  </tr>
</table>

Replace `--gripper_name` with any supported gripper, e.g. `franka_umi`, `franka_panda`, `robotiq_2f_85`, `robotiq_2f_140`, `unitree_g1`, `inspire_hand`, `surge_hand`, `barrett_hand`, `galaxea_g1`, `ezgripper`. See [Supported Grippers](#supported-grippers) for the full list. By default the demo uses the **GraspMoE** planner (grasps sampled with diffusion ∪ Oriented-Bounding grasps, all scored by the discriminator). To fall back to the diffusion-only planner: `--planner diffusion`. To restrict the GraspMoE OBB sweep to the top face only: `--moe_obb_density sparse` (single centroid pose) or `--moe_obb_density dense` (positions along the OBB's longer XY axis, top-down only).

### Predicting grasps for objects from scene point clouds

The scene point cloud demo loads full-scene point clouds (with per-object segmentation), runs GraspGenX inference on every segmented object, and visualizes the predicted grasps in the context of the full scene. Grasps that would collide with the rest of the scene are filtered out. You can access the GUI by visiting [http://localhost:8080](http://localhost:8080) on a browser.

```bash
uv run python scripts/demo_scene_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85
```

<table cellspacing="20">
  <tr>
    <td style="padding: 0 15px;"><img src="assets/figs/scene_pc_1.png" height="300" width="420" title="scenepc1"></td>
    <td style="padding: 0 15px;"><img src="assets/figs/scene_pc_2.png" height="300" width="420" title="scenepc2"></td>
    <td style="padding: 0 15px;"><img src="assets/figs/scene_pc_3.png" height="300" width="300" title="scenepc3"></td>
    <td style="padding: 0 15px;"><img src="assets/figs/scene_pc_4.png" height="300" width="300" title="scenepc4"></td>
  </tr>
</table>

By default, the model runs GraspMoE (Grasp Mixture-of-Experts) sampling with both the diffusion model or Oriented-Bounding-Box (OBB)-based heuristics. To use just the diffusion model, you can use `--planner diffusion` and to remove collision checking, you can add `--no-filter_collisions`.

### Predicting grasps for object meshes

The script samples points from the mesh surface, runs GraspGenX inference, and visualizes both the mesh and predicted grasps. You can access the GUI by visiting [http://localhost:8080](http://localhost:8080) on a browser.

```bash
uv run python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/banana.obj \
    --mesh_scale 1.0 \
    --gripper_name inspire_hand \
    --grasp_threshold -1.0 --return_topk --topk_num_grasps 100 --plot_top_mesh
```

Supported mesh formats: `.obj`, `.stl`, `.ply`, and USD (`.usd`, `.usda`, `.usdc`, `.usdz`). USD files are loaded via [scene_synthesizer](https://github.com/NVlabs/scene_synthesizer). For headless/batch inference (no visualization), add `--no-visualization --output_file /tmp/grasps.yml`.

<table>
  <tr>
    <td><img src="assets/figs/obj_mesh_arx_x5.png" width="300" title="arx_x5"></td>
    <td><img src="assets/figs/obj_mesh_surge_hand.png" width="268" title="arx_x5"></td>
    <td><img src="assets/figs/obj_mesh_unitree_g1.png" width="258" title="arx_x5"></td>
  </tr>
</table>

## End-to-End Demos

Beyond predicting grasps, [`end2end/`](end2end/README.md) wires GraspGenX into
a full closed-loop pick-and-place pipeline: **GraspGenX** grasps → **cuRobo**
collision-free motion planning → **Newton/MuJoCo** physics replay (gravity +
contacts) → **MP4** render + **USD** export for IsaacSim / Omniverse. Four
example demos ship out of the box — a Franka Panda (single object → bin, and a
3-object clutter scene → bin) and a UR10e with two multi-finger hands
(arx_x5 pick-and-lift, surge_hand pick → bin).

This pipeline needs an extra dependency stack (cuRobo, Newton, MuJoCo, CoACD,
USD, fcl):

```bash
uv sync --extra end2end
```

See [`end2end/README.md`](end2end/README.md) for the run command, config
layout, and visualization tools.

## Integrating a New Gripper

GraspGenX runs zero-shot on any gripper given a URDF + meshes. The only thing GraspGenX needs that a stock URDF does not provide is a `config.json` describing the open/close joint states and the swept-volume bounding boxes that condition the model. The interactive **gripper config wizard** generates that file (and copies the URDF/meshes into the right place) for you; everything else — visual mesh, collision mesh, point-cloud cache, etc. — is produced automatically downstream.

### Run the wizard

```bash
uv run python scripts/gripper_config_wizard.py \
    --urdf /path/to/your/gripper.urdf \
    --name <gripper_name> \
    --port 8081
```

For example, to onboard the right Sharpa hand from a sibling `gripper_descriptions` checkout:

```bash
uv run python scripts/gripper_config_wizard.py \
    --urdf ../gripper_descriptions/gripper_descriptions/assets/sharpa/right_sharpa_wave.urdf \
    --name sharpa_right \
    --port 8081
```

Open `http://localhost:8081` and step through the GUI. There are six panels — the wizard prompts you in order and you click "Confirm" to advance:

1. **Align Base Frame.** Rotate so `+Z` is the approach axis (along the fingers) and `+X` is the closing direction.
2. **Confirm Open Configuration.** Drive joint sliders to the fully-open pose.
3. **Annotate Open Sweep Volume.** Drag the blue box to enclose the inner finger volume at the open state.
4. **Annotate Half-Open Sweep Volume.** Set the closed pose, then drag the orange box to enclose the inner volume at the half-open state.
5. **Review Animation.** Watch open ↔ close with both boxes overlaid; go back if anything looks off.
6. **Confirm and Save.** Pick the gripper type (`parallel_2f`, `revolute_2f`, `revolute_3f`) and symmetry flag, then save.

### What gets written

On save, the wizard writes a fresh directory at `<gripper_descriptions_root>/assets/x_grippers/<name>/` (resolved via `$GRASPGENX_GRIPPER_CFG_DIR`, or the auto-cloned `<repo>/ext/gripper_descriptions/` — see [Gripper Descriptions Asset Repo](#gripper-descriptions-asset-repo)) containing:

- `gripper.urdf` and `meshes/` — copied from your source URDF.
- `config.json` — the open/close joint states, fingertip, swept-volume boxes, gripper type, and base rotation you annotated in the GUI. This is the file the model conditions on.
- `vis_mesh.obj` — merged visual mesh of the gripper at the open pose.

The collision mesh and point-cloud caches needed at inference time are generated lazily on first use; you don't need to run anything else.

### Verify

Spin up the descriptor viewer to sanity-check the saved config (animates open ↔ close with both sweep volumes overlaid):

```bash
uv run python scripts/vis_gripper_desc.py --gripper <gripper_name> --port 8081
```

Then run inference against a point cloud as you would for any built-in gripper. Any of the demo scripts should work:

```bash
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name <gripper_name> \
    --plot_top_mesh
```

### Please upstream your gripper

If you've onboarded a gripper that isn't already in [`gripper_descriptions`](https://huggingface.co/datasets/adithyamurali/gripper_descriptions), please consider opening a PR with the saved `assets/x_grippers/<name>/` directory. The community benefits enormously from a growing library of curated grippers, and your config will let everyone else skip the wizard for that hand.

### Procedural Gripper Assets

Procedural gripper meshes and URDFs are stored under `assets/proc_grippers/`, organized by kinematic family. Each gripper directory contains the URDF, collision meshes, and swept volume point cloud.


## REST API Server

`scripts/serve_grasp_predictor.py` serves a GraspGenX checkpoint over HTTP.
It speaks the request/response contract in
[`docs/api/predict.md`](docs/api/predict.md) — the **same contract as the
DexGraspNet 2.0 server** — so one ROS 2 client can be pointed at either
backend without code changes. Send a segmented object point cloud, get back a
ranked list of 6-DoF grasps in the input frame.

Install the extra dependencies with `uv sync --extra rest` (the Docker image
already has them).

### Run it

```bash
docker run --rm --gpus all --ipc=host --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/code \
  -v /path/to/GraspGenX:/code \
  -w /code \
  x_grasp:3.0 \
  python scripts/serve_grasp_predictor.py \
    --gripper arx_x5 \
    --host 0.0.0.0 \
    --port 8000 \
    --viser-port 8080
```

Checkpoints and gripper assets are resolved automatically from
`ext/graspgenx_checkpoints` and `ext/gripper_descriptions` (see [Setup
Checkpoints and Gripper Assets](#setup-checkpoints-and-gripper-assets)); the
bind mount above makes them visible inside the container. The model loads
before the port opens, so a successful connection means the server is ready —
roughly 40 s including the startup warmup inference.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Object point cloud (+ optional scene clutter) → ranked grasps |
| `GET /healthz` | Liveness probe |
| `GET /config` | Active gripper, checkpoint, frame convention, point-count limits |
| `GET /version` | Build metadata |

```bash
curl -s localhost:8000/config | python -m json.tool
```

### Grasp frame

GraspGenX natively predicts the pose of the gripper's **base link** with
**+Z = approach** and **+X = jaw-closing**. Since other stacks label the same
physical pose differently, the server can emit any of three conventions:

| `--grasp-frame` | Approach | Jaw | For |
|---|---|---|---|
| `dexgraspnet2` *(default)* | +X | +Y | Drop-in for a DexGraspNet 2.0 client |
| `graspgenx` | +Z | +X | The model-native pose |
| `tcp_z_approach` | +Z | +Y | Robot TCPs with Z along the fingers |

`--tcp-offset <m>` (or `--tcp-at-fingertip`) shifts the returned origin along
the approach axis — use it to report your hand's actual TCP rather than the
gripper base. `/config` reports `grasp_frame`, `approach_axis_local`,
`jaw_axis_local`, `tcp_offset` and `grasp_reference` so a client can adapt
automatically instead of hardcoding a convention.

### Clutter

Passing `scene_points` alongside `point_cloud` makes the server collision-check
each predicted grasp's gripper mesh against the surrounding geometry and drop
the blocked ones. This is most useful for bin picking and dense clutter. For an
isolated object on a table, leave it out: the fingers legitimately come within
a millimeter or two of the support surface, so a geometric filter rejects
almost everything unless you strip the support plane client-side first.
`meta.num_collision_rejected` tells you how many were dropped.

### Web visualizer

Pass `--viser-port 8080` and the server also serves a live [viser](https://viser.studio/)
scene at `http://<host>:8080`. Every `/predict` call re-renders it: the target
cloud in orange, the scene clutter in grey, the top grasps as score-coloured
gripper outlines, and a solid gripper mesh at the best one, plus a side panel
with the request id, point counts and score range. Open it in a browser and
leave it up while your client runs — it is the quickest way to see whether the
poses coming back actually make sense on the object.

Poses are drawn in the GraspGenX **native** frame (+Z approach, +X jaw)
whatever `--grasp-frame` the HTTP response uses, since that is the frame the
gripper mesh lives in. `--viser-top-k` controls how many grasps are drawn.
Visualizer failures are logged and swallowed — they never affect a response.

### Trying it against live ROS 2 data

`scripts/dev_ros2_predict_smoke_test.py` is a throwaway dev tool that pulls a
depth image, camera info and a segmentation mask off the ROS 2 graph, builds
the request exactly as a real client would, and prints the result with
self-consistency checks. It works against either backend:

```bash
source /opt/ros/jazzy/setup.bash
python scripts/dev_ros2_predict_smoke_test.py --url http://localhost:8000 --publish
```

See [`docs/api/predict.md`](docs/api/predict.md) for the full schema, the
error table, and the behavioural differences from DexGraspNet 2.0.

## Agentic Workflows

See `mcp/` and `client-server/`.


## Testing

The `end2end` takes a while longer and can be skipped:
```bash
uv run --no-sync pytest -m "not end2end"
```

## FAQ

### How does GraspGenX differ from GraspGen?

GraspGen trains a **separate model per gripper**. GraspGenX trains a **single model** that generalizes across grippers by conditioning on a swept-volume representation of the gripper. This means GraspGenX can generate grasps for novel grippers not seen during training, without re-training.

### What grippers does GraspGenX support?

GraspGenX supports any gripper for which you can compute a swept volume. The released model was trained on 32 procedural grippers spanning 6 kinematic families (parallel 2-finger, revolute 2-finger, revolute 3-finger — each with two design variants). It generalizes zero-shot to multiple real-world grippers.

### How do I add a new gripper?

See the [Integrating a New Gripper](#integrating-a-new-gripper) section. The interactive wizard generates `config.json` (and copies the URDF/meshes) for any gripper given just its URDF.

### Can I use the GraspGen checkpoints with GraspGenX?

No. GraspGen and GraspGenX use different model architectures (per-gripper vs. cross-embodiment conditioning). You need GraspGenX-specific checkpoints.


### How do I report a bug?

Please post a GitHub issue and we will follow up, or email us directly.

## License

This project is licensed under the [Apache License 2.0](LICENSE). The model checkpoints are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

Please email us (admurali@nvidia.com) if you are using this repository for commerical deployment and require any support.

## Citation

If you found this work useful, please consider citing:

```
@inproceedings{graspgenx2026,
  title     = {GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping},
  author    = {Han, Beining and Chao, Yu-Wei and Coumans, Erwin and Eppner, Clemens and Sundaralingam, Balakumar and Deng, Jia and Birchfield, Stan and Murali, Adithyavairavan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
## Third-Party Software

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Contact

Please reach out to [Adithya Murali](http://adithyamurali.com) (admurali@nvidia.com) for further enquiries.
