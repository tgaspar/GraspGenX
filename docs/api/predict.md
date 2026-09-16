# `POST /predict` — Grasp prediction API

Contract between a ROS 2 client and the GraspGenX inference server running
inside the Docker container. This document is the source of truth for both
sides.

This is the **same wire contract** as the DexGraspNet 2.0 server
(`DexGraspNet2/docs/api/predict.md`), so one client can point at either
backend. Where the two backends genuinely cannot behave identically, this
document says so explicitly and the GraspGenX server exposes a CLI flag to
pick the behaviour — see [Compatibility with DexGraspNet 2.0](#compatibility-with-dexgraspnet-20).

---

## TL;DR

Send the point cloud of the object you want to grasp. **Optionally** also send
the surrounding-scene points so grasps colliding with the clutter get dropped.
Get a ranked list of grasp poses back, in the same coordinate frame as the
input. All geometry in meters / radians / unit quaternions.

```
POST /predict
Content-Type: application/json

Body (minimal):
  { "point_cloud": ..., "num_grasps": 20 }

Body (clutter-aware):
  { "point_cloud": ..., "scene_points": ..., "num_grasps": 20 }

→ 200 OK
   { "grasps": [ {translation, rotation_quat, ...}, ... ], "meta": { ... } }
```

**Segmentation is the client's responsibility.** The client separates
target-object points from everything else. The target goes in `point_cloud`;
optionally, the rest of the scene goes in `scene_points` so the server can
reject grasps whose gripper geometry would collide with the surroundings.

---

## Endpoint

```
POST /predict
Content-Type: application/json
Accept:       application/json
```

No authentication assumed (the server is expected to live on a local network or
`localhost` only). If auth is ever added, it'll be a `X-Api-Key` header; plan
for that column being absent for now.

---

## Server assumptions

- The server is **configured at startup** for one specific gripper (`--gripper
  arx_x5`, `franka_panda`, …). Clients do not specify the gripper per-request;
  the server answers with whatever it was loaded with. Run multiple containers
  on different ports to serve multiple grippers.
- `GET /config` is provided so a client can introspect which gripper / model /
  frame convention is active (see below).
- Frame: the server does **no frame transformations**. All input points are
  interpreted verbatim, and all returned grasp poses are in the **same frame**
  as the input point cloud. The client is responsible for any TF gymnastics.
- Units: positions in meters, quaternions in `(x, y, z, w)` ordering (ROS
  convention), rotation matrices are row-major 3×3.
- **Input expectation**: every point in `point_cloud` belongs to the object you
  want grasps on. Optional `scene_points` supplies the *rest* of the scene
  (non-target objects, table, bin walls) so the server can collision-check each
  predicted grasp's gripper mesh against it.

---

## Grasp-frame axis convention

This is the part most worth reading carefully.

GraspGenX natively predicts the 6-DoF pose of the **gripper's base link** — the
URDF root, i.e. the frame where the gripper bolts onto the flange — with

| Axis | Role |
|---|---|
| **+Z** | approach direction — fingers extend along +Z, hand moves +Z to reach the object |
| **+X** | jaw-closing axis — the two fingers sit at `± aperture/2` along X |
| **+Y** | the thin direction (irrelevant to planning) |

The fingertips sit at `+Z · fingertip_depth` from that origin (0.143 m for
`arx_x5`), and the fully-open aperture is `gripper_max_width` (0.085 m for
`arx_x5`). Both are reported by `/config`.

Because different stacks label the same physical pose differently, the server
can **emit** the rotation in any of three conventions, selected at startup with
`--grasp-frame`:

| `--grasp-frame` | Approach axis | Jaw-closing axis | Use when |
|---|---|---|---|
| `dexgraspnet2` **(default)** | **+X** | **+Y** | Drop-in for a client already written against the DexGraspNet 2.0 server |
| `graspgenx` | +Z | +X | You want the model-native pose (matches GraspGenX demos / visualizers) |
| `tcp_z_approach` | +Z | **+Y** | Your robot's TCP has Z along the fingers and the jaws open along Y |

Internally the emitted rotation is `R_out = R_native @ C`, where `C` is a pure
change of basis; the physical pose is identical in all three, only the axis
*labels* move.

The origin of the returned pose can be shifted along the approach axis with
`--tcp-offset <meters>` (or `--tcp-at-fingertip`, which uses the gripper's
`fingertip_depth`). `/config.grasp_reference` reports what you asked for:

```
grasp_reference = "gripper_base_link"                          # --tcp-offset 0.0 (default)
grasp_reference = "gripper_base_link+0.1430m_along_approach"   # --tcp-at-fingertip on arx_x5
```

Derived quantities clients frequently want (shown for the default
`dexgraspnet2` frame):

```
approach_vec_world  = rotation_matrix @ [1, 0, 0]   # also returned as `approach_axis`
left_finger_world   = translation + rotation_matrix @ [0, +gripper_width/2, 0]
right_finger_world  = translation + rotation_matrix @ [0, -gripper_width/2, 0]
```

For `graspgenx` / `tcp_z_approach` the approach column is `[0, 0, 1]` instead;
for `graspgenx` the jaw column is `[1, 0, 0]`. `/config` reports both as
`approach_axis_local` and `jaw_axis_local`, so a client can stay
convention-agnostic:

```python
cfg = requests.get(f"{BASE}/config").json()
a = np.array(cfg["approach_axis_local"])     # e.g. [1, 0, 0]
j = np.array(cfg["jaw_axis_local"])          # e.g. [0, 1, 0]
approach_world = np.array(g["rotation_matrix"]) @ a
```

`approach_axis` in each grasp is already the world-frame approach direction, so
most clients never need that math.

---

## Request body

JSON object with the following fields.

| Field | Type | Required | Description |
|---|---|---|---|
| `point_cloud` | object (see below) | ✓ | `(N, 3)` array of XYZ points in meters, filtered to only the target object. |
| `scene_points` | object (see below) | ✗ | `(M, 3)` array of XYZ points for the *rest* of the scene (non-target objects, table, container, etc.). When present, the server collision-checks every predicted grasp's gripper mesh against these points and drops the ones that would hit something. `N + M` must not exceed `max_points`. |
| `num_grasps` | int | ✗ | Max number of grasps to return. Default: `20`. Server may return fewer. |
| `min_score` | float | ✗ | Filter: drop grasps whose discriminator confidence is below this value. Default: `0.0` (keep all). GraspGenX scores live in `[0, 1]` — see [Scores](#scores). |
| `category_sampling` | bool / null | ✗ | **Accepted and ignored** by this backend; echoed in `meta.category_sampling`. GraspGenX conditions only on the target-object cloud and has no per-instance seed-sampling knob. Present so a DexGraspNet 2.0 client can send it without a 4xx. |

### When to use `scene_points`

- **Isolated object on a table / bench** → omit it. Cleaner request, and the
  model output is unchanged either way.
- **Object in a bin / bowl / stacked pile / next to clutter** → include it. The
  gripper geometry is checked against the clutter and blocked approaches get
  dropped, so the top-K is far more likely to be executable.
- **You already have a scene PC from your perception stack** → include
  everything that is NOT the target object.

Tuning: `--collision-threshold` (default 0.02 m) sets how close a gripper
surface sample may come to a scene point before the grasp counts as colliding.
`meta.num_collision_rejected` tells you how many were dropped; if that number
is most of your candidates, either your `scene_points` still contain the target
object, or the threshold is too aggressive.

### Encoded-array subschema for `point_cloud`

The point cloud is packed as a numpy array in JSON without ballooning the
payload:

```json
{
  "dtype":    "float32" | "float64",
  "shape":    [N, 3],
  "data_b64": "AAAA..."
}
```

Semantics:
- `dtype`: numpy dtype string. Server accepts `float32` and `float64`.
- `shape`: declared shape, must be `[N, 3]` with `N >= 1`.
- `data_b64`: raw little-endian bytes, base64-encoded. Standard
  `base64.b64encode(array.tobytes())` on the client side.
- Server validates `prod(shape) * dtype_size == len(decoded_bytes)` and rejects
  mismatches with `400`.

### Example request (minimal, no scene context)

```json
{
  "point_cloud": {
    "dtype": "float32",
    "shape": [4096, 3],
    "data_b64": "..."
  },
  "num_grasps": 10,
  "min_score": 0.5
}
```

### Example request (clutter-aware)

```json
{
  "point_cloud": {
    "dtype": "float32",
    "shape": [2048, 3],
    "data_b64": "..."
  },
  "scene_points": {
    "dtype": "float32",
    "shape": [8192, 3],
    "data_b64": "..."
  },
  "num_grasps": 10
}
```

---

## Response body (success, 200)

```json
{
  "grasps": [
    {
      "translation":      [x, y, z],
      "rotation_quat":    [qx, qy, qz, qw],
      "rotation_matrix":  [[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]],
      "score":            0.96,
      "gripper_width":    0.085,
      "joint_angles":     null,
      "joint_names":      null,
      "approach_axis":    [0.0, 0.0, -1.0],
      "object_id":        1,
      "branch":           "obb"
    }
  ],
  "meta": {
    "hand":             "arx_x5",
    "model_name":       "GraspGenX/release",
    "checkpoint":       "epoch_736.pth+epoch_1056.pth",
    "frame_id":         "unchanged_from_input",
    "num_input_points": 4096,
    "inference_ms":     640.9,
    "server_version":   "0.1.0"
  }
}
```

### Grasp object — field-by-field

| Field | Type | Always present? | Description |
|---|---|---|---|
| `translation` | `[float, float, float]` | ✓ | Position of the grasp reference point in the **input point-cloud frame**, meters. Which point on the hand this is, is server-configured (`--tcp-offset`) and reported via `/config.grasp_reference`. Default: the gripper base link. |
| `rotation_quat` | `[qx, qy, qz, qw]` | ✓ | Orientation as a unit quaternion, ROS `(x, y, z, w)` ordering. Sign-canonicalised to `qw >= 0`. |
| `rotation_matrix` | `[[float]*3]*3` | ✓ | Same orientation as a row-major 3×3. Redundant with `rotation_quat` but both are returned so the client doesn't have to convert. Axis roles depend on `--grasp-frame`. |
| `score` | `float` | ✓ | GraspGenX discriminator confidence in `[0, 1]`. Higher is better; the list is sorted descending. |
| `gripper_width` | `float` or `null` | ✓ (unless `--width-mode none`) | Target jaw opening at the grasp moment, meters. See [Gripper width](#gripper-width) — GraspGenX does not predict this, so how it is filled is a server setting. |
| `joint_angles` | `null` | ✓ | Always `null`. GraspGenX predicts a 6-DoF pose plus an opening, never a joint vector — even for the multi-finger grippers in its library. |
| `joint_names` | `null` | ✓ | Always `null`, for the same reason. |
| `approach_axis` | `[float, float, float]` | ✓ | Unit vector **in the input point-cloud frame** giving the direction the hand travels to reach the grasp from its pregrasp pose. Usable directly for trajectory planning, no rotation math needed. |
| `object_id` | `int` | ✓ | Always `1` by default: GraspGenX only generates grasps on the target-object cloud, never on clutter. Configurable via `--object-id`. |
| `branch` | `"diff"` \| `"obb"` | ✓ | **GraspGenX extra** (not in the DexGraspNet 2.0 contract). Which GraspMoE branch produced this grasp: `diff` = diffusion sample, `obb` = oriented-bounding-box sweep candidate. Both are scored by the same discriminator. Always `"diff"` when `--planner diffusion`. Safe to ignore. |

### Meta object

| Field | Type | Description |
|---|---|---|
| `hand` | string | The gripper the server was started with, e.g. `"arx_x5"`. |
| `model_name` | string | Checkpoint label, e.g. `"GraspGenX/release"`. |
| `checkpoint` | string | `"<generator>.pth+<discriminator>.pth"`. |
| `frame_id` | string | Always `"unchanged_from_input"` — reminder that grasps are in the input frame. |
| `num_input_points` | int | Count of target-object points (N) the server received. |
| `num_scene_points` | int | Count of `scene_points` (M) received; `0` if omitted. |
| `context_mode` | string | `"object_only"`, `"scene_collision_filtered"`, `"scene_ignored"` (server started with `--no-scene-collision-filter`), or `"scene_collision_filter_failed"` (the filter raised and results were returned unfiltered). |
| `category_sampling` | bool | Echo of the request field. Always a no-op on this backend. |
| `inference_ms` | float | Wall-clock ms spent in planning + collision filtering (excluding HTTP overhead). |
| `server_version` | string | Semver-ish version of the server build. |
| `request_id` | string | Short UUID for this request; matches filenames under `<log_dir>/predictions/`. |
| `debug_html` / `debug_usd` | null | Always `null`. Present for schema parity with the DexGraspNet 2.0 server, which can render per-request Plotly / USD dumps. |
| `debug_npz` | string or null | Path to the per-request NPZ dump when the server was started with `--dump-npz`. |
| `backend` | string | `"graspgenx"`. **Extra** — the cheapest way for a client to log which server answered. |
| `api_version` | string | Contract version, `"0.1.0"`. **Extra.** |
| `gripper`, `grasp_frame`, `tcp_offset`, `planner`, `width_mode` | — | **Extras** echoing the server's configuration, so a grasp log is self-describing. |
| `num_collision_rejected` | int | **Extra.** How many candidate grasps the `scene_points` collision filter dropped. |

### Scores

GraspGenX's `score` is the discriminator's confidence, bounded in `[0, 1]`;
values above ~0.8 are strong. This differs from the DexGraspNet 2.0 server,
whose score is an unbounded logit-like quantity (values in the tens are
routine). **A hardcoded `min_score` that works against one backend will not
transfer to the other.** Query `/config.backend` and pick your threshold
accordingly, or just leave `min_score` at `0.0` and take the top-K, which is
backend-agnostic.

### Gripper width

GraspGenX predicts a 6-DoF pose; it does **not** predict a jaw opening. The
server fills `gripper_width` according to `--width-mode`:

| `--width-mode` | `gripper_width` is | Notes |
|---|---|---|
| `aperture` **(default)** | the gripper's fully-open aperture (`/config.gripper_max_width`) — a constant | Safe: the jaws are wide open on approach and you close on contact / to a force target. |
| `measured` | the span of the object points that fall inside the jaw sweep volume, plus `--width-clearance` (default 1 cm), clamped to the aperture | A geometric estimate from the input cloud, not a model output. Useful for pre-shaping the jaws; falls back to the aperture when no object points land inside the jaws. |
| `none` | `null` | For clients that command the gripper by force and never read the field. |

---

## Errors

| Status | Condition | Body |
|---|---|---|
| 400 | Malformed JSON, missing `point_cloud`, encoded-array shape/dtype mismatch, `num_grasps < 1`, `point_cloud.shape` not `[N, 3]`, `scene_points.shape` not `[M, 3]`. | `{"error": "<code>", "message": "<human description>"}` |
| 413 | Total `N + M` exceeds `max_points` (default limit: 65536). | `{"error": "payload_too_large", "max_points": 65536}` |
| 422 | `point_cloud` or `scene_points` contains NaN or Inf values, or `point_cloud` has fewer than `min_points` target-object points (default: 128; `scene_points` is unconstrained on the low end). | `{"error": "invalid_geometry", "message": "..."}` |
| 503 | Server starting up, checkpoint not yet loaded, or GPU unavailable. | `{"error": "not_ready", "retry_after_seconds": 5}` |
| 500 | Unhandled model exception. | `{"error": "internal_error", "message": "..."}` |

Error codes are stable machine-readable strings; messages are human-readable
and may change. Note that FastAPI wraps these bodies in a `detail` object, i.e.
the response is `{"detail": {"error": ..., "message": ...}}` — the same shape
the DexGraspNet 2.0 server produces.

Two cases are caught by pydantic before the handler runs, so they come back as
**422** with pydantic's own `detail` *list* rather than the table's code above:
a missing `point_cloud`, and `num_grasps < 1`. Both servers behave identically
here, since both declare the same schema.

A request that is well-formed but yields nothing graspable returns **200** with
`"grasps": []`, not an error. Clients must handle the empty list.

---

## Auxiliary endpoints

### `GET /healthz` — liveness

- 200 `{"status": "ok"}` when the server is up and the model is loaded.
- 503 `{"status": "loading"}` during startup / checkpoint load.
- Intended for supervisor probes. Cheap; does not run inference.

Note that the model is loaded *before* the HTTP port opens, so in practice a
connection refused means "still loading" and a successful connect means ready.

### `GET /config` — introspection

Returns what the server is configured with. Call once at client startup.

```json
{
  "hand":                      "arx_x5",
  "model_name":                "GraspGenX/release",
  "checkpoint":                "epoch_736.pth+epoch_1056.pth",
  "joint_names":               null,
  "grasp_reference":           "gripper_base_link",
  "default_num_grasps":        20,
  "min_points":                128,
  "max_points":                65536,
  "server_version":            "0.1.0",

  "backend":                   "graspgenx",
  "api_version":               "0.1.0",
  "gripper":                   "arx_x5",
  "grasp_frame":               "dexgraspnet2",
  "grasp_frame_description":   "+X approach, +Y jaw-closing (DexGraspNet 2.0 convention)",
  "approach_axis_local":       [1.0, 0.0, 0.0],
  "jaw_axis_local":            [0.0, 1.0, 0.0],
  "tcp_offset":                0.0,
  "gripper_max_width":         0.085,
  "fingertip_depth":           0.143,
  "planner":                   "graspmoe",
  "width_mode":                "aperture",
  "approach_axis_convention":  "true_approach",
  "scene_points_mode":         "collision_filter",
  "supports_category_sampling": false
}
```

The first block is the fields the DexGraspNet 2.0 server also returns; the
second is GraspGenX-specific and is what lets a client adapt automatically to
whichever backend is running.

### `GET /version` — build metadata

```json
{
  "server_version": "0.1.0",
  "api_version":    "0.1.0",
  "backend":        "graspgenx",
  "model_commit":   "",
  "built_at":       "2026-09-16T12:43:20Z"
}
```

---

## Compatibility with DexGraspNet 2.0

The request schema is **identical** — the same JSON body works against both
servers unmodified. The response schema is identical too, plus a few additive
GraspGenX-only fields (`branch`, `meta.backend`, …) that a strict client can
ignore.

Four behavioural differences are worth knowing, three of which have a CLI flag:

| # | Difference | What to do |
|---|---|---|
| 1 | **Grasp-frame axes.** GraspGenX is natively +Z approach / +X jaw; DexGraspNet 2.0 is +X approach / +Y jaw. | Default `--grasp-frame dexgraspnet2` already matches DexGraspNet 2.0. Use `--grasp-frame tcp_z_approach` if your robot TCP is +Z approach / +Y jaw and you'd rather drop the client-side fixup. |
| 2 | **Grasp origin.** GraspGenX returns the gripper *base link*; DexGraspNet 2.0 returns the midpoint between the jaws at the finger base. These are different physical points (0.143 m apart on `arx_x5`). | Set `--tcp-offset` to shift along the approach axis, or `--tcp-at-fingertip`. There is no automatic equivalence — the two models describe different hands, so pick the offset that matches *your* hand's TCP. |
| 3 | **`approach_axis`.** The DexGraspNet 2.0 server returns `rotation_matrix @ [0,0,1]` (its frame's +Z), even though its documented approach axis is +X — so its `approach_axis` is not actually the approach direction. GraspGenX returns the true approach direction. | Default `--approach-axis-convention true_approach` is the correct one. If your client is calibrated against DexGraspNet 2.0's behaviour, `--approach-axis-convention frame_z` reproduces it exactly. |
| 4 | **Score range.** GraspGenX: `[0, 1]`. DexGraspNet 2.0: unbounded (tens). | No flag — the scores come from different heads. Branch on `/config.backend`, or use top-K instead of an absolute `min_score`. |

Two smaller notes:

- `scene_points` is handled differently by design. DexGraspNet 2.0 feeds the
  clutter to its backbone with a target/scene segmentation mask and can return
  grasps anchored on the clutter (`object_id == 0`). GraspGenX conditions on
  the target cloud alone, so it uses `scene_points` as **collision geometry**
  and returns only target grasps. The intent — don't propose approaches that
  are physically blocked — is the same; the mechanism differs.
- `object_id` is `1` here by default, per the contract. The live DexGraspNet
  2.0 server emits `-1` in object-only mode. A client filtering on `== 1`
  therefore drops everything from DexGraspNet 2.0 today; if yours mirrors that
  quirk, `--object-id -1` matches it.

---

## Example end-to-end call from Python

```python
import base64, numpy as np, requests

BASE = "http://localhost:8000"

def _encode(arr: np.ndarray) -> dict:
    return {
        "dtype":    str(arr.dtype),
        "shape":    list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }

# Target-object points — e.g. from your depth + segmentation mask.
target_pc = np.random.randn(2048, 3).astype(np.float32)

# Everything else in the scene — table, other objects, etc.
# Omit this field entirely if you only have / want the target.
scene_pc = np.random.randn(8192, 3).astype(np.float32)

body = {
    "point_cloud":  _encode(target_pc),
    "scene_points": _encode(scene_pc),   # optional
    "num_grasps":   10,
}

r = requests.post(f"{BASE}/predict", json=body, timeout=30.0)
r.raise_for_status()
data = r.json()

grasps = data["grasps"]
print(f"backend:      {data['meta'].get('backend', 'dexgraspnet2')}")
print(f"context_mode: {data['meta']['context_mode']}")
print(f"Inference:    {data['meta']['inference_ms']:.1f} ms")
print(f"{len(grasps)} grasps, "
      f"{data['meta'].get('num_collision_rejected', 0)} rejected by collision")
if grasps:
    g = grasps[0]
    print(f"Best: t={g['translation']} "
          f"approach={g['approach_axis']} width={g['gripper_width']}")
```

---

## ROS 2 integration notes

Conversion from `sensor_msgs/PointCloud2` to the request body:

```python
from sensor_msgs_py import point_cloud2

# Convert PointCloud2 → (N, 3) float32
xyz = point_cloud2.read_points_numpy(
    msg, field_names=("x", "y", "z"), skip_nans=True
).astype(np.float32)
```

The `frame_id` of the ROS message must match the frame you expect the returned
grasps in. The server does not transform frames. Typical flow:

- Upstream perception (depth + segmentation mask, RGBD fusion, …) produces a
  per-object point cloud in `camera_depth_optical_frame`.
- Optionally transform to `base_link` using `tf2_ros.Buffer` before sending.
- Server returns grasps in that same frame.
- Node publishes the grasps as `geometry_msgs/PoseArray` with
  `header.frame_id` matching.

**Segmentation is done entirely client-side.** If your scene has a bowl of
apples and you want to grasp one apple, publish ONE apple's point cloud as
`point_cloud` per `/predict` call; put everything else (the bowl, the other
apples, the table) in `scene_points`. Multiple apples = multiple calls.

Building `scene_points` when you already have per-object masks:

```python
# `full_cloud` is (N_total, 3) float32 from PointCloud2
# `target_mask` is a bool array of shape (N_total,) marking the object you want
target_pc = full_cloud[target_mask]
scene_pc  = full_cloud[~target_mask]
# Subsample the scene to keep the request under max_points and the collision
# check fast:
if len(scene_pc) > 16000:
    idx = np.random.choice(len(scene_pc), 16000, replace=False)
    scene_pc = scene_pc[idx]
```

### Handling the response in your ROS 2 node

```python
data = response.json()

pose_array = PoseArray()
pose_array.header.frame_id = input_cloud_frame_id  # same frame as you sent
pose_array.header.stamp = self.get_clock().now().to_msg()
for g in data["grasps"]:                 # already sorted by score, descending
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = g["translation"]
    qx, qy, qz, qw = g["rotation_quat"]
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    pose_array.poses.append(pose)
publisher.publish(pose_array)

# Gripper-width per grasp — publish on a side topic
widths = [g["gripper_width"] for g in data["grasps"]]
```

A few things to be aware of in your ROS node:

- **Check the grasp frame once at startup.** `GET /config` reports
  `grasp_frame`, `approach_axis_local` and `jaw_axis_local`. Reading those
  instead of hardcoding an axis convention means switching backends (or
  switching `--grasp-frame`) needs no client change.
- **`grasp_reference` is not the same point on both backends.** Before you
  send a pose to the motion planner, make sure the TF frame you plan for is
  the one the server is reporting. `/config.grasp_reference` plus
  `/config.fingertip_depth` give you everything needed to build the offset.
- `data["meta"]["request_id"]` is worth logging — it pins each grasp set to
  the corresponding `.npz` dump on the server when `--dump-npz` is on, which
  is invaluable for post-hoc cross-check against your RViz / Isaac Sim view.
- A response can legitimately be `"grasps": []`. Handle it as "try again with
  a better view" rather than as an error.

---

## Versioning

- This API is at version **0.1.0**. Breaking changes bump minor version until
  1.0; additive changes bump patch.
- The server advertises its version in every `/predict` response and via
  `/version`.
- The client SHOULD log a warning if `meta.server_version` mismatches the
  version it was developed against.

---

## Deliberately out of scope (for now)

- **Streaming / websocket / gRPC bidirectional**: single request/response only.
- **Multi-gripper servers**: one gripper per server process. Run multiple
  containers if needed. (The ZMQ transport under `client-server/` does support
  per-request gripper selection, if that is what you need.)
- **Segmentation / scene parsing**: the server has no object detector, no
  instance classifier, no foreground/background heuristic. The client produces
  the target/scene split; the server only acts on what it's given.
- **Reachability / IK filtering**: `scene_points` buys you gripper-vs-clutter
  collision rejection and nothing more. Arm reachability, self-collision and
  IK remain the client's job.
