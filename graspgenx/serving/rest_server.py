# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI grasp-prediction server for GraspGenX.

Single responsibility: expose a loaded GraspGenX checkpoint + one gripper over
HTTP, speaking the exact wire contract documented in ``docs/api/predict.md``.
That contract is shared with DexGraspNet 2.0, so a client can point at either
server without code changes.

The server is configured at startup for ONE gripper and does not transform
frames — all grasps come back in the same coordinate frame as the input point
cloud.

See ``scripts/serve_grasp_predictor.py`` for the CLI entry point.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

SERVER_VERSION = "0.1.0"
API_VERSION = "0.1.0"
BACKEND = "graspgenx"
BUILT_AT = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

_ALLOWED_PC_DTYPES = ("float32", "float64")


# ---------------------------------------------------------------------------
# Grasp-frame conventions
# ---------------------------------------------------------------------------
#
# GraspGenX predicts the pose of the gripper's *base link* (the URDF root, i.e.
# the flange-mount frame) with the native convention
#
#     +Z = approach axis (fingers extend along +Z)
#     +X = jaw-closing axis (the two fingers sit at +/- aperture/2 along X)
#     +Y = the thin direction
#
# Other stacks label the same physical pose differently. Each entry below is
# the change-of-basis matrix ``C`` that maps *output-frame* axes onto *native*
# axes, so the emitted rotation is ``R_out = R_native @ C``:
#
#     C @ approach_axis_out = (0, 0, 1)        # native approach
#     C @ jaw_axis_out      = (1, 0, 0)        # native closing
#
# The world-frame approach direction is ``R_native[:, 2]`` regardless of the
# convention chosen, which is why ``approach_axis`` in the response is
# convention-independent.

GRASP_FRAMES: Dict[str, Dict[str, Any]] = {
    # Native GraspGenX: +Z approach, +X jaw.
    "graspgenx": {
        "C": np.eye(3),
        "approach_axis_local": [0.0, 0.0, 1.0],
        "jaw_axis_local": [1.0, 0.0, 0.0],
        "description": "+Z approach, +X jaw-closing (GraspGenX native)",
    },
    # DexGraspNet 2.0 / GraspNet-1Billion: +X approach, +Y jaw.
    "dexgraspnet2": {
        "C": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "approach_axis_local": [1.0, 0.0, 0.0],
        "jaw_axis_local": [0.0, 1.0, 0.0],
        "description": "+X approach, +Y jaw-closing (DexGraspNet 2.0 convention)",
    },
    # Common robot TCP convention: +Z approach, +Y jaw.
    "tcp_z_approach": {
        "C": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "approach_axis_local": [0.0, 0.0, 1.0],
        "jaw_axis_local": [0.0, 1.0, 0.0],
        "description": "+Z approach, +Y jaw-closing (robot TCP convention)",
    },
}

WIDTH_MODES = ("aperture", "measured", "none")
APPROACH_AXIS_CONVENTIONS = ("true_approach", "frame_z")


# ---------------------------------------------------------------------------
# Pydantic schemas (mirror docs/api/predict.md)
# ---------------------------------------------------------------------------


class EncodedArray(BaseModel):
    dtype: str
    shape: List[int]
    data_b64: str


class PredictRequest(BaseModel):
    point_cloud: EncodedArray
    scene_points: Optional[EncodedArray] = None
    num_grasps: int = Field(default=20, ge=1)
    min_score: float = 0.0
    # Accepted for wire compatibility with the DexGraspNet 2.0 contract.
    # GraspGenX has no per-instance seed-sampling knob (it is conditioned on
    # the target object cloud alone), so this field is a no-op here; it is
    # echoed back in `meta.category_sampling` so logs line up across servers.
    category_sampling: Optional[bool] = None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class ServerState:
    """Everything loaded at startup. One instance per process."""

    sampler: Any = None
    gripper: Any = None
    gripper_name: str = ""
    model_name: str = ""
    checkpoint: str = ""
    gen_checkpoint: str = ""
    dis_checkpoint: str = ""
    loaded: bool = False
    model_commit: str = ""

    # Geometry / conventions.
    grasp_frame: str = "dexgraspnet2"
    frame_C: np.ndarray = np.eye(3)
    tcp_offset: float = 0.0
    grasp_reference: str = ""
    max_aperture: float = 0.0
    fingertip_depth: float = 0.0
    # `approach_axis` semantics. "true_approach" reports the real approach
    # direction. "frame_z" reports the emitted frame's +Z column instead —
    # bug-compatible with the DexGraspNet 2.0 server, which computes
    # `rotation_matrix @ tcp_local_z` even though its +X is the approach axis.
    approach_axis_convention: str = "true_approach"
    # Value written into each grasp's `object_id`. The contract says 1 for
    # target-object grasps; the DexGraspNet 2.0 server emits -1 in
    # object-only mode.
    object_id: int = 1

    # Inference knobs.
    planner: str = "graspmoe"
    num_diffusion_samples: int = 200
    oversample_factor: int = 4
    width_mode: str = "aperture"
    width_clearance: float = 0.01
    min_points: int = 128
    max_points: int = 65536
    default_num_grasps: int = 20

    # Scene-collision filtering (our analogue of the DGN2 `scene_points` input).
    scene_collision_filter: bool = True
    collision_threshold: float = 0.005
    num_collision_samples: int = 2000
    scene_exclusion_radius: float = 0.01
    collision_batch_size: int = 8
    gripper_surface_points: Optional[np.ndarray] = None

    # Observability.
    log_dir: Optional[Path] = None
    requests_jsonl: Optional[Path] = None
    dump_npz: bool = False


STATE = ServerState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_array(enc: EncodedArray, allowed_dtypes: tuple) -> np.ndarray:
    """Decode an EncodedArray payload. Raises HTTPException(400) on any
    validation failure, matching the error codes in docs/api/predict.md."""
    if enc.dtype not in allowed_dtypes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "bad_dtype",
                "message": f"dtype must be one of {allowed_dtypes}, got {enc.dtype}",
            },
        )
    try:
        raw = base64.b64decode(enc.data_b64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_base64", "message": str(exc)},
        )
    try:
        arr = np.frombuffer(raw, dtype=np.dtype(enc.dtype)).reshape(enc.shape)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "shape_dtype_mismatch",
                "message": (
                    f"decoded bytes do not match shape={enc.shape} "
                    f"dtype={enc.dtype}: {exc}"
                ),
            },
        )
    # np.frombuffer yields a read-only view; downstream torch/numpy code wants
    # a writable array.
    return np.array(arr, copy=True)


def _decode_point_cloud(
    enc: EncodedArray, field: str, min_points: Optional[int]
) -> np.ndarray:
    """Decode + validate one (N, 3) point-cloud field."""
    pc = _decode_array(enc, _ALLOWED_PC_DTYPES)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "bad_shape",
                "message": f"{field} must be (N, 3); got {pc.shape}",
            },
        )
    if not np.isfinite(pc).all():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_geometry",
                "message": f"{field} contains NaN or Inf values",
            },
        )
    if min_points is not None and pc.shape[0] < min_points:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_geometry",
                "message": (
                    f"need at least {min_points} points in {field}; "
                    f"got {pc.shape[0]}"
                ),
            },
        )
    return pc.astype(np.float32, copy=False)


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> List[float]:
    """3x3 rotation matrix -> (x, y, z, w) quaternion (ROS ordering).

    Shepperd's method: pick the branch with the largest denominator so the
    conversion stays numerically stable near each of the four singularities.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    # Canonical sign: keep w >= 0 so repeated calls give comparable numbers.
    if q[3] < 0:
        q = -q
    return q.tolist()


def _measure_grasp_widths(
    grasps_native: np.ndarray, object_pc: np.ndarray
) -> np.ndarray:
    """Per-grasp jaw opening measured from the object geometry.

    For each grasp, object points are expressed in the native grasp frame
    (+Z approach, +X closing) and clipped to the gripper's open sweep box in
    the Y (thin) and Z (approach) directions. The span of the surviving points
    along X, plus ``width_clearance``, is the commanded opening — clamped to
    the gripper's aperture. Grasps with no object points inside the box fall
    back to the full aperture.
    """
    aperture = STATE.max_aperture
    K = len(grasps_native)
    widths = np.full((K,), aperture, dtype=np.float64)
    if K == 0 or len(object_pc) == 0:
        return widths

    sweep = np.asarray(STATE.gripper.sweep_volume, dtype=np.float64)
    extents, offset = sweep[:3], sweep[3:]
    y_lo, y_hi = offset[1] - extents[1] / 2.0, offset[1] + extents[1] / 2.0
    z_lo, z_hi = offset[2] - extents[2] / 2.0, offset[2] + extents[2] / 2.0

    pts = np.asarray(object_pc, dtype=np.float64)
    for k in range(K):
        R = grasps_native[k][:3, :3].astype(np.float64)
        t = grasps_native[k][:3, 3].astype(np.float64)
        local = (pts - t) @ R  # == (R.T @ (p - t)).T
        inside = (
            (local[:, 1] >= y_lo)
            & (local[:, 1] <= y_hi)
            & (local[:, 2] >= z_lo)
            & (local[:, 2] <= z_hi)
        )
        if not inside.any():
            continue
        x = local[inside, 0]
        # Symmetric two-finger jaws close around the grasp axis, so the
        # commanded opening is driven by the farthest contact from centre.
        span = 2.0 * float(np.abs(x).max())
        widths[k] = float(np.clip(span + STATE.width_clearance, 0.0, aperture))
    return widths


def _prune_scene_near_target(
    scene_pts: np.ndarray, target_pts: np.ndarray, radius: float
) -> np.ndarray:
    """Drop scene points within ``radius`` of any target point.

    A mask-derived `scene_points` cloud hugs the target: segmentation
    boundaries bleed, and the surface the object rests on continues right up
    to its footprint. Left in, those points make every grasp that encloses the
    object look like a collision. Removing a thin shell around the target
    leaves the real obstacles (bin walls, neighbouring objects) intact.
    """
    if radius <= 0 or len(scene_pts) == 0 or len(target_pts) == 0:
        return scene_pts

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp64 for the same reason as collision_filter.py: cdist's matmul path
    # loses far too much precision on nearby fp32 points to threshold against.
    scene_t = torch.as_tensor(scene_pts, dtype=torch.float64, device=device)
    target_t = torch.as_tensor(target_pts, dtype=torch.float64, device=device)

    keep = torch.ones(len(scene_t), dtype=torch.bool, device=device)
    # Chunk the scene so the (chunk, |target|) distance matrix stays bounded.
    chunk = max(1, int(2_000_000 // max(len(target_t), 1)))
    for s0 in range(0, len(scene_t), chunk):
        block = scene_t[s0 : s0 + chunk]
        min_d = torch.cdist(block, target_t, p=2).amin(dim=1)
        keep[s0 : s0 + chunk] = min_d > radius

    out = scene_pts[keep.detach().cpu().numpy()]
    logger.info(
        f"[collision] scene pruning: {len(out)}/{len(scene_pts)} points kept "
        f"(dropped everything within {radius:.3f} m of the target)"
    )
    return out


def _grasp_to_response(
    pose_native: np.ndarray,
    score: float,
    width: Optional[float],
    branch: str,
) -> Dict[str, Any]:
    """Convert one native (4, 4) GraspGenX pose into the spec-compliant dict."""
    R_native = np.asarray(pose_native[:3, :3], dtype=np.float64)
    t_native = np.asarray(pose_native[:3, 3], dtype=np.float64)

    # World-frame approach direction — independent of the output convention.
    approach_world = R_native @ np.array([0.0, 0.0, 1.0])

    R_out = R_native @ STATE.frame_C
    t_out = t_native + STATE.tcp_offset * approach_world

    if STATE.approach_axis_convention == "frame_z":
        reported_axis = R_out[:, 2]
    else:
        reported_axis = approach_world

    return {
        "translation": t_out.tolist(),
        "rotation_quat": _rotation_matrix_to_quat_xyzw(R_out),
        "rotation_matrix": R_out.tolist(),
        "score": float(score),
        "gripper_width": None if width is None else float(width),
        # GraspGenX is a parallel/multi-finger *conditioned* model: it predicts
        # a 6-DoF pose plus an opening, never a joint vector.
        "joint_angles": None,
        "joint_names": None,
        "approach_axis": reported_axis.tolist(),
        # GraspGenX only ever generates grasps on the target object cloud, so
        # every returned grasp is a target grasp. The field exists so clients
        # written against the DexGraspNet 2.0 contract keep working unchanged.
        "object_id": STATE.object_id,
        # GraspGenX-specific extra: which GraspMoE branch produced this grasp.
        "branch": branch,
    }


def _append_request_log(record: Dict[str, Any]) -> None:
    if STATE.requests_jsonl is None:
        return
    try:
        with STATE.requests_jsonl.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append to requests.jsonl: {exc}")


def _dump_npz(
    request_id: str,
    point_cloud: np.ndarray,
    poses_native: np.ndarray,
    scores: np.ndarray,
    widths: np.ndarray,
    scene_points: Optional[np.ndarray],
) -> Optional[Path]:
    """Save raw input + output arrays for offline replay / visualization."""
    if STATE.log_dir is None:
        return None
    try:
        out_dir = STATE.log_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.npz"
        save_args: Dict[str, Any] = dict(
            point_cloud=point_cloud.astype(np.float32, copy=False),
            grasps_native=np.asarray(poses_native, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            gripper_widths=np.asarray(widths, dtype=np.float32),
            gripper_name=np.array(STATE.gripper_name),
            grasp_frame=np.array(STATE.grasp_frame),
            tcp_offset=np.array(STATE.tcp_offset, dtype=np.float32),
        )
        if scene_points is not None and scene_points.size > 0:
            save_args["scene_points"] = scene_points.astype(np.float32, copy=False)
        np.savez(out_path, **save_args)
        return out_path
    except Exception as exc:
        logger.warning(f"NPZ dump failed for request {request_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app() -> FastAPI:
    app = FastAPI(
        title="GraspGenX Grasp Predictor",
        version=API_VERSION,
        description="HTTP interface over a pretrained GraspGenX model.",
    )

    # -- /healthz -----------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> Any:
        if STATE.loaded:
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "loading"})

    # -- /config ------------------------------------------------------------

    @app.get("/config")
    def config() -> Any:
        if not STATE.loaded:
            raise HTTPException(
                status_code=503,
                detail={"error": "not_ready", "retry_after_seconds": 5},
            )
        frame = GRASP_FRAMES[STATE.grasp_frame]
        return {
            # Contract fields (identical set to the DexGraspNet 2.0 server).
            "hand": STATE.gripper_name,
            "model_name": STATE.model_name,
            "checkpoint": STATE.checkpoint,
            "joint_names": None,
            "grasp_reference": STATE.grasp_reference,
            "default_num_grasps": STATE.default_num_grasps,
            "min_points": STATE.min_points,
            "max_points": STATE.max_points,
            "server_version": SERVER_VERSION,
            # GraspGenX-specific introspection.
            "backend": BACKEND,
            "api_version": API_VERSION,
            "gripper": STATE.gripper_name,
            "grasp_frame": STATE.grasp_frame,
            "grasp_frame_description": frame["description"],
            "approach_axis_local": frame["approach_axis_local"],
            "jaw_axis_local": frame["jaw_axis_local"],
            "tcp_offset": STATE.tcp_offset,
            "gripper_max_width": STATE.max_aperture,
            "fingertip_depth": STATE.fingertip_depth,
            "planner": STATE.planner,
            "width_mode": STATE.width_mode,
            "approach_axis_convention": STATE.approach_axis_convention,
            "collision_threshold": STATE.collision_threshold,
            "scene_exclusion_radius": STATE.scene_exclusion_radius,
            "scene_points_mode": (
                "collision_filter" if STATE.scene_collision_filter else "ignored"
            ),
            "supports_category_sampling": False,
        }

    # -- /version -----------------------------------------------------------

    @app.get("/version")
    def version() -> Any:
        return {
            "server_version": SERVER_VERSION,
            "api_version": API_VERSION,
            "backend": BACKEND,
            "model_commit": STATE.model_commit,
            "built_at": BUILT_AT,
        }

    # -- /predict -----------------------------------------------------------

    @app.post("/predict")
    def predict(req: PredictRequest) -> Any:
        if not STATE.loaded or STATE.sampler is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "not_ready", "retry_after_seconds": 5},
            )

        pc = _decode_point_cloud(req.point_cloud, "point_cloud", STATE.min_points)
        n = pc.shape[0]

        scene_pts: Optional[np.ndarray] = None
        if req.scene_points is not None:
            scene_pts = _decode_point_cloud(req.scene_points, "scene_points", None)

        num_scene = int(scene_pts.shape[0]) if scene_pts is not None else 0
        total_points = n + num_scene
        if total_points > STATE.max_points:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "payload_too_large",
                    "max_points": STATE.max_points,
                    "message": (
                        f"point_cloud ({n}) + scene_points ({num_scene}) = "
                        f"{total_points} > {STATE.max_points}"
                    ),
                },
            )

        request_id = uuid.uuid4().hex[:12]
        # Oversample so the min_score filter still has candidates left to
        # return after truncation to num_grasps. When scene_points is present
        # the top-k cap has to come off entirely: collision filtering runs
        # *after* the planner, and a small pre-filter cap regularly leaves
        # nothing once the blocked approaches are dropped.
        will_collision_filter = (
            scene_pts is not None
            and len(scene_pts) > 0
            and STATE.scene_collision_filter
        )
        topk = -1 if will_collision_filter else max(
            req.num_grasps * STATE.oversample_factor, 64
        )

        t0 = time.time()
        try:
            from graspgenx.samplers.planner import run_planner_on_object

            poses, scores, tags, _obb = run_planner_on_object(
                pc,
                STATE.sampler,
                planner=STATE.planner,
                grasp_threshold=-1.0,
                num_grasps=STATE.num_diffusion_samples,
                topk_num_grasps=topk,
            )
        except Exception as exc:
            logger.exception("Inference failed")
            _append_request_log(
                {
                    "request_id": request_id,
                    "time": _dt.datetime.utcnow().isoformat() + "Z",
                    "num_input_points": int(n),
                    "num_scene_points": num_scene,
                    "num_grasps_requested": req.num_grasps,
                    "error": str(exc),
                }
            )
            raise HTTPException(
                status_code=500,
                detail={"error": "internal_error", "message": str(exc)},
            )

        poses = np.asarray(poses, dtype=np.float32).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        tags = list(tags) if tags is not None else ["diff"] * len(poses)

        # Scene-context handling. GraspGenX conditions only on the target
        # cloud, so `scene_points` is used the way it is meant to help: as
        # clutter geometry that predicted grasps must not collide with.
        num_collision_rejected = 0
        context_mode = "object_only"
        if will_collision_filter and len(poses) > 0:
            context_mode = "scene_collision_filtered"
            try:
                from graspgenx.utils.collision_filter import filter_colliding_grasps

                scene_for_collision = _prune_scene_near_target(
                    scene_pts, pc, STATE.scene_exclusion_radius
                )
                free = filter_colliding_grasps(
                    scene_pc=scene_for_collision,
                    grasp_poses=poses,
                    gripper_surface_points=STATE.gripper_surface_points,
                    gripper_collision_mesh=(
                        None
                        if STATE.gripper_surface_points is not None
                        else STATE.gripper.collision_mesh
                    ),
                    collision_threshold=STATE.collision_threshold,
                    num_collision_samples=STATE.num_collision_samples,
                    batch_size=STATE.collision_batch_size,
                )
                num_collision_rejected = int((~free).sum())
                poses = poses[free]
                scores = scores[free]
                tags = [t for t, keep in zip(tags, free) if keep]
            except Exception as exc:
                # A collision-filter failure must not sink the whole request —
                # degrade to unfiltered results and say so in the log.
                logger.warning(
                    f"[{request_id}] scene collision filter failed, returning "
                    f"unfiltered grasps: {exc}"
                )
                context_mode = "scene_collision_filter_failed"
        elif scene_pts is not None and len(scene_pts) > 0:
            context_mode = "scene_ignored"

        # Rank descending: GraspMoE returns the diffusion and OBB branches
        # concatenated, not globally sorted.
        order = np.argsort(-scores)
        poses, scores = poses[order], scores[order]
        tags = [tags[i] for i in order]

        keep = scores >= float(req.min_score)
        poses, scores = poses[keep], scores[keep]
        tags = [t for t, k in zip(tags, keep) if k]

        poses = poses[: req.num_grasps]
        scores = scores[: req.num_grasps]
        tags = tags[: req.num_grasps]

        if STATE.width_mode == "measured":
            widths = _measure_grasp_widths(poses, pc)
        elif STATE.width_mode == "aperture":
            widths = np.full((len(poses),), STATE.max_aperture, dtype=np.float64)
        else:
            widths = np.full((len(poses),), np.nan, dtype=np.float64)

        inference_ms = (time.time() - t0) * 1000.0

        grasps_out = [
            _grasp_to_response(
                poses[i],
                scores[i],
                None if STATE.width_mode == "none" else widths[i],
                tags[i] if i < len(tags) else "diff",
            )
            for i in range(len(poses))
        ]

        npz_path: Optional[Path] = None
        if STATE.dump_npz and len(poses) > 0:
            npz_path = _dump_npz(request_id, pc, poses, scores, widths, scene_pts)

        _append_request_log(
            {
                "request_id": request_id,
                "time": _dt.datetime.utcnow().isoformat() + "Z",
                "num_input_points": int(n),
                "num_scene_points": num_scene,
                "num_grasps_requested": req.num_grasps,
                "num_grasps_returned": len(grasps_out),
                "num_collision_rejected": num_collision_rejected,
                "top_score": float(scores[0]) if len(scores) else None,
                "inference_ms": round(inference_ms, 3),
                "npz_path": str(npz_path) if npz_path else None,
            }
        )

        return {
            "grasps": grasps_out,
            "meta": {
                "hand": STATE.gripper_name,
                "model_name": STATE.model_name,
                "checkpoint": STATE.checkpoint,
                "frame_id": "unchanged_from_input",
                "num_input_points": int(n),
                "num_scene_points": num_scene,
                "context_mode": context_mode,
                # Echoed for log parity with the DexGraspNet 2.0 server; the
                # GraspGenX backend has no such sampling mode.
                "category_sampling": bool(req.category_sampling)
                if req.category_sampling is not None
                else False,
                "inference_ms": round(inference_ms, 3),
                "server_version": SERVER_VERSION,
                "request_id": request_id,
                "debug_html": None,
                "debug_usd": None,
                "debug_npz": str(npz_path) if npz_path else None,
                # GraspGenX-specific extras.
                "backend": BACKEND,
                "api_version": API_VERSION,
                "gripper": STATE.gripper_name,
                "grasp_frame": STATE.grasp_frame,
                "tcp_offset": STATE.tcp_offset,
                "planner": STATE.planner,
                "width_mode": STATE.width_mode,
                "num_collision_rejected": num_collision_rejected,
            },
        }

    return app


# ---------------------------------------------------------------------------
# Startup / wiring
# ---------------------------------------------------------------------------


def setup_log_dir(
    log_dir: Optional[Path],
    gripper_name: str,
    dump_npz: bool,
) -> Path:
    """Create the run directory, attach a file log handler, prepare paths."""
    import logging

    if log_dir is None:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        log_dir = Path(".logs") / f"{ts}_server_{gripper_name}"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    STATE.log_dir = log_dir
    STATE.requests_jsonl = log_dir / "requests.jsonl"
    STATE.dump_npz = bool(dump_npz)
    logger.info(f"Server log directory: {log_dir}")
    return log_dir


def initialise_state(
    *,
    gripper: str,
    checkpoints: Optional[str] = None,
    gen_pth: Optional[str] = None,
    dis_pth: Optional[str] = None,
    assets_dir: Optional[str] = None,
    grasp_frame: str = "dexgraspnet2",
    tcp_offset: float = 0.0,
    tcp_at_fingertip: bool = False,
    planner: str = "graspmoe",
    num_diffusion_samples: int = 200,
    oversample_factor: int = 4,
    width_mode: str = "aperture",
    width_clearance: float = 0.01,
    approach_axis_convention: str = "true_approach",
    object_id: int = 1,
    min_points: int = 128,
    max_points: int = 65536,
    default_num_grasps: int = 20,
    scene_collision_filter: bool = True,
    collision_threshold: float = 0.005,
    num_collision_samples: int = 2000,
    scene_exclusion_radius: float = 0.01,
    collision_batch_size: int = 8,
    use_tensorrt: bool = False,
    tensorrt_precision: str = "fp32",
    model_commit: str = "",
    warmup: bool = True,
) -> None:
    """Load the model + gripper and populate STATE. Call before serving."""
    import os

    from graspgenx._setup_dependencies import (
        get_checkpoints_version_dir,
        get_gripper_descriptions_assets,
    )
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg

    if grasp_frame not in GRASP_FRAMES:
        raise ValueError(
            f"Unknown --grasp-frame {grasp_frame!r}; "
            f"choose one of {sorted(GRASP_FRAMES)}"
        )
    if width_mode not in WIDTH_MODES:
        raise ValueError(
            f"Unknown --width-mode {width_mode!r}; choose one of {WIDTH_MODES}"
        )
    if approach_axis_convention not in APPROACH_AXIS_CONVENTIONS:
        raise ValueError(
            f"Unknown --approach-axis-convention {approach_axis_convention!r}; "
            f"choose one of {APPROACH_AXIS_CONVENTIONS}"
        )

    ckpt_root = Path(checkpoints) if checkpoints else Path(get_checkpoints_version_dir())
    cfg = load_model_cfg(
        str(ckpt_root / "gen"), str(ckpt_root / "dis"), gen_pth, dis_pth
    )

    if assets_dir is None:
        # resolve_gripper_info() checks the gripper_descriptions package first
        # and only falls back to assets_dir, so the repo's assets/ (which holds
        # the procedural grippers) is the right fallback root.
        assets_dir = str(Path(__file__).resolve().parents[2] / "assets")
        # Make sure the curated x_grippers checkout is materialised.
        try:
            get_gripper_descriptions_assets()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"gripper_descriptions setup hook failed: {exc}")

    logger.info(f"Loading gripper {gripper!r} (assets_dir={assets_dir})")
    sampler = GraspGenXSampler(
        cfg,
        gripper,
        assets_dir=assets_dir,
        use_tensorrt=use_tensorrt,
        tensorrt_precision=tensorrt_precision,
    )
    info = sampler.get_gripper_info()

    frame = GRASP_FRAMES[grasp_frame]
    fingertip_depth = float(info.depth)
    if tcp_at_fingertip:
        tcp_offset = fingertip_depth

    # sweep_volume is [extents_xyz, offset_xyz]; extents[0] is the jaw
    # aperture at the fully-open state (see x_grippers.py).
    max_aperture = float(np.asarray(info.sweep_volume)[0])

    STATE.sampler = sampler
    STATE.gripper = info
    STATE.gripper_name = gripper
    STATE.model_name = f"GraspGenX/{ckpt_root.name}"
    STATE.gen_checkpoint = os.path.basename(str(cfg.eval.gen_checkpoint))
    STATE.dis_checkpoint = os.path.basename(str(cfg.eval.dis_checkpoint))
    STATE.checkpoint = f"{STATE.gen_checkpoint}+{STATE.dis_checkpoint}"
    STATE.model_commit = model_commit

    STATE.grasp_frame = grasp_frame
    STATE.frame_C = np.asarray(frame["C"], dtype=np.float64)
    STATE.tcp_offset = float(tcp_offset)
    STATE.max_aperture = max_aperture
    STATE.fingertip_depth = fingertip_depth
    STATE.grasp_reference = (
        "gripper_base_link"
        if abs(STATE.tcp_offset) < 1e-9
        else f"gripper_base_link+{STATE.tcp_offset:.4f}m_along_approach"
    )

    STATE.planner = planner
    STATE.num_diffusion_samples = int(num_diffusion_samples)
    STATE.oversample_factor = int(oversample_factor)
    STATE.width_mode = width_mode
    STATE.width_clearance = float(width_clearance)
    STATE.approach_axis_convention = approach_axis_convention
    STATE.object_id = int(object_id)
    STATE.min_points = int(min_points)
    STATE.max_points = int(max_points)
    STATE.default_num_grasps = int(default_num_grasps)

    STATE.scene_collision_filter = bool(scene_collision_filter)
    STATE.collision_threshold = float(collision_threshold)
    STATE.num_collision_samples = int(num_collision_samples)
    STATE.scene_exclusion_radius = float(scene_exclusion_radius)
    STATE.collision_batch_size = int(collision_batch_size)
    if STATE.scene_collision_filter:
        # Sample the gripper collision mesh once instead of per request.
        try:
            import trimesh

            sampled, _ = trimesh.sample.sample_surface(
                info.collision_mesh, STATE.num_collision_samples
            )
            STATE.gripper_surface_points = np.asarray(sampled, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not pre-sample the gripper collision mesh "
                f"({exc}); falling back to per-request sampling."
            )
            STATE.gripper_surface_points = None

    if warmup:
        _run_warmup()

    STATE.loaded = True
    logger.info(
        f"Loaded GraspGenX ({STATE.checkpoint}) for gripper {gripper!r}: "
        f"grasp_frame={grasp_frame} ({frame['description']}), "
        f"grasp_reference={STATE.grasp_reference}, "
        f"aperture={max_aperture:.4f} m, fingertip_depth={fingertip_depth:.4f} m, "
        f"planner={planner}, width_mode={width_mode}."
    )

    if STATE.log_dir is not None:
        try:
            with (STATE.log_dir / "config.json").open("w") as f:
                json.dump(
                    {
                        "server_version": SERVER_VERSION,
                        "backend": BACKEND,
                        "checkpoints": str(ckpt_root),
                        "gen_checkpoint": STATE.gen_checkpoint,
                        "dis_checkpoint": STATE.dis_checkpoint,
                        "gripper": gripper,
                        "grasp_frame": grasp_frame,
                        "grasp_reference": STATE.grasp_reference,
                        "tcp_offset": STATE.tcp_offset,
                        "gripper_max_width": max_aperture,
                        "fingertip_depth": fingertip_depth,
                        "planner": planner,
                        "width_mode": width_mode,
                        "approach_axis_convention": approach_axis_convention,
                        "object_id": STATE.object_id,
                        "min_points": STATE.min_points,
                        "max_points": STATE.max_points,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to write config.json: {exc}")


def _run_warmup() -> None:
    """One throwaway inference so the first real request isn't paying for
    lazy CUDA kernel / JIT compilation."""
    try:
        from graspgenx.samplers.planner import run_planner_on_object

        rng = np.random.default_rng(0)
        pc = (rng.random((2048, 3)).astype(np.float32) - 0.5) * np.float32(0.08)
        t0 = time.time()
        run_planner_on_object(
            pc,
            STATE.sampler,
            planner=STATE.planner,
            grasp_threshold=-1.0,
            num_grasps=32,
            topk_num_grasps=8,
        )
        logger.info(f"Warmup inference done in {time.time() - t0:.1f} s")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Warmup inference failed (non-fatal): {exc}")
