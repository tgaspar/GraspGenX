#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a GraspGenX checkpoint over HTTP.

Speaks the request/response contract documented in `docs/api/predict.md` —
the same contract the DexGraspNet 2.0 server implements — so a client can be
pointed at either backend without changing any code.

Typical usage inside the container:
    python scripts/serve_grasp_predictor.py \\
        --gripper arx_x5 \\
        --host 0.0.0.0 --port 8000

See `docs/api/predict.md` for the full schema and the grasp-frame discussion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a GraspGenX grasp predictor over HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- model / gripper ----------------------------------------------------
    parser.add_argument(
        "--gripper",
        required=True,
        help="Gripper to serve, e.g. arx_x5, franka_panda, robotiq_2f_85. "
        "One gripper per server process. `python scripts/list_grippers.py` "
        "lists what is available.",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Checkpoint root containing gen/ and dis/ subdirectories. "
        "Defaults to the auto-downloaded ext/graspgenx_checkpoints/release "
        "(override globally with $GRASPGENX_CHECKPOINT_DIR).",
    )
    parser.add_argument("--gen-pth", default=None, help="Generator .pth filename.")
    parser.add_argument("--dis-pth", default=None, help="Discriminator .pth filename.")
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Fallback asset root holding x_grippers/ and proc_grippers/. "
        "The curated gripper_descriptions checkout is searched first.",
    )

    # -- output conventions -------------------------------------------------
    parser.add_argument(
        "--grasp-frame",
        default="dexgraspnet2",
        choices=["dexgraspnet2", "graspgenx", "tcp_z_approach"],
        help="Axis convention of the returned rotations. "
        "'dexgraspnet2': +X approach, +Y jaw (drop-in for a DexGraspNet 2.0 "
        "client). 'graspgenx': +Z approach, +X jaw (model native). "
        "'tcp_z_approach': +Z approach, +Y jaw (common robot TCP convention).",
    )
    parser.add_argument(
        "--tcp-offset",
        type=float,
        default=0.0,
        help="Shift the returned grasp origin this many meters along the "
        "approach axis. 0.0 keeps GraspGenX's native reference, the gripper "
        "base/mount link.",
    )
    parser.add_argument(
        "--tcp-at-fingertip",
        action="store_true",
        help="Shorthand for --tcp-offset <gripper fingertip depth>, putting "
        "the returned origin at the fingertip plane.",
    )
    parser.add_argument(
        "--width-mode",
        default="aperture",
        choices=["aperture", "measured", "none"],
        help="How `gripper_width` is filled. 'aperture': the gripper's "
        "fully-open jaw opening (GraspGenX does not predict a width). "
        "'measured': span of the object points inside the jaw sweep volume "
        "plus --width-clearance. 'none': return null.",
    )
    parser.add_argument(
        "--width-clearance",
        type=float,
        default=0.01,
        help="Meters added to the measured object span in --width-mode measured.",
    )
    parser.add_argument(
        "--approach-axis-convention",
        default="true_approach",
        choices=["true_approach", "frame_z"],
        help="What each grasp's `approach_axis` reports. 'true_approach': the "
        "real approach direction. 'frame_z': the emitted frame's +Z column, "
        "which is bug-compatible with the DexGraspNet 2.0 server (it returns "
        "rotation_matrix @ tcp_local_z even though its approach axis is +X).",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="Value written into every grasp's `object_id`. The contract says "
        "1 for target-object grasps; the DexGraspNet 2.0 server emits -1 in "
        "object-only mode, so set -1 if your client mirrors that.",
    )

    # -- inference knobs ----------------------------------------------------
    parser.add_argument(
        "--planner",
        default="graspmoe",
        choices=["graspmoe", "diffusion"],
        help="graspmoe: diffusion samples plus OBB-swept candidates, all "
        "scored by the discriminator. diffusion: diffusion samples only.",
    )
    parser.add_argument(
        "--num-diffusion-samples",
        type=int,
        default=200,
        help="Diffusion samples drawn per request before ranking.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=4,
        help="Internal top-k is num_grasps * this, so the min_score filter "
        "still leaves candidates to return.",
    )
    parser.add_argument(
        "--default-num-grasps",
        type=int,
        default=20,
        help="Value advertised via /config as default_num_grasps.",
    )
    parser.add_argument("--min-points", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=65536)
    parser.add_argument(
        "--no-scene-collision-filter",
        dest="scene_collision_filter",
        action="store_false",
        help="Ignore the request's `scene_points` instead of using it to "
        "reject grasps whose gripper geometry would hit the clutter.",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.005,
        help="Meters; gripper surface samples closer than this to a scene "
        "point count as a collision. Kept small because a gripper reaching an "
        "object on a table is legitimately within centimeters of the table.",
    )
    parser.add_argument(
        "--scene-exclusion-radius",
        type=float,
        default=0.01,
        help="Meters; drop `scene_points` within this distance of the target "
        "cloud before collision checking. Removes segmentation bleed and the "
        "object's own contact patch, which would otherwise make every "
        "enclosing grasp look like a collision. 0 disables.",
    )
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=2000,
        help="Surface samples drawn from the gripper collision mesh.",
    )
    parser.add_argument(
        "--collision-batch-size",
        type=int,
        default=8,
        help="Grasps per vectorized collision check. Peak GPU memory is "
        "roughly batch * num_collision_samples * len(scene_points) * 4 bytes.",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Compile the diffusion denoiser with TensorRT (needs the "
        "'tensorrt' extra). Falls back to eager PyTorch if unavailable.",
    )
    parser.add_argument(
        "--tensorrt-precision", default="fp32", choices=["fp32", "fp16"]
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip the startup warmup inference (first request will be slow).",
    )

    # -- serving / observability -------------------------------------------
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for run.log, config.json, requests.jsonl and NPZ "
        "dumps. Default: .logs/<timestamp>_server_<gripper>/",
    )
    parser.add_argument(
        "--dump-npz",
        action="store_true",
        help="Save the input cloud and output grasp arrays per request to "
        "<log-dir>/predictions/<request_id>.npz.",
    )
    parser.add_argument(
        "--model-commit", default="", help="Build tag reported via /version."
    )

    args = parser.parse_args(argv)

    import uvicorn

    from graspgenx.serving import rest_server

    rest_server.setup_log_dir(args.log_dir, args.gripper, args.dump_npz)
    rest_server.initialise_state(
        gripper=args.gripper,
        checkpoints=args.checkpoints,
        gen_pth=args.gen_pth,
        dis_pth=args.dis_pth,
        assets_dir=args.assets_dir,
        grasp_frame=args.grasp_frame,
        tcp_offset=args.tcp_offset,
        tcp_at_fingertip=args.tcp_at_fingertip,
        planner=args.planner,
        num_diffusion_samples=args.num_diffusion_samples,
        oversample_factor=args.oversample_factor,
        width_mode=args.width_mode,
        width_clearance=args.width_clearance,
        approach_axis_convention=args.approach_axis_convention,
        object_id=args.object_id,
        min_points=args.min_points,
        max_points=args.max_points,
        default_num_grasps=args.default_num_grasps,
        scene_collision_filter=args.scene_collision_filter,
        collision_threshold=args.collision_threshold,
        num_collision_samples=args.num_collision_samples,
        scene_exclusion_radius=args.scene_exclusion_radius,
        collision_batch_size=args.collision_batch_size,
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
        model_commit=args.model_commit,
        warmup=args.warmup,
    )

    app = rest_server.build_app()
    print(f"Starting GraspGenX REST server on http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
