#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""THROWAWAY dev tool — not part of the API surface, not covered by tests.

Grabs live data off the ROS 2 graph, builds a `/predict` request exactly the way
a real client would, posts it to the GraspGenX (or DexGraspNet 2.0) server, and
prints / republishes the result so you can eyeball it in RViz.

Two ways to get the target-object point cloud:

  --source depth   (default)  deproject /camera/depth through /camera/camera_info,
                              keep the pixels where /yolo_masker/debug/mask > 0.
                              This mirrors the production pipeline: depth camera
                              + segmentation model -> segmented object cloud.
  --source cloud              take an existing PointCloud2 topic verbatim
                              (e.g. /grasp_estimator/debug/cropped_pointcloud).

Examples
--------
    source /opt/ros/jazzy/setup.bash

    # Straight smoke test against the GraspGenX server
    python scripts/dev_ros2_predict_smoke_test.py --url http://localhost:8001

    # Send the rest of the scene too, so clutter-blocked grasps get dropped
    python scripts/dev_ros2_predict_smoke_test.py --url http://localhost:8001 --with-scene

    # Point the same script at the DexGraspNet 2.0 server to compare
    python scripts/dev_ros2_predict_smoke_test.py --url http://localhost:8000

    # Republish for RViz (PoseArray + per-grasp approach arrows)
    python scripts/dev_ros2_predict_smoke_test.py --url http://localhost:8001 --publish

Needs only rclpy + numpy + requests (urllib is used, so requests is optional).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseArray
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as exc:  # pragma: no cover - dev tool
    sys.exit(
        f"ROS 2 python packages not importable ({exc}).\n"
        f"Run `source /opt/ros/jazzy/setup.bash` first."
    )


# ---------------------------------------------------------------------------
# ROS helpers
# ---------------------------------------------------------------------------

_IMAGE_DTYPES = {
    "32FC1": (np.float32, 1),
    "16UC1": (np.uint16, 1),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
}


def image_to_numpy(msg: Image) -> np.ndarray:
    """Minimal cv_bridge replacement for the encodings this script sees."""
    if msg.encoding not in _IMAGE_DTYPES:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    dtype, channels = _IMAGE_DTYPES[msg.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    arr = np.frombuffer(msg.data, dtype=dtype)
    arr = arr.reshape(msg.height, msg.step // dtype.itemsize)[
        :, : msg.width * channels
    ]
    if channels > 1:
        arr = arr.reshape(msg.height, msg.width, channels)
    return arr


def collect_messages(node, specs: dict, timeout_s: float) -> dict:
    """Subscribe to {name: (topic, msg_type)} and wait for one message each."""
    got: dict = {}
    subs = []

    def make_cb(name):
        def cb(msg):
            got.setdefault(name, msg)

        return cb

    for name, (topic, msg_type) in specs.items():
        subs.append(
            node.create_subscription(
                msg_type, topic, make_cb(name), qos_profile_sensor_data
            )
        )

    t0 = time.time()
    while time.time() - t0 < timeout_s and len(got) < len(specs):
        rclpy.spin_once(node, timeout_sec=0.1)

    missing = {n: specs[n][0] for n in specs if n not in got}
    if missing:
        raise TimeoutError(
            f"timed out after {timeout_s:.0f}s waiting for: "
            + ", ".join(f"{n} ({t})" for n, t in missing.items())
        )
    for s in subs:
        node.destroy_subscription(s)
    return got


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    from sensor_msgs_py import point_cloud2

    xyz = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z"), skip_nans=True
    )
    return np.asarray(xyz, dtype=np.float32).reshape(-1, 3)


def deproject(
    depth: np.ndarray, mask: np.ndarray | None, K: np.ndarray, max_depth: float
) -> np.ndarray:
    """Depth image -> (N, 3) XYZ in the camera optical frame.

    Standard pinhole deprojection: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.
    """
    if depth.dtype == np.uint16:  # millimetres
        depth = depth.astype(np.float32) / 1000.0
    depth = depth.astype(np.float32)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < max_depth)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}")
        valid &= mask > 0

    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def http_get(url: str, timeout: float = 15.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def http_post(url: str, body: dict, timeout: float = 120.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")}


def subsample(pc: np.ndarray, limit: int, rng) -> np.ndarray:
    if len(pc) <= limit:
        return pc
    return pc[rng.choice(len(pc), limit, replace=False)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Send live ROS 2 perception data to a /predict server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--url", default="http://localhost:8001", help="Server base URL.")
    ap.add_argument(
        "--source",
        default="depth",
        choices=["depth", "cloud"],
        help="Build the object cloud from depth+mask, or take a PointCloud2 topic.",
    )
    ap.add_argument("--depth-topic", default="/camera/depth")
    ap.add_argument("--mask-topic", default="/yolo_masker/debug/mask")
    ap.add_argument("--info-topic", default="/camera/camera_info")
    ap.add_argument(
        "--cloud-topic",
        default="/grasp_estimator/debug/cropped_pointcloud",
        help="Used when --source cloud.",
    )
    ap.add_argument(
        "--masked-depth-topic",
        default="/grasp_estimator/debug/depth",
        help="Fallback object depth when --mask-topic has no publisher.",
    )
    ap.add_argument("--num-grasps", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument(
        "--with-scene",
        action="store_true",
        help="Also send the non-target points (full depth minus mask) as "
        "`scene_points` so clutter-blocked grasps get dropped.",
    )
    ap.add_argument("--max-object-points", type=int, default=20000)
    ap.add_argument("--max-scene-points", type=int, default=16000)
    ap.add_argument("--max-depth", type=float, default=3.0, help="Meters.")
    ap.add_argument("--timeout", type=float, default=20.0, help="ROS wait, seconds.")
    ap.add_argument(
        "--publish",
        action="store_true",
        help="Republish the grasps as PoseArray + approach-arrow MarkerArray.",
    )
    ap.add_argument("--pose-topic", default="/graspgenx_test/grasps")
    ap.add_argument("--marker-topic", default="/graspgenx_test/grasp_markers")
    ap.add_argument("--repeat", type=int, default=1, help="Number of requests.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # -- server introspection ----------------------------------------------
    try:
        _, cfg = http_get(args.url + "/config")
    except Exception as exc:
        return _fail(f"cannot reach {args.url}/config: {exc}")

    backend = cfg.get("backend", "dexgraspnet2 (no `backend` field)")
    print(f"\n=== server {args.url} ===")
    print(f"  backend          : {backend}")
    print(f"  hand             : {cfg.get('hand')}")
    print(f"  checkpoint       : {cfg.get('checkpoint')}")
    print(f"  grasp_reference  : {cfg.get('grasp_reference')}")
    print(f"  grasp_frame      : {cfg.get('grasp_frame', 'n/a (assume +X approach)')}")
    print(f"  approach_local   : {cfg.get('approach_axis_local', '[1,0,0] assumed')}")
    print(f"  jaw_local        : {cfg.get('jaw_axis_local', '[0,1,0] assumed')}")
    print(f"  point budget     : {cfg.get('min_points')}..{cfg.get('max_points')}")

    rclpy.init()
    node = rclpy.create_node("graspgenx_predict_smoke_test")
    pose_pub = marker_pub = None
    if args.publish:
        pose_pub = node.create_publisher(PoseArray, args.pose_topic, 1)
        marker_pub = node.create_publisher(MarkerArray, args.marker_topic, 1)

    try:
        # -- gather perception data ----------------------------------------
        if args.source == "cloud":
            msgs = collect_messages(
                node, {"cloud": (args.cloud_topic, PointCloud2)}, args.timeout
            )
            object_pc = pointcloud2_to_xyz(msgs["cloud"])
            frame_id = msgs["cloud"].header.frame_id
            scene_pc = None
            print(f"\nobject cloud from {args.cloud_topic}: {len(object_pc)} points")
        else:
            specs = {
                "info": (args.info_topic, CameraInfo),
                "depth": (args.depth_topic, Image),
                "mask": (args.mask_topic, Image),
            }
            try:
                msgs = collect_messages(node, specs, args.timeout)
                mask = image_to_numpy(msgs["mask"])
            except TimeoutError as exc:
                print(f"\n[warn] {exc}")
                print(f"[warn] falling back to masked depth {args.masked_depth_topic}")
                specs = {
                    "info": (args.info_topic, CameraInfo),
                    "depth": (args.masked_depth_topic, Image),
                }
                msgs = collect_messages(node, specs, args.timeout)
                mask = None

            K = np.asarray(msgs["info"].k, dtype=np.float64).reshape(3, 3)
            depth = image_to_numpy(msgs["depth"])
            frame_id = msgs["depth"].header.frame_id
            print(f"\nintrinsics fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
                  f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}  ({depth.shape[1]}x{depth.shape[0]})")
            if mask is not None:
                print(f"mask: {int((mask > 0).sum())} / {mask.size} pixels set")

            object_pc = deproject(depth, mask, K, args.max_depth)
            print(f"object cloud: {len(object_pc)} points")

            scene_pc = None
            if args.with_scene:
                if mask is None:
                    print("[warn] --with-scene needs a mask; skipping scene_points")
                else:
                    full = collect_messages(
                        node, {"d": (args.depth_topic, Image)}, args.timeout
                    )["d"]
                    scene_pc = deproject(
                        image_to_numpy(full), (mask == 0), K, args.max_depth
                    )
                    print(f"scene cloud: {len(scene_pc)} points")

        if len(object_pc) == 0:
            return _fail("segmented object cloud is empty — is the sim publishing?")

        lo = int(cfg.get("min_points", 128))
        if len(object_pc) < lo:
            print(f"[warn] only {len(object_pc)} object points, server wants >= {lo}")

        object_pc = subsample(object_pc, args.max_object_points, rng)
        if scene_pc is not None:
            scene_pc = subsample(scene_pc, args.max_scene_points, rng)

        print(f"frame_id: {frame_id}")
        print(f"object bbox min {object_pc.min(0).round(3)} "
              f"max {object_pc.max(0).round(3)} centroid {object_pc.mean(0).round(3)}")

        # -- request --------------------------------------------------------
        body = {
            "point_cloud": encode_array(object_pc),
            "num_grasps": args.num_grasps,
            "min_score": args.min_score,
        }
        if scene_pc is not None and len(scene_pc):
            body["scene_points"] = encode_array(scene_pc)

        approach_local = np.asarray(
            cfg.get("approach_axis_local", [1.0, 0.0, 0.0]), dtype=float
        )

        for attempt in range(args.repeat):
            t0 = time.time()
            status, data = http_post(args.url + "/predict", body)
            rtt = (time.time() - t0) * 1000.0
            if status != 200:
                return _fail(f"HTTP {status}: {json.dumps(data)[:500]}")

            meta = data["meta"]
            grasps = data["grasps"]
            print(f"\n--- request {attempt + 1}/{args.repeat} ---")
            print(f"HTTP 200 in {rtt:.0f} ms  (server inference {meta['inference_ms']:.0f} ms)")
            print(f"  request_id   : {meta['request_id']}")
            print(f"  context_mode : {meta['context_mode']}")
            print(f"  points in    : {meta['num_input_points']} object, "
                  f"{meta.get('num_scene_points', 0)} scene")
            if "num_collision_rejected" in meta:
                print(f"  collision    : {meta['num_collision_rejected']} rejected")
            print(f"  grasps out   : {len(grasps)}")

            if not grasps:
                print("  (empty result — nothing graspable in this view)")
                continue

            print(f"\n  {'#':>2} {'score':>7} {'width':>7}  "
                  f"{'translation':<26} {'approach':<26} branch")
            for i, g in enumerate(grasps):
                R = np.asarray(g["rotation_matrix"])
                t = np.asarray(g["translation"])
                a = np.asarray(g["approach_axis"])
                w = g["gripper_width"]
                print(f"  {i:>2} {g['score']:>7.3f} "
                      f"{'   n/a' if w is None else f'{w:>7.4f}'}  "
                      f"{np.array2string(t.round(3)):<26} "
                      f"{np.array2string(a.round(3)):<26} {g.get('branch', '-')}")

                # Cheap self-consistency checks on the first grasp.
                if i == 0:
                    err_orth = np.abs(R @ R.T - np.eye(3)).max()
                    det = np.linalg.det(R)
                    axis_from_R = R @ approach_local
                    print(f"\n  sanity: det(R)={det:.6f}  "
                          f"orthonormality err={err_orth:.2e}")
                    print(f"          R @ approach_axis_local = "
                          f"{axis_from_R.round(4)}")
                    print(f"          reported approach_axis  = {a.round(4)}  "
                          f"{'MATCH' if np.allclose(axis_from_R, a, atol=1e-4) else 'DIFFERS'}")
                    # How far the fingertips still are from the reported
                    # origin: --tcp-offset may already have moved it forward.
                    remaining = float(cfg.get("fingertip_depth", 0.0)) - float(
                        cfg.get("tcp_offset", 0.0)
                    )
                    tip = t + a * remaining
                    print(f"          origin {t.round(3)} + {remaining:.3f} m "
                          f"-> fingertip plane {tip.round(3)} "
                          f"(object centroid {object_pc.mean(0).round(3)})")

            if args.publish:
                _publish(node, pose_pub, marker_pub, grasps, frame_id)
                print(f"\n  published {len(grasps)} poses on {args.pose_topic} "
                      f"and markers on {args.marker_topic}")
                # Give the publishers a moment to flush before we shut down.
                for _ in range(20):
                    rclpy.spin_once(node, timeout_sec=0.05)

        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _publish(node, pose_pub, marker_pub, grasps, frame_id):
    stamp = node.get_clock().now().to_msg()

    pa = PoseArray()
    pa.header.frame_id = frame_id
    pa.header.stamp = stamp
    markers = MarkerArray()

    for i, g in enumerate(grasps):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [
            float(v) for v in g["translation"]
        ]
        qx, qy, qz, qw = [float(v) for v in g["rotation_quat"]]
        pose.orientation.x, pose.orientation.y = qx, qy
        pose.orientation.z, pose.orientation.w = qz, qw
        pa.poses.append(pose)

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "approach"
        m.id = i
        m.type = Marker.ARROW
        m.action = Marker.ADD
        start = np.asarray(g["translation"], dtype=float)
        end = start + np.asarray(g["approach_axis"], dtype=float) * 0.05
        m.points = [_pt(start), _pt(end)]
        m.scale.x, m.scale.y, m.scale.z = 0.004, 0.008, 0.0
        # Best grasp green, fading to red down the ranking.
        frac = i / max(len(grasps) - 1, 1)
        m.color.r, m.color.g, m.color.b, m.color.a = frac, 1.0 - frac, 0.1, 0.9
        markers.markers.append(m)

    pose_pub.publish(pa)
    marker_pub.publish(markers)


def _pt(xyz):
    from geometry_msgs.msg import Point

    p = Point()
    p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return p


def _fail(msg: str) -> int:
    print(f"\nERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
