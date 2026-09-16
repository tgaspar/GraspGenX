# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the REST grasp-prediction server.

Everything here is CPU-only and model-free: the frame algebra, the array
codec, the width estimator and the response shape are all exercised against a
stubbed sampler, so the whole file runs without a GPU or a checkpoint.

The wire contract these assertions encode lives in ``docs/api/predict.md``.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="needs the 'rest' extra")
from fastapi.testclient import TestClient  # noqa: E402

from graspgenx.serving import rest_server as rs  # noqa: E402


# ---------------------------------------------------------------------------
# Frame conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(rs.GRASP_FRAMES))
def test_frame_matrices_are_proper_rotations(name):
    C = np.asarray(rs.GRASP_FRAMES[name]["C"], dtype=np.float64)
    assert np.allclose(C @ C.T, np.eye(3)), f"{name}: C is not orthogonal"
    assert np.isclose(np.linalg.det(C), 1.0), f"{name}: C is a reflection"


@pytest.mark.parametrize("name", sorted(rs.GRASP_FRAMES))
def test_frame_maps_declared_axes_onto_native(name):
    """C must send the declared output axes onto GraspGenX's native ones.

    Native convention: +Z approach, +X jaw-closing.
    """
    frame = rs.GRASP_FRAMES[name]
    C = np.asarray(frame["C"], dtype=np.float64)
    approach_out = np.asarray(frame["approach_axis_local"], dtype=np.float64)
    jaw_out = np.asarray(frame["jaw_axis_local"], dtype=np.float64)

    assert np.allclose(C @ approach_out, [0, 0, 1]), f"{name}: approach axis"
    assert np.allclose(C @ jaw_out, [1, 0, 0]), f"{name}: jaw axis"


@pytest.mark.parametrize("name", sorted(rs.GRASP_FRAMES))
def test_emitted_rotation_preserves_physical_axes(name):
    """R_out @ approach_axis_local must equal the true world approach.

    This is the property a client actually relies on: whichever convention the
    server is configured with, rotating that convention's local approach axis
    by the emitted matrix has to land on the same physical direction.
    """
    rng = np.random.default_rng(7)
    R_native, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R_native) < 0:
        R_native[:, 0] *= -1

    frame = rs.GRASP_FRAMES[name]
    R_out = R_native @ np.asarray(frame["C"], dtype=np.float64)

    approach_world = R_native @ np.array([0.0, 0.0, 1.0])
    jaw_world = R_native @ np.array([1.0, 0.0, 0.0])

    assert np.allclose(R_out @ frame["approach_axis_local"], approach_world)
    assert np.allclose(R_out @ frame["jaw_axis_local"], jaw_world)


def test_dexgraspnet2_frame_is_x_approach_y_jaw():
    """Pin the two conventions we claim compatibility with, explicitly."""
    C = rs.GRASP_FRAMES["dexgraspnet2"]["C"]
    R_native = np.eye(3)
    R_out = R_native @ C
    # Native approach is +Z; in the DexGraspNet 2.0 frame it must be column 0.
    assert np.allclose(R_out[:, 0], [0, 0, 1])
    # Native jaw is +X; in the DexGraspNet 2.0 frame it must be column 1.
    assert np.allclose(R_out[:, 1], [1, 0, 0])


def test_tcp_z_approach_frame_is_z_approach_y_jaw():
    C = rs.GRASP_FRAMES["tcp_z_approach"]["C"]
    R_out = np.eye(3) @ C
    assert np.allclose(R_out[:, 2], [0, 0, 1])  # approach stays +Z
    assert np.allclose(R_out[:, 1], [1, 0, 0])  # jaw moves to +Y


# ---------------------------------------------------------------------------
# Quaternion conversion
# ---------------------------------------------------------------------------


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_quaternion_round_trips_for_random_rotations():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        q = rs._rotation_matrix_to_quat_xyzw(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert q[3] >= 0.0, "w should be sign-canonicalised non-negative"
        assert np.allclose(_quat_to_matrix(q), R, atol=1e-9)


@pytest.mark.parametrize(
    "R",
    [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),  # 180 deg about X — trace branch fails here
        np.diag([-1.0, 1.0, -1.0]),  # 180 deg about Y
        np.diag([-1.0, -1.0, 1.0]),  # 180 deg about Z
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ],
)
def test_quaternion_handles_singular_branches(R):
    """Each Shepperd branch, including the three 180-degree cases where the
    naive trace formula divides by zero."""
    q = rs._rotation_matrix_to_quat_xyzw(R)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(_quat_to_matrix(q), R, atol=1e-9)


# ---------------------------------------------------------------------------
# Array codec
# ---------------------------------------------------------------------------


def _encode(arr):
    return rs.EncodedArray(
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        data_b64=base64.b64encode(arr.tobytes()).decode("ascii"),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_decode_array_round_trip(dtype):
    arr = np.arange(30, dtype=dtype).reshape(10, 3)
    out = rs._decode_array(_encode(arr), rs._ALLOWED_PC_DTYPES)
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)
    assert out.flags.writeable, "downstream torch/numpy code needs a writable array"


def test_decode_array_rejects_bad_dtype():
    arr = np.arange(30, dtype=np.int32).reshape(10, 3)
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_array(_encode(arr), rs._ALLOWED_PC_DTYPES)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "bad_dtype"


def test_decode_array_rejects_shape_byte_mismatch():
    arr = np.zeros((10, 3), dtype=np.float32)
    enc = _encode(arr)
    enc.shape = [11, 3]
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_array(enc, rs._ALLOWED_PC_DTYPES)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "shape_dtype_mismatch"


def test_decode_array_rejects_bad_base64():
    enc = rs.EncodedArray(dtype="float32", shape=[10, 3], data_b64="!!!nope!!!")
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_array(enc, rs._ALLOWED_PC_DTYPES)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "bad_base64"


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 4), dtype=np.float32),
        np.zeros((10,), dtype=np.float32),
    ],
)
def test_decode_point_cloud_rejects_bad_shape(bad):
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_point_cloud(_encode(bad), "point_cloud", None)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "bad_shape"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_decode_point_cloud_rejects_non_finite(bad_value):
    arr = np.zeros((10, 3), dtype=np.float32)
    arr[3, 1] = bad_value
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_point_cloud(_encode(arr), "point_cloud", None)
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "invalid_geometry"


def test_decode_point_cloud_enforces_min_points():
    arr = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(fastapi.HTTPException) as exc:
        rs._decode_point_cloud(_encode(arr), "point_cloud", 128)
    assert exc.value.status_code == 422
    assert "at least 128" in exc.value.detail["message"]


# ---------------------------------------------------------------------------
# Fixtures for the response-shape tests
# ---------------------------------------------------------------------------


class _StubGripper:
    """Stands in for XGripperInfo with the arx_x5 numbers."""

    # [extents_xyz, offset_xyz] of the open sweep box.
    sweep_volume = np.array([0.085, 0.016, 0.06, 0.0, 0.0, 0.12])
    depth = 0.143
    collision_mesh = None


@pytest.fixture
def configured_state(monkeypatch):
    """Populate STATE the way initialise_state would, without a model."""
    monkeypatch.setattr(rs.STATE, "gripper", _StubGripper())
    monkeypatch.setattr(rs.STATE, "gripper_name", "arx_x5")
    monkeypatch.setattr(rs.STATE, "model_name", "GraspGenX/release")
    monkeypatch.setattr(rs.STATE, "checkpoint", "gen.pth+dis.pth")
    monkeypatch.setattr(rs.STATE, "max_aperture", 0.085)
    monkeypatch.setattr(rs.STATE, "fingertip_depth", 0.143)
    monkeypatch.setattr(rs.STATE, "grasp_reference", "gripper_base_link")
    monkeypatch.setattr(rs.STATE, "grasp_frame", "dexgraspnet2")
    monkeypatch.setattr(
        rs.STATE, "frame_C", np.asarray(rs.GRASP_FRAMES["dexgraspnet2"]["C"])
    )
    monkeypatch.setattr(rs.STATE, "tcp_offset", 0.0)
    monkeypatch.setattr(rs.STATE, "width_mode", "aperture")
    monkeypatch.setattr(rs.STATE, "approach_axis_convention", "true_approach")
    monkeypatch.setattr(rs.STATE, "object_id", 1)
    monkeypatch.setattr(rs.STATE, "loaded", True)
    monkeypatch.setattr(rs.STATE, "log_dir", None)
    monkeypatch.setattr(rs.STATE, "requests_jsonl", None)
    monkeypatch.setattr(rs.STATE, "dump_npz", False)
    return rs.STATE


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------


def _pose(R=None, t=(0.1, 0.2, 0.3)):
    P = np.eye(4)
    P[:3, :3] = np.eye(3) if R is None else R
    P[:3, 3] = t
    return P


def test_grasp_response_has_exactly_the_contract_fields(configured_state):
    out = rs._grasp_to_response(_pose(), 0.9, 0.085, "diff")
    contract = {
        "translation",
        "rotation_quat",
        "rotation_matrix",
        "score",
        "gripper_width",
        "joint_angles",
        "joint_names",
        "approach_axis",
        "object_id",
    }
    # Every DexGraspNet 2.0 field is present; `branch` is our only addition.
    assert contract <= set(out)
    assert set(out) - contract == {"branch"}
    assert out["joint_angles"] is None and out["joint_names"] is None


def test_approach_axis_is_native_z_regardless_of_frame(configured_state, monkeypatch):
    """The reported approach direction must not depend on the output frame."""
    rng = np.random.default_rng(3)
    R_native, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R_native) < 0:
        R_native[:, 0] *= -1
    expected = R_native @ np.array([0.0, 0.0, 1.0])

    for name, frame in rs.GRASP_FRAMES.items():
        monkeypatch.setattr(rs.STATE, "grasp_frame", name)
        monkeypatch.setattr(rs.STATE, "frame_C", np.asarray(frame["C"]))
        out = rs._grasp_to_response(_pose(R_native), 0.5, None, "diff")
        assert np.allclose(out["approach_axis"], expected), name
        # ...and rotating the frame's own local axis by R_out agrees with it.
        R_out = np.asarray(out["rotation_matrix"])
        assert np.allclose(R_out @ frame["approach_axis_local"], expected), name


def test_frame_z_convention_reports_the_output_frames_z(configured_state, monkeypatch):
    """Bug-compatibility mode with the DexGraspNet 2.0 server."""
    monkeypatch.setattr(rs.STATE, "approach_axis_convention", "frame_z")
    rng = np.random.default_rng(11)
    R_native, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R_native) < 0:
        R_native[:, 0] *= -1

    out = rs._grasp_to_response(_pose(R_native), 0.5, None, "diff")
    R_out = np.asarray(out["rotation_matrix"])
    assert np.allclose(out["approach_axis"], R_out[:, 2])
    # In the default dexgraspnet2 frame the real approach is column 0, so the
    # two conventions must actually disagree — otherwise the flag is pointless.
    assert not np.allclose(R_out[:, 2], R_out[:, 0])


def test_tcp_offset_shifts_along_the_approach_axis(configured_state, monkeypatch):
    R_native = np.eye(3)  # native approach = world +Z
    base = rs._grasp_to_response(_pose(R_native), 0.5, None, "diff")
    monkeypatch.setattr(rs.STATE, "tcp_offset", 0.143)
    moved = rs._grasp_to_response(_pose(R_native), 0.5, None, "diff")

    delta = np.asarray(moved["translation"]) - np.asarray(base["translation"])
    assert np.allclose(delta, 0.143 * np.array([0.0, 0.0, 1.0]))
    # Orientation is untouched by the offset.
    assert np.allclose(moved["rotation_matrix"], base["rotation_matrix"])


def test_object_id_is_configurable(configured_state, monkeypatch):
    assert rs._grasp_to_response(_pose(), 0.5, None, "diff")["object_id"] == 1
    monkeypatch.setattr(rs.STATE, "object_id", -1)
    assert rs._grasp_to_response(_pose(), 0.5, None, "diff")["object_id"] == -1


# ---------------------------------------------------------------------------
# Width estimation
# ---------------------------------------------------------------------------


def test_measured_width_recovers_a_known_object_span(configured_state, monkeypatch):
    monkeypatch.setattr(rs.STATE, "width_clearance", 0.01)
    # Identity grasp: native frame == world. The open sweep box spans
    # y in [-0.008, 0.008] and z in [0.09, 0.15]; put a 4 cm wide slab of
    # points inside it, centred on the grasp axis.
    rng = np.random.default_rng(5)
    n = 500
    pts = np.stack(
        [
            rng.uniform(-0.02, 0.02, n),  # +/- 2 cm about the axis => 4 cm span
            rng.uniform(-0.005, 0.005, n),
            rng.uniform(0.10, 0.14, n),
        ],
        axis=-1,
    ).astype(np.float32)

    widths = rs._measure_grasp_widths(np.stack([_pose(t=(0, 0, 0))]), pts)
    assert widths.shape == (1,)
    # 2 * max|x| + clearance, and max|x| approaches 0.02 with 500 samples.
    assert widths[0] == pytest.approx(0.05, abs=2e-3)


def test_measured_width_clamps_to_the_aperture(configured_state, monkeypatch):
    monkeypatch.setattr(rs.STATE, "width_clearance", 0.01)
    pts = np.array([[0.5, 0.0, 0.12], [-0.5, 0.0, 0.12]], dtype=np.float32)
    widths = rs._measure_grasp_widths(np.stack([_pose(t=(0, 0, 0))]), pts)
    assert widths[0] == pytest.approx(rs.STATE.max_aperture)


def test_measured_width_falls_back_when_jaws_are_empty(configured_state):
    """No object points inside the sweep box => open the jaws fully."""
    pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)  # far outside the box
    widths = rs._measure_grasp_widths(np.stack([_pose(t=(0, 0, 0))]), pts)
    assert widths[0] == pytest.approx(rs.STATE.max_aperture)


def test_measured_width_handles_empty_inputs(configured_state):
    assert rs._measure_grasp_widths(np.zeros((0, 4, 4)), np.zeros((0, 3))).shape == (0,)


# ---------------------------------------------------------------------------
# Scene pruning
# ---------------------------------------------------------------------------


def test_scene_pruning_drops_only_points_near_the_target():
    pytest.importorskip("torch")
    target = np.zeros((1, 3), dtype=np.float32)
    scene = np.array(
        [[0.005, 0, 0], [0.02, 0, 0], [0.5, 0, 0]], dtype=np.float32
    )
    kept = rs._prune_scene_near_target(scene, target, radius=0.01)
    assert len(kept) == 2
    assert np.allclose(kept, scene[1:])


def test_scene_pruning_is_a_no_op_at_zero_radius():
    scene = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    kept = rs._prune_scene_near_target(scene, scene, radius=0.0)
    assert kept is scene


# ---------------------------------------------------------------------------
# Endpoints that need no model
# ---------------------------------------------------------------------------


def test_endpoints_report_not_ready_before_load(monkeypatch):
    monkeypatch.setattr(rs.STATE, "loaded", False)
    monkeypatch.setattr(rs.STATE, "sampler", None)
    client = TestClient(rs.build_app())

    r = client.get("/healthz")
    assert r.status_code == 503 and r.json() == {"status": "loading"}

    assert client.get("/config").status_code == 503

    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [200, 3],
            "data_b64": base64.b64encode(
                np.zeros((200, 3), dtype=np.float32).tobytes()
            ).decode(),
        }
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "not_ready"


def test_version_is_available_before_the_model_loads(monkeypatch):
    monkeypatch.setattr(rs.STATE, "loaded", False)
    payload = TestClient(rs.build_app()).get("/version").json()
    assert payload["backend"] == "graspgenx"
    assert payload["api_version"] == rs.API_VERSION


def test_config_reports_the_active_conventions(configured_state):
    cfg = TestClient(rs.build_app()).get("/config").json()
    # Contract fields the DexGraspNet 2.0 server also returns.
    for key in (
        "hand",
        "model_name",
        "checkpoint",
        "joint_names",
        "grasp_reference",
        "default_num_grasps",
        "min_points",
        "max_points",
        "server_version",
    ):
        assert key in cfg, key
    assert cfg["backend"] == "graspgenx"
    assert cfg["joint_names"] is None
    assert cfg["approach_axis_local"] == [1.0, 0.0, 0.0]
    assert cfg["jaw_axis_local"] == [0.0, 1.0, 0.0]
    assert cfg["supports_category_sampling"] is False


def test_predict_rejects_oversized_payloads(configured_state, monkeypatch):
    monkeypatch.setattr(rs.STATE, "sampler", object())
    monkeypatch.setattr(rs.STATE, "max_points", 100)
    monkeypatch.setattr(rs.STATE, "min_points", 1)
    arr = np.zeros((200, 3), dtype=np.float32)
    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [200, 3],
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
        }
    }
    r = TestClient(rs.build_app()).post("/predict", json=body)
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "payload_too_large"


def test_predict_returns_a_well_formed_response(configured_state, monkeypatch):
    """Full /predict path with the planner stubbed out — no GPU, no weights."""
    monkeypatch.setattr(rs.STATE, "sampler", object())
    monkeypatch.setattr(rs.STATE, "min_points", 1)
    monkeypatch.setattr(rs.STATE, "oversample_factor", 4)

    poses = np.stack([_pose(t=(0, 0, 0.1)), _pose(t=(0, 0, 0.2)), _pose(t=(0, 0, 0.3))])
    # Deliberately unsorted: GraspMoE concatenates its two branches without a
    # global sort, so the server has to rank them itself.
    scores = np.array([0.4, 0.95, 0.7], dtype=np.float32)
    tags = ["diff", "obb", "diff"]

    import graspgenx.samplers.planner as planner_mod

    monkeypatch.setattr(
        planner_mod, "run_planner_on_object", lambda *a, **k: (poses, scores, tags, None)
    )

    arr = np.zeros((300, 3), dtype=np.float32)
    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [300, 3],
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
        },
        "num_grasps": 2,
        "category_sampling": True,
    }
    data = TestClient(rs.build_app()).post("/predict", json=body).json()

    grasps = data["grasps"]
    assert len(grasps) == 2, "num_grasps must cap the result"
    assert [g["score"] for g in grasps] == [pytest.approx(0.95), pytest.approx(0.7)]
    assert [g["branch"] for g in grasps] == ["obb", "diff"]

    meta = data["meta"]
    assert meta["num_input_points"] == 300
    assert meta["num_scene_points"] == 0
    assert meta["context_mode"] == "object_only"
    assert meta["frame_id"] == "unchanged_from_input"
    assert meta["backend"] == "graspgenx"
    # category_sampling is a no-op here but must be echoed, not rejected.
    assert meta["category_sampling"] is True
    assert meta["debug_html"] is None and meta["debug_usd"] is None


def test_predict_applies_min_score(configured_state, monkeypatch):
    monkeypatch.setattr(rs.STATE, "sampler", object())
    monkeypatch.setattr(rs.STATE, "min_points", 1)

    poses = np.stack([_pose(), _pose()])
    scores = np.array([0.9, 0.2], dtype=np.float32)

    import graspgenx.samplers.planner as planner_mod

    monkeypatch.setattr(
        planner_mod,
        "run_planner_on_object",
        lambda *a, **k: (poses, scores, ["diff", "diff"], None),
    )

    arr = np.zeros((300, 3), dtype=np.float32)
    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [300, 3],
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
        },
        "num_grasps": 10,
        "min_score": 0.5,
    }
    data = TestClient(rs.build_app()).post("/predict", json=body).json()
    assert len(data["grasps"]) == 1
    assert data["grasps"][0]["score"] == pytest.approx(0.9)


def test_predict_returns_empty_list_rather_than_an_error(configured_state, monkeypatch):
    """Nothing graspable is a 200 with an empty list, per the contract."""
    monkeypatch.setattr(rs.STATE, "sampler", object())
    monkeypatch.setattr(rs.STATE, "min_points", 1)

    import graspgenx.samplers.planner as planner_mod

    monkeypatch.setattr(
        planner_mod,
        "run_planner_on_object",
        lambda *a, **k: (np.zeros((0, 4, 4)), np.zeros((0,)), [], None),
    )

    arr = np.zeros((300, 3), dtype=np.float32)
    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [300, 3],
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
        }
    }
    r = TestClient(rs.build_app()).post("/predict", json=body)
    assert r.status_code == 200
    assert r.json()["grasps"] == []


def test_predict_surfaces_model_failures_as_500(configured_state, monkeypatch):
    monkeypatch.setattr(rs.STATE, "sampler", object())
    monkeypatch.setattr(rs.STATE, "min_points", 1)

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    import graspgenx.samplers.planner as planner_mod

    monkeypatch.setattr(planner_mod, "run_planner_on_object", boom)

    arr = np.zeros((300, 3), dtype=np.float32)
    body = {
        "point_cloud": {
            "dtype": "float32",
            "shape": [300, 3],
            "data_b64": base64.b64encode(arr.tobytes()).decode(),
        }
    }
    r = TestClient(rs.build_app()).post("/predict", json=body)
    assert r.status_code == 500
    assert r.json()["detail"]["error"] == "internal_error"
    assert "CUDA out of memory" in r.json()["detail"]["message"]
