# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference serving layer for GraspGenX.

Two transports live here:

* ``zmq_server`` / ``zmq_client`` — a REQ/REP wire protocol (msgpack +
  msgpack-numpy) so lightweight clients, including the MCP bridge under
  ``mcp/`` and the CLI under ``client-server/``, can drive grasp inference
  without loading any model weights themselves. Needs the ``serve`` extra.
* ``rest_server`` — a FastAPI app speaking the HTTP contract in
  ``docs/api/predict.md``, shared with the DexGraspNet 2.0 server so one ROS
  client can talk to either backend. Needs the ``rest`` extra.

Submodules are resolved lazily so that importing one transport does not
require the other's dependencies to be installed.
"""

from graspgenx.serving.types import SweepVolumeParams

__all__ = [
    "GraspGenXClient",
    "GraspGenXZMQServer",
    "SweepVolumeParams",
    "rest_server",
]

_LAZY = {
    "GraspGenXClient": ("graspgenx.serving.zmq_client", "GraspGenXClient"),
    "GraspGenXZMQServer": ("graspgenx.serving.zmq_server", "GraspGenXZMQServer"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    if name == "rest_server":
        import importlib

        return importlib.import_module("graspgenx.serving.rest_server")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
