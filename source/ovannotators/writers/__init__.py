# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writers layer for ovannotators.

This package provides dataset-format writers that consume per-frame render
vars (numpy arrays) and persist them in a target layout.

Importing this package auto-registers all built-in writers into
:class:`WriterRegistry`.  Third-party writers can register via the
``ovannotators.writers`` entry-point group::

    [project.entry-points."ovannotators.writers"]
    my_writer = "mypackage.writers:MyWriter"

Re-exported names
-----------------
Writer
    Protocol that all writers must satisfy.
WriterRegistry
    Central registry; use ``WriterRegistry.get("name")`` to retrieve a class
    and ``WriterRegistry.list_names()`` to enumerate available writers.
BasicWriter, BasicWriterConfig
COCOWriter, COCOWriterConfig
KITTIWriter, KITTIWriterConfig
YOLOWriter, YOLOWriterConfig
BOPWriter, BOPWriterConfig
LeRobotWriter, LeRobotWriterConfig
CosmosTransferWriter, CosmosTransferWriterConfig
"""

from __future__ import annotations

# Importlib entry-point auto-registration (third-party writers)
import importlib.metadata as _imeta

from ovannotators.writers._base import Writer
from ovannotators.writers._registry import WriterRegistry

# Import concrete writers to trigger @WriterRegistry.register at module load.
from ovannotators.writers.basic import BasicWriter, BasicWriterConfig
from ovannotators.writers.bop import BOPWriter, BOPWriterConfig
from ovannotators.writers.coco import COCOWriter, COCOWriterConfig
from ovannotators.writers.kitti import KITTIWriter, KITTIWriterConfig
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)
from ovannotators.writers.lerobot import LeRobotWriter, LeRobotWriterConfig
from ovannotators.writers.yolo import YOLOWriter, YOLOWriterConfig

# All seven writers are now registered.
__all__ = [
    "Writer",
    "WriterRegistry",
    "BasicWriter",
    "BasicWriterConfig",
    "COCOWriter",
    "COCOWriterConfig",
    "KITTIWriter",
    "KITTIWriterConfig",
    "YOLOWriter",
    "YOLOWriterConfig",
    "BOPWriter",
    "BOPWriterConfig",
    "LeRobotWriter",
    "LeRobotWriterConfig",
    "CosmosTransferWriter",
    "CosmosTransferWriterConfig",
]


# Load any installed third-party writers.
def _load_entry_points() -> None:
    try:
        eps = _imeta.entry_points(group="ovannotators.writers")
    except Exception:
        return
    for ep in eps:
        try:
            ep.load()
        except Exception:
            pass


_load_entry_points()
