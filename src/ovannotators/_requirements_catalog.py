# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in dataset discovery definitions, loaded on first discovery call."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any
from threading import RLock

from ovannotators.exceptions import ConfigurationError
from ovannotators.requirements import (
    DatasetProfile,
    DatasetRequirements,
    register_dataset_profiles,
)

_registered = False
_registration_lock = RLock()


def register_builtins() -> None:
    """Register built-in definitions without constructing writers or codecs."""
    with _registration_lock:
        _register_builtins()


def _register_builtins() -> None:
    global _registered
    if _registered:
        return
    from ovannotators import writers as w
    from ovannotators import profile_writers as p

    specs = (
        (
            w.BasicWriter,
            w.BasicWriterConfig,
            (
                DatasetProfile(
                    "default", "Direct per-frame outputs", {"render_vars": ["LdrColor"]}
                ),
            ),
        ),
        (
            w.COCOWriter,
            w.COCOWriterConfig,
            (
                DatasetProfile(
                    "detection",
                    "COCO image and bounding-box annotations",
                    constraints={
                        "include_segmentation": False,
                        "include_panoptic": False,
                    },
                ),
                DatasetProfile(
                    "instance_segmentation",
                    "COCO per-instance RLE masks",
                    constraints={
                        "include_segmentation": True,
                        "include_panoptic": False,
                    },
                ),
                DatasetProfile(
                    "panoptic",
                    "Existing instance-derived COCO panoptic export",
                    defaults={"include_segmentation": True},
                    constraints={
                        "include_panoptic": True,
                    },
                ),
            ),
        ),
        (
            w.YOLOWriter,
            w.YOLOWriterConfig,
            (
                DatasetProfile(
                    "detection",
                    "Ultralytics detection labels",
                    constraints={"task": "detect"},
                ),
                DatasetProfile(
                    "instance_segmentation",
                    "Ultralytics instance polygons",
                    constraints={"task": "segment"},
                ),
            ),
        ),
        (
            w.OpenLABELWriter,
            w.OpenLABELWriterConfig,
            (
                DatasetProfile(
                    "detection",
                    "OpenLABEL image boxes",
                    constraints={"include_segmentation": False},
                ),
                DatasetProfile(
                    "instance_segmentation",
                    "OpenLABEL image boxes and instance polygons",
                    defaults={"include_segmentation": True},
                    constraints={"include_segmentation": True},
                ),
            ),
        ),
        (
            w.KITTIWriter,
            w.KITTIWriterConfig,
            (DatasetProfile("default", "KITTI object detection"),),
        ),
        (
            w.BOPWriter,
            w.BOPWriterConfig,
            (DatasetProfile("default", "BOP pose, masks, and depth"),),
        ),
        (
            w.LeRobotWriter,
            w.LeRobotWriterConfig,
            (DatasetProfile("default", "Legacy image-only LeRobot layout"),),
        ),
        (
            w.CosmosTransferWriter,
            w.CosmosTransferWriterConfig,
            (DatasetProfile("default", "RGB and 8-bit conditioning videos"),),
        ),
        (
            p.CosmosMultiviewWriter,
            p.CosmosMultiviewWriterConfig,
            (
                DatasetProfile(
                    "default", "Cosmos 3.5 candidate registration and materialization"
                ),
            ),
        ),
    )
    for writer, config, profiles in specs:
        register_dataset_profiles(writer, config, profiles, partial(_resolve, writer))
    _registered = True


def _resolve(
    writer: type, profile: str, values: Mapping[str, Any]
) -> DatasetRequirements:
    name = writer.name
    render_vars = set(getattr(writer, "required_render_vars", ()))
    recommended: set[str] = set()
    scene = ()
    temporal = ()
    non_render = ()
    notes = ()
    sensors = ()
    cameras: int | None = 1
    streams = ()
    representations: dict[str, Any] = {}
    for key in ("fps", "video_fps", "depth_scale", "depth_max_m", "meters_per_unit"):
        if key in values and (not math.isfinite(values[key]) or values[key] <= 0):
            raise ConfigurationError(f"{key} must be positive")
    if name in {"coco", "yolo", "openlabel", "kitti", "bop"}:
        recommended.add("SemanticIdMap")
        scene = (
            "Semantic labels on target objects; matching IDs across boxes and masks.",
        )
        representations["image"] = values.get("image_format", "png")
    if name == "basic":
        selected = values["render_vars"]
        render_vars = set(selected or ())
        from ovannotators.writers.basic import _VAR_FILE_MAP

        sensor_vars = {"PointCloud", "RtxSensorCpu", "RtxSensorGpu"}
        cameras = None
        if selected is not None:
            if set(selected) & (set(_VAR_FILE_MAP) - sensor_vars):
                cameras = 1
            elif set(selected) <= sensor_vars:
                cameras = 0
            if set(selected) & sensor_vars:
                sensors = (
                    "Bind the requested sensor outputs to the corresponding lidar/radar producer; they are not camera outputs.",
                )
        representations["image"] = values["image_format"]
        notes = (
            "An explicit render_vars selection is the capture request; the writer itself tolerates missing selected outputs.",
        )
        if selected is None:
            notes += (
                "render_vars=None writes supplied outputs and does not mean request every renderer output.",
            )
    elif name == "coco":
        if values["include_segmentation"] or values["include_panoptic"]:
            render_vars.add("SemanticInstanceSegmentation")
        if profile == "panoptic":
            notes = (
                "The existing export derives panoptic segments from instance masks; no separate stuff-label authoring API is provided.",
            )
    elif name == "yolo":
        if values["task"] == "segment":
            render_vars.add("SemanticInstanceSegmentation")
    elif name == "openlabel":
        if values["include_segmentation"]:
            render_vars.add("SemanticInstanceSegmentation")
    elif name == "kitti":
        if values["write_depth"]:
            render_vars.add("DistanceToCameraSD")
            representations["depth"] = {
                "capture": "DistanceToCameraSD",
                "storage": "uint16 PNG",
                "units": "metres",
                "stored_units_per_metre": values["depth_scale"],
            }
    elif name == "bop":
        representations["depth"] = {
            "capture": "DistanceToCameraSD",
            "storage": "uint16 PNG",
            "millimetres_per_stored_unit": values["depth_scale"],
        }
    elif name == "lerobot":
        keys = values["camera_keys"]
        if not keys or len(set(keys)) != len(keys):
            raise ConfigurationError("camera_keys must be non-empty and unique")
        temporal = (f"One episode; synthesized timestamps at {values['fps']} fps.",)
        representations["image"] = "uint8 RGB H.264 video"
        notes = ("Every camera key receives the same incoming image.",)
    elif name == "cosmos_transfer":
        from ovannotators.writers.cosmos_transfer import (
            _required_vars_for_config,
            CosmosTransferWriterConfig,
        )

        render_vars = set(
            _required_vars_for_config(CosmosTransferWriterConfig(**values))
        )
        temporal = (
            f"Constant frame rate {values['fps']} fps; explicit clip boundaries.",
        )
        if "depth" in values["conditioning_types"]:
            representations["depth"] = {
                "capture": "DistanceToCameraSD",
                "storage": "8-bit inverse-depth H.264 conditioning",
                "units": "clip-normalized, not physical depth",
            }
    elif name == "cosmos_multiview":
        cameras = None
        materialization = values.get("materialization")
        selected = (
            tuple(materialization.selected_stream_ids)
            if materialization is not None
            else ()
        )
        non_render = (
            "Episode and stream descriptors, source timestamps, provenance, and owner-supplied QA policy.",
        ) + selected
        temporal = (
            "Explicit episodes; timestamp-aligned half-open windows and per-stream alignment policies.",
        )
        scene = (
            "Camera streams with geometry need intrinsics, image size, camera model, reference frame ID, and an SE(3) transform.",
        )
        representations["streams"] = (
            dict(materialization.output_encoding_by_stream_type)
            if materialization is not None
            else {}
        )
        notes = (
            "Stream descriptors determine camera count, modalities, dtype, shape, units, and geometry; selected stream IDs alone do not identify cameras.",
            "This is a candidate profile; discovery does not establish external consumer acceptance.",
        )
    return DatasetRequirements(
        name,
        profile,
        frozenset(render_vars),
        frozenset(recommended),
        cameras,
        streams,
        non_render,
        scene,
        temporal,
        representations,
        notes,
        sensor_requirements=sensors,
    )
