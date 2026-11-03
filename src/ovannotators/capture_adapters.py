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

"""Explicit adapters from finalized writer output to WebDataset samples.

Experimental: the adapter contract may change before it is declared stable.
Adapters declare a ``frame``, ``episode`` or ``clip`` sample unit.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence

from ovannotators._internal.mp4_probe import probe_mp4_video
from ovannotators.backends.webdataset import SampleFiles, _component_suffix
from ovannotators.config import config_to_dict
from ovannotators.exceptions import ConfigurationError, PackagingError
from ovannotators.writers.basic import BasicWriter
from ovannotators.writers.bop import BOPWriter
from ovannotators.writers.coco import COCOWriter
from ovannotators.writers.kitti import KITTIWriter
from ovannotators.writers.yolo import YOLOWriter, YOLOWriterConfig


_SAMPLE_UNITS = ("frame", "episode", "clip")


@dataclass(frozen=True)
class ClipSpan:
    """Identify one closed clip of an episode.

    Attributes:
        clip_index: Native clip index reported by ``CaptureAdapter.cut_clip``
            or ``CaptureAdapter.discover_clips``; unique within an episode.
        frame_ids: Ordered episode-local frame indices written into the clip.
    """

    clip_index: int
    frame_ids: tuple[int, ...]


@dataclass
class AdapterContext:
    """Carry dataset identity and shared format state across episode adapters.

    Attributes:
        dataset_id: WebDataset namespace.
        episode_id: Current episode identity.
        generated_root: Temporary directory for files an adapter generates,
            such as per-sample annotation records. Files under it are packaged
            but have no native source path. Removed after the episode.
        frame_ids: Ordered native frame indices.
        frame_files: File names the writer requested during the latest frame
            write. Recorded from the requested paths, not read back from disk,
            so portability checks do not depend on the host file system.
        state: Dataset-wide adapter state shared by every episode of one
            capture or export, such as a category map. Adapters own their keys.
        shared_paths: Native paths of episode-wide documents that every sample
            depends on, keyed by a role name such as ``instances``.
        clips: Closed clips in order, for ``clip`` sample-unit adapters only.
            Managed capture appends one span per clip boundary; finalized
            export fills it from ``CaptureAdapter.discover_clips``.
    """

    dataset_id: str
    episode_id: str
    generated_root: Path
    frame_ids: list[int] = field(default_factory=list)
    frame_files: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    shared_paths: dict[str, str] = field(default_factory=dict)
    clips: list[ClipSpan] = field(default_factory=list)


class CaptureAdapter(ABC):
    """Adapt one writer's finalized episode output into samples.

    Experimental. Subclasses must define ``sample_schema``, ``create_writer``
    and ``collect_samples``; the remaining hooks have permissive defaults.
    Built-in adapters are selected by exact writer type; pass an adapter
    explicitly to package a custom writer or a built-in writer subclass.

    ``sample_unit`` selects what one sample represents and is checked when
    the adapter is resolved:

    - ``"frame"`` (default): one sample per frame.
    - ``"episode"``: exactly one sample holding the complete episode, such as
      an episode video bundle. Shard sample counts then count episodes. The
      sample is never split, even with ``allow_episode_spanning=True``, so an
      episode larger than a shard needs
      ``oversized_episode_policy="dedicated_shard"``.
    - ``"clip"``: one sample per clip. The producer closes clips with
      ``CaptureEpisode.end_clip``, which calls ``cut_clip``; clip adapters
      must override ``cut_clip``. The clips of one episode stay in one shard
      unless spanning is enabled.

    Set ``allow_native_overwrite = True`` when the writer legitimately
    rewrites a native file during an episode, such as a clip manifest that
    ``CosmosTransferWriter.cut`` updates after every clip. By default the
    episode disk backend rejects overwrites. The flag applies to the whole
    episode backend, including frame writes: the backend can no longer detect
    a writer that reuses one file name for two frames, so set it only for a
    writer known to overwrite one specific native file.
    """

    sample_schema: str
    sample_unit: str = "frame"
    allow_native_overwrite: bool = False

    def validate(self, writer: Any) -> None:
        """Reject unsupported writer configurations before capture starts."""

    def describe_writer(self, writer: Any) -> dict[str, Any]:
        """Return the JSON writer description recorded in the manifest.

        The default records ``writer.name`` (or the class name) and
        ``writer.config`` converted with ``config_to_dict``.
        """
        config = getattr(writer, "config", None)
        if is_dataclass(config):
            config = config_to_dict(config)
        return {
            "writer": getattr(writer, "name", type(writer).__name__),
            "writer_config": config,
        }

    @abstractmethod
    def create_writer(self, template: Any) -> Any:
        """Create a clean writer for one capture episode."""

    def validate_frame_output(
        self, frame_id: int, writer: Any, root: Path, context: AdapterContext
    ) -> None:
        """Check output immediately after each frame write."""

    def cut_clip(
        self, writer: Any, backend: Any, root: Path, context: AdapterContext
    ) -> int:
        """Close the writer's current clip and return its native clip index.

        Called by ``CaptureEpisode.end_clip`` and, for frames written after
        the last boundary, before writer finalization. Call a public writer
        method, such as ``CosmosTransferWriter.cut(backend)``; do not change
        private writer state. The episode then appends the returned index and
        the frames written since the previous boundary to ``context.clips``.

        Args:
            writer: Episode writer created by ``create_writer``.
            backend: Episode disk backend that the writer writes to.
            root: Native output root of ``backend``.
            context: Current episode context.

        Returns:
            Nonnegative clip index, unique within the episode.

        Raises:
            PackagingError: By default; clip adapters must override it.
        """
        raise PackagingError(f"{type(self).__name__} does not support clips")

    def discover_frames(self, root: Path, writer: Any) -> list[int]:
        """Return ordered frame IDs from an existing finalized native dataset.

        Used by finalized export for the ``frame`` and ``episode`` units.

        Raises:
            PackagingError: By default; override to support finalized export.
        """
        raise PackagingError(
            f"{type(self).__name__} does not support finalized-dataset export"
        )

    def discover_clips(self, root: Path, writer: Any) -> list[ClipSpan]:
        """Return ordered clips from an existing finalized native dataset.

        Used by finalized export for the ``clip`` unit instead of
        ``discover_frames``. Spans obey the capture rules: unique clip
        indices, nonempty clips and strictly increasing frame IDs across the
        episode. The export sets ``context.frame_ids`` to their concatenation.

        Raises:
            PackagingError: By default; override to support finalized export.
        """
        raise PackagingError(
            f"{type(self).__name__} does not support finalized-dataset export"
        )

    @abstractmethod
    def collect_samples(
        self, root: Path, writer: Any, context: AdapterContext
    ) -> list[SampleFiles]:
        """Collect complete samples after writer finalization.

        Return one sample per frame for the ``frame`` unit, exactly one sample
        for the ``episode`` unit, or one sample per ``context.clips`` entry,
        in order, for the ``clip`` unit; episode and clip sample counts are
        checked before packaging. Component files must be under ``root``
        (native output) or ``context.generated_root`` (generated by the
        adapter).
        """

    def dataset_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Return dataset-wide manifest metadata derived from adapter state."""
        return {}


class BasicFrameAdapter(CaptureAdapter):
    """Package BasicWriter's finalized per-frame files."""

    sample_schema = "basic-frame-v1"

    def validate(self, writer: Any) -> None:
        """Require the supported BasicWriter frame configuration."""
        if not isinstance(writer, BasicWriter):
            raise ConfigurationError("BasicFrameAdapter requires a BasicWriter")
        if writer.config.video:
            raise ConfigurationError(
                "BasicWriter video output requires an explicit episode sample "
                "adapter; pass adapter=BasicVideoEpisodeAdapter()"
            )

    def create_writer(self, template: BasicWriter) -> BasicWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def validate_frame_output(
        self, frame_id: int, writer: BasicWriter, root: Path, context: AdapterContext
    ) -> None:
        """Reject an empty frame or a non-portable file name immediately."""
        names = sorted(context.frame_files)
        if not names:
            raise PackagingError(
                f"BasicWriter produced no files for frame {frame_id} in episode "
                f"{context.episode_id}; check render_vars and the writer selection"
            )
        for name in names:
            _component_suffix(name)

    def discover_frames(self, root: Path, writer: BasicWriter) -> list[int]:
        """Read native frame directories and reject ambiguous numbering."""
        digits = writer.config.frame_id_digits
        frames = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            if not path.name.isdigit() or len(path.name) < digits:
                raise PackagingError(f"Unexpected BasicWriter directory: {path.name}")
            frame = int(path.name)
            if f"{frame:0{digits}d}" != path.name:
                raise PackagingError(f"Ambiguous BasicWriter frame path: {path.name}")
            frames.append(frame)
        if not frames:
            raise PackagingError(f"No BasicWriter frame directories in {root}")
        return sorted(frames)

    def collect_samples(
        self, root: Path, writer: BasicWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Collect frame files with their original component names."""
        result = []
        for frame in context.frame_ids:
            directory = root / f"{frame:0{writer.config.frame_id_digits}d}"
            files = {}
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    raise PackagingError(
                        f"Unexpected entry in BasicWriter frame: {path.name}"
                    )
                files[_component_suffix(path.name)] = path
            if not files:
                raise PackagingError(f"BasicWriter frame {frame} contains no files")
            result.append(
                SampleFiles(frame, files, {"writer": "basic", "frame_index": frame})
            )
        return result


class COCOFrameAdapter(CaptureAdapter):
    """Package COCOWriter images and per-image annotation records.

    Image and annotation IDs are renumbered to be unique across the dataset;
    category IDs are mapped consistently across episodes by category name.
    """

    sample_schema = "coco-instance-v1"

    def validate(self, writer: Any) -> None:
        """Require a COCO writer instance."""
        if not isinstance(writer, COCOWriter):
            raise ConfigurationError("COCOFrameAdapter requires a COCOWriter")

    def create_writer(self, template: COCOWriter) -> COCOWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def discover_frames(self, root: Path, writer: COCOWriter) -> list[int]:
        """Read frame identities from finalized COCO images."""
        document = self._document(root, writer)
        frames = []
        for image in document["images"]:
            name = Path(image["file_name"])
            if (
                name.suffix != f".{writer.config.image_format}"
                or not name.stem.isdigit()
            ):
                raise PackagingError(f"Unsupported COCO image name: {name}")
            frames.append(int(name.stem))
        if len(frames) != len(set(frames)):
            raise PackagingError("Duplicate COCO frame identity")
        if not frames:
            raise PackagingError(f"No COCO images in {root}")
        if frames != sorted(frames):
            raise PackagingError(
                f"COCO images in {root} must be listed in increasing frame "
                "order, as DatasetCapture requires"
            )
        return frames

    def _document(self, root: Path, writer: COCOWriter) -> dict[str, Any]:
        cfg = writer.config
        annotations = root / "annotations"
        instances = annotations / f"instances_{cfg.split}.json"
        if not instances.is_file():
            found = sorted(p.name for p in annotations.glob("instances_*.json"))
            raise PackagingError(
                f"Missing {instances.name} for split {cfg.split!r} in {root}; "
                f"found {found or 'none'}"
            )
        panoptic = (annotations / f"panoptic_{cfg.split}.json").is_file()
        if panoptic != cfg.include_panoptic:
            raise PackagingError(
                f"COCO output in {root} {'has' if panoptic else 'lacks'} "
                f"panoptic_{cfg.split}.json but include_panoptic="
                f"{cfg.include_panoptic}"
            )
        return json.loads(instances.read_text())

    def collect_samples(
        self, root: Path, writer: COCOWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Split finalized COCO documents and reconcile dataset categories."""
        cfg = writer.config
        document = self._document(root, writer)
        context.shared_paths["instances"] = f"annotations/instances_{cfg.split}.json"
        if cfg.include_panoptic:
            context.shared_paths["panoptic"] = f"annotations/panoptic_{cfg.split}.json"
        if len(context.frame_ids) != len(document["images"]):
            raise PackagingError("COCO image count does not match captured frames")
        state = context.state
        categories = state.setdefault("categories", {})
        state.setdefault("next_image_id", 1)
        state.setdefault("next_annotation_id", 1)
        remap = {}
        for category in document["categories"]:
            name = category["name"]
            if name not in categories:
                categories[name] = len(categories) + 1
            remap[category["id"]] = categories[name]
        sample_categories = [
            {**c, "id": remap[c["id"]]} for c in document["categories"]
        ]
        image_ids = {}
        for image in document["images"]:
            image_ids[image["id"]] = state["next_image_id"]
            state["next_image_id"] += 1
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in document["annotations"]:
            for field_name, known in (("image_id", image_ids), ("category_id", remap)):
                if annotation[field_name] not in known:
                    raise PackagingError(
                        f"COCO annotation {annotation.get('id')} in episode "
                        f"{context.episode_id} references unknown "
                        f"{field_name} {annotation[field_name]!r}"
                    )
            annotations.setdefault(annotation["image_id"], []).append(
                {
                    **annotation,
                    "id": state["next_annotation_id"],
                    "image_id": image_ids[annotation["image_id"]],
                    "category_id": remap[annotation["category_id"]],
                }
            )
            state["next_annotation_id"] += 1
        panoptic = {}
        if cfg.include_panoptic:
            panoptic_doc = json.loads(
                (root / f"annotations/panoptic_{cfg.split}.json").read_text()
            )
            panoptic = {a["image_id"]: a for a in panoptic_doc["annotations"]}
        result = []
        for frame, image in zip(context.frame_ids, document["images"]):
            name = Path(image["file_name"])
            if name.stem != f"{frame:06d}":
                raise PackagingError("COCO image order does not match frame identity")
            key = f"{context.dataset_id}/episode-{context.episode_id}/sample-{frame}"
            files = {cfg.image_format: root / "images" / image["file_name"]}
            record = {
                "schema_version": 1,
                "image": {
                    **image,
                    "id": image_ids[image["id"]],
                    "file_name": f"{key}.{cfg.image_format}",
                },
                "annotations": annotations.get(image["id"], []),
                "categories": sample_categories,
            }
            if image["id"] in panoptic:
                entry = panoptic[image["id"]]
                files["panoptic.png"] = (
                    root / f"annotations/panoptic_{cfg.split}" / entry["file_name"]
                )
                record["panoptic"] = {
                    **entry,
                    "image_id": image_ids[image["id"]],
                    "file_name": key + ".panoptic.png",
                    "segments_info": [
                        {**s, "category_id": remap[s["category_id"]]}
                        for s in entry["segments_info"]
                    ],
                }
            path = context.generated_root / f"sample-{frame}.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            files["json"] = path
            result.append(
                SampleFiles(
                    frame,
                    files,
                    {"writer": "coco", "frame_index": frame, "split": cfg.split},
                )
            )
        return result

    def dataset_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Return the dataset-wide category table."""
        return {
            "categories": [
                {"id": value, "name": name, "supercategory": ""}
                for name, value in state.get("categories", {}).items()
            ]
        }


class BOPFrameAdapter(CaptureAdapter):
    """Package BOPWriter frames with their scene JSON entries and every object mask.

    Each sample holds one frame of the writer's ``{scene_id:06d}`` scene:
    ``rgb.png``, ``depth.png``, every ``mask.<suffix>.png`` and
    ``mask_visib.<suffix>.png`` named after the frame, and a generated ``json``
    record with that frame's ``scene_camera``, ``scene_gt`` and
    ``scene_gt_info`` entries. Encoded bytes and JSON values are unchanged.
    """

    sample_schema = "bop-frame-v1"

    _DOCUMENTS = ("scene_camera", "scene_gt", "scene_gt_info")
    _MASK_DIRS = ("mask", "mask_visib")

    def validate(self, writer: Any) -> None:
        """Require a BOP writer instance."""
        if not isinstance(writer, BOPWriter):
            raise ConfigurationError("BOPFrameAdapter requires a BOPWriter")

    def create_writer(self, template: BOPWriter) -> BOPWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def validate_frame_output(
        self, frame_id: int, writer: BOPWriter, root: Path, context: AdapterContext
    ) -> None:
        """Require the frame's RGB and depth images immediately."""
        scene = root / f"{writer.config.scene_id:06d}"
        for directory in ("rgb", "depth"):
            if not (scene / directory / f"{frame_id:06d}.png").is_file():
                raise PackagingError(
                    f"BOPWriter produced no {directory} image for frame {frame_id} "
                    f"in episode {context.episode_id}"
                )

    def discover_frames(self, root: Path, writer: BOPWriter) -> list[int]:
        """Read frame identities from the configured scene's camera document."""
        return sorted(self._documents(root, writer)["scene_camera"])

    def _documents(
        self, root: Path, writer: BOPWriter
    ) -> dict[str, dict[int, Any] | None]:
        """Load scene documents keyed by integer frame ID.

        ``scene_camera.json`` is required and defines the frames. A missing
        ``scene_gt.json`` or ``scene_gt_info.json`` maps to ``None``.
        """
        name = f"{writer.config.scene_id:06d}"
        if not (root / name / "scene_camera.json").is_file():
            found = sorted(p.name for p in root.iterdir() if p.is_dir())
            raise PackagingError(
                f"Missing {name}/scene_camera.json for scene_id "
                f"{writer.config.scene_id} in {root}; found {found or 'none'}"
            )
        documents: dict[str, dict[int, Any] | None] = {}
        for role in self._DOCUMENTS:
            path = root / name / f"{role}.json"
            if not path.is_file():
                documents[role] = None
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise PackagingError(
                    f"{name}/{role}.json is not valid UTF-8 JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise PackagingError(f"{name}/{role}.json is not a frame mapping")
            entries = {}
            for key, value in raw.items():
                if not key.isdigit() or str(int(key)) != key:
                    raise PackagingError(f"Unsupported BOP frame key {key!r} in {role}")
                entries[int(key)] = value
            documents[role] = entries
        frames = set(documents["scene_camera"])
        if not frames:
            raise PackagingError(f"No BOP frames in {name}/scene_camera.json")
        for role in self._DOCUMENTS[1:]:
            entries = documents[role]
            if entries is None:
                continue
            extra = sorted(set(entries) - frames)
            if extra:
                raise PackagingError(
                    f"{name}/{role}.json frames {extra} are absent from "
                    "scene_camera.json"
                )
            missing = sorted(frames - set(entries))
            if missing:
                raise PackagingError(
                    f"{name}/{role}.json has no entry for frames {missing}; "
                    "BOPWriter writes one per frame"
                )
        for role in self._DOCUMENTS[1:]:
            for frame, objects in (documents[role] or {}).items():
                if not isinstance(objects, list) or not all(
                    isinstance(entry, dict) for entry in objects
                ):
                    raise PackagingError(
                        f"{name}/{role}.json frame {frame} is not a list of objects"
                    )
        scale = float(writer.config.depth_scale)
        for frame, camera in documents["scene_camera"].items():
            if not isinstance(camera, dict):
                raise PackagingError(
                    f"{name}/scene_camera.json frame {frame} is not an object"
                )
            intrinsics = camera.get("cam_K")
            if (
                not isinstance(intrinsics, list)
                or len(intrinsics) != 9
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in intrinsics
                )
            ):
                raise PackagingError(
                    f"{name}/scene_camera.json frame {frame} lacks cam_K (9 numbers)"
                )
            if camera.get("depth_scale") != scale:
                raise PackagingError(
                    f"BOP frame {frame} depth_scale {camera.get('depth_scale')} "
                    f"does not match the writer depth_scale {scale}"
                )
        return documents

    def collect_samples(
        self, root: Path, writer: BOPWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Split finalized scene documents into per-frame records.

        Raises:
            PackagingError: If frames disagree with ``scene_camera.json``, a
                frame lacks RGB or depth, values are non-finite, or a file
                under ``root`` belongs to no sample or scene document.
        """
        scene_id = writer.config.scene_id
        name = f"{scene_id:06d}"
        documents = self._documents(root, writer)
        if set(documents["scene_camera"]) != set(context.frame_ids):
            raise PackagingError(
                "BOP scene_camera.json frames do not match the captured frames"
            )
        claimed = set()
        for role, entries in documents.items():
            if entries is not None:
                context.shared_paths[role] = f"{name}/{role}.json"
                claimed.add(root / name / f"{role}.json")
        object_ids = set(context.state.get("object_ids", []))
        # Index mask files by frame stem once instead of globbing per frame.
        masks_by_frame: dict[str, dict[str, list[Path]]] = {}
        for directory in self._MASK_DIRS:
            index: dict[str, list[Path]] = {}
            for path in (root / name / directory).glob("*_*.png"):
                index.setdefault(path.name.split("_", 1)[0], []).append(path)
            masks_by_frame[directory] = index
        result = []
        for frame in context.frame_ids:
            stem = f"{frame:06d}"
            files: dict[str, Path] = {}
            for directory in ("rgb", "depth"):
                path = root / name / directory / f"{stem}.png"
                if not path.is_file():
                    raise PackagingError(f"Missing {name}/{directory}/{stem}.png")
                files[f"{directory}.png"] = path
            for directory in self._MASK_DIRS:
                for path in sorted(masks_by_frame[directory].get(stem, ())):
                    suffix = path.name[len(stem) + 1 : -len(".png")]
                    files[_component_suffix(f"{directory}.{suffix}.png")] = path
            claimed.update(files.values())
            record: dict[str, Any] = {
                "schema_version": 1,
                "scene_id": scene_id,
                "frame_id": frame,
            }
            for role, entries in documents.items():
                record[role] = None if entries is None else entries.get(frame)
            masks = []
            # _documents validated scene_gt and scene_gt_info per frame.
            for entry in record["scene_gt"] or []:
                obj_id = entry.get("obj_id")
                if (
                    isinstance(obj_id, bool)
                    or not isinstance(obj_id, int)
                    or obj_id < 0
                ):
                    raise PackagingError(f"BOP frame {frame} has obj_id {obj_id!r}")
                references: dict[str, Any] = {"obj_id": obj_id}
                for directory in self._MASK_DIRS:
                    component = f"{directory}.{obj_id:06d}.png"
                    references[directory] = component if component in files else None
                masks.append(references)
                object_ids.add(obj_id)
            record["masks"] = masks
            try:
                text = json.dumps(record, allow_nan=False)
            except ValueError as exc:
                raise PackagingError(
                    f"BOP frame {frame} contains non-finite values: {exc}"
                ) from exc
            path = context.generated_root / f"sample-{frame}.json"
            path.write_text(text, encoding="utf-8")
            files["json"] = path
            result.append(
                SampleFiles(
                    frame,
                    files,
                    {"writer": "bop", "frame_index": frame, "scene_id": scene_id},
                )
            )
        unclaimed = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if not p.is_dir() and p not in claimed
        )
        if unclaimed:
            raise PackagingError(
                f"BOP output in {root} has files outside any frame sample or "
                f"scene document: {unclaimed[:5]}"
            )
        context.state["object_ids"] = sorted(object_ids)
        return result

    def dataset_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Return the sorted BOP object IDs observed across the dataset."""
        return {"object_ids": list(state.get("object_ids", []))}


_YOLO_SPLITS = ("train", "val")


class _DataYamlSink:
    """Backend stand-in that keeps the bytes ``YOLOWriter.finalize`` writes."""

    def __init__(self) -> None:
        self.written: dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        self.written[path] = data


def _writer_data_yaml(names: list[str]) -> str:
    """Return the ``data.yaml`` text a ``YOLOWriter`` emits for ``names``."""
    sink = _DataYamlSink()
    YOLOWriter(YOLOWriterConfig(classes=names)).finalize(sink)  # type: ignore[arg-type]
    return sink.written["data.yaml"].decode("utf-8")


class YOLOFrameAdapter(CaptureAdapter):
    """Package YOLOWriter images and labels with dataset-wide class IDs.

    Each sample keeps its native image and label rows. Only the class index
    column is remapped, by class name, to one dataset-wide table; normalized
    box and polygon text is copied unchanged. Explicit ``classes`` define the
    table; otherwise it extends in first-seen order across episodes. A frame's
    original ``train``/``val`` assignment is per-sample metadata; episodes are
    not repartitioned by split.
    """

    sample_schema = "yolo-frame-v1"

    def validate(self, writer: Any) -> None:
        """Require a YOLO writer with unique explicit class names."""
        if not isinstance(writer, YOLOWriter):
            raise ConfigurationError("YOLOFrameAdapter requires a YOLOWriter")
        classes = writer.config.classes
        if classes is not None and len(set(classes)) != len(classes):
            raise ConfigurationError("YOLOWriter classes must be unique to package")

    def create_writer(self, template: YOLOWriter) -> YOLOWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def validate_frame_output(
        self, frame_id: int, writer: YOLOWriter, root: Path, context: AdapterContext
    ) -> None:
        """Require one image and its label file in a single split."""
        self._frame_split(root, writer, frame_id)

    def discover_frames(self, root: Path, writer: YOLOWriter) -> list[int]:
        """Read frame identities from finalized YOLO split directories."""
        fmt = writer.config.image_format
        self._class_names(root)
        for kind in ("images", "labels"):
            directory = root / kind
            extra = (
                sorted(
                    p.name for p in directory.iterdir() if p.name not in _YOLO_SPLITS
                )
                if directory.is_dir()
                else []
            )
            if extra:
                raise PackagingError(f"Unexpected YOLO {kind} entries: {extra}")
        frames: dict[int, str] = {}
        for split in _YOLO_SPLITS:
            for kind, suffix in (("images", f".{fmt}"), ("labels", ".txt")):
                directory = root / kind / split
                if not directory.is_dir():
                    continue
                for path in directory.iterdir():
                    stem = path.name.removesuffix(suffix)
                    if (
                        not path.is_file()
                        or not path.name.endswith(suffix)
                        or not stem.isdigit()
                        or f"{int(stem):06d}" != stem
                    ):
                        raise PackagingError(
                            f"Unsupported YOLO file: {kind}/{split}/{path.name}"
                        )
                    if kind == "images":
                        if int(stem) in frames:
                            raise PackagingError(
                                f"YOLO frame {stem} appears in both splits"
                            )
                        frames[int(stem)] = split
                    elif frames.get(int(stem)) != split:
                        raise PackagingError(
                            f"YOLO label without matching image: "
                            f"labels/{split}/{path.name}"
                        )
        if not frames:
            raise PackagingError(f"No YOLO images in {root}")
        return sorted(frames)

    def collect_samples(
        self, root: Path, writer: YOLOWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Remap label class indices to the dataset table and record splits."""
        cfg = writer.config
        names = self._class_names(root)
        if cfg.classes is not None and names != list(cfg.classes):
            raise PackagingError(
                f"data.yaml names {names} in {root} do not match the writer's "
                f"explicit classes {list(cfg.classes)}"
            )
        context.shared_paths["data"] = "data.yaml"
        classes: list[str] = context.state.setdefault(
            "classes", list(cfg.classes or [])
        )
        counts = context.state.setdefault(
            "split_counts", dict.fromkeys(_YOLO_SPLITS, 0)
        )
        for name in names:
            if name not in classes:
                classes.append(name)
        remap = [classes.index(name) for name in names]
        native_ids = remap == list(range(len(names)))
        sample_classes = [
            {"id": remap[index], "name": name, "source_id": index}
            for index, name in enumerate(names)
        ]
        result = []
        for frame in context.frame_ids:
            split = self._frame_split(root, writer, frame)
            stem = f"{frame:06d}"
            label = root / "labels" / split / f"{stem}.txt"
            text = self._remap_labels(label, remap, cfg.task)
            files: dict[str, Path] = {
                cfg.image_format: root / "images" / split / f"{stem}.{cfg.image_format}"
            }
            if native_ids:
                files["txt"] = label
            else:
                generated = context.generated_root / f"sample-{frame}.txt"
                generated.write_bytes(text.encode("utf-8"))
                files["txt"] = generated
                files["native.txt"] = label
            record = {
                "schema_version": 1,
                "task": cfg.task,
                "split": split,
                "frame_index": frame,
                "objects": text.count("\n"),
                "classes": sample_classes,
            }
            path = context.generated_root / f"sample-{frame}.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            files["json"] = path
            counts[split] += 1
            result.append(
                SampleFiles(
                    frame,
                    files,
                    {"writer": "yolo", "frame_index": frame, "split": split},
                )
            )
        return result

    def dataset_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Return the dataset-wide class table and per-split sample counts."""
        return {
            "classes": [
                {"id": index, "name": name}
                for index, name in enumerate(state.get("classes", []))
            ],
            "split_counts": dict(
                state.get("split_counts", dict.fromkeys(_YOLO_SPLITS, 0))
            ),
        }

    @staticmethod
    def _frame_split(root: Path, writer: YOLOWriter, frame: int) -> str:
        stem = f"{frame:06d}"
        fmt = writer.config.image_format
        found = [
            split
            for split in _YOLO_SPLITS
            if (root / "images" / split / f"{stem}.{fmt}").is_file()
        ]
        if len(found) != 1:
            raise PackagingError(
                f"YOLO frame {frame} needs one {stem}.{fmt} image in images/train "
                f"or images/val; found {len(found)}"
            )
        if not (root / "labels" / found[0] / f"{stem}.txt").is_file():
            raise PackagingError(f"Missing YOLO label labels/{found[0]}/{stem}.txt")
        return found[0]

    @staticmethod
    def _class_names(root: Path) -> list[str]:
        """Parse YOLOWriter's data.yaml; reject anything it would not emit."""
        path = root / "data.yaml"
        if not path.is_file():
            raise PackagingError(f"Missing YOLO data.yaml in {root}")
        text = path.read_text(encoding="utf-8")
        prefix = "names: "
        line = next((ln for ln in text.splitlines() if ln.startswith(prefix)), "")
        try:
            names = ast.literal_eval(line.removeprefix(prefix))
        except (ValueError, SyntaxError):
            names = None
        if (
            not isinstance(names, list)
            or not all(isinstance(n, str) for n in names)
            or _writer_data_yaml(names) != text
        ):
            raise PackagingError(f"{path} is not a YOLOWriter data.yaml")
        if len(set(names)) != len(names):
            raise PackagingError(f"Duplicate class names in {path}")
        return names

    @staticmethod
    def _remap_labels(path: Path, remap: list[int], task: str) -> str:
        """Replace only each row's class index; keep coordinate text verbatim."""
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PackagingError(f"YOLO label is not UTF-8 text: {path}") from exc
        if text and not text.endswith("\n"):
            raise PackagingError(f"YOLO label lacks a final newline: {path}")
        rows = []
        for row in text[:-1].split("\n") if text else []:
            token, _, rest = row.partition(" ")
            fields = rest.split(" ")
            count = len(fields)
            # YOLOWriter writes a 4-field box row when a segment instance has no
            # polygon, so segment rows are that box row or at least 3 points.
            polygon = count >= 6 and count % 2 == 0
            valid = (
                token.isdigit()
                and int(token) < len(remap)
                and (count == 4 or (task == "segment" and polygon))
            )
            try:
                valid = valid and all(math.isfinite(float(f)) for f in fields)
            except ValueError:
                valid = False
            if not valid:
                raise PackagingError(f"Unsupported YOLO {task} label row in {path}")
            rows.append(f"{remap[int(token)]} {rest}\n")
        return "".join(rows)


class BasicVideoEpisodeAdapter(CaptureAdapter):
    """Package BasicWriter video output as one complete episode sample.

    Requires ``BasicWriterConfig(video=True)``. It is never selected
    automatically: pass ``adapter=BasicVideoEpisodeAdapter()`` to
    ``DatasetCapture`` or ``export_finalized_dataset``.

    The sample holds every episode MP4 with its encoded bytes unchanged, as
    the ``<stream>.mp4`` component, plus the per-frame files selected by
    ``frame_files``, as ``<frame directory>.<file name>`` components (for
    example ``0003.camera_params.json``). A generated ``episode.json``
    component records each stream's codec, coded size, timescale, frame
    count, presentation timestamps and the episode frame index of every video
    frame, plus the selected files of every frame.

    Args:
        frame_files: ``fnmatch`` patterns, matched case-sensitively against
            per-frame file names, that select the original frame artifacts
            to include. The default ``("*.json",)`` selects per-frame JSON
            annotations and metadata, such as camera parameters and bounding
            boxes, which the videos do not carry. Use ``("*",)`` for every
            per-frame file or ``()`` for videos only.

    Raises:
        ConfigurationError: If ``frame_files`` is not a sequence of nonempty
            strings.
    """

    sample_schema = "basic-video-episode-v1"
    sample_unit = "episode"

    def __init__(self, frame_files: Sequence[str] = ("*.json",)) -> None:
        """Initialize the adapter with a frame-artifact selection."""
        if isinstance(frame_files, str) or not all(
            isinstance(pattern, str) and pattern for pattern in frame_files
        ):
            raise ConfigurationError(
                "frame_files must be a sequence of nonempty file-name patterns"
            )
        self.frame_files = tuple(frame_files)

    def validate(self, writer: Any) -> None:
        """Require a BasicWriter in video mode with a portable video directory."""
        if not isinstance(writer, BasicWriter):
            raise ConfigurationError("BasicVideoEpisodeAdapter requires a BasicWriter")
        if not writer.config.video:
            raise ConfigurationError(
                "BasicVideoEpisodeAdapter requires BasicWriterConfig(video=True); "
                "use BasicFrameAdapter for frame output"
            )
        parts = writer.config.video_dir.strip("/").split("/")
        if any(part in {".", ".."} for part in parts) or parts[0].isdigit():
            raise ConfigurationError(
                f"Unsupported BasicWriter video_dir {writer.config.video_dir!r}; "
                "it must not contain '.' or '..' or start with a frame-like "
                "numeric directory"
            )

    def create_writer(self, template: BasicWriter) -> BasicWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def validate_frame_output(
        self, frame_id: int, writer: BasicWriter, root: Path, context: AdapterContext
    ) -> None:
        """Reject an empty frame or a selected file without a portable name."""
        name = f"{frame_id:0{writer.config.frame_id_digits}d}"
        names = sorted(context.frame_files)
        if not names:
            raise PackagingError(
                f"BasicWriter produced no files for frame {frame_id} in episode "
                f"{context.episode_id}; check render_vars and the writer selection"
            )
        for file_name in names:
            if self._selected(file_name):
                self._frame_component(name, file_name)

    def discover_frames(self, root: Path, writer: BasicWriter) -> list[int]:
        """Read native frame directories, skipping the video directory."""
        digits = writer.config.frame_id_digits
        top = writer.config.video_dir.strip("/").split("/")[0]
        frames = []
        for path in root.iterdir():
            if not path.is_dir() or (top and path.name == top):
                continue
            if not path.name.isdigit() or len(path.name) < digits:
                raise PackagingError(f"Unexpected BasicWriter directory: {path.name}")
            frame = int(path.name)
            if f"{frame:0{digits}d}" != path.name:
                raise PackagingError(f"Ambiguous BasicWriter frame path: {path.name}")
            frames.append(frame)
        if not frames:
            raise PackagingError(f"No BasicWriter frame directories in {root}")
        return sorted(frames)

    def collect_samples(
        self, root: Path, writer: BasicWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Collect the episode videos and selected frame files as one sample.

        Raises:
            PackagingError: If the episode has no video, a video cannot be
                read, its frame count differs from the number of frames that
                hold its source render var, or a component name is invalid.
        """
        cfg = writer.config
        files: dict[str, Path] = {}
        frames = []
        stems: dict[int, set[str]] = {}
        for frame in context.frame_ids:
            name = f"{frame:0{cfg.frame_id_digits}d}"
            directory = root / name
            if not directory.is_dir():
                raise PackagingError(f"Missing BasicWriter frame directory: {name}")
            components = {}
            stems[frame] = set()
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    raise PackagingError(
                        f"Unexpected entry in BasicWriter frame: {name}/{path.name}"
                    )
                stems[frame].add(path.name.partition(".")[0])
                if self._selected(path.name):
                    component = self._frame_component(name, path.name)
                    files[component] = path
                    components[path.name] = component
            frames.append({"frame_index": frame, "components": components})

        video_root = root / cfg.video_dir.strip("/")
        videos = (
            sorted(p for p in video_root.glob("*.mp4") if p.is_file())
            if video_root.is_dir()
            else []
        )
        if not videos:
            raise PackagingError(
                f"BasicWriter wrote no video for episode {context.episode_id}; "
                "check video_vars and that frames contain a video-capable render var"
            )
        streams = {}
        for video in videos:
            stream = video.stem
            component = _component_suffix(f"{stream}.mp4")
            try:
                track = probe_mp4_video(video)
            except ValueError as exc:
                raise PackagingError(f"Unreadable video {video.name}: {exc}") from exc
            source = stream.removesuffix("_colorized")
            frame_ids = [f for f in context.frame_ids if source in stems[f]]
            if len(frame_ids) != track.frame_count:
                raise PackagingError(
                    f"Video {video.name} has {track.frame_count} frames but "
                    f"{len(frame_ids)} frames of episode {context.episode_id} "
                    f"contain {source}; cannot pair video frames with frame indices"
                )
            files[component] = video
            streams[stream] = {
                "component": component,
                "codec": track.codec,
                "width": track.width,
                "height": track.height,
                "timescale": track.timescale,
                "frame_count": track.frame_count,
                "duration": track.duration,
                "pts": list(track.timestamps),
                "frame_ids": frame_ids,
            }
        record = {
            "schema_version": 1,
            "episode_id": context.episode_id,
            "frame_ids": list(context.frame_ids),
            "video_fps": cfg.video_fps,
            "frame_file_patterns": list(self.frame_files),
            "streams": streams,
            "frames": frames,
        }
        path = context.generated_root / "episode.json"
        path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        files["episode.json"] = path
        metadata = {
            "writer": "basic",
            "episode_id": context.episode_id,
            "frame_count": len(context.frame_ids),
            "streams": sorted(streams),
        }
        return [SampleFiles("episode", files, metadata)]

    def dataset_metadata(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Record the frame-artifact selection in the manifest."""
        return {"frame_file_patterns": list(self.frame_files)}

    def _selected(self, name: str) -> bool:
        return any(fnmatchcase(name, pattern) for pattern in self.frame_files)

    @staticmethod
    def _frame_component(directory: str, name: str) -> str:
        component = f"{directory}.{name}"
        try:
            return _component_suffix(component)
        except PackagingError as exc:
            raise PackagingError(
                f"Selected frame file {directory}/{name} has no portable component "
                f"name ({component!r}); narrow frame_files"
            ) from exc


class KITTIFrameAdapter(CaptureAdapter):
    """Package KITTIWriter image, label, calibration and optional depth files.

    Each native file becomes a distinct component with unchanged bytes:
    ``image_2.png``, ``label_2.txt``, ``calib.txt`` and, when
    ``write_depth`` is set, ``depth.png``. Label text, calibration matrices
    and stored uint16 depth values are not parsed or rewritten. The writer
    stored depth as metres times ``depth_scale``, clipped to 0-65535 and
    truncated to an integer, so fractions are lost and saturated values cannot
    be recovered. An empty ``label_2.txt`` is a negative example.
    """

    sample_schema = "kitti-frame-v1"
    _LAYOUT = (
        ("image_2", "png", "image_2.png"),
        ("label_2", "txt", "label_2.txt"),
        ("calib", "txt", "calib.txt"),
    )
    _DEPTH = ("depth", "png", "depth.png")

    def validate(self, writer: Any) -> None:
        """Require a KITTI writer instance."""
        if not isinstance(writer, KITTIWriter):
            raise ConfigurationError("KITTIFrameAdapter requires a KITTIWriter")

    def create_writer(self, template: KITTIWriter) -> KITTIWriter:
        """Create an empty writer of the template's type and configuration."""
        return type(template)(copy.deepcopy(template.config))

    def _layout(self, writer: KITTIWriter) -> tuple[tuple[str, str, str], ...]:
        return self._LAYOUT + ((self._DEPTH,) if writer.config.write_depth else ())

    def validate_frame_output(
        self, frame_id: int, writer: KITTIWriter, root: Path, context: AdapterContext
    ) -> None:
        """Require every configured KITTI file for the frame."""
        for directory, extension, _ in self._layout(writer):
            if not (root / directory / f"{frame_id:06d}.{extension}").is_file():
                raise PackagingError(
                    f"KITTIWriter produced no {directory} file for frame "
                    f"{frame_id} in episode {context.episode_id}"
                )

    def discover_frames(self, root: Path, writer: KITTIWriter) -> list[int]:
        """Read frame IDs and require identical frame sets in every directory.

        Raises:
            PackagingError: If a directory is missing or unexpected, a file name
                is not a canonical frame name, or the directories disagree.
        """
        layout = self._layout(writer)
        expected = {directory for directory, _, _ in layout}
        found = {path.name for path in root.iterdir()}
        if found != expected:
            raise PackagingError(
                f"KITTI output in {root} has {sorted(found)}; expected "
                f"{sorted(expected)} for write_depth={writer.config.write_depth}"
            )
        for directory, _, _ in layout:
            if not (root / directory).is_dir():
                raise PackagingError(
                    f"KITTI entry {directory} in {root} is not a directory"
                )
        frames = None
        for directory, extension, _ in layout:
            current = set()
            for path in (root / directory).iterdir():
                stem = path.name.removesuffix(f".{extension}")
                if (
                    not path.is_file()
                    or stem == path.name
                    or not stem.isdigit()
                    or f"{int(stem):06d}" != stem
                ):
                    raise PackagingError(
                        f"Unexpected KITTI entry: {directory}/{path.name}"
                    )
                current.add(int(stem))
            if frames is not None and current != frames:
                raise PackagingError(
                    f"KITTI {directory} frames differ from {layout[0][0]} in {root}"
                )
            frames = current
        if not frames:
            raise PackagingError(f"No KITTI frames in {root}")
        return sorted(frames)

    def collect_samples(
        self, root: Path, writer: KITTIWriter, context: AdapterContext
    ) -> list[SampleFiles]:
        """Collect each frame's native files as separate components."""
        if self.discover_frames(root, writer) != context.frame_ids:
            raise PackagingError("KITTI frames do not match captured frames")
        layout = self._layout(writer)
        result = []
        for frame in context.frame_ids:
            stem = f"{frame:06d}"
            files = {
                component: root / directory / f"{stem}.{extension}"
                for directory, extension, component in layout
            }
            metadata: dict[str, Any] = {"writer": "kitti", "frame_index": frame}
            if writer.config.write_depth:
                metadata["depth_scale"] = writer.config.depth_scale
            result.append(SampleFiles(frame, files, metadata))
        return result


_BUILTIN_ADAPTERS: Mapping[type, type[CaptureAdapter]] = {
    BasicWriter: BasicFrameAdapter,
    COCOWriter: COCOFrameAdapter,
    KITTIWriter: KITTIFrameAdapter,
    BOPWriter: BOPFrameAdapter,
    YOLOWriter: YOLOFrameAdapter,
}


def resolve_adapter(writer: Any, adapter: CaptureAdapter | None) -> CaptureAdapter:
    """Select an explicit adapter or an exact built-in writer match.

    Raises:
        ConfigurationError: If no adapter applies, the adapter lacks a
            non-empty string ``sample_schema`` or ``sample_unit``, its
            ``sample_unit`` is unsupported, ``allow_native_overwrite`` is not
            a boolean, a clip adapter does not override ``cut_clip``, the
            adapter rejects the writer, or the writer description is not
            JSON-serializable.
    """
    if adapter is None:
        adapter_type = _BUILTIN_ADAPTERS.get(type(writer))
        if adapter_type is None:
            names = ", ".join(sorted(t.__name__ for t in _BUILTIN_ADAPTERS))
            raise ConfigurationError(
                f"Adapters are selected automatically only for exact built-in "
                f"writer types ({names}); pass adapter= (for example "
                "BasicFrameAdapter()) for subclasses and other writers, or use "
                "SampleFiles"
            )
        adapter = adapter_type()
    elif not isinstance(adapter, CaptureAdapter):
        raise ConfigurationError("adapter must be a CaptureAdapter instance")
    for attribute in ("sample_schema", "sample_unit"):
        value = getattr(adapter, attribute, None)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(
                f"{type(adapter).__name__} must define {attribute} as a "
                f"non-empty string, got {value!r}; the value is recorded in "
                "the manifest at finalize"
            )
    if adapter.sample_unit not in _SAMPLE_UNITS:
        raise ConfigurationError(
            f"Unsupported sample_unit {adapter.sample_unit!r}; "
            f"use one of {', '.join(_SAMPLE_UNITS)}"
        )
    if not isinstance(adapter.allow_native_overwrite, bool):
        raise ConfigurationError("allow_native_overwrite must be a boolean")
    if (
        adapter.sample_unit == "clip"
        and type(adapter).cut_clip is CaptureAdapter.cut_clip
    ):
        raise ConfigurationError(
            f"{type(adapter).__name__} uses clip samples and must override cut_clip"
        )
    adapter.validate(writer)
    try:
        json.dumps(adapter.describe_writer(writer), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Writer description must be JSON-serializable: {exc}; "
            "override CaptureAdapter.describe_writer"
        ) from exc
    return adapter


def _check_clips(clips: Sequence[ClipSpan]) -> None:
    """Validate the ordered clip spans of one episode.

    Raises:
        PackagingError: If there are no clips, a span is malformed or empty, a
            clip index repeats, or frame IDs do not strictly increase across
            the episode.
    """
    if not clips:
        raise PackagingError("A clip-unit episode must contain at least one clip")
    indices: set[int] = set()
    previous = -1
    for span in clips:
        if not isinstance(span, ClipSpan):
            raise PackagingError("Clips must be ClipSpan instances")
        index = span.clip_index
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PackagingError(f"Invalid clip index: {index!r}")
        if index in indices:
            raise PackagingError(f"Duplicate clip index: {index}")
        indices.add(index)
        if not span.frame_ids:
            raise PackagingError(f"Clip {index} contains no frames")
        for frame in span.frame_ids:
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame <= previous
            ):
                raise PackagingError(
                    f"Clip {index} frame indices must be nonnegative and "
                    "strictly increasing across the episode"
                )
            previous = frame


def source_paths(
    sample: SampleFiles, root: Path, generated_root: Path
) -> dict[str, str]:
    """Map native components to POSIX paths relative to ``root``.

    Generated components are omitted. Paths are normalized but symlinks are
    not followed, so an alias keeps its own name. Raises ``PackagingError`` if a
    component, or a symlink's target, lies outside both roots.
    """
    native = Path(os.path.abspath(root))
    generated = Path(os.path.abspath(generated_root))
    real_roots = (native.resolve(), generated.resolve())
    paths = {}
    for component, source in sample.files.items():
        path = Path(os.path.abspath(source))
        if not any(path.resolve().is_relative_to(r) for r in real_roots):
            raise PackagingError(f"Component {component} is outside the native root")
        if path.is_relative_to(generated):
            continue
        if not path.is_relative_to(native):
            raise PackagingError(f"Component {component} is outside the native root")
        paths[component] = path.relative_to(native).as_posix()
    return paths
