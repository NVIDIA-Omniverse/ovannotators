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

"""Manage episode capture and adapt finalized writer output to WebDataset samples.

Experimental: ``DatasetCapture``, ``CaptureEpisode`` and
``export_finalized_dataset`` may change before they are declared stable.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.backends.webdataset import WebDatasetPackager, _identifier
from ovannotators.capture_adapters import (
    AdapterContext,
    CaptureAdapter,
    ClipSpan,
    _check_clips,
    resolve_adapter,
    source_paths,
)
from ovannotators.exceptions import ConfigurationError, PackagingError


class _PackagingSession:
    """Shared adapter-to-packager pipeline for capture and finalized export."""

    def __init__(
        self,
        writer: Any,
        packager: WebDatasetPackager,
        adapter: CaptureAdapter | None,
        provenance: str,
    ) -> None:
        if not isinstance(packager, WebDatasetPackager):
            raise ConfigurationError("Packaging requires a WebDatasetPackager")
        self.adapter = resolve_adapter(writer, adapter)
        self.packager = packager
        self.writer = writer
        self.provenance = provenance
        self.state: dict[str, Any] = {}
        self.episodes: set[str] = set()
        self.source_paths: dict[str, dict[str, Any]] = {}

    def context(self, episode_id: str, generated_root: Path) -> AdapterContext:
        return AdapterContext(
            dataset_id=self.packager.config.dataset_id,
            episode_id=episode_id,
            generated_root=generated_root,
            state=self.state,
        )

    def package(self, root: Path, writer: Any, context: AdapterContext) -> None:
        """Collect, record and pack one episode; roll back state on failure."""
        snapshot = copy.deepcopy(self.state)
        unit = self.adapter.sample_unit
        try:
            if unit == "clip":
                _check_clips(context.clips)
            samples = list(self.adapter.collect_samples(root, writer, context))
            if unit == "episode" and len(samples) != 1:
                raise PackagingError(
                    f"Episode adapter returned {len(samples)} samples for episode "
                    f"{context.episode_id}; expected exactly one"
                )
            if unit == "clip" and len(samples) != len(context.clips):
                raise PackagingError(
                    f"Clip adapter returned {len(samples)} samples for "
                    f"{len(context.clips)} clips in episode {context.episode_id}"
                )
            paths = {
                str(sample.key): source_paths(sample, root, context.generated_root)
                for sample in samples
            }
            self.packager.pack_episode(context.episode_id, samples)
        except BaseException:
            self.state.clear()
            self.state.update(snapshot)
            raise
        record: dict[str, Any] = {
            "shared": dict(context.shared_paths),
            "samples": paths,
        }
        if unit == "clip":
            record["clips"] = [
                {
                    "clip_index": span.clip_index,
                    "sample": str(sample.key),
                    "frame_ids": list(span.frame_ids),
                }
                for span, sample in zip(context.clips, samples)
            ]
        self.source_paths[context.episode_id] = record
        self.episodes.add(context.episode_id)

    def finalize(self) -> None:
        metadata = {
            **self.adapter.describe_writer(self.writer),
            "sample_schema": self.adapter.sample_schema,
            "sample_unit": self.adapter.sample_unit,
            "provenance": self.provenance,
            **self.adapter.dataset_metadata(self.state),
        }
        document = {
            "schema_version": 1,
            "provenance": self.provenance,
            "episodes": self.source_paths,
        }
        self.packager.finalize(
            metadata,
            auxiliary={
                "source-paths.json": json.dumps(document, allow_nan=False).encode()
            },
        )


class _RecordingDisk(DiskBackend):
    """Disk backend that records the file names a writer asks it to create.

    Names are recorded from the requested path rather than read back from the
    staging directory, so the portability check does not depend on how the host
    file system treats characters such as ``:`` (an NTFS stream separator).
    """

    def __init__(self, config: DiskBackendConfig) -> None:
        super().__init__(config)
        self.written: list[str] = []

    def write(self, path: str, data: bytes) -> None:
        self.written.append(path.rsplit("/", 1)[-1])
        super().write(path, data)


class DatasetCapture:
    """Capture explicit episodes with a format writer and a sample adapter.

    Experimental: the managed capture API may change before it is declared
    stable. ``WebDatasetPackager`` and ``SampleFiles`` are the stable
    packaging surface.

    Each episode uses a fresh writer and temporary disk backend, so deferred
    annotations are finalized before packaging. The supplied writer is a
    configuration template, not a previously populated writer. Native writer
    output remains unchanged. A failed episode is discarded and the capture
    stays open, so the producer may catch the error and retry or skip that
    episode. A storage failure aborts the capture. Context exit finalizes
    packaging on success and aborts on an uncaught error. Exact BasicWriter and
    COCOWriter types use built-in adapters; pass ``adapter=`` to package other
    writers. The adapter's ``sample_unit`` decides whether samples are frames,
    whole episodes or clips closed with ``CaptureEpisode.end_clip``. Basic
    episode-wide video output is not a frame sample.
    """

    def __init__(
        self,
        writer: Any,
        packager: WebDatasetPackager,
        *,
        adapter: CaptureAdapter | None = None,
    ) -> None:
        """Initialize managed capture and validate the writer and adapter.

        Args:
            writer: Fresh format-writer configuration template.
            packager: Explicit WebDataset packaging destination.
            adapter: Explicit adapter for a custom writer or a built-in writer
                subclass. Exact BasicWriter and COCOWriter types select a
                built-in adapter when omitted.

        Raises:
            ConfigurationError: If no adapter applies or the writer description
                cannot be recorded in the manifest.
        """
        if not isinstance(packager, WebDatasetPackager):
            raise ConfigurationError("DatasetCapture requires a WebDatasetPackager")
        self._session = _PackagingSession(writer, packager, adapter, "capture")
        self.adapter = self._session.adapter
        self.packager = packager
        self._active: CaptureEpisode | None = None
        self._state = "new"

    def __enter__(self) -> DatasetCapture:
        if self._state != "new":
            raise PackagingError("DatasetCapture contexts are single-use")
        self._state = "open"
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.abort()
        else:
            try:
                self.finalize()
            except BaseException:
                self.abort()
                raise

    def episode(self, episode_id: str | int) -> CaptureEpisode:
        """Create an episode context; concurrent/nested episodes are rejected.

        Args:
            episode_id: Dataset-wide unique portable episode identifier.

        Returns:
            Context in which complete frames may be written.
        """
        if self._state != "open" or self._active is not None:
            raise PackagingError(
                "Open a dataset context and finish its current episode first"
            )
        identity = _identifier(episode_id)
        if identity in self._session.episodes:
            raise PackagingError(f"Duplicate episode: {identity}")
        return CaptureEpisode(self, identity)

    def finalize(self) -> None:
        """Finalize the dataset after all episode contexts have completed."""
        if self._state == "complete":
            return
        if self._state != "open" or self._active is not None:
            raise PackagingError(
                f"Cannot finalize a {self._state} dataset or during an episode"
            )
        self._session.finalize()
        self._state = "complete"

    def abort(self) -> None:
        """Clean temporary resources and abort packaging without marking success."""
        if self._state not in {"complete", "aborted"}:
            if self._active is not None:
                self._active._cleanup()
            self.packager.abort()
            self._state = "aborted"

    def _episode_failed(self) -> None:
        """Abort the capture only when the packager can no longer continue."""
        if not self.packager.is_open:
            self.abort()


class CaptureEpisode:
    """Stage frames for one episode and package them on successful context exit."""

    def __init__(self, capture: DatasetCapture, episode_id: str) -> None:
        """Create an episode belonging to a managed capture.

        Args:
            capture: Owning capture session.
            episode_id: Validated episode identifier.
        """
        self._capture = capture
        self.episode_id = episode_id
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._root: Path | None = None
        self._context: AdapterContext | None = None
        self._writer: Any = None
        self._disk: _RecordingDisk | None = None
        self._frames: list[int] = []
        self._clip_start = 0
        self._state = "new"

    def __enter__(self) -> CaptureEpisode:
        capture = self._capture
        if (
            self._state != "new"
            or capture._state != "open"
            or capture._active is not None
        ):
            raise PackagingError("Episode contexts are single-use and cannot be nested")
        session = capture._session
        if self.episode_id in session.episodes:
            raise PackagingError(f"Duplicate episode: {self.episode_id}")
        self._tmp = tempfile.TemporaryDirectory(prefix="ovannotators-episode-")
        self._root = Path(self._tmp.name) / "native"
        generated = Path(self._tmp.name) / "generated"
        self._root.mkdir()
        generated.mkdir()
        self._context = session.context(self.episode_id, generated)
        self._disk = _RecordingDisk(
            DiskBackendConfig(
                root_dir=self._root,
                overwrite=session.adapter.allow_native_overwrite,
            )
        )
        self._writer = session.adapter.create_writer(session.writer)
        capture._active = self
        self._state = "open"
        return self

    def write_frame(self, frame_id: int, render_vars: dict[str, np.ndarray]) -> None:
        """Write a complete frame using the selected format writer.

        Args:
            frame_id: Nonnegative strictly increasing episode-local frame index;
                gaps are allowed and indices may restart in the next episode.
            render_vars: Original frame mapping; COCO requires LdrColor and
                SemanticBoundingBox2DTight, with optional SemanticIdMap.

        Raises:
            PackagingError: If frame identity is invalid or the adapter rejects
                the frame output, such as a BasicWriter frame without files. A
                failed write discards the episode; the capture stays open for
                another episode.
        """
        if self._state != "open" or self._capture._state != "open":
            raise PackagingError("Frame writes require an open episode context")
        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
            or (self._frames and frame_id <= self._frames[-1])
        ):
            raise PackagingError(
                "Frame indices must be nonnegative and strictly increasing"
            )
        try:
            self._disk.written.clear()
            self._writer.write_frame(frame_id, render_vars, self._disk)
            self._context.frame_files = list(self._disk.written)
            self._capture.adapter.validate_frame_output(
                frame_id, self._writer, self._root, self._context
            )
        except BaseException:
            self._discard()
            raise
        self._frames.append(frame_id)

    def end_clip(self) -> int:
        """Close the current clip of a ``clip`` sample-unit adapter.

        Frames written since the previous boundary form the clip. The adapter's
        ``cut_clip`` tells the writer to close it, and the span is recorded in
        the episode's ``AdapterContext.clips``. Frames written after the last
        boundary are closed as a final clip on successful episode exit, so a
        producer only calls this between clips, for example between scene
        variations.

        Returns:
            Native clip index reported by the adapter.

        Raises:
            PackagingError: If the episode is not open, the adapter's sample
                unit is not ``clip`` or no frame was written since the previous
                boundary; the episode stays open. If the adapter or writer
                fails to close the clip, the episode is discarded and the
                capture stays open.
        """
        if self._state != "open" or self._capture._state != "open":
            raise PackagingError("Clip boundaries require an open episode context")
        adapter = self._capture.adapter
        if adapter.sample_unit != "clip":
            raise PackagingError(
                f"{type(adapter).__name__} packages {adapter.sample_unit} samples; "
                "clip boundaries require a clip sample-unit adapter"
            )
        if len(self._frames) == self._clip_start:
            raise PackagingError(
                f"Cannot end an empty clip in episode {self.episode_id}; "
                "write at least one frame first"
            )
        return self._cut_clip()

    def _cut_clip(self) -> int:
        frames = tuple(self._frames[self._clip_start :])
        try:
            index = self._capture.adapter.cut_clip(
                self._writer, self._disk, self._root, self._context
            )
            span = ClipSpan(index, frames)
            _check_clips([*self._context.clips, span])
        except BaseException:
            self._discard()
            raise
        self._context.clips.append(span)
        self._clip_start = len(self._frames)
        return index

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self._discard()
            self._cleanup()
            return
        try:
            if self._state == "failed":
                raise PackagingError(
                    f"Cannot complete failed episode {self.episode_id}"
                )
            if self._state != "open" or self._capture._state != "open":
                raise PackagingError("Cannot complete an aborted episode")
            if not self._frames:
                raise PackagingError(f"Episode {self.episode_id} contains no frames")
            if (
                self._capture.adapter.sample_unit == "clip"
                and len(self._frames) > self._clip_start
            ):
                self._cut_clip()
            self._writer.finalize(self._disk)
            self._context.frame_ids = self._frames.copy()
            self._capture._session.package(self._root, self._writer, self._context)
            self._state = "complete"
        except BaseException:
            self._discard()
            raise
        finally:
            self._cleanup()

    def _discard(self) -> None:
        """Drop this episode's output and keep the capture open if possible."""
        if self._state == "open":
            self._state = "failed"
        if self._tmp is not None:
            self._tmp.cleanup()
        self._capture._episode_failed()

    def _cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        if self._capture._active is self:
            self._capture._active = None
        if self._state == "open":
            self._state = "aborted"


def export_finalized_dataset(
    writer: Any,
    episodes: Mapping[str | int, str | Path],
    packager: WebDatasetPackager,
    *,
    adapter: CaptureAdapter | None = None,
) -> None:
    """Package existing finalized local episode datasets with a writer adapter.

    Experimental. Source directories are read, never modified. The writer is a
    configuration template that must match the native output; the built-in
    COCO adapter checks the split and panoptic setting against the files. The
    manifest records ``provenance: finalized-export``. Frame and episode
    adapters discover frames with ``discover_frames``; clip adapters discover
    clip spans with ``discover_clips``.

    Args:
        writer: Writer configuration matching the existing native datasets.
        episodes: Episode IDs mapped to finalized native directory roots.
        packager: New WebDataset packaging destination.
        adapter: Explicit adapter for a custom writer or a built-in writer
            subclass.

    Raises:
        ConfigurationError: If no adapter applies or the adapter rejects the
            writer. The packager is aborted.
        PackagingError: If an episode root is missing or empty, an identity is
            duplicated, or native output does not match the writer configuration.
            The packager is aborted and no completion manifest is written.
    """
    if not isinstance(packager, WebDatasetPackager):
        raise ConfigurationError(
            "export_finalized_dataset requires a WebDatasetPackager"
        )
    try:
        session = _PackagingSession(writer, packager, adapter, "finalized-export")
        for identity, path in episodes.items():
            episode_id = _identifier(identity)
            if episode_id in session.episodes:
                raise PackagingError(f"Duplicate episode: {episode_id}")
            root = Path(path)
            if not root.is_dir():
                raise PackagingError(f"Native episode root does not exist: {root}")
            with tempfile.TemporaryDirectory(
                prefix="ovannotators-export-"
            ) as generated:
                context = session.context(episode_id, Path(generated))
                if session.adapter.sample_unit == "clip":
                    context.clips = list(session.adapter.discover_clips(root, writer))
                    _check_clips(context.clips)
                    context.frame_ids = [
                        frame for span in context.clips for frame in span.frame_ids
                    ]
                else:
                    context.frame_ids = session.adapter.discover_frames(root, writer)
                session.package(root, writer, context)
        session.finalize()
    except BaseException:
        packager.abort()
        raise
