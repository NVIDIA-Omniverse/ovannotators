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

"""Executable WebDataset capture and existing-file packaging examples."""

import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.cpu_only]


def test_coco_webdataset_capture(tmp_path, sample_render_vars):
    output_dir = tmp_path / "capture"
    episodes = [
        ("episode001", [sample_render_vars, sample_render_vars]),
        ("episode002", [sample_render_vars]),
    ]
    # [snippet:doc-webdataset-capture]
    from ovannotators import DatasetCapture
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import COCOWriter, COCOWriterConfig

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        storage,
        WebDatasetPackagerConfig(
            dataset_id="training-run",
            max_episodes_per_shard=1,
        ),
    )
    writer = COCOWriter(COCOWriterConfig(image_format="jpg"))
    with DatasetCapture(writer=writer, packager=packager) as dataset:
        for episode_id, frames in episodes:
            with dataset.episode(episode_id) as episode:
                for frame_index, render_vars in enumerate(frames):
                    episode.write_frame(frame_index, render_vars)
    # [/snippet:doc-webdataset-capture]
    manifest = json.loads((output_dir / "training-run/manifest.json").read_text())
    assert manifest["samples"] == 3
    assert [s["samples"] for s in manifest["shards"]] == [2, 1]
    with tarfile.open(output_dir / manifest["shards"][0]["path"]) as archive:
        key = "training-run/episode-episode001/sample-0"
        image = Image.open(io.BytesIO(archive.extractfile(key + ".jpg").read()))
        assert image.size == (640, 480)
        record = json.load(archive.extractfile(key + ".json"))
        assert len(record["annotations"]) == 2
        assert record["image"]["file_name"] == key + ".jpg"


def test_package_existing_samples(tmp_path):
    output_dir = tmp_path / "packaged"
    image_path = tmp_path / "image.png"
    annotation_path = tmp_path / "annotations.json"
    Image.new("RGB", (8, 6), "navy").save(image_path)
    annotation_path.write_text('{"label": "object"}')
    # [snippet:doc-webdataset-pack-files]
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        SampleFiles,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    with WebDatasetPackager(
        storage,
        WebDatasetPackagerConfig(dataset_id="converted", max_shard_bytes="512 MiB"),
    ) as packager:
        packager.pack_episode(
            "episode001",
            [
                SampleFiles(
                    key="frame000001",
                    files={"png": image_path, "json": annotation_path},
                    metadata={"frame_index": 1},
                )
            ],
        )
    # [/snippet:doc-webdataset-pack-files]
    with tarfile.open(output_dir / "converted/shard-000000.tar") as archive:
        assert (
            archive.extractfile(
                "converted/episode-episode001/sample-frame000001.png"
            ).read()
            == image_path.read_bytes()
        )
        assert (
            json.load(
                archive.extractfile(
                    "converted/episode-episode001/sample-frame000001.metadata.json"
                )
            )["metadata"]["frame_index"]
            == 1
        )


def test_explicit_custom_adapter(tmp_path):
    from dataclasses import dataclass

    from ovannotators import (
        AdapterContext,
        CaptureAdapter,
        DatasetCapture,
        PackagingError,
        SampleFiles,
    )
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )

    # [snippet:doc-webdataset-custom-adapter]
    @dataclass
    class TextConfig:
        prefix: str = "frame"

    class TextWriter:
        name = "text"

        def __init__(self, config=None):
            self.config = config or TextConfig()

        def write_frame(self, frame_id, render_vars, backend):
            backend.write(f"{frame_id:04d}.txt", render_vars["text"].encode())

        def finalize(self, backend):
            pass

    class TextAdapter(CaptureAdapter):
        sample_schema = "text-frame-v1"

        def validate(self, writer):
            if not isinstance(writer, TextWriter):
                raise TypeError("TextAdapter requires TextWriter")

        def create_writer(self, template):
            return TextWriter(TextConfig(prefix=template.config.prefix))

        def validate_frame_output(self, frame_id, writer, root, context):
            if not (root / f"{frame_id:04d}.txt").is_file():
                raise PackagingError(f"Missing text frame {frame_id}")

        def discover_frames(self, root, writer):
            return sorted(int(path.stem) for path in root.glob("*.txt"))

        def collect_samples(self, root, writer, context: AdapterContext):
            return [
                SampleFiles(frame, {"txt": root / f"{frame:04d}.txt"})
                for frame in context.frame_ids
            ]

    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "custom"))
    packager = WebDatasetPackager(storage, WebDatasetPackagerConfig(dataset_id="text"))
    with DatasetCapture(TextWriter(), packager, adapter=TextAdapter()) as capture:
        with capture.episode("one") as episode:
            episode.write_frame(0, {"text": "hello"})
    # [/snippet:doc-webdataset-custom-adapter]
    with tarfile.open(tmp_path / "custom/text/shard-000000.tar") as archive:
        assert archive.extractfile("text/episode-one/sample-0.txt").read() == b"hello"
    assert (
        json.loads((tmp_path / "custom/text/manifest.json").read_text())["metadata"][
            "sample_schema"
        ]
        == "text-frame-v1"
    )


def test_export_finalized_basic_output(tmp_path):
    from ovannotators import BasicWriter, BasicWriterConfig
    from ovannotators.backends import DiskBackend, DiskBackendConfig

    native = tmp_path / "native"
    writer = BasicWriter(BasicWriterConfig())
    native_backend = DiskBackend(DiskBackendConfig(root_dir=native))
    writer.write_frame(
        0, {"LdrColor": np.zeros((2, 2, 3), dtype=np.uint8)}, native_backend
    )
    writer.finalize(native_backend)

    output_dir = tmp_path / "exported"
    # [snippet:doc-webdataset-export-finalized]
    from ovannotators import BasicWriter, BasicWriterConfig, export_finalized_dataset
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )

    writer = BasicWriter(BasicWriterConfig())
    destination = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        destination, WebDatasetPackagerConfig(dataset_id="existing")
    )
    export_finalized_dataset(writer, {"episode001": native}, packager)
    # [/snippet:doc-webdataset-export-finalized]
    assert (tmp_path / "exported/existing/manifest.json").exists()


def test_read_webdataset_output(tmp_path, sample_render_vars):
    from ovannotators import DatasetCapture
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import COCOWriter, COCOWriterConfig

    output_dir = tmp_path / "capture"
    packager = WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=output_dir)),
        WebDatasetPackagerConfig(dataset_id="training-run", max_episodes_per_shard=1),
    )
    with DatasetCapture(COCOWriter(COCOWriterConfig()), packager) as dataset:
        for episode_id in ("episode001", "episode002"):
            with dataset.episode(episode_id) as episode:
                episode.write_frame(0, sample_render_vars)

    # [snippet:doc-webdataset-read]
    import hashlib
    import json
    import tarfile

    import webdataset as wds

    manifest = json.loads((output_dir / "training-run/manifest.json").read_text())
    shard_paths = []
    for shard in manifest["shards"]:
        path = output_dir / shard["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"Corrupt shard: {shard['path']}")
        shard_paths.append(path)

    # Standard library: group adjacent members by sample key.
    for path in shard_paths:
        with tarfile.open(path) as archive:
            for member in archive:
                key, _, suffix = member.name.partition(".")
                if suffix == "metadata.json":
                    identity = json.load(archive.extractfile(member))
                    print(key, identity["episode_id"], identity["sample_id"])

    # WebDataset reader: one dict per sample, keyed by component suffix. It
    # opens URLs, so pass file URLs (a Windows drive path reads as a scheme).
    shard_urls = [f"file:{path.resolve().as_posix()}" for path in shard_paths]
    for sample in wds.WebDataset(shard_urls, shardshuffle=False).decode("pil"):
        image, record = sample["jpg"], sample["json"]
        print(sample["__key__"], image.size, len(record["annotations"]))
    # [/snippet:doc-webdataset-read]
    samples = list(wds.WebDataset(shard_urls, shardshuffle=False).decode("pil"))
    assert [s["__key__"] for s in samples] == [
        "training-run/episode-episode001/sample-0",
        "training-run/episode-episode002/sample-0",
    ]
    assert all(s["metadata.json"]["dataset_id"] == "training-run" for s in samples)
    assert len(shard_paths) == 2


def test_explicit_clip_adapter(tmp_path):
    output_dir = tmp_path / "clips"
    # [snippet:doc-webdataset-clip-adapter]
    from ovannotators import (
        CaptureAdapter,
        ClipSpan,
        DatasetCapture,
        PackagingError,
        SampleFiles,
    )
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )

    class TextClipWriter:
        name = "text-clip"

        def __init__(self):
            self._lines, self._clips = [], []

        def write_frame(self, frame_id, render_vars, backend):
            self._lines.append((frame_id, render_vars["text"]))

        def cut(self, backend):
            if not self._lines:
                return None
            clip = len(self._clips)
            frames = [frame for frame, _ in self._lines]
            backend.write(
                f"{clip:06d}.txt", "\n".join(text for _, text in self._lines).encode()
            )
            backend.write(f"{clip:06d}.frames", " ".join(map(str, frames)).encode())
            self._clips.append(clip)
            self._lines = []
            return clip

        def finalize(self, backend):
            self.cut(backend)

    class TextClipAdapter(CaptureAdapter):
        sample_schema = "text-clip-v1"
        sample_unit = "clip"

        def create_writer(self, template):
            return TextClipWriter()

        def cut_clip(self, writer, backend, root, context):
            clip = writer.cut(backend)  # a public writer method
            if clip is None:
                raise PackagingError("No frames to cut")
            return clip

        def discover_clips(self, root, writer):
            # Finalized export reads clip identity from the native files.
            return [
                ClipSpan(int(path.stem), tuple(map(int, path.read_text().split())))
                for path in sorted(root.glob("*.frames"))
            ]

        def collect_samples(self, root, writer, context):
            return [
                SampleFiles(
                    span.clip_index,
                    {"txt": root / f"{span.clip_index:06d}.txt"},
                    {"frame_ids": list(span.frame_ids)},
                )
                for span in context.clips
            ]

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(storage, WebDatasetPackagerConfig(dataset_id="clips"))
    with DatasetCapture(
        TextClipWriter(), packager, adapter=TextClipAdapter()
    ) as capture:
        with capture.episode("one") as episode:
            episode.write_frame(0, {"text": "a"})
            episode.write_frame(1, {"text": "b"})
            episode.end_clip()  # closes clip 0 with frames 0 and 1
            episode.write_frame(2, {"text": "c"})
        # Successful episode exit closes the trailing clip 1 with frame 2.
    # [/snippet:doc-webdataset-clip-adapter]
    with tarfile.open(tmp_path / "clips/clips/shard-000000.tar") as archive:
        assert archive.extractfile("clips/episode-one/sample-0.txt").read() == b"a\nb"
        assert archive.extractfile("clips/episode-one/sample-1.txt").read() == b"c"
        meta = json.load(
            archive.extractfile("clips/episode-one/sample-1.metadata.json")
        )
        assert meta["metadata"] == {"frame_ids": [2]}
    manifest = json.loads((tmp_path / "clips/clips/manifest.json").read_text())
    assert manifest["metadata"]["sample_unit"] == "clip"
    assert manifest["samples"] == 2


def test_basic_video_episode_capture(tmp_path, monkeypatch, sample_render_vars):
    av = pytest.importorskip("av")
    if "libx264" not in av.codecs_available:
        pytest.skip("PyAV lacks libx264")
    from ovannotators._internal import codecs
    from tests.test_webdataset_basic_video import _X264Encoder

    # Stand in for NVENC with a CPU H.264 encoder; the writer path is unchanged.
    monkeypatch.setattr(
        codecs,
        "get_video_encoder",
        lambda path, fps, w, h, **kwargs: _X264Encoder(fps, w, h),
    )
    output_dir = tmp_path / "video"
    frames = [
        {
            "LdrColor": np.full((48, 64, 3), level, dtype=np.uint8),
            "CameraParams": sample_render_vars["CameraParams"],
        }
        for level in (40, 160)
    ]

    # [snippet:doc-webdataset-basic-video]
    import json

    import webdataset as wds

    from ovannotators import BasicVideoEpisodeAdapter, DatasetCapture
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import BasicWriter, BasicWriterConfig

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="video-run")
    )
    writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
    # Video mode is opt-in: the default Basic adapter packages frames only.
    adapter = BasicVideoEpisodeAdapter(frame_files=("camera_params.json",))
    with DatasetCapture(writer, packager, adapter=adapter) as dataset:
        with dataset.episode("episode001") as episode:
            for frame_index, render_vars in enumerate(frames):
                episode.write_frame(frame_index, render_vars)

    manifest = json.loads((output_dir / "video-run/manifest.json").read_text())
    shards = [
        f"file:{(output_dir / shard['path']).resolve().as_posix()}"
        for shard in manifest["shards"]
    ]
    for sample in wds.WebDataset(shards, shardshuffle=False):  # one per episode
        record = json.loads(sample["episode.json"])
        stream = record["streams"]["ldr_color"]
        mp4_bytes = sample[stream["component"]]  # encoded bytes, not transcoded
        seconds = [pts / stream["timescale"] for pts in stream["pts"]]
        frame_ids = stream["frame_ids"]  # episode frame index of each video frame
        first = record["frames"][0]["components"]["camera_params.json"]
        camera = json.loads(sample[first])  # original per-frame camera JSON
        print(sample["__key__"], len(mp4_bytes), seconds, frame_ids, sorted(camera))
    # [/snippet:doc-webdataset-basic-video]
    assert manifest["metadata"]["sample_unit"] == "episode"
    assert mp4_bytes[4:8] == b"ftyp"
    assert seconds == [0.0, 1 / 24]
    assert frame_ids == [0, 1]
    assert first == "0000.camera_params.json" and camera


def test_kitti_webdataset_export_and_read(tmp_path, sample_render_vars):
    from ovannotators.backends import DiskBackend, DiskBackendConfig
    from ovannotators.writers import KITTIWriter, KITTIWriterConfig

    kitti_dir = tmp_path / "kitti"
    output_dir = tmp_path / "packaged"
    native = DiskBackend(DiskBackendConfig(root_dir=kitti_dir))
    native_writer = KITTIWriter(KITTIWriterConfig(write_depth=True))
    native_writer.write_frame(0, sample_render_vars, native)
    native_writer.finalize(native)

    # [snippet:doc-webdataset-kitti]
    import json

    import webdataset as wds

    from ovannotators import export_finalized_dataset
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import KITTIWriter, KITTIWriterConfig

    # The writer configuration must match the finalized KITTI directory.
    writer = KITTIWriter(KITTIWriterConfig(write_depth=True))
    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="kitti-run")
    )
    export_finalized_dataset(writer, {"drive001": kitti_dir}, packager)

    manifest = json.loads((output_dir / "kitti-run/manifest.json").read_text())
    # The reader opens URLs, so pass file URLs (a Windows drive letter is a scheme).
    shards = [
        f"file:{(output_dir / shard['path']).resolve().as_posix()}"
        for shard in manifest["shards"]
    ]
    for sample in wds.WebDataset(shards, shardshuffle=False):
        image_png = sample["image_2.png"]  # original image_2 bytes
        label_rows = sample["label_2.txt"].decode().splitlines()  # native rows
        calib_text = sample["calib.txt"].decode()  # P0-P3, R0_rect, Tr_* rows
        depth_png = sample["depth.png"]  # 16-bit PNG: clipped metres * depth_scale
        print(sample["__key__"], len(image_png), len(depth_png), len(label_rows))
        print(calib_text.splitlines()[0])
    # [/snippet:doc-webdataset-kitti]
    assert manifest["metadata"]["sample_schema"] == "kitti-frame-v1"
    assert image_png == (kitti_dir / "image_2/000000.png").read_bytes()
    assert label_rows[0].split()[0] == "car" and len(label_rows[0].split()) == 15
    assert calib_text == (kitti_dir / "calib/000000.txt").read_text()
    assert depth_png == (kitti_dir / "depth/000000.png").read_bytes()


def test_yolo_webdataset_split_filter(tmp_path, sample_render_vars):
    output_dir = tmp_path / "yolo"
    boxes = sample_render_vars["SemanticBoundingBox2DTight"]
    negative = {**sample_render_vars, "SemanticBoundingBox2DTight": boxes[:0]}
    episodes = [
        ("episode001", [sample_render_vars, negative, sample_render_vars]),
        ("episode002", [sample_render_vars]),
    ]
    # [snippet:doc-webdataset-yolo-split-filter]
    import json

    import webdataset as wds

    from ovannotators import DatasetCapture
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import YOLOWriter, YOLOWriterConfig

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="yolo-run")
    )
    writer = YOLOWriter(YOLOWriterConfig(task="detect", val_split=0.5))
    with DatasetCapture(writer=writer, packager=packager) as dataset:
        for episode_id, frames in episodes:
            with dataset.episode(episode_id) as episode:
                for frame_index, render_vars in enumerate(frames):
                    episode.write_frame(frame_index, render_vars)

    manifest = json.loads((output_dir / "yolo-run/manifest.json").read_text())
    split_counts = manifest["metadata"]["split_counts"]  # samples per split
    print(split_counts)
    shards = [
        f"file:{(output_dir / shard['path']).resolve().as_posix()}"
        for shard in manifest["shards"]
    ]
    # Episodes keep frames of both splits together; select a split when reading.
    train = (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .select(lambda sample: sample["json"]["split"] == "train")
    )
    for sample in train:
        image, labels = sample["jpg"], sample["txt"]  # dataset-wide class IDs
        print(sample["__key__"], image.size, labels)
    # [/snippet:doc-webdataset-yolo-split-filter]
    assert manifest["metadata"]["sample_schema"] == "yolo-frame-v1"
    assert split_counts == {"train": 1, "val": 3}
    assert manifest["metadata"]["classes"]
    samples = list(train)
    assert [s["__key__"] for s in samples] == ["yolo-run/episode-episode001/sample-1"]
    assert image.size == (640, 480)
    assert labels == ""
