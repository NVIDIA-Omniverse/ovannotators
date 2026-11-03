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

"""Verify BasicWriter video episodes packaged by BasicVideoEpisodeAdapter.

CPU tests encode real H.264 through the writer's video path, with PyAV's
libx264 standing in for NVENC and the production MP4 muxer, then decode the
packaged bytes. The GPU test uses the real NVENC encoder.
"""

import io
import json
import shutil
import struct
from fractions import Fraction

import numpy as np
import pytest
import webdataset as wds

from ovannotators import (
    AdapterContext,
    BasicVideoEpisodeAdapter,
    BasicWriter,
    BasicWriterConfig,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    export_finalized_dataset,
)
from ovannotators._internal.mp4_mux import mux_h264_annexb
from ovannotators._internal.mp4_probe import probe_mp4_video
from ovannotators.exceptions import ConfigurationError, PackagingError

pytestmark = [pytest.mark.integration]

H, W = 48, 64


class _X264Encoder:
    """CPU H.264 encoder with NvVideoEncoder's interface and MP4 muxing."""

    payloads: list[bytes] = []

    def __init__(self, fps, width, height):
        import av

        self.fps, self.width, self.height = fps, width, height
        self.ctx = av.CodecContext.create("libx264", "w")
        self.ctx.width, self.ctx.height = width, height
        self.ctx.pix_fmt = "yuv420p"
        self.ctx.time_base = Fraction(1, fps)
        self.ctx.framerate = Fraction(fps)
        self.ctx.max_b_frames = 0
        self.ctx.options = {"bf": "0", "preset": "ultrafast", "tune": "zerolatency"}
        self.stream = b""
        self.count = 0

    def write_frame(self, array):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        frame.pts = self.count
        self.count += 1
        self.stream += b"".join(bytes(p) for p in self.ctx.encode(frame))

    def close(self):
        self.stream += b"".join(bytes(p) for p in self.ctx.encode(None))
        payload = mux_h264_annexb(self.stream, self.fps, self.width, self.height)
        _X264Encoder.payloads.append(payload)
        return payload


@pytest.fixture
def x264(monkeypatch):
    av = pytest.importorskip("av")
    if "libx264" not in av.codecs_available:
        pytest.skip("PyAV lacks libx264")
    from ovannotators._internal import codecs

    _X264Encoder.payloads = []
    monkeypatch.setattr(
        codecs,
        "get_video_encoder",
        lambda path, fps, w, h, **kwargs: _X264Encoder(fps, w, h),
    )
    return _X264Encoder


def _frame(level, sample_render_vars, size=(H, W)):
    """A frame whose RGB and depth are uniform, so decoded frames identify it."""
    return {
        "LdrColor": np.full((*size, 3), level, dtype=np.uint8),
        "DistanceToCameraSD": np.full(size, 1.0 + level / 32, dtype=np.float32),
        "CameraParams": sample_render_vars["CameraParams"],
        "SemanticBoundingBox2DTight": sample_render_vars["SemanticBoundingBox2DTight"],
    }


def _packager(tmp_path, name="out", **kwargs):
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / name))
    return WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="video", **kwargs)
    )


def _manifest(tmp_path, name="out"):
    return json.loads((tmp_path / name / "video/manifest.json").read_text())


def _samples(tmp_path, name="out"):
    manifest = _manifest(tmp_path, name)
    shards = [str(tmp_path / name / s["path"]) for s in manifest["shards"]]
    return {s["__key__"]: s for s in wds.WebDataset(shards, shardshuffle=False)}


def _decode(payload):
    import av

    with av.open(io.BytesIO(payload)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def _native(tmp_path, frames, config, name="native"):
    root = tmp_path / name
    backend = DiskBackend(DiskBackendConfig(root_dir=root))
    writer = BasicWriter(config)
    for frame_id, render_vars in frames:
        writer.write_frame(frame_id, render_vars, backend)
    writer.finalize(backend)
    return root


EPISODES = {"a": [(0, 40), (2, 120), (5, 200)], "b": [(0, 90), (1, 160)]}


@pytest.mark.cpu_only
def test_capture_packages_one_episode_sample_with_paired_streams(
    tmp_path, x264, sample_render_vars
):
    writer = BasicWriter(BasicWriterConfig(video=True, video_fps=12))
    with DatasetCapture(
        writer, _packager(tmp_path), adapter=BasicVideoEpisodeAdapter()
    ) as capture:
        for episode_id, frames in EPISODES.items():
            with capture.episode(episode_id) as episode:
                for frame_id, level in frames:
                    episode.write_frame(frame_id, _frame(level, sample_render_vars))

    manifest = _manifest(tmp_path)
    assert manifest["samples"] == 2
    metadata = manifest["metadata"]
    assert metadata["sample_unit"] == "episode"
    assert metadata["sample_schema"] == "basic-video-episode-v1"
    assert metadata["frame_file_patterns"] == ["*.json"]
    assert metadata["writer_config"]["video"] is True

    samples = _samples(tmp_path)
    assert set(samples) == {
        "video/episode-a/sample-episode",
        "video/episode-b/sample-episode",
    }
    # Encoded bytes are packaged unchanged: two streams per episode, in order.
    payloads = iter(x264.payloads)
    for episode_id, frames in EPISODES.items():
        sample = samples[f"video/episode-{episode_id}/sample-episode"]
        record = json.loads(sample["episode.json"])
        frame_ids = [f for f, _ in frames]
        assert record["frame_ids"] == frame_ids
        assert record["video_fps"] == 12
        assert sorted(record["streams"]) == [
            "distance_to_camera_colorized",
            "ldr_color",
        ]
        assert {
            sample["ldr_color.mp4"],
            sample["distance_to_camera_colorized.mp4"],
        } == {
            next(payloads),
            next(payloads),
        }
        for stream in record["streams"].values():
            assert stream["codec"] == "avc1"
            assert (stream["width"], stream["height"]) == (W, H)
            assert stream["frame_count"] == len(frames)
            assert stream["frame_ids"] == frame_ids
            step = stream["timescale"] // 12
            assert stream["pts"] == [i * step for i in range(len(frames))]
            assert len(_decode(sample[stream["component"]])) == len(frames)
        # Video frame i is the frame with index frame_ids[i].
        decoded = _decode(sample["ldr_color.mp4"])
        for image, (_, level) in zip(decoded, frames):
            assert abs(float(image.mean()) - level) < 4
        # Default selection: per-frame JSON only, under directory-qualified names.
        frame_components = sorted(k for k in sample if k[:1].isdigit())
        digits = [f"{f:04d}" for f in frame_ids]
        assert frame_components == sorted(
            f"{d}.{n}"
            for d in digits
            for n in ("bbox_2d_tight.json", "camera_params.json")
        )
        assert json.loads(sample[f"{digits[1]}.camera_params.json"])
        assert record["frames"][1] == {
            "frame_index": frame_ids[1],
            "components": {
                "bbox_2d_tight.json": f"{digits[1]}.bbox_2d_tight.json",
                "camera_params.json": f"{digits[1]}.camera_params.json",
            },
        }
        meta = json.loads(sample["metadata.json"])["metadata"]
        assert meta == {
            "writer": "basic",
            "episode_id": episode_id,
            "frame_count": len(frames),
            "streams": ["distance_to_camera_colorized", "ldr_color"],
        }

    source = json.loads(
        (tmp_path / "out" / manifest["auxiliary"][0]["path"]).read_bytes()
    )
    paths = source["episodes"]["a"]["samples"]["episode"]
    assert paths["ldr_color.mp4"] == "videos/ldr_color.mp4"
    assert paths["0005.camera_params.json"] == "0005/camera_params.json"
    assert "episode.json" not in paths


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("frame_files", "expected"),
    [
        (("*",), {"ldr_color.png", "distance_to_camera.exr", "camera_params.json"}),
        (("ldr_color.png",), {"ldr_color.png"}),
        ((), set()),
    ],
)
def test_finalized_export_selects_frame_files(
    tmp_path, x264, sample_render_vars, frame_files, expected
):
    native = _native(
        tmp_path,
        [(0, _frame(30, sample_render_vars)), (1, _frame(220, sample_render_vars))],
        BasicWriterConfig(video=True, video_vars=["LdrColor"]),
    )
    before = {p: p.read_bytes() for p in native.rglob("*") if p.is_file()}
    adapter = BasicVideoEpisodeAdapter(frame_files=frame_files)
    export_finalized_dataset(
        BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"])),
        {"ep": native},
        _packager(tmp_path),
        adapter=adapter,
    )
    assert {p: p.read_bytes() for p in native.rglob("*") if p.is_file()} == before
    manifest = _manifest(tmp_path)
    assert manifest["metadata"]["provenance"] == "finalized-export"
    (sample,) = _samples(tmp_path).values()
    assert sample["ldr_color.mp4"] == (native / "videos/ldr_color.mp4").read_bytes()
    for frame in ("0000", "0001"):
        selected = {k.split(".", 1)[1] for k in sample if k.startswith(frame + ".")}
        assert expected <= selected
        if frame_files != ("*",):
            assert selected == expected
        for name in selected:
            assert sample[f"{frame}.{name}"] == (native / frame / name).read_bytes()
    record = json.loads(sample["episode.json"])
    assert record["frame_file_patterns"] == list(frame_files)
    assert record["streams"]["ldr_color"]["frame_ids"] == [0, 1]


@pytest.mark.cpu_only
def test_export_with_root_video_dir(tmp_path, x264, sample_render_vars):
    config = BasicWriterConfig(video=True, video_vars=["LdrColor"], video_dir="")
    native = _native(tmp_path, [(3, _frame(10, sample_render_vars))], config)
    export_finalized_dataset(
        BasicWriter(config),
        {"ep": native},
        _packager(tmp_path),
        adapter=BasicVideoEpisodeAdapter(),
    )
    (sample,) = _samples(tmp_path).values()
    assert sample["ldr_color.mp4"] == (native / "ldr_color.mp4").read_bytes()
    assert json.loads(sample["episode.json"])["streams"]["ldr_color"]["frame_ids"] == [
        3
    ]


@pytest.mark.cpu_only
def test_selection_defaults_and_configuration_errors(tmp_path):
    packager = _packager(tmp_path)
    video = BasicWriter(BasicWriterConfig(video=True))
    # The exact-type default stays BasicFrameAdapter, which rejects video mode.
    with pytest.raises(ConfigurationError, match="BasicVideoEpisodeAdapter"):
        DatasetCapture(video, packager)
    with pytest.raises(ConfigurationError, match="video=True"):
        DatasetCapture(BasicWriter(), packager, adapter=BasicVideoEpisodeAdapter())
    with pytest.raises(ConfigurationError, match="requires a BasicWriter"):
        DatasetCapture(object(), packager, adapter=BasicVideoEpisodeAdapter())
    for bad in ("*.json", ["*.json", ""], [3]):
        with pytest.raises(ConfigurationError, match="frame_files"):
            BasicVideoEpisodeAdapter(frame_files=bad)
    for video_dir in ("../videos", "0001", "a/./b"):
        writer = BasicWriter(BasicWriterConfig(video=True, video_dir=video_dir))
        with pytest.raises(ConfigurationError, match="video_dir"):
            DatasetCapture(writer, packager, adapter=BasicVideoEpisodeAdapter())


@pytest.mark.cpu_only
def test_episode_without_video_is_discarded_and_capture_continues(
    tmp_path, x264, sample_render_vars
):
    writer = BasicWriter(BasicWriterConfig(video=True))
    with DatasetCapture(
        writer, _packager(tmp_path), adapter=BasicVideoEpisodeAdapter()
    ) as capture:
        with pytest.raises(PackagingError, match="no video"):
            with capture.episode("none") as episode:
                episode.write_frame(
                    0, {"CameraParams": sample_render_vars["CameraParams"]}
                )
        with pytest.raises(PackagingError, match="no files"):
            with capture.episode("empty") as episode:
                episode.write_frame(0, {})
        with capture.episode("ok") as episode:
            episode.write_frame(0, _frame(50, sample_render_vars))
    assert _manifest(tmp_path)["episodes"] == ["ok"]


@pytest.mark.cpu_only
def test_unpairable_stream_is_rejected(tmp_path, x264, sample_render_vars):
    # The writer drops a frame whose resolution differs from the stream's,
    # so its video no longer has one frame per frame that holds LdrColor.
    writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
    with DatasetCapture(
        writer, _packager(tmp_path), adapter=BasicVideoEpisodeAdapter()
    ) as capture:
        with pytest.raises(PackagingError, match="cannot pair"):
            with capture.episode("a") as episode:
                episode.write_frame(0, _frame(10, sample_render_vars))
                episode.write_frame(1, _frame(10, sample_render_vars, size=(32, 32)))
        with capture.episode("b") as episode:
            episode.write_frame(0, _frame(10, sample_render_vars))


@pytest.mark.cpu_only
def test_long_frame_component_name_fails_at_write(tmp_path, x264, sample_render_vars):
    # "00000000.distance_to_camera_colorized.png" exceeds 40 characters.
    writer = BasicWriter(BasicWriterConfig(video=True, frame_id_digits=8))
    capture = DatasetCapture(
        writer, _packager(tmp_path), adapter=BasicVideoEpisodeAdapter(("*",))
    )
    with pytest.raises(PackagingError, match="narrow frame_files"):
        with capture:
            with capture.episode("a") as episode:
                episode.write_frame(0, _frame(10, sample_render_vars))
    assert not (tmp_path / "out/video/manifest.json").exists()


@pytest.mark.cpu_only
def test_oversized_episode_is_never_split(tmp_path, x264, sample_render_vars):
    frames = [(i, _frame(40 * i, sample_render_vars)) for i in range(3)]
    native = _native(tmp_path, frames, BasicWriterConfig(video=True))
    limits = {"max_shard_bytes": 10240, "allow_episode_spanning": True}
    adapter = BasicVideoEpisodeAdapter(("*",))
    writer = BasicWriter(BasicWriterConfig(video=True))
    with pytest.raises(PackagingError):
        export_finalized_dataset(
            writer,
            {"ep": native},
            _packager(tmp_path, "err", **limits),
            adapter=adapter,
        )
    export_finalized_dataset(
        writer,
        {"ep": native},
        _packager(tmp_path, oversized_episode_policy="dedicated_shard", **limits),
        adapter=adapter,
    )
    manifest = _manifest(tmp_path)
    assert manifest["samples"] == 1
    assert len(manifest["shards"]) == 1 and manifest["shards"][0]["oversized"]


def _corrupt_video(native):
    (native / "videos/ldr_color.mp4").write_bytes(b"\x00\x00\x00\x08free")


def _truncate_video_extended_header(native):
    (native / "videos/ldr_color.mp4").write_bytes(struct.pack(">I4s", 1, b"mdat"))


def _remove_frames(native):
    for name in ("0000", "0001"):
        shutil.rmtree(native / name)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_corrupt_video, "Unreadable video ldr_color.mp4"),
        (_truncate_video_extended_header, "Unreadable video ldr_color.mp4"),
        (lambda native: (native / "extra").mkdir(), "Unexpected BasicWriter directory"),
        (lambda native: (native / "00001").mkdir(), "Ambiguous BasicWriter frame"),
        (lambda native: (native / "0000/sub").mkdir(), "Unexpected entry"),
        (_remove_frames, "No BasicWriter frame directories"),
    ],
)
def test_export_rejects_inconsistent_native_output(
    tmp_path, x264, sample_render_vars, corrupt, message
):
    config = BasicWriterConfig(video=True, video_vars=["LdrColor"])
    frames = [(0, _frame(10, sample_render_vars)), (1, _frame(20, sample_render_vars))]
    native = _native(tmp_path, frames, config)
    corrupt(native)
    with pytest.raises(PackagingError, match=message):
        export_finalized_dataset(
            BasicWriter(config),
            {"ep": native},
            _packager(tmp_path),
            adapter=BasicVideoEpisodeAdapter(),
        )
    assert not (tmp_path / "out/video/manifest.json").exists()


@pytest.mark.cpu_only
def test_collect_rejects_missing_frame_directory(tmp_path, x264, sample_render_vars):
    config = BasicWriterConfig(video=True, video_vars=["LdrColor"])
    native = _native(tmp_path, [(0, _frame(10, sample_render_vars))], config)
    context = AdapterContext("video", "ep", tmp_path, frame_ids=[0, 7])
    with pytest.raises(PackagingError, match="Missing BasicWriter frame directory"):
        BasicVideoEpisodeAdapter().collect_samples(native, BasicWriter(config), context)


def _box(kind, payload):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full(kind, version, payload):
    return _box(kind, struct.pack(">I", version << 24) + payload)


def _moov(*, mdhd_version=0, ctts=None, handler=b"vide", stts=((3, 100),)):
    entry = _box(b"hvc1", bytes(24) + struct.pack(">HH", 32, 16) + bytes(50))
    stbl = [
        _full(b"stsd", 0, struct.pack(">I", 1) + entry),
        _full(
            b"stts",
            0,
            struct.pack(">I", len(stts))
            + b"".join(struct.pack(">II", c, d) for c, d in stts),
        ),
        _full(b"stsz", 0, struct.pack(">II", 0, 3) + bytes(12)),
    ]
    if ctts is not None:
        version, offsets = ctts
        fmt = ">Ii" if version else ">II"
        stbl.append(
            _full(
                b"ctts",
                version,
                struct.pack(">I", len(offsets))
                + b"".join(struct.pack(fmt, 1, o) for o in offsets),
            )
        )
    times = (
        struct.pack(">QQIQ", 0, 0, 1000, 300)
        if mdhd_version
        else struct.pack(">IIII", 0, 0, 1000, 300)
    )
    mdia = (
        _full(b"mdhd", mdhd_version, times + bytes(4))
        + _full(b"hdlr", 0, bytes(4) + handler + bytes(13))
        + _box(b"minf", _box(b"stbl", b"".join(stbl)))
    )
    return _box(b"moov", _box(b"trak", _box(b"mdia", mdia)))


@pytest.mark.unit
@pytest.mark.cpu_only
def test_probe_reads_container_timing_variants(tmp_path):
    path = tmp_path / "v.mp4"
    # 64-bit mdat header before moov, version-1 mdhd, signed ctts reordering.
    mdat = struct.pack(">I4sQ", 1, b"mdat", 16 + 4) + b"data"
    path.write_bytes(
        _box(b"ftyp", b"isom")
        + mdat
        + _moov(mdhd_version=1, ctts=(1, [200, -100, -100]))
    )
    track = probe_mp4_video(path)
    assert (track.codec, track.width, track.height) == ("hvc1", 32, 16)
    assert (track.timescale, track.frame_count, track.duration) == (1000, 3, 300)
    assert track.timestamps == (0, 100, 200)
    path.write_bytes(_moov(ctts=(0, [0, 0, 0]), stts=((1, 100), (2, 50))))
    assert probe_mp4_video(path).timestamps == (0, 100, 150)


@pytest.mark.unit
@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_box(b"ftyp", b"isom"), "no moov"),
        (_box(b"moov", b"")[:6], "no moov"),
        (struct.pack(">I4s", 4, b"free"), "Malformed"),
        (struct.pack(">I4s", 0, b"mdat"), "no moov"),
        (_moov(handler=b"soun"), "no video track"),
        (_moov(stts=((2, 100),)), "stts covers 2"),
        (_moov(ctts=(0, [0])), "ctts"),
        (struct.pack(">I4s", 100, b"moov") + b"x", "Truncated"),
        (struct.pack(">I4s", 1, b"mdat"), "Truncated MP4 box header"),
        (struct.pack(">I4sI", 1, b"mdat", 0), "Truncated MP4 box header"),
        (struct.pack(">I4sQ", 1, b"moov", 8), "Malformed"),
        (struct.pack(">I4sQ", 1, b"mdat", 2**63) + b"x", "Truncated MP4 box"),
        (struct.pack(">I4sQ", 1, b"moov", 2**64 - 1) + b"x", "Truncated MP4 box"),
        (_box(b"moov", _box(b"trak", _box(b"mdia", b""))), "no video track"),
    ],
)
def test_probe_rejects_malformed_files(tmp_path, payload, message):
    path = tmp_path / "bad.mp4"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        probe_mp4_video(path)


def _nvenc_works():
    try:
        from ovannotators._internal.codecs import get_video_encoder

        encoder = get_video_encoder("", 24, 256, 256)
        encoder.write_frame(np.zeros((256, 256, 3), dtype=np.uint8))
        return bool(encoder.close())
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.e2e
def test_gpu_nvenc_episode_packages_and_decodes(tmp_path, sample_render_vars):
    """Encode with NVENC through BasicWriter, package, decode and pair frames."""
    pytest.importorskip("PyNvVideoCodec")
    pytest.importorskip("av")
    if not _nvenc_works():
        pytest.skip("PyNvVideoCodec cannot encode on this machine")
    # NVENC rejects frames narrower than about 160 pixels.
    size = (256, 256)
    frames = [(i * 3, 20 + 25 * i) for i in range(8)]
    writer = BasicWriter(BasicWriterConfig(video=True, video_fps=24))
    with DatasetCapture(
        writer, _packager(tmp_path), adapter=BasicVideoEpisodeAdapter()
    ) as capture:
        with capture.episode("gpu") as episode:
            for frame_id, level in frames:
                episode.write_frame(frame_id, _frame(level, sample_render_vars, size))
    (sample,) = _samples(tmp_path).values()
    record = json.loads(sample["episode.json"])
    frame_ids = [f for f, _ in frames]
    assert sorted(record["streams"]) == ["distance_to_camera_colorized", "ldr_color"]
    for name, stream in record["streams"].items():
        decoded = _decode(sample[f"{name}.mp4"])
        assert stream["codec"] == "avc1"
        assert len(decoded) == stream["frame_count"] == len(frames)
        assert stream["frame_ids"] == frame_ids
        assert len(stream["pts"]) == len(frames)
        assert stream["pts"] == sorted(stream["pts"])
        assert decoded[0].shape[:2] == size
    for image, (_, level) in zip(_decode(sample["ldr_color.mp4"]), frames):
        assert abs(float(image.mean()) - level) < 6
