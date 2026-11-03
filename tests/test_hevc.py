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

"""HEVC container, writer configuration and real lossless RGB round trips."""

import json
import shutil
import struct
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import warp as wp

from ovannotators._internal.codecs import NvVideoEncoder, get_video_encoder
from ovannotators._internal.mp4_mux import mux_hevc_annexb
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.writers import (
    BasicWriter,
    BasicWriterConfig,
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
    LeRobotWriter,
    LeRobotWriterConfig,
)

# Parameter sets captured from NVENC HEVC 8-bit 4:4:4, 256x256, lossless P5.
_VPS = bytes.fromhex("40010c01ffff0408000003009e0800000300003c970240")
_SPS = bytes.fromhex(
    "42 01 01 04 08 00 00 03 00 9e 08 00 00 03 00 00 3c 90 01 01 00 80 "
    "b2 cb a5 21 08 c8 bf f1 80 80 80 00 00 03 00 80 00 00 0f 30 02 ef "
    "28 80 01 24 f8 00 01 6e 36 08"
)
_PPS = bytes.fromhex("4401c0937c4cc9")
_IDR = b"\x26\x01\x80\x12"
_P = b"\x02\x01\x80\x34"


def _annexb(*nals):
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


def _box_payload(data, kind):
    pos = data.index(kind)
    size = struct.unpack_from(">I", data, pos - 4)[0]
    return data[pos + 4 : pos - 4 + size]


@pytest.mark.parametrize(
    "config", [BasicWriterConfig, CosmosTransferWriterConfig, LeRobotWriterConfig]
)
@pytest.mark.parametrize(
    "options",
    [
        {"video_codec": "av1"},
        {"video_lossless": True},
        {"video_preset": "P8"},
        {"video_lossless": "yes"},
    ],
)
def test_invalid_writer_video_options(config, options):
    with pytest.raises(ValueError, match="video_"):
        config(**options)


@pytest.mark.parametrize(
    "codec,lossless,preset,format",
    [
        ("h264", False, None, "ARGB"),
        ("hevc", False, "P4", "ARGB"),
        ("hevc", True, None, "YUV444"),
    ],
)
def test_encoder_configuration_and_frame_packing(codec, lossless, preset, format):
    module = MagicMock()
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        encoder = get_video_encoder(
            "", 24, 256, 256, codec=codec, lossless=lossless, preset=preset
        )
        rgb = np.random.default_rng(8).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        encoder.write_frame(rgb)
    args, kwargs = module.CreateEncoder.call_args
    assert args == (256, 256, format, True)
    assert kwargs["codec"] == codec and kwargs["bf"] == 0
    packed = module.CreateEncoder.return_value.Encode.call_args.args[0]
    if lossless:
        assert kwargs["tuning_info"] == "lossless"
        assert kwargs["preset"] == "P5"
        np.testing.assert_array_equal(packed.transpose(1, 2, 0)[:, :, [2, 0, 1]], rgb)
    else:
        np.testing.assert_array_equal(packed[:, :, :3], rgb[:, :, ::-1])
        assert "tuning_info" not in kwargs
        if preset is None:
            assert "preset" not in kwargs


@pytest.mark.parametrize(
    "fps,w,h", [(0, 256, 256), (24, 0, 256), (24, 256, -1), (24, 257, 256)]
)
def test_lossless_rejects_invalid_geometry_before_loading_gpu(fps, w, h):
    with pytest.raises(ValueError):
        NvVideoEncoder(codec="hevc", lossless=True).open("", fps, w, h)


@pytest.mark.parametrize(
    "dtype,shape",
    [(np.float32, (256, 256, 3)), (np.uint8, (256, 256, 4)), (np.uint8, (128, 256, 3))],
)
def test_lossless_rejects_non_rgb_frames(dtype, shape):
    with patch.dict("sys.modules", {"PyNvVideoCodec": MagicMock()}):
        enc = get_video_encoder("", 24, 256, 256, codec="hevc", lossless=True)
        with pytest.raises(ValueError, match="uint8 RGB"):
            enc.write_frame(np.zeros(shape, dtype=dtype))


def test_hevc_mux_samples_parameter_sets_and_timing():
    # Two slices in the first picture; a suffix SEI stays with that picture.
    extra_slice = b"\x26\x01\x00\x56"
    suffix = b"\x50\x01\x80"
    stream = _annexb(_VPS, _SPS, _PPS, _IDR, extra_slice, suffix, _P)
    mp4 = mux_hevc_annexb(stream, 24, 256, 256, rgb=True)
    hvcc = _box_payload(mp4, b"hvcC")
    assert hvcc[0] == 1
    assert hvcc[16] & 3 == 3  # 4:4:4
    assert hvcc[17] & 7 == 0  # 8-bit
    assert hvcc[22] == 3  # VPS/SPS/PPS arrays
    assert b"hvc1" in mp4
    mdat = _box_payload(mp4, b"mdat")
    expected = [_IDR, extra_slice, suffix, _P]
    assert mdat == b"".join(struct.pack(">I", len(n)) + n for n in expected)
    sizes = _box_payload(mp4, b"stsz")
    assert struct.unpack_from(">I", sizes, 8)[0] == 2
    assert struct.unpack_from(">II", _box_payload(mp4, b"stts"), 8) == (2, 1000)
    assert struct.unpack_from(">II", _box_payload(mp4, b"stss"), 4) == (1, 1)
    offset = struct.unpack_from(">I", _box_payload(mp4, b"stco"), 8)[0]
    assert mp4[offset:] == mdat
    assert _VPS not in mdat and _PPS not in mdat


@pytest.mark.parametrize(
    "nals,match",
    [
        ([], "missing"),
        ([_VPS, _SPS, _IDR], "missing"),
        ([_VPS, _SPS, _PPS], "No coded"),
        ([_VPS, _SPS, _PPS, b"\x26"], "header"),
        ([_VPS, _SPS, _PPS, b"\x26\x01\x00"], "first slice"),
        ([_VPS, _SPS, _PPS, _IDR, _PPS + b"x"], "Changing"),
        ([_VPS, _SPS[:5], _PPS, _IDR], "Truncated"),
    ],
)
def test_hevc_mux_rejects_malformed_streams(nals, match):
    with pytest.raises(ValueError, match=match):
        mux_hevc_annexb(_annexb(*nals), 24, 256, 256)


@pytest.mark.parametrize("packet_style", ["bytes", "mappings"])
def test_hevc_encoder_muxes_both_packet_apis(packet_style):
    module = MagicMock()
    payload = _annexb(_VPS, _SPS, _PPS, _IDR)
    module.CreateEncoder.return_value.Encode.return_value = (
        payload if packet_style == "bytes" else [{"data": payload}]
    )
    module.CreateEncoder.return_value.EndEncode.return_value = b""
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        enc = get_video_encoder("", 24, 256, 256, codec="hevc", lossless=True)
        enc.write_frame(np.zeros((256, 256, 3), np.uint8))
        result = enc.close()
    assert b"hvcC" in result and b"avcC" not in result


def test_hevc_flush_errors_are_not_silenced():
    module = MagicMock()
    module.CreateEncoder.return_value.EndEncode.side_effect = RuntimeError(
        "flush failed"
    )
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        enc = get_video_encoder("", 24, 256, 256, codec="hevc")
        enc.write_frame(np.zeros((256, 256, 3), np.uint8))
        with pytest.raises(RuntimeError, match="flush failed"):
            enc.close()


@pytest.mark.parametrize(
    "writer_type,config_type",
    [
        (BasicWriter, BasicWriterConfig),
        (CosmosTransferWriter, CosmosTransferWriterConfig),
        (LeRobotWriter, LeRobotWriterConfig),
    ],
)
def test_writers_forward_video_options_and_metadata(tmp_path, writer_type, config_type):
    options = dict(video_codec="hevc", video_lossless=True, video_preset="P6")
    if writer_type is BasicWriter:
        options.update(video=True, colorize=False)
    elif writer_type is CosmosTransferWriter:
        options.update(conditioning_types=[])
    else:
        pytest.importorskip("pyarrow")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    rgb = np.random.default_rng(3).integers(0, 256, (256, 256, 3), np.uint8)
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.close.return_value = b"mp4"
        writer = writer_type(config_type(**options))
        render_vars = {"LdrColor": rgb}
        if writer_type is CosmosTransferWriter:
            render_vars.update(
                DistanceToCameraSD=np.ones((256, 256), np.float32),
                SemanticInstanceSegmentation=np.ones((256, 256), np.uint32),
            )
        writer.write_frame(0, render_vars, backend)
        writer.finalize(backend)
        kwargs = factory.call_args.kwargs
        assert (kwargs["codec"], kwargs["lossless"], kwargs["preset"]) == (
            "hevc",
            True,
            "P6",
        )
        np.testing.assert_array_equal(
            factory.return_value.write_frame.call_args.args[0], rgb
        )
    if writer_type is LeRobotWriter:
        info = json.loads(backend.read("meta/info.json"))
        video = info["features"]["observation.images.top"]["video_info"]
        assert video["video.codec"] == "hevc" and video["video.pix_fmt"] == "gbrp"
    elif writer_type is CosmosTransferWriter:
        info = json.loads(backend.read("manifest.json"))
        assert info["clips"][0]["video_encoding"] == {
            "codec": "hevc",
            "lossless": True,
            "pixel_format": "gbrp",
        }


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    "codec,lossless", [("h264", False), ("hevc", False), ("hevc", True)]
)
@pytest.mark.parametrize("residency", ["host", "device", "mixed"])
def test_real_video_roundtrip(tmp_path, codec, lossless, residency):
    pytest.importorskip("PyNvVideoCodec")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for independent video verification")
    rng = np.random.default_rng(17)
    frames = rng.integers(0, 256, (4, 256, 256, 3), np.uint8)
    frames[0, :32] = [255, 0, 0]
    frames[0, 32:64] = [0, 255, 0]
    frames[0, 64:96] = [0, 0, 255]
    enc = get_video_encoder("", 24, 256, 256, codec=codec, lossless=lossless)
    for i, frame in enumerate(frames):
        # A mixed session starts on the device so later host frames upload; a
        # host-started session downloads CUDA frames by design.
        on_device = residency == "device" or (residency == "mixed" and i % 2 == 0)
        source = wp.array(frame, device="cuda:0") if on_device else frame
        with patch.object(
            wp.array, "numpy", side_effect=AssertionError("raw host copy")
        ):
            enc.write_frame(source)
    path = tmp_path / "video.mp4"
    path.write_bytes(enc.close())
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert stream["codec_name"] == codec
    assert stream["nb_frames"] == "4" and stream["avg_frame_rate"] == "24/1"
    assert stream["width"] == stream["height"] == 256
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    decoded = np.frombuffer(result.stdout, np.uint8).reshape(frames.shape)
    if lossless:
        assert stream["pix_fmt"] == "gbrp" and stream["color_range"] == "pc"
        np.testing.assert_array_equal(decoded, frames)
        # NVDEC's native output holds the same G/B/R planes, without CSC.
        import PyNvVideoCodec as nvc

        decoder = nvc.SimpleDecoder(str(path), use_device_memory=False)
        native = decoder.get_batch_frames(len(frames))
        assert len(native) == len(frames)
        for actual, expected in zip(native, frames, strict=True):
            planes = np.from_dlpack(actual).reshape(3, 256, 256)
            np.testing.assert_array_equal(
                planes.transpose(1, 2, 0)[:, :, [2, 0, 1]], expected
            )
    else:
        assert stream["pix_fmt"] == "yuv420p"
        # Solid primaries catch channel-order errors without expecting lossy equality.
        np.testing.assert_allclose(
            decoded[0, 8:24].mean(axis=(0, 1)), [255, 0, 0], atol=5
        )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.parametrize(
    "writer_type,config_type,preset",
    [
        (BasicWriter, BasicWriterConfig, "P1"),
        (CosmosTransferWriter, CosmosTransferWriterConfig, "P5"),
        (LeRobotWriter, LeRobotWriterConfig, "P7"),
    ],
)
def test_lossless_writer_dataset_roundtrip(tmp_path, writer_type, config_type, preset):
    pytest.importorskip("PyNvVideoCodec")
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required for independent dataset video verification")
    options = dict(video_codec="hevc", video_lossless=True, video_preset=preset)
    if writer_type is BasicWriter:
        options.update(video=True, render_vars=["LdrColor"])
    elif writer_type is CosmosTransferWriter:
        options.update(conditioning_types=["depth", "segmentation"])
    else:
        pytest.importorskip("pyarrow")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    writer = writer_type(config_type(**options))
    # RGBA input exercises writer normalization and explicit alpha discard.
    frames = np.random.default_rng(9).integers(0, 256, (3, 256, 320, 4), np.uint8)
    for index, frame in enumerate(frames):
        writer.write_frame(
            index,
            {
                "LdrColor": frame,
                "DistanceToCameraSD": np.ones((256, 320), np.float32),
                "SemanticInstanceSegmentation": np.ones((256, 320), np.uint32),
            },
            backend,
        )
    expected_videos = [frames[:, :, :, :3]]
    if writer_type is CosmosTransferWriter:
        expected_videos = [
            np.stack(writer._colorize_depth_frames()),
            np.stack(writer._colorize_seg_frames()),
            frames[:, :, :, :3],
        ]
    writer.finalize(backend)
    video_paths = sorted(tmp_path.rglob("*.mp4"))
    assert len(video_paths) == len(expected_videos)
    for video_path, expected in zip(video_paths, expected_videos, strict=True):
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        decoded = np.frombuffer(result.stdout, np.uint8).reshape(3, 256, 320, 3)
        np.testing.assert_array_equal(decoded, expected)
    before = video_paths[0].read_bytes()
    writer.finalize(backend)
    assert video_paths[0].read_bytes() == before


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.parametrize(
    "writer_type,config_type",
    [
        (BasicWriter, BasicWriterConfig),
        (CosmosTransferWriter, CosmosTransferWriterConfig),
        (LeRobotWriter, LeRobotWriterConfig),
    ],
)
def test_lossless_device_writer_roundtrip(tmp_path, writer_type, config_type):
    """All writers preserve borrowed CUDA pixels without raw host copies."""
    pytest.importorskip("PyNvVideoCodec")
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required for independent verification")
    if writer_type is LeRobotWriter:
        pytest.importorskip("pyarrow")
    options = dict(video_codec="hevc", video_lossless=True)
    if writer_type is BasicWriter:
        options.update(video=True, render_vars=["LdrColor"])
    elif writer_type is CosmosTransferWriter:
        options.update(conditioning_types=[])
    writer = writer_type(config_type(**options))
    # Isolate video output from optional still-image encoder fallback.
    backend = MagicMock()
    frames = np.random.default_rng(42).integers(0, 256, (3, 192, 256, 4), np.uint8)
    borrowed = wp.empty(frames.shape[1:], dtype=wp.uint8, device="cuda:0")
    with patch.object(wp.array, "numpy", side_effect=AssertionError("raw host copy")):
        with patch.object(CosmosTransferWriter, "_encode_jpeg", return_value=b"jpeg"):
            for i, frame in enumerate(frames):
                wp.copy(borrowed, wp.array(frame, device="cuda:0"))
                writer.write_frame(i, {"LdrColor": borrowed}, backend)
            borrowed.zero_()
            writer.finalize(backend)
    videos = [
        call.args[1]
        for call in backend.write.call_args_list
        if call.args[0].endswith(".mp4")
    ]
    assert len(videos) == 1
    path = tmp_path / "device.mp4"
    path.write_bytes(videos[0])
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    decoded = np.frombuffer(result.stdout, np.uint8).reshape(3, 192, 256, 3)
    np.testing.assert_array_equal(decoded, frames[..., :3])


def test_basic_lossless_never_crops_or_skips_frames():
    writer = BasicWriter(
        BasicWriterConfig(video=True, video_codec="hevc", video_lossless=True)
    )
    with patch.dict("sys.modules", {"PyNvVideoCodec": MagicMock()}):
        with pytest.raises(ValueError, match="positive frame dimensions"):
            writer._append_video_frame("rgb", np.zeros((0, 256, 3), np.uint8))
        with pytest.raises(ValueError, match="even dimensions"):
            writer._append_video_frame("rgb", np.zeros((256, 257, 3), np.uint8))
        writer._append_video_frame("rgb", np.zeros((256, 256, 3), np.uint8))
        with pytest.raises(ValueError, match="fixed frame resolution"):
            writer._append_video_frame("rgb", np.zeros((258, 256, 3), np.uint8))


def test_lossless_encoder_rejects_missing_output():
    module = MagicMock()
    module.CreateEncoder.return_value.Encode.return_value = b""
    module.CreateEncoder.return_value.EndEncode.return_value = b""
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        enc = get_video_encoder("", 24, 256, 256, codec="hevc", lossless=True)
        enc.write_frame(np.zeros((256, 256, 3), np.uint8))
        with pytest.raises(RuntimeError, match="no data"):
            enc.close()


def test_unsupported_encoder_fails_without_fallback():
    module = MagicMock()
    module.CreateEncoder.side_effect = ValueError("unsupported device")
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        with pytest.raises(
            RuntimeError, match="hevc.*lossless=True.*unsupported device"
        ) as error:
            enc = get_video_encoder("", 24, 256, 256, codec="hevc", lossless=True)
            enc.write_frame(np.zeros((256, 256, 3), np.uint8))
    assert isinstance(error.value.__cause__, ValueError)
    module.CreateEncoder.assert_called_once()


@pytest.mark.parametrize(
    "writer_type,config_type",
    [
        (CosmosTransferWriter, CosmosTransferWriterConfig),
        (LeRobotWriter, LeRobotWriterConfig),
    ],
)
def test_failed_video_finalize_can_be_retried(tmp_path, writer_type, config_type):
    if writer_type is LeRobotWriter:
        pytest.importorskip("pyarrow")
    options = dict(video_codec="hevc", video_lossless=True)
    if writer_type is CosmosTransferWriter:
        options["conditioning_types"] = []
    writer = writer_type(config_type(**options))
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
    writer.write_frame(
        0,
        {
            "LdrColor": np.zeros((256, 256, 3), np.uint8),
            "DistanceToCameraSD": np.ones((256, 256), np.float32),
            "SemanticInstanceSegmentation": np.ones((256, 256), np.uint32),
        },
        backend,
    )
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.side_effect = RuntimeError("GPU unavailable")
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            writer.finalize(backend)
        assert not writer._finalized
        factory.side_effect = None
        factory.return_value.close.return_value = b"mp4"
        writer.finalize(backend)
    assert writer._finalized
    assert len(list(tmp_path.rglob("*.mp4"))) == 1


def test_cosmos_lossless_rejects_rgb_resizing():
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(video_codec="hevc", video_lossless=True)
    )
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        with pytest.raises(ValueError, match="fixed frame resolution"):
            writer._encode_video([np.zeros((256, 258, 3), np.uint8)], 256, 256)
    factory.return_value.close.assert_called_once()


def test_cosmos_encode_failure_closes_encoder_and_keeps_original_error():
    writer = CosmosTransferWriter(CosmosTransferWriterConfig())
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.write_frame.side_effect = RuntimeError("NVENC failed")
        factory.return_value.close.side_effect = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="NVENC failed"):
            writer._encode_video([np.zeros((256, 256, 3), np.uint8)], 256, 256)
    factory.return_value.close.assert_called_once()


def test_lerobot_encode_failure_closes_encoder_and_keeps_original_error():
    from ovannotators.writers.lerobot import _encode_video

    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.write_frame.side_effect = ValueError("bad frame")
        factory.return_value.close.side_effect = RuntimeError("close failed")
        with pytest.raises(ValueError, match="bad frame"):
            _encode_video([np.zeros((256, 256, 3), np.uint8)], 24)
    factory.return_value.close.assert_called_once()


@pytest.mark.parametrize("shape", [(0, 256, 3), (192, 0, 3), (191, 256, 3)])
def test_cosmos_lossless_rejects_invalid_first_frame_before_buffering(shape):
    """Invalid first-frame dimensions cannot reach finalization's thumbnail."""
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(
            video_codec="hevc", video_lossless=True, conditioning_types=[]
        )
    )
    backend = MagicMock()
    with pytest.raises(ValueError, match="positive, even dimensions"):
        writer.write_frame(0, {"LdrColor": np.zeros(shape, np.uint8)}, backend)
    assert writer._resolution is None
    assert not writer._rgb_frames
    writer.finalize(backend)
    backend.write.assert_not_called()


@pytest.mark.parametrize("shape", [(0, 256, 3), (192, 0, 3), (191, 256, 3)])
def test_lerobot_lossless_rejects_invalid_first_frame_before_buffering(shape):
    """Invalid first-frame dimensions fail at write time, not after the episode."""
    writer = LeRobotWriter(LeRobotWriterConfig(video_codec="hevc", video_lossless=True))
    backend = MagicMock()
    with pytest.raises(ValueError, match="positive, even dimensions"):
        writer.write_frame(0, {"LdrColor": np.zeros(shape, np.uint8)}, backend)
    assert writer._image_shape is None
    assert not writer._frames
    assert not writer._rows
    writer.finalize(backend)
    backend.write.assert_not_called()


@pytest.mark.parametrize("empty_shape", [(0, 256, 3), (192, 0, 3)])
@pytest.mark.parametrize("empty_first", [False, True])
def test_basic_public_lossless_rejects_empty_frames(empty_shape, empty_first):
    """Empty selected RGB frames raise before serialization or silent skipping."""
    writer = BasicWriter(
        BasicWriterConfig(
            video=True,
            video_codec="hevc",
            video_lossless=True,
            render_vars=["LdrColor"],
        )
    )
    backend = MagicMock()
    valid = np.zeros((192, 256, 3), np.uint8)
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.close.return_value = b"mp4"
        if not empty_first:
            writer.write_frame(0, {"LdrColor": valid}, backend)
        encoded_before = factory.return_value.write_frame.call_count
        written_before = backend.write_array.call_count
        with pytest.raises(ValueError, match="positive frame dimensions"):
            writer.write_frame(
                1, {"LdrColor": np.zeros(empty_shape, np.uint8)}, backend
            )
        assert factory.return_value.write_frame.call_count == encoded_before
        assert backend.write_array.call_count == written_before
        writer.write_frame(2, {"LdrColor": valid}, backend)
        writer.finalize(backend)
        assert factory.return_value.write_frame.call_count == encoded_before + 1
        backend.write.assert_called_once_with("videos/ldr_color.mp4", b"mp4")


@pytest.mark.integration
@pytest.mark.parametrize("changed_shape", [(192, 320, 3), (256, 256, 3)])
def test_lerobot_lossless_rejects_resolution_changes_across_chunks(
    tmp_path, changed_shape
):
    """A rejected frame cannot create chunks inconsistent with dataset metadata."""
    pytest.importorskip("pyarrow")
    writer = LeRobotWriter(
        LeRobotWriterConfig(video_codec="hevc", video_lossless=True, frames_per_chunk=1)
    )
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    valid = np.zeros((192, 256, 3), np.uint8)
    writer.write_frame(0, {"LdrColor": valid}, backend)
    with pytest.raises(ValueError, match="fixed frame resolution"):
        writer.write_frame(1, {"LdrColor": np.zeros(changed_shape, np.uint8)}, backend)
    writer.write_frame(1, {"LdrColor": valid}, backend)
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.close.return_value = b"mp4"
        writer.finalize(backend)
        assert factory.call_count == 2
        for call in factory.call_args_list:
            assert call.args[2:] == (256, 192)
    info = json.loads(backend.read("meta/info.json"))
    assert info["total_frames"] == 2
    assert info["features"]["observation.images.top"]["shape"] == [192, 256, 3]
    import pyarrow.parquet as pq

    for chunk in range(2):
        rows = pq.read_table(
            tmp_path / f"data/chunk-{chunk:03d}/episode_000000.parquet"
        )
        assert rows["frame_index"].to_pylist() == [chunk]
    assert len(list(tmp_path.rglob("*.mp4"))) == 2


@pytest.mark.integration
@pytest.mark.parametrize("changed_shape", [(192, 320, 3), (256, 256, 3)])
def test_cosmos_lossless_rejects_resolution_change_before_buffering(
    tmp_path, changed_shape
):
    """Invalid RGB cannot poison deferred encoding or conditioning buffers."""
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(
            video_codec="hevc",
            video_lossless=True,
            conditioning_types=["depth", "segmentation"],
        )
    )
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    render_vars = {
        "LdrColor": np.zeros((192, 256, 4), np.uint8),
        "DistanceToCameraSD": np.ones((192, 256), np.float32),
        "SemanticInstanceSegmentation": np.ones((192, 256), np.uint32),
    }
    writer.write_frame(0, render_vars, backend)
    first_rgb = writer._rgb_frames[0]
    with pytest.raises(ValueError, match="fixed frame resolution"):
        writer.write_frame(
            1,
            {**render_vars, "LdrColor": np.zeros(changed_shape, np.uint8)},
            backend,
        )
    assert writer._resolution == (256, 192)
    assert writer._rgb_frames == [first_rgb]
    assert len(writer._depth_frames) == len(writer._seg_frames) == 1
    assert not writer._clips
    assert not backend.list()
    # Channel normalization does not constitute a resolution change.
    writer.write_frame(
        1, {**render_vars, "LdrColor": np.zeros((192, 256, 3), np.uint8)}, backend
    )
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.close.return_value = b"mp4"
        writer.finalize(backend)
        assert factory.call_count == 3
        assert factory.return_value.write_frame.call_count == 6
    manifest = json.loads(backend.read("manifest.json"))
    assert manifest["clips"][0]["num_frames"] == 2
    assert manifest["clips"][0]["resolution"] == [256, 192]
    assert len(list(tmp_path.rglob("*.mp4"))) == 3
    assert not writer._rgb_frames


@pytest.mark.parametrize(
    "fps,expected",
    [(24, (24, 1)), (23.976, (24000, 1001)), (29.97, (30000, 1001)), (12.5, (25, 2))],
)
def test_fractional_frame_rates_are_exact(fps, expected):
    from ovannotators._internal.video_config import video_frame_rate

    rate = video_frame_rate(fps)
    assert (rate.numerator, rate.denominator) == expected


@pytest.mark.parametrize(
    "fps,timing", [(24, (24000, 1000)), (29.97, (30000, 1001)), (12.5, (25, 2))]
)
def test_mux_timing_uses_rational_frame_rate(fps, timing):
    mp4 = mux_hevc_annexb(_annexb(_VPS, _SPS, _PPS, _IDR), fps, 256, 256)
    timescale = struct.unpack_from(">I", _box_payload(mp4, b"mdhd"), 12)[0]
    delta = struct.unpack_from(">II", _box_payload(mp4, b"stts"), 8)[1]
    assert (timescale, delta) == timing


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "24"])
@pytest.mark.parametrize(
    "config_type,field",
    [
        (BasicWriterConfig, "video_fps"),
        (CosmosTransferWriterConfig, "fps"),
        (LeRobotWriterConfig, "fps"),
    ],
)
def test_writer_configs_reject_invalid_frame_rates(config_type, field, fps):
    options = {field: fps}
    if config_type is BasicWriterConfig:
        options["video"] = True
    with pytest.raises(ValueError, match=field):
        config_type(**options)


def test_encoder_accepts_fractional_frame_rate():
    module = MagicMock()
    module.CreateEncoder.return_value.Encode.return_value = _annexb(
        _VPS, _SPS, _PPS, _IDR
    )
    module.CreateEncoder.return_value.EndEncode.return_value = b""
    with patch.dict("sys.modules", {"PyNvVideoCodec": module}):
        enc = get_video_encoder("", 29.97, 256, 256, codec="hevc")
        enc.write_frame(np.zeros((256, 256, 3), np.uint8))
        mp4 = enc.close()
    assert module.CreateEncoder.call_args.kwargs["fps"] == "30"
    assert struct.unpack_from(">II", _box_payload(mp4, b"stts"), 8)[1] == 1001


class _FakeCudaFrame:
    """A CUDA-looking frame whose pixels a patched ``as_warp`` supplies."""

    def __init__(self, pixels):
        self.pixels = pixels

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.pixels.shape, "typestr": "|u1", "version": 3}


def test_host_started_session_downloads_later_cuda_frames():
    """A second NVENC session would emit parameter sets the muxer rejects."""
    module = MagicMock()
    encoder = module.CreateEncoder.return_value
    encoder.Encode.return_value = b""
    encoder.EndEncode.return_value = b""
    host = np.zeros((192, 256, 3), np.uint8)
    device = np.full((193, 257, 3), 7, np.uint8)
    with (
        patch.dict("sys.modules", {"PyNvVideoCodec": module}),
        patch(
            "ovannotators._internal.video_frames.as_warp",
            side_effect=lambda f: wp.array(f.pixels, device="cpu"),
        ),
    ):
        enc = get_video_encoder("", 24, 256, 192)
        enc.write_frame(host)
        enc.write_frame(_FakeCudaFrame(device))
        enc.close()
    module.CreateEncoder.assert_called_once()
    assert module.CreateEncoder.call_args.args[3] is True  # host input
    encoder.EndEncode.assert_called_once()
    frames = [call.args[0] for call in encoder.Encode.call_args_list]
    assert [f.shape for f in frames] == [(192, 256, 4), (192, 256, 4)]
    assert np.all(frames[1][..., :3] == 7)


def test_basic_finalize_writes_remaining_streams_after_a_failure():
    writer = BasicWriter(
        BasicWriterConfig(video=True, video_vars=None, render_vars=None)
    )
    failing, working = MagicMock(), MagicMock()
    failing.close.side_effect = RuntimeError("flush failed")
    working.close.return_value = b"mp4"
    writer._video_encoders = {"ldr_color": failing, "normals": working}
    writer._seg_colorizer = MagicMock()
    backend = MagicMock()
    with pytest.raises(RuntimeError, match="ldr_color") as error:
        writer.finalize(backend)
    assert str(error.value.__cause__) == "flush failed"
    backend.write.assert_called_once_with("videos/normals.mp4", b"mp4")
    writer._seg_colorizer.save.assert_called_once()
    writer.finalize(backend)  # repeated calls stay no-ops
    backend.write.assert_called_once()


def test_cosmos_manifest_always_reports_video_encoding(tmp_path):
    writer = CosmosTransferWriter(CosmosTransferWriterConfig(conditioning_types=[]))
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    writer.write_frame(0, {"LdrColor": np.zeros((64, 64, 3), np.uint8)}, backend)
    with patch("ovannotators._internal.codecs.get_video_encoder") as factory:
        factory.return_value.close.return_value = b"mp4"
        with patch.object(CosmosTransferWriter, "_encode_jpeg", return_value=b"jpg"):
            writer.finalize(backend)
    clip = json.loads(backend.read("manifest.json"))["clips"][0]
    assert clip["video_encoding"] == {
        "codec": "h264",
        "lossless": False,
        "pixel_format": "yuv420p",
    }


def test_lerobot_without_cameras_skips_rgb_snapshots():
    pytest.importorskip("pyarrow")
    writer = LeRobotWriter(LeRobotWriterConfig(camera_keys=[]))
    with patch(
        "ovannotators.writers.lerobot.snapshot_rgb",
        side_effect=AssertionError("snapshot without cameras"),
    ):
        writer.write_frame(0, {"LdrColor": np.zeros((4, 4, 3), np.uint8)}, MagicMock())
    assert not writer._frames and len(writer._rows) == 1
