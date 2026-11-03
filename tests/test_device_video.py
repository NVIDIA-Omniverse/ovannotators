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

"""Device residency, borrowed-buffer ownership and actual NVENC round trips."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import warp as wp

from ovannotators._internal.video_frames import device_rgb, snapshot_rgb
from ovannotators.writers._helpers import to_rgb_uint8


@pytest.mark.parametrize("shape", [(7, 9), (7, 9, 1), (7, 9, 3), (7, 9, 4)])
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.float16, np.float32, np.float64, np.int32]
)
def test_device_packing_matches_cpu_on_warp_cpu(shape, dtype):
    rng = np.random.default_rng(13)
    src = (
        rng.random(shape) * (2 if np.issubdtype(dtype, np.floating) else 255)
    ).astype(dtype)
    expected = to_rgb_uint8(src)[:6, :8]
    packed = device_rgb(wp.array(src, device="cpu"), crop_even=True, bgra=True).numpy()
    np.testing.assert_array_equal(packed[..., :3], expected[..., ::-1])
    assert np.all(packed[..., 3] == 255)


def test_buffered_cpu_snapshot_owns_pixels():
    src = np.full((8, 8, 3), 17, np.uint8)
    snapshot = snapshot_rgb(src)
    src.fill(99)
    assert np.all(snapshot == 17)


class _DlpackOnlyFrame:
    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, stream=None):
        return self.array.__dlpack__(stream=stream)


@pytest.mark.integration
def test_dlpack_alias_and_canny_preparation_on_cpu():
    from ovannotators._internal.video_frames import (
        as_warp,
        device_gray,
        device_edges_rgb,
    )

    pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    original = wp.array(pixels, device="cpu")
    imported = as_warp(_DlpackOnlyFrame(original))
    assert imported.ptr == original.ptr
    gray = device_gray(imported)
    original.zero_()
    np.testing.assert_allclose(gray.numpy(), [[0.2989, 0.587, 0.114]], atol=1e-6)

    edges = wp.array(np.array([[0, 1, 0]], np.uint8), device="cpu")
    rgb = device_edges_rgb(edges)
    np.testing.assert_array_equal(
        rgb.numpy(), [[[0, 0, 0], [255, 255, 255], [0, 0, 0]]]
    )


def test_device_packing_rejects_invalid_channel_count():
    with pytest.raises(ValueError, match="frame shape"):
        device_rgb(wp.zeros((8, 8, 2), dtype=wp.uint8, device="cpu"))


@pytest.mark.gpu
@pytest.mark.integration
def test_cai_only_frame_aliases_real_cuda_buffer(cuda_device):
    from ovannotators._internal.video_frames import as_warp

    original = wp.array(np.full((8, 8, 3), 17, np.uint8), device=cuda_device)

    class CaiOnly:
        __cuda_array_interface__ = original.__cuda_array_interface__

    frame = CaiOnly()
    with wp.ScopedDevice(cuda_device):
        imported = as_warp(frame)
    assert imported.ptr == original.ptr
    assert imported.device == original.device
    original.zero_()
    assert np.all(imported.numpy() == 0)


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_cai_only_frame_explicit_device():
    from ovannotators._internal.video_frames import as_warp

    if wp.get_cuda_device_count() < 2:
        pytest.skip("two CUDA devices required for explicit owning-device test")
    original = wp.array(np.full((8, 8, 3), 17, np.uint8), device="cuda:1")

    class CaiOnly:
        __cuda_array_interface__ = original.__cuda_array_interface__

    with wp.ScopedDevice("cuda:0"):
        imported = as_warp(CaiOnly(), device="cuda:1")
    assert imported.ptr == original.ptr
    assert imported.device == original.device
    np.testing.assert_array_equal(imported.numpy(), original.numpy())


@pytest.mark.gpu
@pytest.mark.integration
def test_dlpack_only_normals_write_exr_and_video(cuda_device, tmp_path):
    pytest.importorskip("PyNvVideoCodec")
    av = pytest.importorskip("av", reason="independent video decoding")
    from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
    from ovannotators._internal.codecs import read_exr
    from ovannotators.writers.basic import BasicWriter, BasicWriterConfig

    normals = np.zeros((192, 256, 3), np.float32)
    normals[..., 0] = 1
    frame = _DlpackOnlyFrame(wp.array(normals, device=cuda_device))
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path)))
    writer = BasicWriter(
        BasicWriterConfig(video=True, colorize=False, render_vars=["NormalSD"])
    )
    writer.write_frame(0, {"NormalSD": frame}, backend)
    writer.finalize(backend)

    np.testing.assert_array_equal(read_exr(str(tmp_path / "0000/normal.exr")), normals)
    with av.open(io.BytesIO(backend.read("videos/normal_colorized.mp4"))) as video:
        decoded = list(video.decode(video=0))
    assert len(decoded) == 1
    np.testing.assert_allclose(
        decoded[0].to_ndarray(format="rgb24").mean(axis=(0, 1)),
        [255, 127, 127],
        atol=5,
    )


@pytest.fixture
def cuda_device():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    return wp.get_cuda_device()


@pytest.mark.gpu
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_gpu_strided_normalization_without_download(cuda_device, channels, dtype):
    src = np.random.default_rng(3).random((194, 260, channels))
    if dtype == np.uint8:
        src *= 255
    src = src.astype(dtype)
    if channels == 0:
        src = (
            np.random.default_rng(3)
            .integers(0, 255, (194, 260), dtype=np.uint8)
            .astype(dtype)
        )
    device_src = wp.array(src, device=cuda_device)[:, ::2]
    with patch.object(wp.array, "numpy", side_effect=AssertionError("raw host copy")):
        result = device_rgb(device_src, crop_even=True)
    np.testing.assert_array_equal(result.numpy(), to_rgb_uint8(src[:, ::2]))


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.parametrize("kind", ["basic", "lerobot", "cosmos"])
def test_writers_encode_gpu_frames_without_download(cuda_device, kind):
    pytest.importorskip("PyNvVideoCodec")
    av = pytest.importorskip(
        "av", reason="PyAV used only to independently decode NVENC output"
    )
    from ovannotators.writers.basic import BasicWriter, BasicWriterConfig
    from ovannotators.writers.lerobot import LeRobotWriter, LeRobotWriterConfig
    from ovannotators.writers.cosmos_transfer import (
        CosmosTransferWriter,
        CosmosTransferWriterConfig,
    )

    backend = MagicMock()
    written = {}
    backend.write.side_effect = lambda path, data: written.__setitem__(path, data)
    if kind == "basic":
        writer = BasicWriter(BasicWriterConfig(video=True, render_vars=["LdrColor"]))
    elif kind == "lerobot":
        pytest.importorskip("pyarrow")
        writer = LeRobotWriter(LeRobotWriterConfig(camera_keys=["front"]))
    else:
        writer = CosmosTransferWriter(CosmosTransferWriterConfig(conditioning_types=[]))
        # Thumbnail encoding is a separate image-codec concern.
        writer._encode_jpeg = lambda *args, **kwargs: b"jpeg"

    device_src = wp.empty((192, 256, 4), dtype=wp.uint8, device=cuda_device)
    with patch.object(wp.array, "numpy", side_effect=AssertionError("raw host copy")):
        for frame_id, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            src = np.zeros((192, 256, 4), dtype=np.uint8)
            src[..., :3] = color
            src[..., 3] = 255
            # Reuse a renderer-like allocation, produced on a separate stream.
            with wp.ScopedStream(wp.Stream(cuda_device)):
                wp.copy(device_src, wp.array(src, device=cuda_device))
            writer.write_frame(
                frame_id,
                {
                    "LdrColor": device_src,
                    "DistanceToCameraSD": None,
                    "SemanticInstanceSegmentation": None,
                },
                backend,
            )
        device_src.zero_()
        writer.finalize(backend)
    videos = [data for path, data in written.items() if path.endswith(".mp4")]
    assert len(videos) == 1
    frames = list(av.open(io.BytesIO(videos[0])).decode(video=0))
    assert len(frames) == 3
    for i, frame in enumerate(frames):
        rgb = frame.to_ndarray(format="rgb24").mean(axis=(0, 1))
        assert rgb[i] > 240
        assert np.max(np.delete(rgb, i)) < 15


@pytest.mark.gpu
def test_rgb_canny_keeps_images_on_device(cuda_device):
    from ovannotators.writers.cosmos_transfer import (
        CosmosTransferWriter,
        CosmosTransferWriterConfig,
    )

    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(conditioning_types=["canny"])
    )
    src = np.zeros((192, 256, 3), np.uint8)
    src[48:144, 64:192] = 255
    writer.write_frame(
        0,
        {
            "LdrColor": wp.array(src, device=cuda_device),
            "DistanceToCameraSD": None,
            "SemanticInstanceSegmentation": None,
        },
        MagicMock(),
    )
    numpy = wp.array.numpy

    def scalar_only(array, *args, **kwargs):
        assert array.size == 1, "Only Canny convergence/reduction scalars may reach CPU"
        return numpy(array, *args, **kwargs)

    with patch.object(wp.array, "numpy", scalar_only):
        frames = writer._compute_canny_frames()
    assert frames[0].device == cuda_device
    assert np.any(frames[0].numpy() == 255)


@pytest.mark.gpu
@pytest.mark.e2e
def test_nvenc_mixed_host_and_device_frames(cuda_device):
    pytest.importorskip("PyNvVideoCodec")
    av = pytest.importorskip("av")
    from ovannotators._internal.codecs import get_video_encoder

    enc = get_video_encoder("", 24, 256, 192)
    for i, gpu in enumerate([False, True, False, True]):
        src = np.zeros((192, 256, 3), np.uint8)
        src[..., i % 3] = 255
        enc.write_frame(wp.array(src, device=cuda_device) if gpu else src)
    decoded = list(av.open(io.BytesIO(enc.close())).decode(video=0))
    assert len(decoded) == 4
    for i, frame in enumerate(decoded):
        rgb = frame.to_ndarray(format="rgb24").mean(axis=(0, 1))
        assert rgb[i % 3] > 240
        assert np.max(np.delete(rgb, i % 3)) < 15


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_normal_colorization_stays_on_device(cuda_device, dtype):
    from ovannotators.writers.basic import _colorize_normals

    normals = np.random.default_rng(19).uniform(-1, 1, (64, 80, 4)).astype(dtype)
    normals[0, 0] = 0
    expected = _colorize_normals(normals)
    src = wp.array(normals, device=cuda_device)
    with patch.object(wp.array, "numpy", side_effect=AssertionError("raw host copy")):
        result = _colorize_normals(src, keep_device=True)
    assert result.device == cuda_device
    np.testing.assert_array_equal(result.numpy(), expected)


class _FakeWarpArray:
    """Stand-in for ``wp.array`` that records how a frame was imported."""

    def __init__(self, data=None, device=None, copy=True):
        if data is not None:
            # Warp requires a sequence even for a CAI-only input. Preserve
            # that boundary contract so this fake catches missing adapters.
            assert len(data) == data.__cuda_array_interface__["shape"][0]
        self.data, self.device, self.copy = data, device, copy


class _CaiOnlyFrame:
    """CUDA array interface without DLPack, like some third-party buffers."""

    @property
    def __cuda_array_interface__(self):
        return {"shape": (2, 2, 3), "typestr": "|u1", "data": (1, False), "version": 3}


def test_as_warp_imports_cuda_array_interface_only_frames(monkeypatch):
    from types import SimpleNamespace

    from ovannotators._internal import video_frames

    current = SimpleNamespace(is_cuda=True, name="current")
    monkeypatch.setattr(
        video_frames,
        "wp",
        SimpleNamespace(
            array=_FakeWarpArray,
            get_cuda_device_count=lambda: 1,
            get_device=lambda: current,
            get_cuda_device=lambda: pytest.fail("current device is already CUDA"),
        ),
    )
    frame = _CaiOnlyFrame()
    assert video_frames.is_device_frame(frame)

    imported = video_frames.as_warp(frame)
    assert imported.data.frame is frame
    assert (imported.device, imported.copy) == (current, False)

    explicit = SimpleNamespace(is_cuda=True, name="explicit")
    assert video_frames.as_warp(frame, device=explicit).device is explicit


def test_as_warp_uses_a_cuda_device_when_the_current_one_is_cpu(monkeypatch):
    from types import SimpleNamespace

    from ovannotators._internal import video_frames

    cuda = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(
        video_frames,
        "wp",
        SimpleNamespace(
            array=_FakeWarpArray,
            get_cuda_device_count=lambda: 1,
            get_device=lambda: SimpleNamespace(is_cuda=False),
            get_cuda_device=lambda: cuda,
        ),
    )
    assert video_frames.as_warp(_CaiOnlyFrame()).device is cuda


def test_as_warp_resolves_the_owning_gpu_on_multi_gpu_hosts(monkeypatch):
    """A CAI-only frame on cuda:1 is never aliased as the current cuda:0."""
    from types import SimpleNamespace

    from ovannotators._internal import video_frames

    devices = [SimpleNamespace(is_cuda=True, ordinal=i) for i in range(2)]
    monkeypatch.setattr(
        video_frames,
        "wp",
        SimpleNamespace(
            array=_FakeWarpArray,
            get_cuda_device_count=lambda: 2,
            get_device=lambda: devices[0],
            get_cuda_device=lambda ordinal=0: devices[ordinal],
        ),
    )
    queried = []
    monkeypatch.setattr(
        video_frames,
        "_pointer_device_ordinal",
        lambda ptr: queried.append(ptr) or 1,
    )
    assert video_frames.as_warp(_CaiOnlyFrame()).device is devices[1]
    assert queried == [1]

    monkeypatch.setattr(video_frames, "_pointer_device_ordinal", lambda ptr: None)
    with pytest.raises(ValueError, match="pass device="):
        video_frames.as_warp(_CaiOnlyFrame())
    explicit = video_frames.as_warp(_CaiOnlyFrame(), device=devices[1])
    assert explicit.device is devices[1]


def test_as_warp_rejects_frames_without_a_cuda_interface():
    from ovannotators._internal.video_frames import as_warp

    with pytest.raises(TypeError, match="cuda_array_interface"):
        as_warp(object())


@pytest.mark.parametrize("dtype", [wp.vec3, wp.vec4])
def test_vector_typed_normals_use_the_device_path(dtype):
    """2D vec3/vec4 NormalSD matches the host colorization."""
    from ovannotators.writers.basic import _colorize_normals

    normals = np.random.default_rng(23).uniform(-1, 1, (6, 5, dtype._length_))
    normals = normals.astype(np.float32)
    src = wp.array(normals, dtype=dtype, device="cpu")
    with patch(
        "ovannotators._internal.video_frames.is_device_frame", return_value=True
    ):
        result = _colorize_normals(src)
    np.testing.assert_array_equal(result, _colorize_normals(normals))


def test_device_normals_reject_too_few_channels():
    from ovannotators.writers.basic import _colorize_normals

    src = wp.array(np.zeros((4, 4, 2), np.float32), device="cpu")
    with patch(
        "ovannotators._internal.video_frames.is_device_frame", return_value=True
    ):
        with pytest.raises(ValueError, match="C>=3"):
            _colorize_normals(src)


def test_canny_grayscale_matches_between_host_and_device():
    """A dark uint8 frame is scaled by 255 on both paths."""
    from ovannotators._internal.video_frames import device_gray
    from ovannotators.writers.cosmos_transfer import CosmosTransferWriter

    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, 4:] = 1
    host = CosmosTransferWriter._to_gray01(rgb)
    device = device_gray(wp.array(rgb, device="cpu")).numpy()
    np.testing.assert_allclose(host, device, atol=1e-6)
    assert host.max() < 0.01
