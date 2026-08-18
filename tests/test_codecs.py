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

"""Tests for ovannotators/_internal/codecs.py — image codecs + video encoders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovannotators._internal.codecs import (
    NvImageCodecBackend,
    NvVideoEncoder,
    PillowCodecBackend,
    get_image_codec,
    get_video_encoder,
)


def _has_nvimgcodec() -> bool:
    """Check whether GPU image encode/decode is actually usable.

    Mirrors the source-side detection: the bindings can import on a host with
    no CUDA driver/device, where real GPU encode/decode still fails, so we gate
    on a working GPU codec rather than mere importability.
    """
    from ovannotators._internal.codecs import _gpu_image_codec_available

    return _gpu_image_codec_available()


# ---------------------------------------------------------------------------
# PillowCodecBackend
# ---------------------------------------------------------------------------


class TestPillowCodecBackend:
    def setup_method(self):
        self.codec = PillowCodecBackend()

    def test_encode_jpeg(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        data = self.codec.encode(img, "jpg")
        assert isinstance(data, bytes)
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_encode_png(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        data = self.codec.encode(img, "png")
        assert isinstance(data, bytes)
        # PNG magic bytes
        assert data[:4] == b"\x89PNG"

    def test_encode_jpeg_alias(self):
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        data = self.codec.encode(img, "jpeg")
        assert data[:2] == b"\xff\xd8"

    def test_decode_roundtrip_png(self):
        img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        encoded = self.codec.encode(img, "png")
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(decoded, img)

    def test_quality_param(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        low_q = self.codec.encode(img, "jpg", quality=10)
        high_q = self.codec.encode(img, "jpg", quality=95)
        # Higher quality = larger file
        assert len(high_q) > len(low_q)


# ---------------------------------------------------------------------------
# NvImageCodecBackend (mocked)
# ---------------------------------------------------------------------------


class TestNvImageCodecBackend:
    @pytest.mark.skipif(
        not _has_nvimgcodec(), reason="nvidia-nvimgcodec-cu12 not installed"
    )
    def test_encode_png(self, tmp_path):
        """Test encode writes a valid PNG file."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        backend = NvImageCodecBackend()
        backend.write(str(tmp_path / "test.png"), img)
        assert (tmp_path / "test.png").exists()
        assert (tmp_path / "test.png").stat().st_size > 0

    @pytest.mark.skipif(
        not _has_nvimgcodec(), reason="nvidia-nvimgcodec-cu12 not installed"
    )
    def test_write_dispatches_exr_to_openexr(self, tmp_path):
        """Test that .exr files are written via OpenEXR."""
        hdr = np.random.rand(16, 16, 4).astype(np.float16)
        backend = NvImageCodecBackend()
        backend.write(str(tmp_path / "test.exr"), hdr)
        assert (tmp_path / "test.exr").exists()

    def test_encode_mocked(self, tmp_path):
        """Test encode path with mocked nvimgcodec (CI-friendly)."""
        mock_nvimgcodec = MagicMock()
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.encode.return_value = b"fake_png_bytes"
        mock_nvimgcodec.Encoder.return_value = mock_encoder_instance

        backend = NvImageCodecBackend()
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        with patch.dict(
            "sys.modules",
            {
                "nvidia.nvimgcodec": mock_nvimgcodec,
                "nvidia": MagicMock(nvimgcodec=mock_nvimgcodec),
            },
        ):
            backend.write(str(tmp_path / "out.png"), img)

        assert (tmp_path / "out.png").exists()

    def test_encode_ensures_contiguous(self, tmp_path):
        """Non-contiguous arrays are made contiguous before encode."""
        mock_nvimgcodec = MagicMock()
        encoded_data = []

        def capture_encode(arr, **kwargs):
            encoded_data.append(arr.copy())
            return b"fake_png"

        mock_encoder_instance = MagicMock()
        mock_encoder_instance.encode.side_effect = capture_encode
        mock_nvimgcodec.Encoder.return_value = mock_encoder_instance

        backend = NvImageCodecBackend()

        # Create a non-contiguous array (e.g. from slicing)
        big = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        non_contig = big[::2, ::2, :]  # non-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]

        with patch.dict(
            "sys.modules",
            {
                "nvidia.nvimgcodec": mock_nvimgcodec,
                "nvidia": MagicMock(nvimgcodec=mock_nvimgcodec),
            },
        ):
            backend.write(str(tmp_path / "out.png"), non_contig)

        # The array passed to encoder.encode should be contiguous
        call_args = mock_encoder_instance.encode.call_args
        written_arr = call_args[0][0]
        assert written_arr.flags["C_CONTIGUOUS"]

    def test_decode_mocked(self, tmp_path):
        """Test decode path with mocked nvimgcodec."""
        mock_nvimgcodec = MagicMock()
        expected = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_decoder_instance = MagicMock()
        mock_decoder_instance.decode.return_value = expected
        mock_nvimgcodec.Decoder.return_value = mock_decoder_instance

        backend = NvImageCodecBackend()

        with patch.dict(
            "sys.modules",
            {
                "nvidia.nvimgcodec": mock_nvimgcodec,
                "nvidia": MagicMock(nvimgcodec=mock_nvimgcodec),
            },
        ):
            result = backend.decode(b"fake png data")

        # The raw bytes must be forwarded verbatim to the nvimgcodec decoder...
        mock_decoder_instance.decode.assert_called_once_with(b"fake png data")
        # ...and the decoder's result is returned (wrapped via np.asarray).
        assert np.array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_write_exr_dispatches(self, tmp_path):
        """write() dispatches .exr to _write_exr."""
        backend = NvImageCodecBackend()

        hdr = np.random.rand(16, 16, 3).astype(np.float16)
        with patch("ovannotators._internal.codecs._write_exr") as mock_write_exr:
            backend.write(str(tmp_path / "test.exr"), hdr)
            mock_write_exr.assert_called_once()


# ---------------------------------------------------------------------------
# NvVideoEncoder (mocked)
# ---------------------------------------------------------------------------


# Minimal but parseable H.264 Annex-B stream: SPS (type 7) + PPS (type 8) +
# IDR slice (type 5), each prefixed with a 4-byte start code. The slice payload
# is not real coded data, but it exercises the NAL parsing / MP4 muxing path.
_START = b"\x00\x00\x00\x01"
_FAKE_ANNEXB = (
    _START
    + b"\x67\x42\x00\x1e\xda\x01\x40"  # SPS (profile 0x42, level 0x1e)
    + _START
    + b"\x68\xce\x3c\x80"  # PPS
    + _START
    + b"\x65\x88\x84\x00\x21\x00"  # IDR slice
)


class TestNvVideoEncoder:
    def test_write_frame_before_open_raises(self):
        enc = NvVideoEncoder()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="open.*must be called"):
            enc.write_frame(frame)

    def test_close_without_open_returns_empty(self):
        enc = NvVideoEncoder()
        result = enc.close()
        assert result == b""

    def test_open_and_close_muxes_to_mp4(self):
        mock_pynv = MagicMock()
        mock_encoder_inst = MagicMock()
        mock_encoder_inst.Encode.return_value = _FAKE_ANNEXB
        mock_encoder_inst.EndEncode.return_value = b""
        mock_pynv.CreateEncoder.return_value = mock_encoder_inst

        with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
            enc = NvVideoEncoder()
            enc.open("", 30, 64, 64)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            enc.write_frame(frame)
            result = enc.close()

        # bf=0 (no B-frames) is requested so the stream can be muxed in order.
        _, kwargs = mock_pynv.CreateEncoder.call_args
        assert kwargs.get("bf") == 0
        # Output is a real MP4 container, not the raw H.264 elementary stream.
        assert result[4:8] == b"ftyp"
        assert b"moov" in result
        assert b"mdat" in result

    def test_close_flush_exception_handled(self):
        mock_pynv = MagicMock()
        mock_encoder_inst = MagicMock()
        mock_encoder_inst.Encode.return_value = _FAKE_ANNEXB
        mock_encoder_inst.EndEncode.side_effect = RuntimeError("flush error")
        mock_pynv.CreateEncoder.return_value = mock_encoder_inst

        with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
            enc = NvVideoEncoder()
            enc.open("", 30, 64, 64)
            enc.write_frame(np.zeros((64, 64, 3), dtype=np.uint8))
            # Should not raise — flush errors are caught
            result = enc.close()
            assert result[4:8] == b"ftyp"

    def test_close_muxes_raw_h264_payload(self):
        enc = NvVideoEncoder()
        enc._chunks = [_FAKE_ANNEXB]
        enc._fps = 24
        enc._width = 128
        enc._height = 72
        result = enc.close()
        assert result[4:8] == b"ftyp"
        assert b"mdat" in result

    def test_close_empty_payload_returns_empty(self):
        enc = NvVideoEncoder()
        enc._chunks = []
        enc._fps = 24
        enc._width = 64
        enc._height = 64
        assert enc.close() == b""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EXR support
# ---------------------------------------------------------------------------


class TestExrReadWrite:
    def test_write_read_roundtrip_rgb(self, tmp_path):
        """Float16 RGB roundtrip through EXR."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(32, 64, 3).astype(np.float16)
        path = str(tmp_path / "test.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.shape == (32, 64, 3)
        assert result.dtype == np.float16
        np.testing.assert_array_equal(arr, result)

    def test_write_read_roundtrip_rgba(self, tmp_path):
        """Float16 RGBA roundtrip through EXR."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(16, 16, 4).astype(np.float16)
        path = str(tmp_path / "test.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.shape == (16, 16, 4)
        np.testing.assert_array_equal(arr, result)

    def test_write_exr_converts_float32_to_half(self, tmp_path):
        """Float32 input is converted to float16 for EXR storage."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(8, 8, 3).astype(np.float32)
        path = str(tmp_path / "test.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.dtype == np.float16

    def test_write_read_roundtrip_single_channel(self, tmp_path):
        """Single-channel (H, W, 1) data roundtrips through EXR."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(16, 24, 1).astype(np.float16)
        path = str(tmp_path / "depth.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.shape == (16, 24, 1)
        np.testing.assert_array_equal(arr, result)

    def test_write_read_roundtrip_2d_as_single_channel(self, tmp_path):
        """2-D (H, W) data is stored as a single channel and reads back as 1-ch."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(8, 8).astype(np.float16)
        path = str(tmp_path / "depth2d.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.shape == (8, 8, 1)
        np.testing.assert_array_equal(arr[:, :, np.newaxis], result)

    def test_write_read_roundtrip_two_channel(self, tmp_path):
        """Two-channel (H, W, 2) motion data roundtrips through EXR."""
        from ovannotators._internal.codecs import _write_exr, read_exr

        arr = np.random.rand(12, 20, 2).astype(np.float16)
        path = str(tmp_path / "motion.exr")
        _write_exr(path, arr)
        result = read_exr(path)
        assert result.shape == (12, 20, 2)
        np.testing.assert_array_equal(arr, result)

    def test_write_exr_invalid_shape(self, tmp_path):
        """Unsupported rank or channel count raises ValueError."""
        from ovannotators._internal.codecs import _write_exr

        with pytest.raises(ValueError, match="EXR write expects"):
            _write_exr(str(tmp_path / "bad.exr"), np.zeros((10, 10, 5)))
        with pytest.raises(ValueError, match="EXR write expects"):
            _write_exr(str(tmp_path / "bad.exr"), np.zeros((10,)))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestGetImageCodec:
    def test_returns_codec_with_required_methods(self):
        codec = get_image_codec()
        assert hasattr(codec, "encode")
        assert hasattr(codec, "decode")
        assert hasattr(codec, "write")

    def test_without_gpu_codec_returns_pillow(self):
        with patch(
            "ovannotators._internal.codecs._gpu_image_codec_available",
            return_value=False,
        ):
            codec = get_image_codec()
            assert isinstance(codec, PillowCodecBackend)

    def test_with_gpu_codec_prefers_gpu(self):
        with patch(
            "ovannotators._internal.codecs._gpu_image_codec_available",
            return_value=True,
        ):
            codec = get_image_codec()
            assert isinstance(codec, NvImageCodecBackend)

    def test_gpu_unavailable_without_cuda_device(self):
        mock_wp = MagicMock()
        mock_wp.get_cuda_device_count.return_value = 0
        with patch.dict("sys.modules", {"warp": mock_wp}):
            from ovannotators._internal.codecs import _gpu_image_codec_available

            assert _gpu_image_codec_available() is False


class TestGetVideoEncoder:
    def test_without_pynvvideocodec_raises(self):
        with patch.dict("sys.modules", {"PyNvVideoCodec": None}):
            with pytest.raises(ImportError):
                get_video_encoder("", 30, 64, 64)

    def test_with_pynvvideocodec(self):
        mock_pynv = MagicMock()
        mock_pynv.CreateEncoder.return_value = MagicMock()
        with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
            enc = get_video_encoder("", 30, 64, 64)
            assert isinstance(enc, NvVideoEncoder)


# ---------------------------------------------------------------------------
# NvImageCodecBackend — GPU JPEG params and None-result guard
# ---------------------------------------------------------------------------


def test_nvimgcodec_encode_jpeg_roundtrip():
    if not _has_nvimgcodec():
        pytest.skip("no usable GPU image codec (CUDA device + nvimgcodec)")
    from ovannotators._internal.codecs import NvImageCodecBackend

    backend = NvImageCodecBackend()
    rgb = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    data = backend.encode(rgb, "jpg", quality=90)
    # Valid JPEG SOI marker confirms the GPU JPEG-params path produced output.
    assert isinstance(data, (bytes, bytearray)) and data[:2] == b"\xff\xd8"


def test_nvimgcodec_encode_none_result_raises(monkeypatch):
    # ``importorskip`` is not enough: a bare ``nvidia`` namespace can resolve
    # ``nvidia.nvimgcodec`` to an empty module with no real encode API to patch.
    from ovannotators._internal.optional_deps import nvimgcodec_available

    if not nvimgcodec_available():
        pytest.skip("nvidia.nvimgcodec not installed (only an empty namespace)")
    import nvidia.nvimgcodec as nvimgcodec

    from ovannotators._internal.codecs import NvImageCodecBackend

    class _NullEncoder:
        def encode(self, *args, **kwargs):
            return None

    monkeypatch.setattr(nvimgcodec, "Encoder", lambda *a, **k: _NullEncoder())
    with pytest.raises(RuntimeError):
        NvImageCodecBackend().encode(np.zeros((4, 4, 3), np.uint8), "png")


# ---------------------------------------------------------------------------
# PillowCodecBackend — JPEG/PNG/EXR write paths
# ---------------------------------------------------------------------------


def test_pillow_backend_write_jpg_and_exr(tmp_path):
    from ovannotators._internal.codecs import PillowCodecBackend, read_exr

    backend = PillowCodecBackend()
    rgb = np.random.default_rng(2).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    jpg_path = tmp_path / "out.jpg"
    backend.write(str(jpg_path), rgb, quality=80)
    assert jpg_path.exists() and jpg_path.stat().st_size > 0

    png_path = tmp_path / "out.png"
    backend.write(str(png_path), rgb)
    assert png_path.exists()

    exr_path = tmp_path / "out.exr"
    hdr = np.random.default_rng(3).random((8, 8, 3)).astype(np.float32)
    backend.write(str(exr_path), hdr)
    assert exr_path.exists()
    back = read_exr(str(exr_path))
    assert back.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# EXR encode/write helpers
# ---------------------------------------------------------------------------


def test_write_exr_bad_shape_raises(tmp_path):
    from ovannotators._internal.codecs import _write_exr_file

    # 5 channels is an unsupported EXR shape and must raise.
    with pytest.raises(ValueError):
        _write_exr_file(str(tmp_path / "bad.exr"), np.zeros((4, 4, 5), np.float16))


def test_write_exr_converts_float32(tmp_path):
    from ovannotators._internal.codecs import _write_exr_file, read_exr

    arr = np.random.default_rng(4).random((6, 6, 4)).astype(np.float32)
    out = tmp_path / "rgba.exr"
    _write_exr_file(str(out), arr)
    assert out.exists()
    assert read_exr(str(out)).shape == (6, 6, 4)


def test_encode_exr_bytes(tmp_path):
    """``_encode_exr_bytes`` produces valid EXR bytes via the OpenEXR path."""
    pytest.importorskip("OpenEXR")
    from ovannotators._internal import codecs

    out = codecs._encode_exr_bytes(np.zeros((4, 4, 3), np.float16))
    # EXR magic number (little-endian 20000630 = 0x76 0x2f 0x31 0x01).
    assert isinstance(out, bytes) and out[:4] == b"\x76\x2f\x31\x01"


def test_to_numpy_array_like_fallback():
    """A plain array-like with no ``.numpy()`` is coerced via ``np.asarray``."""
    from ovannotators._internal.codecs import _to_numpy

    out = _to_numpy([[1, 2], [3, 4]])
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [[1, 2], [3, 4]]


def test_to_numpy_calls_numpy_method():
    """An array-like exposing ``.numpy()`` (e.g. a Warp array) is materialized."""
    import warp as wp

    from ovannotators._internal.codecs import _to_numpy

    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = _to_numpy(wp.array(src))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, src)


# ---------------------------------------------------------------------------
# Channel normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_for_encode_squeezes_singleton_channel():
    """A trailing ``(H, W, 1)`` axis is squeezed to ``(H, W)``."""
    from ovannotators._internal.codecs import _normalize_for_encode

    out = _normalize_for_encode(np.zeros((4, 5, 1), dtype=np.uint8), "png")
    assert out.shape == (4, 5)


def test_normalize_for_encode_drops_alpha_for_jpeg():
    """JPEG has no alpha, so an RGBA ``(H, W, 4)`` array drops to RGB."""
    from ovannotators._internal.codecs import _normalize_for_encode

    out = _normalize_for_encode(np.zeros((4, 5, 4), dtype=np.uint8), "jpg")
    assert out.shape == (4, 5, 3)


def test_normalize_gpu_image_squeezes_singleton_and_makes_contiguous():
    """``(H, W, 1)`` device arrays squeeze the channel axis and stay contiguous.

    A Warp array is a real array-like exposing ``ndim``/``shape``/slicing and a
    ``contiguous()`` method, so it drives the device-side normalization path
    without a GPU or a hand-written stub.
    """
    import warp as wp

    from ovannotators._internal.codecs import _normalize_gpu_image

    src = wp.array(np.zeros((4, 5, 1), dtype=np.uint8))
    out = _normalize_gpu_image(src, "png")
    assert out.shape == (4, 5)


def test_normalize_gpu_image_drops_alpha_for_jpeg():
    """RGBA device arrays drop alpha for JPEG, mirroring the host normalizer."""
    import warp as wp

    from ovannotators._internal.codecs import _normalize_gpu_image

    src = wp.array(np.zeros((4, 5, 4), dtype=np.uint8))
    out = _normalize_gpu_image(src, "jpeg")
    assert out.shape == (4, 5, 3)


# ---------------------------------------------------------------------------
# NvImageCodecBackend.encode — JPEG params, GPU path, and None-result guard
# ---------------------------------------------------------------------------


def _mock_nvimgcodec(encode_side_effect=None, encode_return=b"fake"):
    """Build a ``sys.modules`` patch dict for a fake ``nvidia.nvimgcodec``.

    Reuses the repo's accepted GPU-dep test pattern (patch ``sys.modules`` with
    a ``MagicMock`` codec) so the encode/decode branches run on a GPU-less host.
    """
    mock_nvimgcodec = MagicMock()
    mock_encoder = MagicMock()
    if encode_side_effect is not None:
        mock_encoder.encode.side_effect = encode_side_effect
    else:
        mock_encoder.encode.return_value = encode_return
    mock_nvimgcodec.Encoder.return_value = mock_encoder
    modules = {
        "nvidia.nvimgcodec": mock_nvimgcodec,
        "nvidia": MagicMock(nvimgcodec=mock_nvimgcodec),
    }
    return modules, mock_nvimgcodec, mock_encoder


def test_nvimgcodec_encode_jpeg_builds_params():
    """The JPEG branch constructs ``EncodeParams`` from the quality setting."""
    modules, mock_nvimgcodec, mock_encoder = _mock_nvimgcodec(encode_return=b"jpgdata")
    backend = NvImageCodecBackend()
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    with patch.dict("sys.modules", modules):
        out = backend.encode(img, "jpg", quality=70)

    assert out == b"jpgdata"
    # EncodeParams is built (jpeg branch) and the codec name is normalized.
    mock_nvimgcodec.EncodeParams.assert_called_once()
    _, kwargs = mock_encoder.encode.call_args
    assert kwargs["codec"] == "jpeg"


def test_nvimgcodec_encode_gpu_array_zero_copy_path():
    """A device-resident array is encoded on-device without a host round-trip."""
    modules, _, mock_encoder = _mock_nvimgcodec(encode_return=b"gpu_bytes")
    backend = NvImageCodecBackend()
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        patch.dict("sys.modules", modules),
        patch("ovannotators._internal.codecs._is_gpu_array", return_value=True),
    ):
        out = backend.encode(img, "png")

    assert out == b"gpu_bytes"
    # The on-device encode succeeded, so the host fallback never ran.
    mock_encoder.encode.assert_called_once()


def test_nvimgcodec_encode_gpu_failure_falls_back_to_host():
    """If the on-device encode raises, encoding falls back to the host path."""
    # First call (GPU path) raises; second call (host path) succeeds.
    modules, _, mock_encoder = _mock_nvimgcodec(
        encode_side_effect=[RuntimeError("unsupported layout"), b"host_bytes"]
    )
    backend = NvImageCodecBackend()
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        patch.dict("sys.modules", modules),
        patch("ovannotators._internal.codecs._is_gpu_array", return_value=True),
    ):
        out = backend.encode(img, "png")

    assert out == b"host_bytes"
    assert mock_encoder.encode.call_count == 2


def test_nvimgcodec_encode_none_result_raises_mocked():
    """A ``None`` encoder result on the host path raises ``RuntimeError``."""
    modules, _, _ = _mock_nvimgcodec(encode_return=None)
    backend = NvImageCodecBackend()
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    with patch.dict("sys.modules", modules):
        with pytest.raises(RuntimeError, match="failed to encode"):
            backend.encode(img, "png")


# ---------------------------------------------------------------------------
# NvVideoEncoder — non-RGB frame passthrough and flush-chunk accumulation
# ---------------------------------------------------------------------------


def test_write_frame_non_rgb_passthrough():
    """A non-RGB(A) frame (e.g. 2-D) is forwarded to the encoder unchanged."""
    mock_pynv = MagicMock()
    captured = []
    mock_encoder_inst = MagicMock()
    mock_encoder_inst.Encode.side_effect = lambda frame: captured.append(frame) or b""
    mock_encoder_inst.EndEncode.return_value = b""
    mock_pynv.CreateEncoder.return_value = mock_encoder_inst

    with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
        enc = NvVideoEncoder()
        enc.open("", 30, 64, 64)
        gray = np.zeros((64, 64), dtype=np.uint8)
        enc.write_frame(gray)

    # The 2-D frame skips ARGB packing and is passed through as-is.
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], gray)


def test_close_appends_flush_chunk():
    """Bytes returned by ``EndEncode`` are appended before muxing."""
    mock_pynv = MagicMock()
    mock_encoder_inst = MagicMock()
    mock_encoder_inst.Encode.return_value = _FAKE_ANNEXB
    # A non-empty flush payload (extra IDR slice) must be appended to the chunks.
    mock_encoder_inst.EndEncode.return_value = _START + b"\x65\x88\x84\x00\x21\x00"
    mock_pynv.CreateEncoder.return_value = mock_encoder_inst

    with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
        enc = NvVideoEncoder()
        enc.open("", 30, 64, 64)
        enc.write_frame(np.zeros((64, 64, 3), dtype=np.uint8))
        result = enc.close()

    mock_encoder_inst.EndEncode.assert_called_once()
    assert result[4:8] == b"ftyp"


# ---------------------------------------------------------------------------
# _gpu_image_codec_available — warp probe failure
# ---------------------------------------------------------------------------


def test_gpu_image_codec_unavailable_when_warp_probe_raises():
    """A raising CUDA-device probe is treated as "no GPU codec"."""
    mock_wp = MagicMock()
    mock_wp.get_cuda_device_count.side_effect = RuntimeError("warp init failed")
    with patch.dict("sys.modules", {"warp": mock_wp}):
        from ovannotators._internal.codecs import _gpu_image_codec_available

        assert _gpu_image_codec_available() is False
