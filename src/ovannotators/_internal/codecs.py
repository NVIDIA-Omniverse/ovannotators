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

"""Codec abstraction — GPU-accelerated or CPU fallback for image and video."""

from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ovannotators._internal.video_config import (
    VideoCodec,
    VideoPreset,
    validate_video_options,
    video_frame_rate,
)

# Image inputs may be NumPy arrays or any array-like that materializes to one
# (e.g. Warp CPU/CUDA arrays, which expose ``numpy()`` and the array/DLPack
# interfaces). GPU-resident inputs are encoded on-device when the active codec
# supports it; otherwise they are copied to the host first.
ArrayLike = Any


def _is_gpu_array(array: object) -> bool:
    """Return ``True`` if ``array`` is GPU-resident.

    GPU array libraries (Warp CUDA arrays, CuPy, torch CUDA tensors, ...) expose
    the ``__cuda_array_interface__`` attribute; host arrays such as NumPy and
    Warp CPU arrays do not.

    Args:
        array: Object to inspect.

    Returns:
        bool: ``True`` if the object exposes the CUDA array interface.
    """
    return hasattr(array, "__cuda_array_interface__")


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Materialize any supported array-like into a host NumPy array.

    Accepts NumPy arrays and array-likes that expose ``numpy()`` (e.g. Warp
    arrays, where CUDA arrays are copied to the host) or the buffer/array
    interface. This is the host fallback for codecs that cannot consume device
    memory directly.

    Args:
        array: NumPy array or array-like (e.g. a Warp array).

    Returns:
        np.ndarray: Host NumPy array view or copy of ``array``.
    """
    if isinstance(array, np.ndarray):
        return array
    numpy_fn = getattr(array, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn())
    return np.asarray(array)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ImageCodec(Protocol):
    def encode(self, array: ArrayLike, format: str, quality: int = 85) -> bytes:
        """Encode an image array into encoded bytes.

        Args:
            array (np.ndarray): Image array to encode.
            format (str): Target format such as ``"png"`` or ``"jpeg"``.
            quality (int): Encoder quality setting when supported.

        Returns:
            bytes: Encoded image payload.
        """
        ...

    def decode(self, data: bytes) -> np.ndarray:
        """Decode encoded image bytes into a numpy array.

        Args:
            data (bytes): Encoded image payload.

        Returns:
            np.ndarray: Decoded image array.
        """
        ...

    def write(self, path: str, array: ArrayLike, quality: int = 85) -> None:
        """Encode and write an image array to disk.

        Args:
            path (str): Destination path.
            array (ArrayLike): Image array to encode and write (NumPy or Warp).
            quality (int): Encoder quality setting when supported.
        """
        ...


class VideoEncoder(Protocol):
    def open(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Open an output video stream.

        Args:
            path (str): Destination video path.
            fps (float): Output frame rate; fractional rates such as 29.97
                are supported.
            width (int): Output width in pixels.
            height (int): Output height in pixels.
        """
        ...

    def write_frame(self, array: ArrayLike) -> None:
        """Write a single frame to the open stream.

        Args:
            array (ArrayLike): Frame image array (NumPy or Warp).
        """
        ...

    def close(self) -> bytes:
        """Close the stream and return final encoded bytes.

        Returns:
            bytes: Encoded video payload.
        """
        ...


# ---------------------------------------------------------------------------
# Image codecs
# ---------------------------------------------------------------------------


def _normalize_for_encode(array: np.ndarray, format: str) -> np.ndarray:
    """Coerce an array into a shape/channel layout image codecs accept.

    Args:
        array: Image array, possibly with a trailing singleton channel axis
            ``(H, W, 1)`` or an alpha channel ``(H, W, 4)``.
        format: Target encoding (``"jpg"``/``"jpeg"``/``"png"``/...).

    Returns:
        A contiguous array with any trailing singleton channel squeezed to
        ``(H, W)`` and, for JPEG (which has no alpha), the alpha channel
        dropped so RGBA becomes RGB.
    """
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if format in ("jpg", "jpeg") and array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)


def _normalize_gpu_image(array: ArrayLike, format: str) -> ArrayLike:
    """Apply :func:`_normalize_for_encode` channel handling on a device array.

    Mirrors :func:`_normalize_for_encode` using slicing that works on GPU
    array-likes (e.g. Warp CUDA arrays) so the data stays on the device for
    zero-copy GPU encoding.

    Args:
        array: GPU-resident image array.
        format: Target encoding (``"jpg"``/``"jpeg"``/``"png"``/...).

    Returns:
        The device array with any trailing singleton channel squeezed and, for
        JPEG (which has no alpha), the alpha channel dropped.
    """
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if format in ("jpg", "jpeg") and array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    # Channel slicing above yields a strided (non-contiguous) view, which
    # nvImageCodec rejects ("padding is only allowed for rows"), forcing a host
    # fallback. Materialize a contiguous copy on the *device* so the data stays
    # on-GPU (a cheap device-to-device copy, no host round-trip). Warp arrays
    # expose ``contiguous()``; it is a no-op when the view is already dense.
    make_contiguous = getattr(array, "contiguous", None)
    if callable(make_contiguous):
        array = make_contiguous()
    return array


@lru_cache(maxsize=8)
def _cached_call(factory: Any) -> Any:  # noqa: ANN401
    """Construct and cache the result of a zero-arg ``factory``.

    nvImageCodec ``Encoder``/``Decoder`` construction allocates GPU/codec
    resources, which is costly relative to a single encode/decode. Reusing one
    instance across calls removes that per-frame setup.

    The cache is keyed on the ``factory`` callable itself (e.g.
    ``nvidia.nvimgcodec.Encoder``). In production this object is the stable
    bindings class, so the instance is reused; tests that swap the callable
    (via ``monkeypatch`` or by patching ``sys.modules``) pass a different object
    and therefore get a freshly constructed instance, keeping test doubles
    isolated from the real singleton.

    Args:
        factory: A zero-argument callable returning the resource to cache.

    Returns:
        Any: The cached result of ``factory()``.
    """
    return factory()


class NvImageCodecBackend:
    """GPU-accelerated image codec via nvidia.nvimgcodec."""

    def encode(self, array: ArrayLike, format: str, quality: int = 85) -> bytes:
        """Encode an image array (NumPy or Warp) to encoded bytes.

        GPU-resident inputs (e.g. Warp CUDA arrays) are encoded on-device
        without a host round-trip; other inputs are copied to host NumPy first.

        Args:
            array (ArrayLike): Image array to encode (NumPy or Warp).
            format (str): Target format such as ``"png"`` or ``"jpeg"``.
            quality (int): Encoder quality setting when supported.

        Returns:
            bytes: Encoded image payload.
        """
        import nvidia.nvimgcodec as nvimgcodec  # type: ignore[import-untyped]

        codec_name = "jpeg" if format in ("jpg", "jpeg") else format
        params: nvimgcodec.EncodeParams | None = None
        if codec_name == "jpeg":
            params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.QUALITY,
                quality_value=float(quality),
            )
        encoder = _cached_call(nvimgcodec.Encoder)

        # Zero-copy GPU path: hand device-resident arrays (e.g. Warp CUDA
        # arrays) straight to nvimgcodec. Fall back to a host copy if the
        # on-device layout is unsupported by the encoder.
        if _is_gpu_array(array):
            try:
                source = _normalize_gpu_image(array, format)
                result = encoder.encode(source, codec=codec_name, params=params)
                if result is not None:
                    return bytes(result)
            except Exception:
                pass

        np_array = _normalize_for_encode(_to_numpy(array), format)
        result = encoder.encode(np_array, codec=codec_name, params=params)
        if result is None:
            raise RuntimeError(
                f"nvimgcodec failed to encode a {np_array.shape} {np_array.dtype} "
                f"array as {codec_name!r}"
            )
        return bytes(result)

    def decode(self, data: bytes) -> np.ndarray:
        """Decode encoded image bytes into a host NumPy array.

        Args:
            data (bytes): Encoded image payload (e.g. PNG/JPEG bytes).

        Returns:
            np.ndarray: Decoded image array.
        """
        import nvidia.nvimgcodec as nvimgcodec  # type: ignore[import-untyped]

        decoder = _cached_call(nvimgcodec.Decoder)
        return np.asarray(decoder.decode(data))

    def write(self, path: str, array: ArrayLike, quality: int = 85) -> None:
        """Encode an array and write it to a file.

        ``.exr`` paths are dispatched to OpenEXR; all other extensions are
        encoded via nvImageCodec.

        Args:
            path (str): Destination file path; the extension selects the format.
            array (ArrayLike): Image array to encode and write (NumPy or Warp).
            quality (int): Encoder quality setting (used for lossy formats).
        """
        ext = Path(path).suffix.lower()
        if ext == ".exr":
            _write_exr(path, array)
            return

        encoded = self.encode(array, ext.lstrip("."), quality=quality)
        Path(path).write_bytes(encoded)


class PillowCodecBackend:
    """CPU image codec via Pillow (fallback when nvimgcodec unavailable)."""

    def encode(self, array: ArrayLike, format: str, quality: int = 85) -> bytes:
        """Encode an image array (NumPy or Warp) to encoded bytes.

        Args:
            array (ArrayLike): Image array to encode (NumPy or Warp). Warp
                arrays are copied to host NumPy before CPU encoding.
            format (str): Target format such as ``"png"`` or ``"jpeg"``.
            quality (int): Encoder quality setting when supported.

        Returns:
            bytes: Encoded image payload.
        """
        from PIL import Image  # type: ignore[import-untyped]

        array = _normalize_for_encode(_to_numpy(array), format)
        img = Image.fromarray(array)
        buf = io.BytesIO()
        fmt = "JPEG" if format in ("jpg", "jpeg") else format.upper()
        kwargs: dict = {"quality": quality} if fmt == "JPEG" else {}
        img.save(buf, format=fmt, **kwargs)
        return buf.getvalue()

    def decode(self, data: bytes) -> np.ndarray:
        """Decode encoded image bytes into a host NumPy array.

        Args:
            data (bytes): Encoded image payload (e.g. PNG/JPEG bytes).

        Returns:
            np.ndarray: Decoded image array.
        """
        from PIL import Image  # type: ignore[import-untyped]

        return np.asarray(Image.open(io.BytesIO(data)))

    def write(self, path: str, array: ArrayLike, quality: int = 85) -> None:
        """Encode an array and write it to a file.

        ``.exr`` paths are dispatched to OpenEXR; all other extensions are
        encoded via Pillow.

        Args:
            path (str): Destination file path; the extension selects the format.
            array (ArrayLike): Image array to encode and write (NumPy or Warp).
            quality (int): Encoder quality setting (used for lossy formats).
        """
        ext = Path(path).suffix.lower()
        if ext == ".exr":
            _write_exr(path, array)
        else:
            from PIL import Image  # type: ignore[import-untyped]

            img = Image.fromarray(_to_numpy(array))
            fmt = "JPEG" if ext in (".jpg", ".jpeg") else None
            kwargs: dict = {"quality": quality} if fmt == "JPEG" else {}
            img.save(path, format=fmt, **kwargs)


# ---------------------------------------------------------------------------
# EXR support (via OpenEXR library)
# ---------------------------------------------------------------------------


def _normalize_exr_array(array: np.ndarray) -> np.ndarray:
    """Normalize an array to ``(H, W, C)`` with an EXR-compatible dtype.

    Args:
        array: 2-D ``(H, W)`` array (treated as a single channel) or a 3-D
            ``(H, W, C)`` array with ``C`` in ``{1, 2, 3, 4}``.

    Returns:
        A float16, float32, or uint32 ``(H, W, C)`` array. Smaller unsigned
        integers are promoted exactly to uint32.

    Raises:
        ValueError: If the array rank or channel count is unsupported.
        TypeError: If the dtype cannot be stored without losing values.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3 or array.shape[2] not in (1, 2, 3, 4):
        raise ValueError(
            "EXR write expects (H, W), (H, W, 1), (H, W, 2), (H, W, 3), or "
            f"(H, W, 4) array, got {array.shape}"
        )
    if array.dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("uint32")):
        return array
    if array.dtype.kind == "u" and array.dtype.itemsize < 4:
        return array.astype(np.uint32)
    raise TypeError(
        f"EXR write supports float16, float32, and unsigned integers up to uint32; got {array.dtype}"
    )


_EXR_COMPRESSION_TYPES = {
    "none": "NO_COMPRESSION",
    "rle": "RLE_COMPRESSION",
    "zips": "ZIPS_COMPRESSION",
    "zip": "ZIP_COMPRESSION",
    "piz": "PIZ_COMPRESSION",
    "pxr24": "PXR24_COMPRESSION",
    "b44": "B44_COMPRESSION",
    "b44a": "B44A_COMPRESSION",
    "dwaa": "DWAA_COMPRESSION",
    "dwab": "DWAB_COMPRESSION",
    "htj2k256": "HTJ2K256_COMPRESSION",
    "htj2k32": "HTJ2K32_COMPRESSION",
}


def _check_exr_compression(compression: str | None) -> str:
    """Return the lowercase scheme name, or raise for an unusable value.

    Raises:
        TypeError: If ``compression`` is neither a string nor ``None``.
        ValueError: If the name is not a supported scheme.
    """
    if compression is not None and not isinstance(compression, str):
        raise TypeError("EXR compression must be a string or None")
    name = "zip" if compression is None else compression.lower()
    if name not in _EXR_COMPRESSION_TYPES:
        choices = ", ".join(_EXR_COMPRESSION_TYPES)
        raise ValueError(f"Unsupported EXR compression '{compression}'. Use: {choices}")
    return name


def _write_exr_file(
    path: str, array: np.ndarray, compression: str | None = None
) -> None:
    """Write a float16/float32/uint32 array via the OpenEXR Python API.

    Args:
        path: Destination EXR file path.
        array: Array accepted by :func:`_normalize_exr_array`.
        compression: Optional OpenEXR compression scheme name; ``None`` uses ZIP.
    """
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    array = _normalize_exr_array(array)
    h, w, c = array.shape

    pixel_type = Imath.PixelType(
        {
            np.dtype("float16"): Imath.PixelType.HALF,
            np.dtype("float32"): Imath.PixelType.FLOAT,
            np.dtype("uint32"): Imath.PixelType.UINT,
        }[array.dtype]
    )
    channel_names = ["R", "G", "B", "A"][:c]

    header = OpenEXR.Header(w, h)
    header["channels"] = {name: Imath.Channel(pixel_type) for name in channel_names}
    name = _check_exr_compression(compression)
    constant = _EXR_COMPRESSION_TYPES[name]
    if not hasattr(Imath.Compression, constant):
        raise ValueError(f"OpenEXR does not support '{compression}' compression")
    header["compression"] = Imath.Compression(getattr(Imath.Compression, constant))

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(
        {
            name: np.ascontiguousarray(array[:, :, i]).tobytes()
            for i, name in enumerate(channel_names)
        }
    )
    out.close()


def _encode_exr_bytes(array: ArrayLike, compression: str | None = None) -> bytes:
    """Encode an array as EXR and return raw EXR bytes.

    Supports single-channel ``(H, W)`` / ``(H, W, 1)`` arrays, two-channel
    ``(H, W, 2)`` arrays, ``(H, W, 3)`` RGB, and ``(H, W, 4)`` RGBA arrays.
    Accepts NumPy or Warp arrays; Warp arrays are copied to host NumPy first
    (EXR has no on-device encoder here).
    """
    array = _normalize_exr_array(_to_numpy(array))

    # OpenEXR's public Python API is path-oriented in many environments, so
    # encode via a temporary file and read the resulting bytes back.
    with tempfile.TemporaryDirectory(prefix="ovannotators_exr_") as tmpdir:
        tmp_path = Path(tmpdir) / "frame.exr"
        _write_exr_file(str(tmp_path), array, compression=compression)
        return tmp_path.read_bytes()


def _write_exr(path: str, array: ArrayLike) -> None:
    """Write a float16/float32/uint32 array as an EXR file.

    Supports single-channel ``(H, W)`` / ``(H, W, 1)``, two-channel
    ``(H, W, 2)``, ``(H, W, 3)`` RGB, and ``(H, W, 4)`` RGBA arrays (NumPy or
    Warp).
    """
    _write_exr_file(path, _to_numpy(array))


def read_exr(path: str) -> np.ndarray:
    """Read an EXR file into an ``(H, W, C)`` float16/32/uint32 array.

    Channels are returned in RGBA order (subset thereof, depending on which are
    present).

    The stored channel type is preserved for files with matching channel types.

    Args:
        path (str): Path to the EXR file to read.

    Returns:
        np.ndarray: ``(H, W, C)`` float16/32/uint32 array in RGBA order.
    """
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    inp = OpenEXR.InputFile(path)
    dw = inp.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = list(inp.header()["channels"].keys())
    # Ensure RGBA ordering
    ordered = []
    for name in ["R", "G", "B", "A"]:
        if name in channels:
            ordered.append(name)

    arrays = []
    pixel_types = {inp.header()["channels"][name].type.v for name in ordered}
    if len(pixel_types) != 1:
        raise ValueError("EXR channels must have the same pixel type")
    stored_type = pixel_types.pop()
    if stored_type == Imath.PixelType.HALF:
        dtype = np.float16
    elif stored_type == Imath.PixelType.FLOAT:
        dtype = np.float32
    elif stored_type == Imath.PixelType.UINT:
        dtype = np.uint32
    else:
        raise ValueError(f"Unsupported EXR pixel type: {stored_type}")
    pixel_type = Imath.PixelType(stored_type)
    for name in ordered:
        data = inp.channel(name, pixel_type)
        arrays.append(np.frombuffer(data, dtype=dtype).reshape(h, w))

    return np.stack(arrays, axis=-1)


# ---------------------------------------------------------------------------
# Video encoders
# ---------------------------------------------------------------------------


class _NvDeviceFrame:
    """Own a packed Warp surface and expose PyNvVideoCodec's plane interface."""

    def __init__(self, array, *, planar=False):
        self.array = array
        # Planar formats require one CUDA interface per plane, not the packed
        # surface interface used by ARGB. Retain all views and their owner.
        height = array.shape[0] // 3 if planar else array.shape[0]
        self._planes = (
            [array[i * height : (i + 1) * height] for i in range(3)]
            if planar
            else array
        )

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def cuda(self):
        return self._planes


def _crop_is_even(array: Any, height: int, width: int) -> bool:
    """Whether cropping ``array`` to even dimensions yields ``(height, width)``."""
    h, w = array.shape[:2]
    return (h, w) != (height, width) and (h - h % 2, w - w % 2) == (height, width)


class NvVideoEncoder:
    """GPU-accelerated video encoder via PyNvVideoCodec."""

    def __init__(
        self,
        *,
        codec: VideoCodec = "h264",
        lossless: bool = False,
        preset: VideoPreset | None = None,
    ) -> None:
        """Initialize an unopened encoder.

        Call :meth:`open` before :meth:`write_frame`.

        Args:
            codec: H.264 or HEVC.
            lossless: Preserve normalized RGB8 exactly; requires HEVC.
            preset: Optional NVENC speed/compression preset; lossless defaults to P5.
        """
        validate_video_options(codec, lossless, preset)
        self._codec, self._lossless, self._preset = codec, lossless, preset
        self._encoder = None
        self._opened = False
        self._device = None
        self._input_frame = None
        self._stream = None
        self._fps: float = 30
        self._width: int = 0
        self._height: int = 0
        self._chunks: list[bytes] = []
        self._frame_count = 0

    def open(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Prepare an H.264 or HEVC stream; initialize on the first frame.

        Args:
            path (str): Destination video path (recorded for the caller; frames
                are buffered in memory until :meth:`close`).
            fps (float): Output frame rate; fractional rates such as 29.97
                are supported.
            width (int): Output width in pixels.
            height (int): Output height in pixels.

        Raises:
            ValueError: If the frame rate is not positive and finite, or the
                dimensions are not positive integers.
        """
        video_frame_rate(fps)
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
            for v in (width, height)
        ):
            raise ValueError("Dimensions must be positive integers")
        if self._lossless and (width % 2 or height % 2):
            raise ValueError(
                "Lossless HEVC requires even dimensions; frames are never cropped"
            )
        # Input residency is known only when the first frame arrives.
        self._fps = fps
        self._width = width
        self._height = height
        self._chunks = []
        self._frame_count = 0
        self._encoder = None
        self._device = None
        self._input_frame = None
        self._stream = None
        self._opened = True

    def _create_encoder(self, device=None) -> None:
        import PyNvVideoCodec

        # The in-process muxer requires packets in display order.
        # NVENC takes an integer rate for rate control only; the MP4 container
        # carries the exact (possibly fractional) frame timing.
        options = dict(codec=self._codec, fps=str(max(1, round(self._fps))), bf=0)
        if self._preset is not None:
            options["preset"] = self._preset
        if self._lossless:
            options.update(tuning_info="lossless", preset=self._preset or "P5")
        if device is not None:
            import warp as wp

            self._stream = wp.get_stream(device)
            options.update(
                cudacontext=device.context,
                cudastream=self._stream.cuda_stream,
                gpu_id=device.ordinal,
            )
        try:
            self._encoder = PyNvVideoCodec.CreateEncoder(
                self._width,
                self._height,
                "YUV444" if self._lossless else "ARGB",
                device is None,
                **options,
            )
        except Exception as exc:
            # Native initialization can raise PyNvVCException (not necessarily
            # RuntimeError), e.g. missing driver, unsupported NVENC, or no session.
            raise RuntimeError(
                f"NVENC {self._codec} initialization failed (lossless={self._lossless}): {exc}"
            ) from exc
        self._device = device

    def _append_packets(self, encoded: object) -> None:
        """Buffer whatever ``Encode``/``EndEncode`` handed back.

        PyNvVideoCodec changed this return type: 2.1 and earlier return the
        Annex-B bytes directly, while 2.2 returns a list of packet mappings
        carrying the bytes under ``data`` alongside ``timestamp`` and
        ``picture_type``. Accept both so the encoder works across either.

        Args:
            encoded: The return value of ``Encode`` or ``EndEncode``: a
                bytes-like object, or an iterable of packet mappings.
        """
        if not encoded:
            return
        if isinstance(encoded, (bytes, bytearray, memoryview)):
            self._chunks.append(bytes(encoded))
            return
        for packet in encoded:
            data = packet["data"] if isinstance(packet, dict) else packet
            if data:
                self._chunks.append(bytes(data))

    def write_frame(self, array: ArrayLike) -> None:
        """Encode and buffer a single frame.

        RGB(A) inputs are packed into NVENC's ARGB (B, G, R, A byte order)
        buffer format before lossy encoding. Lossless HEVC accepts normalized
        uint8 RGB and rearranges it into full-resolution G/B/R planes without
        color conversion.

        The encoding session's residency is fixed by the first frame. A
        session started by a CUDA frame uploads later host frames; a session
        started by a host frame downloads later CUDA frames, because a second
        NVENC session would emit parameter sets the MP4 muxer cannot merge.

        Args:
            array (ArrayLike): Frame image array (NumPy or Warp). CUDA frames
                must be ready on their device's current Warp stream; writers
                synchronize borrowed renderer buffers when they import them.
                Lossy CUDA frames with odd dimensions are cropped to the
                encoder size on-device.

        Raises:
            RuntimeError: If :meth:`open` has not been called.
        """
        if not self._opened:
            raise RuntimeError(
                "NvVideoEncoder.open() must be called before write_frame()"
            )
        from ovannotators._internal.video_frames import (
            as_warp,
            device_gbr,
            device_rgb,
            is_device_frame,
        )

        if self._lossless:
            import warp as wp

            if is_device_frame(array):
                array = as_warp(array)
            if array.shape != (self._height, self._width, 3) or array.dtype not in (
                np.uint8,
                wp.uint8,
            ):
                raise ValueError("Lossless video requires fixed-size uint8 RGB frames")

        host_session = self._encoder is not None and self._device is None
        if not host_session and (is_device_frame(array) or self._device is not None):
            import warp as wp

            if not is_device_frame(array):
                array = wp.array(_to_numpy(array), device=self._device)
            frame = (
                device_gbr(array)
                if self._lossless
                else device_rgb(
                    array,
                    bgra=True,
                    crop_even=_crop_is_even(array, self._height, self._width),
                    synchronize=False,
                )
            )
            if not self._lossless and tuple(frame.shape[:2]) != (
                self._height,
                self._width,
            ):
                raise ValueError(
                    f"frame shape {frame.shape} must match encoder {(self._height, self._width)}"
                )
            if self._encoder is None:
                self._create_encoder(frame.device)
            elif self._device != frame.device:
                frame = frame.to(self._device)
            # Packing ran on the device's current Warp stream. Order NVENC's
            # stream after it without stalling unrelated device work.
            current = wp.get_stream(frame.device)
            if current is not self._stream:
                self._stream.wait_stream(current)
            # Encode is synchronous; retain the input owner through the call.
            wrapped = _NvDeviceFrame(frame, planar=self._lossless)
            self._append_packets(self._encoder.Encode(wrapped))
            wp.synchronize_stream(self._stream)
            self._input_frame = wrapped
            self._frame_count += 1
            return
        if self._encoder is None:
            self._create_encoder()
        array = _to_numpy(as_warp(array) if is_device_frame(array) else array)
        if not self._lossless and _crop_is_even(array, self._height, self._width):
            array = array[: self._height, : self._width]
        # NVENC's "ARGB" buffer format is A8R8G8B8, i.e. a 32-bit little-endian
        # word, so the in-memory byte order is B, G, R, A. Pack RGB(A) inputs
        # into that order; otherwise channels are swizzled (e.g. a black frame
        # would encode as solid blue).
        if self._lossless:
            frame = np.ascontiguousarray(array[:, :, [1, 2, 0]].transpose(2, 0, 1))
        elif array.ndim == 3 and array.shape[2] in (3, 4):
            h, w = array.shape[:2]
            bgra = np.empty((h, w, 4), dtype=np.uint8)
            bgra[:, :, 0] = array[:, :, 2]  # B
            bgra[:, :, 1] = array[:, :, 1]  # G
            bgra[:, :, 2] = array[:, :, 0]  # R
            bgra[:, :, 3] = array[:, :, 3] if array.shape[2] == 4 else 255  # A
            frame = bgra
        else:
            frame = array
        self._append_packets(self._encoder.Encode(frame))
        self._frame_count += 1

    def close(self) -> bytes:
        """Finish encoding and return the muxed MP4 bytes.

        Flushes any remaining frames, then muxes the raw Annex-B stream
        into an in-memory MP4 container (no FFmpeg/PyAV).

        Returns:
            bytes: The encoded ``.mp4`` payload, or empty bytes if no frames
            were written.

        Raises:
            Exception: Whatever ``EndEncode()`` raised. A failed flush would
                otherwise return a silently truncated stream. The encoder is
                closed and its buffered data is discarded either way.
        """
        try:
            if self._encoder is not None:
                self._append_packets(self._encoder.EndEncode())
        except BaseException:
            self._chunks = []
            raise
        finally:
            self._encoder = None
            self._opened = False
            self._input_frame = None
            self._stream = None
        payload = b"".join(self._chunks)
        if not payload and self._lossless and self._frame_count:
            raise RuntimeError("NVENC produced no data for lossless video frames")
        if not payload:
            return payload
        # NVENC emits a raw Annex-B elementary stream, which is not a
        # playable container. Mux it into MP4 in-process (no FFmpeg/PyAV) so
        # the written ``.mp4`` files are valid for standard players.
        from ovannotators._internal.mp4_mux import mux_h264_annexb, mux_hevc_annexb

        if self._codec == "hevc":
            return mux_hevc_annexb(
                payload, self._fps, self._width, self._height, rgb=self._lossless
            )

        return mux_h264_annexb(
            payload, fps=self._fps, width=self._width, height=self._height
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _gpu_image_codec_available() -> bool:
    """Report whether GPU image encode/decode is usable on this machine.

    Requires both a CUDA device (probed via Warp, a core dependency) and the
    nvImageCodec bindings. The CUDA check matters because the bindings can be
    importable on a host that has no usable driver/device, in which case GPU
    encoding would fail at runtime.

    Returns:
        bool: ``True`` if a CUDA device and ``nvidia.nvimgcodec`` are both
        available, otherwise ``False``.
    """
    try:
        import warp as wp

        if wp.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    from ovannotators._internal.optional_deps import nvimgcodec_available

    return nvimgcodec_available()


def get_image_codec() -> ImageCodec:
    """Select the best available image codec, preferring the GPU.

    nvImageCodec is used whenever a usable CUDA device and the bindings are
    present (see :func:`_gpu_image_codec_available`), since GPU encode/decode is
    substantially faster and keeps GPU-resident arrays on-device. Pillow (a core
    dependency) is the host fallback otherwise.

    Returns:
        ImageCodec: An :class:`NvImageCodecBackend` when GPU encoding is
        available, otherwise a :class:`PillowCodecBackend`.
    """
    if _gpu_image_codec_available():
        return NvImageCodecBackend()
    return PillowCodecBackend()


def get_video_encoder(
    path: str,
    fps: float,
    w: int,
    h: int,
    *,
    codec: VideoCodec = "h264",
    lossless: bool = False,
    preset: VideoPreset | None = None,
) -> VideoEncoder:
    """Create and open a GPU video encoder.

    Args:
        path (str): Destination video path.
        fps (float): Output frame rate; fractional rates such as 29.97 are
            supported.
        w (int): Output width in pixels.
        h (int): Output height in pixels.
        codec: H.264 (default) or HEVC. HEVC requires NVENC, without CPU fallback.
        lossless: Preserve normalized RGB8 exactly; requires HEVC.
        preset: Explicit NVENC speed/compression preset.

    Returns:
        VideoEncoder: An opened :class:`NvVideoEncoder` ready for
        :meth:`~NvVideoEncoder.write_frame`.

    Raises:
        ImportError: If ``PyNvVideoCodec`` is not installed.
    """
    validate_video_options(codec, lossless, preset)
    try:
        import PyNvVideoCodec  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyNvVideoCodec is required for video encoding. "
            "Install it with: pip install PyNvVideoCodec"
        ) from e
    enc: VideoEncoder = NvVideoEncoder(codec=codec, lossless=lossless, preset=preset)
    enc.open(path, fps, w, h)
    return enc
