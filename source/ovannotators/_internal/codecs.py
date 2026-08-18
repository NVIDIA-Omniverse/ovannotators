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
        fps: int,
        width: int,
        height: int,
    ) -> None:
        """Open an output video stream.

        Args:
            path (str): Destination video path.
            fps (int): Output frame rate.
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
    """Normalize an array to ``(H, W, C)`` float16 with ``C`` in ``{1,2,3,4}``.

    Args:
        array: 2-D ``(H, W)`` array (treated as a single channel) or a 3-D
            ``(H, W, C)`` array with ``C`` in ``{1, 2, 3, 4}``.

    Returns:
        A float16 ``(H, W, C)`` array suitable for EXR half-float storage.

    Raises:
        ValueError: If the array rank or channel count is unsupported.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3 or array.shape[2] not in (1, 2, 3, 4):
        raise ValueError(
            "EXR write expects (H, W), (H, W, 1), (H, W, 2), (H, W, 3), or "
            f"(H, W, 4) array, got {array.shape}"
        )
    if array.dtype != np.float16:
        array = array.astype(np.float16)
    return array


def _write_exr_file(path: str, array: np.ndarray) -> None:
    """Write a float16/float32 array as an EXR file via the OpenEXR Python API."""
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    array = _normalize_exr_array(array)
    h, w, c = array.shape

    pixel_type = Imath.PixelType(Imath.PixelType.HALF)
    channel_names = ["R", "G", "B", "A"][:c]

    header = OpenEXR.Header(w, h)
    header["channels"] = {name: Imath.Channel(pixel_type) for name in channel_names}

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(
        {
            name: np.ascontiguousarray(array[:, :, i]).tobytes()
            for i, name in enumerate(channel_names)
        }
    )
    out.close()


def _encode_exr_bytes(array: ArrayLike) -> bytes:
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
        _write_exr_file(str(tmp_path), array)
        return tmp_path.read_bytes()


def _write_exr(path: str, array: ArrayLike) -> None:
    """Write a float16/float32 array as an EXR file.

    Supports single-channel ``(H, W)`` / ``(H, W, 1)``, two-channel
    ``(H, W, 2)``, ``(H, W, 3)`` RGB, and ``(H, W, 4)`` RGBA arrays (NumPy or
    Warp).
    """
    _write_exr_file(path, _to_numpy(array))


def read_exr(path: str) -> np.ndarray:
    """Read an EXR file into an ``(H, W, C)`` float16 array.

    Channels are returned in RGBA order (subset thereof, depending on which are
    present).

    Note:
        Every channel is read as half-float (``Imath.PixelType.HALF``)
        regardless of how it was stored. This matches how ovannotators writes
        EXRs (:func:`_write_exr_file` stores all channels as ``HALF``). A file
        authored elsewhere with 32-bit ``FLOAT`` channels would be reinterpreted
        incorrectly; such files are not produced or expected by this library.

    Args:
        path (str): Path to the EXR file to read.

    Returns:
        np.ndarray: ``(H, W, C)`` float16 array with channels in RGBA order.
    """
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    inp = OpenEXR.InputFile(path)
    dw = inp.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    half = Imath.PixelType(Imath.PixelType.HALF)
    channels = list(inp.header()["channels"].keys())
    # Ensure RGBA ordering
    ordered = []
    for name in ["R", "G", "B", "A"]:
        if name in channels:
            ordered.append(name)

    arrays = []
    for name in ordered:
        data = inp.channel(name, half)
        arrays.append(np.frombuffer(data, dtype=np.float16).reshape(h, w))

    return np.stack(arrays, axis=-1)


# ---------------------------------------------------------------------------
# Video encoders
# ---------------------------------------------------------------------------


class NvVideoEncoder:
    """GPU-accelerated video encoder via PyNvVideoCodec."""

    def __init__(self) -> None:
        """Initialize an unopened encoder.

        Call :meth:`open` before :meth:`write_frame`.
        """
        self._encoder = None
        self._fps: int = 30
        self._width: int = 0
        self._height: int = 0
        self._chunks: list[bytes] = []

    def open(
        self,
        path: str,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        """Open an H.264 output stream and prepare for frame writes.

        Args:
            path (str): Destination video path (recorded for the caller; frames
                are buffered in memory until :meth:`close`).
            fps (int): Output frame rate.
            width (int): Output width in pixels.
            height (int): Output height in pixels.
        """
        import PyNvVideoCodec  # type: ignore[import-untyped]

        self._fps = fps
        self._width = width
        self._height = height
        self._chunks = []
        # bf=0 → no B-frames, so the elementary stream is in display order and
        # can be muxed into MP4 without composition-time reordering tables.
        enc_kwargs: dict[str, object] = dict(codec="h264", fps=str(fps), bf=0)
        self._encoder = PyNvVideoCodec.CreateEncoder(
            width, height, "ARGB", True, **enc_kwargs
        )

    def write_frame(self, array: ArrayLike) -> None:
        """Encode and buffer a single frame.

        RGB(A) inputs are packed into NVENC's ARGB (B, G, R, A byte order)
        buffer format before encoding.

        Args:
            array (ArrayLike): Frame image array (NumPy or Warp). Warp arrays
                are copied to host NumPy first.

        Raises:
            RuntimeError: If :meth:`open` has not been called.
        """
        if self._encoder is None:
            raise RuntimeError(
                "NvVideoEncoder.open() must be called before write_frame()"
            )
        array = _to_numpy(array)
        # NVENC's "ARGB" buffer format is A8R8G8B8, i.e. a 32-bit little-endian
        # word, so the in-memory byte order is B, G, R, A. Pack RGB(A) inputs
        # into that order; otherwise channels are swizzled (e.g. a black frame
        # would encode as solid blue).
        if array.ndim == 3 and array.shape[2] in (3, 4):
            h, w = array.shape[:2]
            bgra = np.empty((h, w, 4), dtype=np.uint8)
            bgra[:, :, 0] = array[:, :, 2]  # B
            bgra[:, :, 1] = array[:, :, 1]  # G
            bgra[:, :, 2] = array[:, :, 0]  # R
            bgra[:, :, 3] = array[:, :, 3] if array.shape[2] == 4 else 255  # A
            frame = bgra
        else:
            frame = array
        chunk = self._encoder.Encode(frame)
        if chunk:
            self._chunks.append(bytes(chunk) if not isinstance(chunk, bytes) else chunk)

    def close(self) -> bytes:
        """Finish encoding and return the muxed MP4 bytes.

        Flushes any remaining frames, then muxes the raw H.264 Annex-B stream
        into an in-memory MP4 container (no FFmpeg/PyAV).

        Returns:
            bytes: The encoded ``.mp4`` payload, or empty bytes if no frames
            were written.
        """
        if self._encoder is not None:
            try:
                remaining = self._encoder.EndEncode()
                if remaining:
                    self._chunks.append(
                        bytes(remaining)
                        if not isinstance(remaining, bytes)
                        else remaining
                    )
            except Exception:
                pass
            self._encoder = None
        payload = b"".join(self._chunks)
        if not payload:
            return payload
        # NVENC emits a raw H.264 Annex-B elementary stream, which is not a
        # playable container. Mux it into MP4 in-process (no FFmpeg/PyAV) so
        # the written ``.mp4`` files are valid for standard players.
        from ovannotators._internal.mp4_mux import mux_h264_annexb

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
    fps: int,
    w: int,
    h: int,
    **kwargs,
) -> VideoEncoder:
    """Create and open a GPU video encoder.

    Args:
        path (str): Destination video path.
        fps (int): Output frame rate.
        w (int): Output width in pixels.
        h (int): Output height in pixels.
        **kwargs: Reserved for future encoder options; currently unused.

    Returns:
        VideoEncoder: An opened :class:`NvVideoEncoder` ready for
        :meth:`~NvVideoEncoder.write_frame`.

    Raises:
        ImportError: If ``PyNvVideoCodec`` is not installed.
    """
    try:
        import PyNvVideoCodec  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyNvVideoCodec is required for video encoding. "
            "Install it with: pip install PyNvVideoCodec"
        ) from e
    enc: VideoEncoder = NvVideoEncoder()
    enc.open(path, fps, w, h)
    return enc
