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

"""OvrtxOutput — AOV extraction from ovrtx FrameOutput via DLPack.

The public camera-output catalog and runtime-tested support matrix live at
https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/outputs.html.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import warp as wp

from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.exceptions import ExtractionError, ValidationError

try:
    import ovrtx
    from ovrtx import Device
except ImportError:
    ovrtx = None  # type: ignore[assignment]
    Device = None  # type: ignore[assignment]

# Guard on the real renderer API, not bare importability: a content-less stub
# ``ovrtx`` can import without exposing ``Renderer``.
_OVRTX_AVAILABLE = ovrtx_available()

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ovrtx emits some annotators as opaque byte buffers (a packed struct table
# plus a trailing entry count) rather than ready-to-use arrays. Decoding those
# payloads into the structured arrays writers consume is ovrtx-specific, so it
# belongs here in the extractor — a writer would otherwise have to know which
# renderer produced its input. Maps each ovrtx ``sourceName`` to its decoder.
_ANNOTATOR_DECODERS: dict[str, Any] | None = None


def _annotator_decoders() -> dict[str, Any]:
    """Return the ``{sourceName: decoder}`` map, importing decoders lazily.

    The decoders live in :mod:`ovannotators.writers._helpers`; importing them
    lazily avoids eagerly loading the writers package while the
    ``renderer_outputs`` package is still initializing.

    Returns:
        Mapping of ovrtx ``sourceName`` to the callable that decodes its raw
        payload into the structured array (or dict) writers expect.
    """
    global _ANNOTATOR_DECODERS
    if _ANNOTATOR_DECODERS is None:
        from ovannotators.writers._helpers import (
            decode_bbox_2d,
            decode_bbox_3d,
            decode_camera_params,
            decode_id_string_map,
            decode_instance_map,
            decode_semantic_id_map,
            decode_stable_id_semantic_id_map,
        )

        _ANNOTATOR_DECODERS = {
            "SemanticBoundingBox2DTight": decode_bbox_2d,
            "SemanticBoundingBox2DLoose": decode_bbox_2d,
            "SemanticBoundingBox3D": decode_bbox_3d,
            "SemanticIdMap": decode_semantic_id_map,
            "StableIdMap": decode_id_string_map,
            "StableIdSemanticIdMap": decode_stable_id_semantic_id_map,
            "InstanceMap": decode_instance_map,
            "CameraParams": decode_camera_params,
        }
    return _ANNOTATOR_DECODERS


class OvrtxOutput:
    """Extracts and interprets AOV data from ovrtx FrameOutput.

    Handles DLPack mapping, tensor copying, and ovrtx-specific AOV
    interpretation (dtype normalization, struct extraction).
    """

    renderer_name = "ovrtx"

    def extract(
        self,
        frame_output: Any,
        var_name: str,
        device: Literal["cpu", "cuda"] = "cuda",
        decode: bool = True,
    ) -> Any:
        """Extract a named render var from an ovrtx frame.

        ovrtx maps the var to host/CUDA memory and exposes it via DLPack; this
        wraps that buffer as a zero-copy view without routing through NumPy for
        the tensor data.

        Single-tensor image AOVs (``LdrColor``, ``DistanceToCameraSD``, ...) are
        returned as a Warp array. Multi-tensor or composite/sensor outputs
        (``PointCloud``, ``GenericModelOutput``, ...) are returned as a nested
        ``{"tensors": {...}, "params": {...}}`` dict, where each tensor is a
        Warp array and each param is a NumPy array (params are small CPU-side
        scalars).

        ovrtx delivers bounding boxes, camera parameters, semantic/stable ID
        maps, ``StableIdSemanticIdMap``, and ``InstanceMap`` as opaque byte
        buffers rather than ready-to-use arrays. When ``decode`` is ``True``
        (the default) these are decoded into the structured NumPy arrays or
        dictionaries the writers consume, so the result can be handed straight
        to a writer. Pass ``decode=False`` to receive the raw buffer instead.

        The returned views keep the underlying ovrtx mapping alive via DLPack
        reference counting: ovrtx defers the buffer's release until the last
        consumer view is dropped, so the data stays valid after the ``map()``
        context exits and no manual lifetime management is required. For CUDA
        extraction the mapping is synchronized before returning, so the data is
        valid for any subsequent access.

        Args:
            frame_output: ovrtx ``FrameOutput`` holding the render vars.
            var_name: ovrtx ``sourceName`` of the render var to extract.
            device: Extraction device. ``"cuda"`` (the default) returns CUDA
                views that keep the data on the device where ovrtx rendered it,
                so it can feed the GPU image codec zero-copy; ``params`` remain
                on the CPU. ``"cpu"`` returns host views, forcing a
                device-to-host copy. ``OvrtxOutput`` only runs where the ovrtx
                renderer (hence a CUDA device) is present, so ``"cuda"`` is the
                natural default.
            decode: When ``True``, decode ovrtx annotator byte buffers (bounding
        boxes, ID maps, instance maps, camera params) into the structured
        arrays or dictionaries writers expect.
                Non-annotator vars are unaffected. Decoded annotator arrays are
                small CPU-side metadata regardless of ``device``.

        Returns:
            For a single-tensor render var, the data as a Warp array on the
            requested device (or a decoded structured NumPy array for annotator
            vars when ``decode`` is ``True``). For a multi-tensor/composite
            render var, a ``{"tensors": dict[str, warp array], "params":
            dict[str, numpy array]}`` mapping.

        Raises:
            ImportError: If ovrtx is not installed.
            ValueError: If ``device`` is unsupported.
            ValidationError: If ``var_name`` is absent from the frame.
            ExtractionError: If the var is present but cannot be mapped/decoded.
        """
        if not _OVRTX_AVAILABLE:
            raise ImportError("ovrtx is not installed. Install it to use OvrtxOutput.")
        if device not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported device '{device}'. Expected 'cpu' or 'cuda'."
            )
        raw_var = frame_output.render_vars.get(var_name)
        if raw_var is None:
            # The var simply isn't in this frame: a validation problem, not a
            # decode failure.
            raise ValidationError(render_var_name=var_name)
        decoder = _annotator_decoders().get(var_name) if decode else None
        map_device = Device.CPU if device == "cpu" else Device.CUDA
        try:
            with raw_var.map(device=map_device) as mapping:
                if device == "cuda":
                    # Producer CUDA work may still be in flight on map() return.
                    # Block until it completes so views are safe for any access.
                    mapping.wait()
                result = self._extract_mapping(mapping)
                # Decode while the mapping is still alive; decoders copy out of
                # the buffer, so the result stays valid once it is released.
                if decoder is not None:
                    result = decoder(result)
                return result
        except (ValidationError, ExtractionError):
            raise
        except Exception as exc:
            # The var exists but mapping/decoding it failed: genuine extraction
            # error (malformed buffer, unsupported dtype, DLPack failure, ...).
            raise ExtractionError(render_var_name=var_name, message=str(exc)) from exc

    def _extract_mapping(self, mapping: Any) -> Any:
        """Build the return value from a mapped ovrtx render var.

        Single-tensor render vars with no params yield a zero-copy Warp array.
        Multi-tensor or composite/sensor render vars (and params-only vars)
        yield a nested ``{"tensors": {...}, "params": {...}}`` dict.

        Args:
            mapping: An ovrtx ``MappedRenderVar`` (already synchronized for
                CUDA mappings).

        Returns:
            A Warp array for single-tensor vars, otherwise the nested
            ``{"tensors": dict[str, warp array], "params": dict[str, numpy
            array]}`` mapping.
        """
        params = mapping.params
        if len(mapping) == 1 and not params:
            # Single-tensor image AOV: return the zero-copy Warp view directly.
            # ovrtx keeps the buffer mapped until this consumer is dropped, so
            # it stays valid after the map() context exits.
            return wp.from_dlpack(mapping)

        # Multi-tensor / composite (sensor) output: expose each channel and
        # param by name. Each view incref's the parent mapping via DLPack, so
        # the buffers stay valid past the map() context just like the
        # single-tensor path.
        tensors = {name: wp.from_dlpack(mapping[name]) for name in mapping.keys()}
        param_arrays = {name: np.from_dlpack(value) for name, value in params.items()}
        return {"tensors": tensors, "params": param_arrays}

    def extract_all(
        self,
        frame_output: Any,
        var_names: list[str] | None = None,
        device: Literal["cpu", "cuda"] = "cuda",
        decode: bool = True,
    ) -> dict[str, Any]:
        """Extract all render vars in a frame, or only a selected subset.

        Each value is a zero-copy view (or nested structure of views) that keeps
        its ovrtx mapping alive until dropped (see :meth:`extract`); no manual
        lifetime management is required.

        Args:
            frame_output: ovrtx ``FrameOutput`` holding the render vars.
            var_names: Optional ovrtx ``sourceName``s to extract. If omitted,
                all names from :meth:`available_vars` are extracted. In this
                auto-all mode, vars that fail extraction are skipped.
            device: Extraction device forwarded to :meth:`extract`.
            decode: Forwarded to :meth:`extract`. When ``True`` (the default),
                ovrtx annotator byte buffers (bounding boxes, camera params) are
                decoded into the structured arrays writers expect, so the
                returned dict can be passed straight to ``writer.write_frame``.

        Returns:
            Mapping of ``sourceName`` to its extracted value. Each value is a
            Warp array for single-tensor vars, a decoded structured NumPy array
            for annotator vars (when ``decode`` is ``True``), or a
            ``{"tensors": ..., "params": ...}`` dict for multi-tensor/composite
            (sensor) vars (see :meth:`extract`).
        """
        explicit_var_names = var_names is not None
        names = self.available_vars(frame_output) if var_names is None else var_names
        extracted: dict[str, Any] = {}
        for name in names:
            try:
                extracted[name] = self.extract(
                    frame_output, name, device=device, decode=decode
                )
            except Exception as exc:
                if explicit_var_names:
                    raise
                logger.warning(
                    "Skipping render var '%s' in extract_all auto mode "
                    "(could not extract): %s",
                    name,
                    exc,
                )
                logger.debug(
                    "Traceback for skipped render var '%s'.", name, exc_info=True
                )
        return extracted

    def available_vars(self, frame_output: Any) -> list[str]:
        """List the render var names present in an ovrtx frame output.

        Args:
            frame_output (Any): An ovrtx ``FrameOutput`` object.

        Returns:
            list[str]: Names of the render vars available for extraction.
        """
        return list(frame_output.render_vars.keys())
