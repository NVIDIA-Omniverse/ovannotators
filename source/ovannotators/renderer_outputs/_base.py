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

"""RendererOutput Protocol — renderer-agnostic interface for AOV extraction."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable


@runtime_checkable
class RendererOutput(Protocol):
    """Renderer-agnostic interface for extracting AOV data from a frame.

    Each renderer provides its own implementation that handles DLPack mapping,
    dtype normalization, coordinate transforms, and struct extraction.
    Writers receive clean, renderer-agnostic arrays.
    """

    renderer_name: ClassVar[str]

    def extract(
        self,
        frame_output: Any,
        var_name: str,
        device: Literal["cpu", "cuda"] = "cpu",
        decode: bool = True,
    ) -> Any:
        """Extract a named render var from the renderer's frame output.

        Args:
            frame_output (Any): Renderer-specific frame output object.
            var_name (str): Name of the render var to extract.
            device (Literal['cpu', 'cuda']): Target device for returned views;
                ``"cpu"`` for host arrays, ``"cuda"`` for device arrays.
            decode (bool): When ``True``, decode any renderer-specific opaque
                payloads (e.g. annotator byte buffers) into the structured
                arrays writers consume. Pass ``False`` to receive the raw
                payload instead. Vars without such payloads are unaffected.

        Returns:
            Any: The extracted render var. Implementations may return a single
                array for single-tensor vars, or a nested
                ``{"tensors": {...}, "params": {...}}`` mapping for
                multi-tensor/composite (e.g. sensor) vars.
        """
        ...  # pragma: no cover

    def extract_all(
        self,
        frame_output: Any,
        var_names: list[str] | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        decode: bool = True,
    ) -> dict[str, Any]:
        """Extract all render vars in the frame, or an explicit subset.

        Args:
            frame_output (Any): Renderer-specific frame output object.
            var_names (list[str] | None): Optional render-var names to extract.
                If ``None``, implementations should extract every var returned
                by :meth:`available_vars` and may skip vars they cannot
                process.
            device (Literal['cpu', 'cuda']): Extraction device.
            decode (bool): Forwarded to :meth:`extract`. When ``True``, decode
                renderer-specific opaque payloads into the structured arrays
                writers consume; pass ``False`` to receive raw payloads.

        Returns:
            dict[str, Any]: Mapping from render-var name to extracted value.
                Each value follows :meth:`extract`: a single array for
                single-tensor vars, or a nested
                ``{"tensors": {...}, "params": {...}}`` mapping for
                multi-tensor/composite (e.g. sensor) vars.
        """
        ...

    def available_vars(self, frame_output: Any) -> list[str]:
        """List the render var names present in a frame output.

        Args:
            frame_output (Any): Renderer-specific frame output object.

        Returns:
            list[str]: Names of the render vars available for extraction.
        """
        ...
