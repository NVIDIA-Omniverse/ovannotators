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

"""Library-wide exception hierarchy for ovannotators."""

from __future__ import annotations


class OVAnnotatorsError(Exception):
    """Base class for every exception raised by ovannotators.

    Catch this to handle any library-specific failure uniformly while letting
    unrelated exceptions propagate.
    """


class ConfigurationError(OVAnnotatorsError):
    """Raised when a component is configured with invalid values.

    This signals a programming/setup error detected at construction time (e.g.
    an unknown plugin name or contradictory config fields), as opposed to a
    per-frame data problem.
    """


class ExtractionError(OVAnnotatorsError):
    """Raised when a renderer output cannot decode a present render var.

    This covers genuine mapping/decode failures (unknown annotator, malformed
    buffer, DLPack import error). A var that is simply absent from the frame is
    reported as :class:`ValidationError` instead.
    """

    def __init__(
        self,
        render_var_name: str,
        frame_id: int | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            render_var_name (str): Name of the render var that failed to decode.
            frame_id (int | None): Frame index where the failure occurred, if
                known. Included in the message when provided.
            message (str | None): Optional custom message; a default is built
                from ``render_var_name``/``frame_id`` when omitted.
        """
        self.render_var_name = render_var_name
        self.frame_id = frame_id
        msg = message or f"Failed to extract render var '{render_var_name}'"
        if frame_id is not None:
            msg += f" at frame {frame_id}"
        super().__init__(msg)


class ValidationError(OVAnnotatorsError):
    """Raised when a frame is missing a render var that was expected.

    This is a per-frame runtime check. Renderer outputs raise it when a
    requested var is absent from the ``FrameOutput``; writers raise its
    :class:`MissingRenderVarError` subclass when a required var is missing.
    """

    def __init__(
        self,
        render_var_name: str,
        frame_id: int | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            render_var_name (str): Name of the missing render var.
            frame_id (int | None): Frame index where the var was missing, if
                known. Included in the message when provided.
            message (str | None): Optional custom message; a default is built
                from ``render_var_name``/``frame_id`` when omitted.
        """
        self.render_var_name = render_var_name
        self.frame_id = frame_id
        msg = message or f"Render var '{render_var_name}' missing from FrameOutput"
        if frame_id is not None:
            msg += f" at frame {frame_id}"
        super().__init__(msg)


class MissingRenderVarError(ValidationError):
    """Raised when a writer is given a frame missing a required render var.

    A specialization of :class:`ValidationError` that also records which writer
    imposed the requirement, so callers can distinguish a writer requirement
    from a generic absent-var validation failure.
    """

    def __init__(
        self,
        writer_name: str,
        missing_render_var: str,
        frame_id: int | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            writer_name (str): ``name`` of the writer that required the var.
            missing_render_var (str): Name of the required-but-absent render var.
            frame_id (int | None): Frame index where the var was missing, if
                known. Included in the message when provided.
            message (str | None): Optional custom message; a default naming the
                writer and var is built when omitted.
        """
        self.writer_name = writer_name
        self.missing_render_var = missing_render_var
        msg = message or (
            f"Writer '{writer_name}' requires render var '{missing_render_var}' "
            "which was not provided"
        )
        super().__init__(
            render_var_name=missing_render_var, frame_id=frame_id, message=msg
        )


class BackendWriteError(OVAnnotatorsError):
    """Raised when a backend fails to persist data to a destination."""

    def __init__(self, path: str, message: str | None = None) -> None:
        """Initialize the error.

        Args:
            path (str): Destination path/key the backend was writing to.
            message (str | None): Optional custom message; a default referencing
                ``path`` is built when omitted.
        """
        self.path = path
        msg = message or f"Backend failed to write to '{path}'"
        super().__init__(msg)


class BackendReadError(OVAnnotatorsError):
    """Raised when a backend fails to read data from a source."""

    def __init__(self, path: str, message: str | None = None) -> None:
        """Initialize the error.

        Args:
            path (str): Source path/key the backend was reading from.
            message (str | None): Optional custom message; a default referencing
                ``path`` is built when omitted.
        """
        self.path = path
        msg = message or f"Backend failed to read from '{path}'"
        super().__init__(msg)
