.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Troubleshooting
===============

Installation cannot find an optional package
--------------------------------------------

Install only the extra required by the workflow. Renderer packages may require
the NVIDIA Python package index; use the complete development command in the
repository's ``README.md``. If a CPU-only workflow unexpectedly installs CUDA
packages, use ``all-cpu`` or ``dev-cpu`` instead of ``all`` or ``dev``.

An optional backend or writer fails at import time
--------------------------------------------------

Match the feature to its extra: ``s3`` for `Amazon S3
<https://aws.amazon.com/s3/>`_, ``ovstorage`` for a built ovstorage Stack,
``lerobot`` for `LeRobot <https://huggingface.co/docs/lerobot/index>`_,
``cosmos-multiview`` for Cosmos 3.5 Lance tables, and
``video`` for
`NVENC <https://developer.nvidia.com/video-codec-sdk>`_ output. Re-run the
failing command through the same environment used for installation, for example
``uv run python ...`` after ``uv sync --extra <name>``.

A third-party writer that does not appear in ``WriterRegistry.list_names()``
may have failed to import from its ``ovannotators.writers`` entry point.
``import ovannotators`` still succeeds and logs a warning,
``Skipping writer entry point '<name>'``, with the import traceback. Follow
the cause the traceback shows: install a missing dependency, or report a plugin
bug or an incompatibility with this ``ovannotators`` version to the plugin's
maintainers.

An ovstorage remote route cannot be used
-----------------------------------------

``OvstorageBackend`` raises ``ConfigurationError`` at construction for the
first two cases below and for ``overwrite=False`` on a route without
fail-if-exists writes; the message names the cause.

``AuthRequired`` means the ovstorage connection needs credentials or its own
interactive sign-in. Follow the upstream `authentication guide
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/skills/ovstorage-user-authenticate-to-backend/SKILL.md>`_
for that backend; another application's login does not authenticate this
Stack. For ``NoRoute``, inspect ``stack.list_address_roots()`` and ensure one
advertised root covers the dataset ``root_uri``. ``PermissionDenied`` after
sign-in means the authenticated identity lacks access to the target address.
The :ref:`OvstorageBackend section <ovstoragebackend>` shows the Stack setup
and remote smoke test.

Registering a name that is already taken raises ``ValueError`` (for example
``Writer name 'coco' is already registered by COCOWriter``). Registries
reject a different class under an existing ``name`` (``renderer_name`` for
renderer outputs); choose a unique name. Re-registering the same class is
harmless.

CUDA, NVENC, or the renderer is unavailable
-------------------------------------------

Confirm that the machine has an NVIDIA GPU, a compatible driver, and
`CUDA 12 <https://docs.nvidia.com/cuda/archive/12.9.2/>`_. Then verify that
`Warp <https://nvidia.github.io/warp/>`_ sees a CUDA device with
``uv run python -c "import warp; warp.init()"``. Core writers and disk output
remain usable without a GPU; follow the CPU-only `NumPy <https://numpy.org/>`_
tutorial when renderer
output is not required.

A writer reports a missing render variable
------------------------------------------

Writers consume canonical ``sourceName`` keys such as ``LdrColor`` and
``DistanceToCameraSD``. Pass the untouched renderer frame to ``OvrtxOutput`` so
it can map authored RenderVar prim paths to those keys. See
:doc:`render_outputs` for the supported names and decoded buffer types.

The example runs but the result is unclear
------------------------------------------

Run examples from the repository root and read the final output message. The
minimal capture writes under ``_out/basic_minimal``. Confirm that the directory
contains files for three frame groups, then open an image and inspect its
metadata. Tests validate library behavior; this manual inspection validates
that the local renderer, GPU stack, and filesystem work together.
