.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Agent Skills
============

ovannotators ships reusable, task-oriented agent instructions in ``skills/``.
The ``.agents/skills`` and ``.claude/skills`` paths are read-only mirrors of
that source tree; edit only ``skills/``.

The following workflows are available:

* ``ovannotators-dataset-quickstart`` for producing and inspecting a small labeled COCO
  dataset, on CPU by default.
* ``ovannotators-dataset-capture`` for choosing a writer, profile, frame source, and backend,
  then capturing and verifying a dataset.
* ``new-writer`` for adding an output-format writer.
* ``new-backend`` for adding a storage backend.

Use the :doc:`api` reference as the public API contract. Every substantial API
example in a shipped skill points to a named snippet under ``tests/docs/python``.
Those snippets and the mirror paths are validated by
``tests/test_skill_snippets.py``; run it with:

.. code-block:: bash

   uv run pytest tests/test_skill_snippets.py tests/docs/python -q

The public quickstart and tutorials are likewise executed by
``tests/test_tutorials.py``. Agent tooling itself is not an ovannotators runtime
dependency.
