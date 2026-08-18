.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Agent Skills
============

ovannotators ships reusable, task-oriented agent instructions in ``skills/``.
The ``.agents/skills`` and ``.claude/skills`` paths are read-only mirrors of
that source tree; edit only ``skills/``.

The following workflows are available:

* ``new-writer`` for adding an output-format writer.
* ``new-backend`` for adding a storage backend.
* ``test-creation``, ``test-classification``, ``test-coverage``, and
  ``test-triage`` for maintaining the test suite.

Use the :doc:`api` reference as the public API contract. Every substantial API
example in a shipped skill points to a named snippet under ``tests/docs/python``.
Those snippets and the mirror paths are validated by
``tests/test_skill_snippets.py``; run it with:

.. code-block:: bash

   uv run pytest tests/test_skill_snippets.py tests/docs/python -q

The public quickstart and tutorials are likewise executed by
``tests/test_tutorials.py``. Agent tooling itself is not an ovannotators runtime
dependency.
