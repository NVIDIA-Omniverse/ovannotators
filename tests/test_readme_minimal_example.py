# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the README minimal capture snippet."""

from __future__ import annotations

import re
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def _extract_minimal_example_python(readme_text: str) -> str:
    """Extract the python snippet under the README Minimal Example section.

    Args:
        readme_text: Full README markdown content.

    Returns:
        The Python code block contents for the Minimal Example section.

    Raises:
        AssertionError: If the Minimal Example python block is missing.
    """
    match = re.search(
        r"### Minimal Example.*?```python\n(.*?)\n```",
        readme_text,
        flags=re.DOTALL,
    )
    assert match is not None, "README Minimal Example python snippet not found"
    return match.group(1)


@pytest.mark.gpu
def test_readme_minimal_example_runs(monkeypatch) -> None:
    """Execute the README minimal example and verify it writes frame outputs.

    The output is written to the repo's ``_out/basic_minimal`` directory (the
    same path the README documents) and deliberately left in place after the
    run so it can be inspected. ``DiskBackend`` is configured with
    ``overwrite=True``, so re-runs replace the previous output cleanly.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to isolate cwd.
    """
    repo_root = Path(__file__).resolve().parents[1]
    readme_path = repo_root / "README.md"
    snippet = _extract_minimal_example_python(readme_path.read_text(encoding="utf-8"))

    scene_path = repo_root / "tests" / "data" / "minimal_capture.usda"
    out_dir = repo_root / "_out" / "basic_minimal"
    snippet = snippet.replace(
        '"tests/data/minimal_capture.usda"',
        repr(str(scene_path.resolve())),
    )
    snippet = snippet.replace(
        '"_out/basic_minimal"',
        repr(str(out_dir.resolve())),
    )

    monkeypatch.chdir(repo_root)
    exec(snippet, {}, {})  # noqa: S102

    frame_images = sorted(out_dir.glob("*/ldr_color.png"))
    assert len(frame_images) == 3
    frame_pixels = []
    for frame_image in frame_images:
        with Image.open(frame_image) as image:
            frame_pixels.append(np.asarray(image).copy())
    assert len({pixels.tobytes() for pixels in frame_pixels}) == 3


@pytest.mark.parametrize("failure_point", [None, "render", "write", "finalize"])
@pytest.mark.parametrize("source_name", ["README.md", "examples/minimal_capture.py"])
def test_minimal_capture_publishes_each_update_before_rendering(
    monkeypatch, source_name: str, failure_point: str | None
) -> None:
    import ovannotators.backends as backends
    import ovannotators.renderer_outputs as renderer_outputs
    import ovannotators.writers as writers

    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / source_name
    source = source_path.read_text(encoding="utf-8")
    if source_name == "README.md":
        source = _extract_minimal_example_python(source)

    events: list[tuple] = []
    written_ordinals: list[int] = []
    frame_refs: list[weakref.ReferenceType] = []

    class FakeBackend:
        def __init__(self, config) -> None:
            self.config = config

        def list(self) -> list[str]:
            return []

    class FakeWriter:
        def __init__(self, config) -> None:
            self.config = config

        def write_frame(self, frame_id: int, data: dict, backend: FakeBackend) -> None:
            del backend
            written_ordinals.append(data["ordinal"])
            events.append(("write", frame_id))
            if failure_point == "write":
                raise RuntimeError("injected write failure")

        def finalize(self, backend: FakeBackend) -> None:
            del backend
            assert all(frame_ref() is None for frame_ref in frame_refs)
            events.append(("finalize",))
            if failure_point == "finalize":
                raise RuntimeError("injected finalize failure")

    class FakeExtractor:
        def extract_all(self, frame) -> dict[str, int]:
            return {"ordinal": frame.ordinal}

    class FakeFrame:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal

    class FakeStage:
        def __init__(self, name: str) -> None:
            self.write_floor = 0

        def advance_write_floor(self, ordinal: int):
            def wait():
                self.write_floor = ordinal
                events.append(("floor", ordinal))

            return SimpleNamespace(wait=wait)

        def destroy(self):
            events.append(("destroy-stage",))

    class FakeRenderer:
        def attach_ovstage(self, stage: FakeStage) -> None:
            self.stage = stage
            events.append(("attach",))

        def detach_ovstage(self):
            assert all(frame_ref() is None for frame_ref in frame_refs)
            events.append(("detach",))

        def destroy(self):
            events.append(("destroy-renderer",))

        def step(self, render_products, *, delta_time: float, ordinal: int):
            assert delta_time > 0
            assert ordinal <= self.stage.write_floor
            events.append(("step", ordinal))
            if failure_point == "render":
                raise RuntimeError("injected render failure")
            frame = FakeFrame(ordinal)
            frame_refs.append(weakref.ref(frame))
            return {
                product: SimpleNamespace(frames=[frame]) for product in render_products
            }

    ovrtx_module = ModuleType("ovrtx")
    ovrtx_module.register_schema_paths = lambda: events.append(("schemas",))
    ovrtx_module.Renderer = FakeRenderer
    ovstage_module = ModuleType("ovstage")
    ovstage_module.Stage = FakeStage
    ovstage_module.population = SimpleNamespace(
        open_usd=lambda stage, path, *, ordinal, time_code: events.append(
            ("open", path, ordinal, time_code)
        ),
        update_from_usd_time=lambda stage, ordinal, time: events.append(
            ("update", ordinal, float(time))
        ),
    )

    def fake_config(**kwargs):
        return kwargs

    monkeypatch.setattr(backends, "DiskBackend", FakeBackend)
    monkeypatch.setattr(backends, "DiskBackendConfig", fake_config)
    monkeypatch.setattr(renderer_outputs, "OvrtxOutput", FakeExtractor)
    monkeypatch.setattr(writers, "BasicWriter", FakeWriter)
    monkeypatch.setattr(writers, "BasicWriterConfig", fake_config)
    monkeypatch.setitem(sys.modules, "ovrtx", ovrtx_module)
    monkeypatch.setitem(sys.modules, "ovstage", ovstage_module)

    def run():
        exec(compile(source, str(source_path), "exec"), {"__name__": "__main__"})  # noqa: S102

    if failure_point:
        with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
            run()
    else:
        run()

    expected = [
        ("schemas",),
        ("attach",),
        ("open", "tests/data/minimal_capture.usda", 1, 0.0),
        ("floor", 1),
        ("update", 2, 0.0),
        ("floor", 2),
        ("step", 2),
        ("write", 0),
        ("update", 3, 2.5),
        ("floor", 3),
        ("step", 3),
        ("write", 1),
        ("update", 4, 5.0),
        ("floor", 4),
        ("step", 4),
        ("write", 2),
        ("finalize",),
        ("detach",),
        ("destroy-stage",),
        ("destroy-renderer",),
    ]
    if failure_point:
        failed_event = {
            "render": ("step", 2),
            "write": ("write", 0),
            "finalize": ("finalize",),
        }[failure_point]
        expected = expected[: expected.index(failed_event) + 1] + expected[-3:]
    assert events == expected
    expected_ordinals = {"render": [], "write": [2]}.get(failure_point, [2, 3, 4])
    assert written_ordinals == expected_ordinals
