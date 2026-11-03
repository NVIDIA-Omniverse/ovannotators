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

"""Executable regression tests for the snippets in ``docs/public/tutorials.rst``.

Each test extracts a tutorial's ``code-block:: python`` directly from the docs
and runs it, so the documentation stays honest: if an API drifts, the matching
tutorial test breaks. The snippets are executed **verbatim** — the scene path
(``tests/data/test_scene.usda``) and render product (``/Render/RenderProduct_001``)
are the literal strings in the docs, run from the repo root just as a reader
would. Nothing about the documented logic is rewritten.

The only test-environment adjustments are: the output directory string is
rebased into ``_out/tutorials/<tutorial>/`` (so results persist for inspection
instead of landing in the cwd), and the deliberately large frame counts are
capped for speed (Tutorial 5 is left uncapped so its clip matches the README's
length). Tutorial 5's illustrative ``scene_variations`` is bound to a single
no-op variation.

ovrtx-dependent tutorials are skipped when ovrtx is unavailable; Tutorial 3
(S3) is syntax/symbol checked but its networked execution is skipped.
"""

from __future__ import annotations

import gc
import json
import re
import runpy
import textwrap
from pathlib import Path

import pytest

from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.writers._registry import WriterRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_RST = REPO_ROOT / "docs" / "public" / "tutorials.rst"
QUICKSTART_PY = REPO_ROOT / "docs" / "public" / "examples" / "quickstart.py"
QUICKSTART_CPU_PY = REPO_ROOT / "examples" / "quickstart_cpu.py"
OUT_ROOT = REPO_ROOT / "_out" / "tutorials"

# Keep the rendered tutorials quick: cap the documented loop counts.
RENDER_STEPS = 24

_OVRTX_AVAILABLE = ovrtx_available()

requires_ovrtx = pytest.mark.skipif(
    not _OVRTX_AVAILABLE,
    reason="ovrtx not installed — skipping rendered tutorial tests",
)


# ---------------------------------------------------------------------------
# Doc extraction
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(?P<title>.+)\n[-=]{3,}\n", re.MULTILINE)


def _sections() -> dict[str, str]:
    """Split the tutorials doc into ``{section_title: section_body}``.

    Returns:
        Mapping of each reStructuredText heading title to the text that
        follows it (up to the next heading).
    """
    text = TUTORIALS_RST.read_text(encoding="utf-8")
    matches = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[match.group("title").strip()] = text[start:end]
    return sections


def _python_blocks(body: str) -> list[str]:
    """Return the dedented ``code-block:: python`` snippets within ``body``.

    Args:
        body: A tutorial section body (as returned by :func:`_sections`).

    Returns:
        The Python code blocks, in document order, with the reST indentation
        stripped.
    """
    lines = body.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        match = re.match(r"^(?P<indent>\s*)\.\. code-block:: python\s*$", lines[i])
        if not match:
            i += 1
            continue
        base = len(match.group("indent"))
        i += 1
        collected: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                collected.append("")
                i += 1
                continue
            if len(line) - len(line.lstrip()) > base:
                collected.append(line)
                i += 1
                continue
            break
        blocks.append(textwrap.dedent("\n".join(collected)).strip("\n"))
    return blocks


def _blocks_for(tutorial_number: int) -> list[str]:
    """Return the Python snippets under ``Tutorial <n>:``.

    Args:
        tutorial_number: The 1-based tutorial index.

    Returns:
        The Python code blocks for that tutorial.

    Raises:
        AssertionError: If the tutorial section or any python block is missing.
    """
    prefix = f"Tutorial {tutorial_number}:"
    for title, body in _sections().items():
        if title.startswith(prefix):
            blocks = _python_blocks(body)
            assert blocks, f"No python code-block found under '{title}'"
            return blocks
    raise AssertionError(f"Tutorial section starting with '{prefix}' not found")


# ---------------------------------------------------------------------------
# Documented output-tree validation
# ---------------------------------------------------------------------------


def _output_block(tutorial_number: int) -> str:
    """Return the dedented ``**Output:**`` literal tree for a tutorial.

    Locates the ``**Output:**`` marker in the tutorial section and captures the
    following reStructuredText ``::`` literal block.

    Args:
        tutorial_number: The 1-based tutorial index.

    Returns:
        The literal directory-tree text, dedented.

    Raises:
        AssertionError: If the section or its Output literal block is missing.
    """
    prefix = f"Tutorial {tutorial_number}:"
    body = next((b for t, b in _sections().items() if t.startswith(prefix)), None)
    assert body is not None, f"Tutorial {tutorial_number} section not found"
    marker = body.find("**Output:**")
    assert marker != -1, f"No **Output:** block for Tutorial {tutorial_number}"
    lines = body[marker:].splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "::"), None)
    assert start is not None, f"No '::' literal block for Tutorial {tutorial_number}"
    collected: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "":
            collected.append("")
            continue
        if line[:1].isspace():
            collected.append(line)
            continue
        break
    return textwrap.dedent("\n".join(collected)).strip("\n")


def _expand_braces(name: str) -> list[str]:
    """Expand a single ``{a,b,c}`` group in a tree entry name.

    Supports the compact ``meta/{info,tasks}.json`` notation. Names without a
    brace group are returned unchanged.

    Args:
        name: A tree entry name, possibly containing one brace group.

    Returns:
        One name per brace option (or the original name).
    """
    match = re.search(r"\{([^}]*)\}", name)
    if not match:
        return [name]
    return [
        name[: match.start()] + opt + name[match.end() :]
        for opt in match.group(1).split(",")
    ]


def _documented_paths(tree: str) -> list[tuple[str, bool]]:
    """Parse an ASCII output tree into ``(relative_path, is_dir)`` entries.

    The first line (e.g. ``./cosmos_data/``) is the depth-0 root and is mapped
    to the produced output directory, so returned paths are relative to it.
    Illustrative placeholders (any entry containing ``...``) are skipped, inline
    ``#`` annotations are stripped, and ``{a,b}`` groups are expanded.

    Args:
        tree: The dedented literal tree text from :func:`_output_block`.

    Returns:
        ``(relative_path, is_dir)`` for each concrete documented entry.
    """
    entries: list[tuple[str, bool]] = []
    stack: dict[int, str] = {0: ""}
    for raw in tree.splitlines():
        if not raw.strip():
            continue
        line = raw.split("#", 1)[0].rstrip()
        connector = max(
            line.find("\u251c\u2500\u2500 "), line.find("\u2514\u2500\u2500 ")
        )
        if connector == -1:
            # Depth-0 root line; maps to the produced output root.
            stack = {0: ""}
            continue
        name = line[connector + 4 :].strip()
        if "..." in name:
            continue
        depth = connector // 4 + 1
        is_dir = name.endswith("/")
        name = name.rstrip("/")
        parent = stack.get(depth - 1, "")
        for expanded in _expand_braces(name):
            entries.append((f"{parent}/{expanded}" if parent else expanded, is_dir))
        stack[depth] = f"{parent}/{name}" if parent else name
        for deeper in [d for d in stack if d > depth]:
            del stack[deeper]
    return entries


def _produced_tree(root: Path) -> str:
    """Return a sorted listing of files under ``root`` for failure messages.

    Args:
        root: Directory to walk.

    Returns:
        A newline-joined list of paths relative to ``root`` (or a placeholder
        when ``root`` is missing/empty).
    """
    if not root.exists():
        return f"(nothing — {root} does not exist)"
    found = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    return "\n".join(f"  {p}" for p in found) or f"(empty — {root} has no entries)"


def _assert_glob(root: Path, pattern: str) -> None:
    """Assert ``root`` contains at least one match for ``pattern``.

    Args:
        root: Directory to glob within.
        pattern: A glob pattern relative to ``root``.

    Raises:
        AssertionError: With the produced tree when nothing matches.
    """
    assert sorted(root.glob(pattern)), (
        f"No match for '{pattern}' under {root}.\nProduced:\n{_produced_tree(root)}"
    )


def _assert_documented_output(root: Path, tutorial_number: int) -> None:
    """Assert every concrete entry in a tutorial's Output tree was produced.

    Args:
        root: The directory the documented tree is rooted at (the produced
            output location the snippet wrote to).
        tutorial_number: The 1-based tutorial index.

    Raises:
        AssertionError: If a documented file/directory is missing under ``root``.
    """
    documented = _documented_paths(_output_block(tutorial_number))
    assert documented, f"Tutorial {tutorial_number} Output tree parsed to nothing"
    for rel, is_dir in documented:
        target = root / rel
        check = target.is_dir() if is_dir else target.is_file()
        assert check, (
            f"Tutorial {tutorial_number} documents "
            f"{'directory' if is_dir else 'file'} '{rel}', but it was not "
            f"produced under {root}.\nProduced:\n{_produced_tree(root)}"
        )


# ---------------------------------------------------------------------------
# Snippet rewriting / execution
# ---------------------------------------------------------------------------


def _lit(path: Path) -> str:
    """Return ``path`` as a Python string literal for snippet substitution."""
    return json.dumps(str(path))


def _cap_loops(code: str, steps: int = RENDER_STEPS) -> str:
    """Shrink documented render-loop counts so rendered tests stay fast.

    Caps both ``num_frames = N`` (the time-sampled loops) and any bare
    ``range(N)`` so a snippet renders at most ``steps`` frames.

    Args:
        code: Snippet source.
        steps: Maximum frame count to allow.

    Returns:
        The snippet with its frame counts capped.
    """
    code = re.sub(
        r"num_frames = (\d+)",
        lambda m: f"num_frames = {min(int(m.group(1)), steps)}",
        code,
    )
    code = re.sub(
        r"range\((\d+)\)",
        lambda m: f"range({min(int(m.group(1)), steps)})",
        code,
    )
    return code


def _teardown_renderer(namespace: dict) -> None:
    """Release ovrtx step results before the renderer, deterministically.

    ovrtx logs ``Leaking step result outputs`` when a renderer is destroyed
    while a ``step()`` result handle is still alive. Snippets run via ``exec``
    keep the loop's final ``outputs`` (plus the zero-copy views the writer
    retains) and the ``Renderer`` together in one namespace, so ordinary GC may
    tear the renderer down first. Hold the renderer aside, drop every other
    binding and collect (releasing the step results), then release the renderer
    last so no handles are outstanding when it is destroyed.

    Args:
        namespace: The globals namespace a snippet executed in. Mutated/cleared.
    """
    try:
        import ovrtx
    except ImportError:
        namespace.clear()
        gc.collect()
        return

    # Keep renderers alive while everything that may hold step results is freed.
    renderers = [v for v in namespace.values() if isinstance(v, ovrtx.Renderer)]
    namespace.clear()
    gc.collect()
    # Now drop the renderers themselves; no step results remain outstanding.
    renderers.clear()
    gc.collect()


def _run(code: str, *, teardown: bool = False) -> dict:
    """Execute a tutorial snippet in a fresh namespace and return it.

    Args:
        code: The (already rewritten) snippet source.
        teardown: When ``True``, deterministically release any ovrtx renderer
            and its step results after execution (see :func:`_teardown_renderer`).
            Use for rendered tutorials whose return value is not needed.

    Returns:
        The globals namespace after execution, for optional inspection. When
        ``teardown`` is ``True`` the namespace has been cleared.
    """
    namespace: dict = {}
    exec(compile(code, "<tutorial>", "exec"), namespace)  # noqa: S102
    if teardown:
        _teardown_renderer(namespace)
    return namespace


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each snippet from the repo root.

    The tutorials open ``tests/data/test_scene.usda`` by its documented relative
    path, so the snippets must execute with the repo root as the working
    directory — exactly as a reader running them from a checkout would.
    """
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture(scope="session")
def out_root() -> Path:
    """Create and return the persistent ``_out/tutorials`` output directory."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUT_ROOT


# ---------------------------------------------------------------------------
# Quick Start — docs/public/examples/quickstart.py (literalinclude'd in index.rst)
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_quickstart_example(out_root: Path) -> None:
    out = out_root / "00_quickstart"
    code = QUICKSTART_PY.read_text(encoding="utf-8")
    code = _cap_loops(code)
    code = code.replace('"./output/coco"', _lit(out / "coco"))
    _run(code, teardown=True)

    dataset = out / "coco"
    assert (dataset / "annotations" / "instances_train.json").is_file(), (
        f"instances_train.json missing.\nProduced:\n{_produced_tree(dataset)}"
    )
    _assert_glob(dataset, "images/*.jpg")


# ---------------------------------------------------------------------------
# CPU Quick Start — examples/quickstart_cpu.py (literalinclude'd in index.rst)
# ---------------------------------------------------------------------------


def test_quickstart_cpu_example(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(QUICKSTART_CPU_PY), run_name="__main__")

    assert "Wrote 15 files to _out/quickstart_cpu" in capsys.readouterr().out
    out = tmp_path / "_out" / "quickstart_cpu"
    assert sorted(p.name for p in (out / "0000").iterdir()) == [
        "distance_to_camera.exr",
        "distance_to_camera_colorized.png",
        "ldr_color.png",
        "normal.exr",
        "normal_colorized.png",
    ]


# ---------------------------------------------------------------------------
# Tutorial 1 — Basic COCO Dataset from ovrtx
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_1_coco(out_root: Path) -> None:
    out = out_root / "01_coco"
    code = _blocks_for(1)[0]
    code = _cap_loops(code)
    code = code.replace('"./output/coco_dataset"', _lit(out / "coco_dataset"))
    _run(code, teardown=True)

    dataset = out / "coco_dataset"
    assert (dataset / "annotations" / "instances_train.json").is_file(), (
        f"instances_train.json missing.\nProduced:\n{_produced_tree(dataset)}"
    )
    _assert_glob(dataset, "images/*.jpg")
    _assert_documented_output(dataset, 1)


# ---------------------------------------------------------------------------
# Tutorial 2 — Multiple Writers, One Render Loop
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_2_multi_writer(out_root: Path) -> None:
    out = out_root / "02_multi_writer"
    code = _blocks_for(2)[0]
    code = _cap_loops(code)
    code = code.replace('"./out/coco"', _lit(out / "coco"))
    code = code.replace('"./out/kitti"', _lit(out / "kitti"))
    code = code.replace('"./out/yolo"', _lit(out / "yolo"))
    _run(code, teardown=True)

    assert (out / "coco" / "annotations" / "instances_train.json").is_file()
    assert (out / "kitti" / "label_2").is_dir()
    assert (out / "yolo" / "data.yaml").is_file()


# ---------------------------------------------------------------------------
# Tutorial 3 — Writing to S3 (networked; syntax/symbol checked only)
# ---------------------------------------------------------------------------


def test_tutorial_3_s3_is_valid() -> None:
    code = _blocks_for(3)[0]
    # The snippet must be valid Python and reference real public symbols.
    compile(code, "<tutorial3>", "exec")
    from ovannotators import S3Backend, S3BackendConfig  # noqa: F401

    assert WriterRegistry.get("coco") is not None
    pytest.skip(
        "Tutorial 3 writes to a live S3 bucket (requires AWS credentials); "
        "validated for syntax and API symbols only."
    )


# ---------------------------------------------------------------------------
# Tutorial 4 — LeRobot Dataset (render loop elided in docs; filled here)
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_4_lerobot(out_root: Path) -> None:
    pytest.importorskip("pyarrow", reason="LeRobot parquet output needs pyarrow")
    pytest.importorskip(
        "PyNvVideoCodec", reason="LeRobot video output needs PyNvVideoCodec"
    )
    out = out_root / "04_lerobot"
    code = _blocks_for(4)[0]
    code = _cap_loops(code)
    code = code.replace('"./lerobot_data"', _lit(out))
    _run(code, teardown=True)

    assert (out / "meta" / "info.json").is_file(), (
        f"meta/info.json missing.\nProduced:\n{_produced_tree(out)}"
    )
    _assert_glob(out, "data/chunk-000/episode_*.parquet")
    # Camera-key-agnostic: the exact key is pinned by the documented Output tree.
    _assert_glob(out, "videos/chunk-000/observation.images.*/episode_*.mp4")
    _assert_documented_output(out, 4)


# ---------------------------------------------------------------------------
# Tutorial 5 — Cosmos Transfer Training Data (capture + Cosmos 3 bridge)
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_5_cosmos_transfer(out_root: Path) -> None:
    out = out_root / "05_cosmos_transfer"
    cosmos_data = out / "cosmos_data"
    transfer_specs = out / "transfer_specs"

    config_block, capture_block, bridge_block = _blocks_for(5)

    # config + capture share a namespace (capture references `config`).
    # Unlike the other rendered tutorials, the documented frame count is left
    # uncapped so the produced clip matches the README example's length.
    capture = config_block + "\n" + capture_block
    capture = capture.replace('"./cosmos_data"', _lit(cosmos_data))
    # `scene_variations` is illustrative in the docs; bind one no-op variation.
    variation_stub = textwrap.dedent(
        """
        class _Variation:
            def apply(self, stage, ordinal):
                return ordinal

        scene_variations = [_Variation()]
        """
    )
    capture = variation_stub + "\n" + capture
    _run(capture, teardown=True)

    manifest = cosmos_data / "manifest.json"
    assert manifest.is_file()
    clip = json.loads(manifest.read_text())["clips"][0]
    assert (cosmos_data / clip["conditioning"]["depth"]).is_file()
    assert sorted(cosmos_data.glob("videos/*_rgb.mp4"))
    _assert_documented_output(cosmos_data, 5)

    # Bridge: turn the captured clip into a Cosmos 3 transfer spec + command.
    bridge = bridge_block.replace('"./cosmos_data"', _lit(cosmos_data))
    bridge = bridge.replace('"./transfer_specs"', _lit(transfer_specs))
    namespace = _run(bridge)

    spec_path = namespace["spec_path"]
    assert Path(spec_path).is_file()
    assert namespace["spec"]["model_mode"] == "video2video"
    assert isinstance(namespace["cmd"], list) and namespace["cmd"]


# ---------------------------------------------------------------------------
# Tutorial 6 — Custom Writer (pure Python)
# ---------------------------------------------------------------------------


def test_tutorial_6_custom_writer(out_root: Path) -> None:
    import numpy as np

    from ovannotators.backends.disk import DiskBackend, DiskBackendConfig

    out = out_root / "06_custom_writer"
    code = _blocks_for(6)[0]

    # The decorator registers "depth_stats" globally; drop a stale registration
    # so re-runs in the same session re-register cleanly.
    WriterRegistry._registry.pop("depth_stats", None)
    namespace = _run(code)

    writer_cls = namespace["DepthStatsWriter"]
    assert WriterRegistry.get("depth_stats") is writer_cls

    backend = DiskBackend(DiskBackendConfig(root_dir=str(out), overwrite=True))
    writer = writer_cls()
    rng = np.random.default_rng(0)
    for frame_id in range(3):
        writer.write_frame(
            frame_id,
            {"DistanceToCameraSD": rng.uniform(0.5, 20.0, (32, 32)).astype(np.float32)},
            backend,
        )
    writer.finalize(backend)

    stats = out / "depth_stats.jsonl"
    assert stats.is_file()
    records = [json.loads(line) for line in stats.read_text().splitlines() if line]
    assert len(records) == 3
    assert {"frame_id", "min_depth", "max_depth", "mean_depth", "coverage"} <= set(
        records[0]
    )


# ---------------------------------------------------------------------------
# Tutorial 7 — Using OvrtxOutput Helper
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_7_ovrtx_output(out_root: Path) -> None:
    out = out_root / "07_ovrtx_output"
    code = _blocks_for(7)[0]
    code = _cap_loops(code)
    code = code.replace('"./output"', _lit(out))
    _run(code, teardown=True)

    assert sorted(out.glob("*/ldr_color.png"))


# ---------------------------------------------------------------------------
# Tutorial 8 — Without ovrtx (Numpy Arrays Directly)
# ---------------------------------------------------------------------------


def test_tutorial_8_numpy_direct(out_root: Path) -> None:
    out = out_root / "08_numpy_direct"
    code = _blocks_for(8)[0]
    code = code.replace('"./output"', _lit(out))
    _run(code)

    assert sorted(out.glob("*/ldr_color.png"))


# ---------------------------------------------------------------------------
# Tutorial 9 — Faster Capture with ovrtx (Async Pipelining)
# ---------------------------------------------------------------------------


@requires_ovrtx
def test_tutorial_9_async_pipeline(out_root: Path) -> None:
    out = out_root / "09_async_pipeline"
    code = _blocks_for(9)[0]
    code = _cap_loops(code)
    code = code.replace('"./output"', _lit(out))
    _run(code, teardown=True)

    # One per-frame directory per rendered frame, matching the capped loop.
    frame_dirs = sorted(p for p in out.glob("*") if p.is_dir())
    assert len(frame_dirs) == RENDER_STEPS, (
        f"expected {RENDER_STEPS} frame dirs.\nProduced:\n{_produced_tree(out)}"
    )
    assert sorted(out.glob("*/ldr_color.png"))


# ---------------------------------------------------------------------------
# Tutorial 10 — Saving and Reloading Configs (pure Python, no renderer)
# ---------------------------------------------------------------------------


def test_tutorial_10_config_serialization(out_root: Path) -> None:
    from ovannotators import (
        COCOWriter,
        COCOWriterConfig,
        DiskBackend,
        DiskBackendConfig,
        config_from_dict,
    )

    out = out_root / "10_config_serialization"
    save_block, load_block = _blocks_for(10)

    # Save snapshot, then reload it from the same root in a fresh namespace.
    save_block = save_block.replace('"./output"', _lit(out))
    load_block = load_block.replace('"./output"', _lit(out))
    _run(save_block)
    namespace = _run(load_block)

    snapshot = out / "run_config.json"
    assert snapshot.is_file(), (
        f"run_config.json missing.\nProduced:\n{_produced_tree(out)}"
    )

    # The reloaded plugins are configured exactly as the documented originals.
    assert isinstance(namespace["backend"], DiskBackend)
    assert isinstance(namespace["writer"], COCOWriter)

    saved = json.loads(snapshot.read_text())
    assert config_from_dict(
        DiskBackendConfig, saved["backend"]["config"]
    ) == DiskBackendConfig(root_dir=str(out), overwrite=True)
    assert (
        config_from_dict(COCOWriterConfig, saved["writer"]["config"])
        == COCOWriterConfig()
    )
