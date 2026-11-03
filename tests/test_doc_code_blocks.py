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

"""Static checks for every Python code block in ``docs/public``.

Covers ``code-block:: python`` and Python ``literalinclude`` directives in
reStructuredText pages and ``python`` fences in MyST Markdown pages. Each block
is checked as the reader sees it, not as the surrounding test runs it:

* the block parses as Python (signature listings belong in ``text`` blocks);
* ruff reports no unused imports, unused variables, or redefinitions. The block
  is wrapped in a function so pyflakes also checks local assignments. Plain
  assignments that end the block are its results and may stay unused; a block
  of only imports is an import listing;
* every name the block uses without defining (``F821``) appears as an inline
  literal in the prose of the block's section, so the reader is told where it
  comes from;
* every ``ovannotators`` import names a public module and resolves.

Execution is covered elsewhere: ``tests/test_tutorials.py`` runs the tutorial
blocks and ``tests/docs/python`` runs the literalinclude snippets.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu_only]

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "public"

# Lint rules applied to each block, beyond undefined names (checked against prose).
_LINT_RULES = ("F401", "F811", "F841")

# Private-module imports that are documented on purpose, keyed by
# ``(page, module)``. Each entry needs a reason; remove it once resolved.
_PRIVATE_IMPORT_EXCEPTIONS = {
    ("api.rst", "ovannotators._internal.codecs"): (
        "codec helpers are documented but not yet exported publicly"
    ),
}

_RST_DIRECTIVE = re.compile(r"^(\s*)\.\. (code-block|literalinclude)::\s*(\S*)\s*$")
_RST_OPTION = re.compile(r"^\s*:([\w-]+):\s*(.*)$")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_INLINE_LITERAL = re.compile(r"``(.+?)``|`([^`]+)`")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class DocBlock:
    page: Path
    line: int
    code: str
    context: str

    @property
    def id(self) -> str:
        return f"{self.page.relative_to(DOCS_DIR)}:{self.line}"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _include_region(path: Path, options: dict[str, str]) -> str:
    """Return the part of ``path`` a ``literalinclude`` shows."""
    text = path.read_text(encoding="utf-8")
    if "start-after" in options:
        text = text.split(options["start-after"], 1)[1].split("\n", 1)[1]
    if "start-at" in options:
        text = options["start-at"] + text.split(options["start-at"], 1)[1]
    if "end-before" in options:
        text = text.split(options["end-before"], 1)[0].rsplit("\n", 1)[0]
    return textwrap.dedent(text)


def _rst_blocks(page: Path) -> list[DocBlock]:
    lines = page.read_text(encoding="utf-8").splitlines()
    found: list[tuple[int, int, str]] = []  # (start, end, code); end is exclusive
    i = 0
    while i < len(lines):
        match = _RST_DIRECTIVE.match(lines[i])
        if not match:
            i += 1
            continue
        start, base = i, len(match.group(1))
        kind, argument = match.group(2), match.group(3)
        options: dict[str, str] = {}
        i += 1
        while i < len(lines) and (option := _RST_OPTION.match(lines[i])):
            options[option.group(1)] = option.group(2)
            i += 1
        body: list[str] = []
        while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > base):
            body.append(lines[i])
            i += 1
        if kind == "code-block" and argument == "python":
            found.append((start, i, textwrap.dedent("\n".join(body)).strip("\n")))
        elif kind == "literalinclude" and options.get("language") == "python":
            found.append((start, i, _include_region(page.parent / argument, options)))
    headings = [
        n - 1
        for n, line in enumerate(lines)
        if n and _RST_UNDERLINE.match(line) and lines[n - 1].strip()
    ]
    return _with_context(page, lines, found, headings)


def _md_blocks(page: Path) -> list[DocBlock]:
    lines = page.read_text(encoding="utf-8").splitlines()
    found: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        opening = re.match(r"^( {0,3})(`{3,}|~{3,})(.*)$", lines[i])
        if not opening:
            i += 1
            continue
        start, language = i, opening.group(3).strip()
        fence = opening.group(2)
        indent = len(opening.group(1))
        i += 1
        body: list[str] = []
        marker = re.escape(fence[0]) + "{" + str(len(fence)) + ",}"
        closing = re.compile("^ {0,3}" + marker + "[ \\t]*$")
        while i < len(lines) and not closing.match(lines[i]):
            line = lines[i]
            body.append(line[min(indent, len(line) - len(line.lstrip(" "))) :])
            i += 1
        i += 1
        if language == "python":
            found.append((start, i, "\n".join(body)))
    headings = [n for n, line in enumerate(lines) if line.startswith("#")]
    return _with_context(page, lines, found, headings)


def _with_context(
    page: Path,
    lines: list[str],
    found: list[tuple[int, int, str]],
    headings: list[int],
) -> list[DocBlock]:
    """Attach the prose from each block's section heading to the next block or heading."""
    code_lines = {n for start, end, _ in found for n in range(start, end)}
    blocks = []
    for index, (start, end, code) in enumerate(found):
        section = max((h for h in headings if h < start), default=0)
        after = min(
            found[index + 1][0] if index + 1 < len(found) else len(lines),
            min((h for h in headings if h > start), default=len(lines)),
        )
        context = "\n".join(
            lines[n] for n in range(section, after) if n not in code_lines
        )
        blocks.append(DocBlock(page, start + 1, code, context))
    return blocks


def _all_blocks() -> list[DocBlock]:
    blocks: list[DocBlock] = []
    for page in sorted(DOCS_DIR.rglob("*.rst")):
        blocks.extend(_rst_blocks(page))
    for page in sorted(DOCS_DIR.rglob("*.md")):
        blocks.extend(_md_blocks(page))
    return blocks


BLOCKS = _all_blocks()


def _literal_names(context: str) -> set[str]:
    names: set[str] = set()
    for match in _INLINE_LITERAL.finditer(context):
        names.update(_IDENTIFIER.findall(match.group(1) or match.group(2)))
    return names


@pytest.fixture(scope="module")
def lint_results(tmp_path_factory) -> dict[str, list[dict]]:
    """Run ruff once over every block, each wrapped in a function body."""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not installed")
    root = tmp_path_factory.mktemp("doc_blocks")
    files = {}
    for number, block in enumerate(BLOCKS):
        path = root / f"block_{number:03d}.py"
        path.write_text(
            "def _doc_block():\n" + textwrap.indent(block.code, "    ") + "\n",
            encoding="utf-8",
        )
        files[path.name] = block.id
    completed = subprocess.run(
        [
            ruff,
            "check",
            "--no-cache",
            "--isolated",
            "--select",
            ",".join((*_LINT_RULES, "F821")),
            "--output-format",
            "json",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        pytest.fail(
            f"Ruff exited with status {completed.returncode}: {completed.stderr}"
        )
    results: dict[str, list[dict]] = {block_id: [] for block_id in files.values()}
    for diagnostic in json.loads(completed.stdout or "[]"):
        results[files[Path(diagnostic["filename"]).name]].append(diagnostic)
    return results


def test_blocks_are_discovered() -> None:
    pages = {block.page.name for block in BLOCKS}
    assert {"api.rst", "tutorials.rst", "index.rst"} <= pages
    assert len(BLOCKS) > 50


def _result_names(tree: ast.Module) -> set[str]:
    """Return the names bound by the plain assignments that end the block."""
    names: set[str] = set()
    for statement in reversed(tree.body):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            break
        names.update(
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
    return names


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.id)
def test_block_parses(block: DocBlock) -> None:
    try:
        ast.parse(block.code)
    except SyntaxError as error:
        pytest.fail(f"{block.id}: line {error.lineno}: {error.msg}")


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.id)
def test_block_is_lint_clean(block: DocBlock, lint_results) -> None:
    try:
        tree = ast.parse(block.code)
    except SyntaxError:
        pytest.skip("reported by test_block_parses")
    import_listing = all(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
    )
    results = _result_names(tree)
    problems = [
        f"line {d['location']['row'] - 1}: {d['code']} {d['message']}"
        for d in lint_results[block.id]
        if d["code"] in _LINT_RULES
        and not (d["code"] == "F401" and import_listing)
        and not (
            d["code"] == "F841" and re.search(r"`(\w+)`", d["message"])[1] in results
        )
    ]
    assert not problems, f"{block.id}:\n" + "\n".join(problems)


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.id)
def test_undefined_names_are_explained(block: DocBlock, lint_results) -> None:
    undefined = {
        re.search(r"`(\w+)`", d["message"]).group(1)
        for d in lint_results[block.id]
        if d["code"] == "F821"
    }
    unexplained = sorted(undefined - _literal_names(block.context))
    assert not unexplained, (
        f"{block.id} uses {unexplained} without defining them; name each one "
        "as an inline literal in the prose around the block"
    )


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.id)
def test_imports_are_public_and_resolve(block: DocBlock) -> None:
    try:
        tree = ast.parse(block.code)
    except SyntaxError:
        pytest.skip("reported by test_block_parses")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [(node.module, [alias.name for alias in node.names])]
        elif isinstance(node, ast.Import):
            modules = [(alias.name, []) for alias in node.names]
        else:
            continue
        for module_name, names in modules:
            if module_name.split(".")[0] != "ovannotators":
                continue
            private = [part for part in module_name.split(".") if part.startswith("_")]
            if private and (block.page.name, module_name) not in (
                _PRIVATE_IMPORT_EXCEPTIONS
            ):
                pytest.fail(f"{block.id} imports private module {module_name}")
            module = importlib.import_module(module_name)
            for name in names:
                assert not (name.startswith("_") and not name.startswith("__")), (
                    f"{block.id} imports private name {module_name}.{name}"
                )
                assert hasattr(module, name), (
                    f"{block.id} imports missing name {module_name}.{name}"
                )


@pytest.mark.parametrize("fence", ["```", "````", "~~~", "~~~~~"])
def test_markdown_fence_forms(tmp_path, fence):
    page = tmp_path / "example.md"
    page.write_text(f"# Example\n{fence}python\nvalue = 1\n{fence}\n", encoding="utf-8")
    blocks = _md_blocks(page)
    assert len(blocks) == 1
    assert blocks[0].code == "value = 1"
    assert blocks[0].line == 2


def test_markdown_shorter_and_other_fences_do_not_close_block(tmp_path):
    page = tmp_path / "example.md"
    code = "text = '''\n```\n~~~~\n'''"
    page.write_text(f"````python\n{code}\n`````\n", encoding="utf-8")
    assert [block.code for block in _md_blocks(page)] == [code]


def test_markdown_unterminated_python_block(tmp_path):
    page = tmp_path / "example.md"
    page.write_text("~~~python\nvalue = 1\n", encoding="utf-8")
    assert [block.code for block in _md_blocks(page)] == ["value = 1"]


@pytest.mark.parametrize("status", [2, -9])
def test_abnormal_ruff_exit_fails_before_json_parsing(
    monkeypatch, tmp_path_factory, status
):
    monkeypatch.setattr(shutil, "which", lambda name: "/unused/ruff")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=status, stdout="", stderr="ruff invocation failed"
        ),
    )
    with pytest.raises(pytest.fail.Exception, match="ruff invocation failed"):
        lint_results.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("indent", [" ", "  ", "   "])
@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_markdown_indented_closer_preserves_next_block(tmp_path, indent, fence):
    page = tmp_path / "example.md"
    page.write_text(
        f"{fence}python\nfirst = 1\n{indent}{fence}\n"
        f"{fence}python\nsecond = 2\n{fence}\n",
        encoding="utf-8",
    )
    assert [block.code for block in _md_blocks(page)] == ["first = 1", "second = 2"]


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_markdown_opening_indent_is_removed_from_body(tmp_path, indent, fence):
    page = tmp_path / "example.md"
    page.write_text(
        f"{indent}{fence}python\n{indent}def render():\n"
        f"{indent}    return 1\n{indent}{fence}\n",
        encoding="utf-8",
    )
    assert [block.code for block in _md_blocks(page)] == ["def render():\n    return 1"]
