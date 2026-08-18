---
name: test-creation
description: Create or update ovannotators tests following repository conventions: meaningful assertions, correct pytest markers, golden-data discipline, and local/CI validation commands.
---

# Test Creation

Use this when adding or changing tests for ovannotators.

## Before writing

1. Read `AGENTS.md` for repo rules and SPDX expectations.
2. Read the source under test; avoid testing guesses.
3. If examples/docs are involved, follow the snippet-source convention in `tests/docs/AGENTS.md`.
4. If output/reference data changes, use the golden-data blessing workflow: agents propose; humans approve.

## Test quality rules

- Every test needs meaningful assertions; do not add coverage-padding tests.
- Prefer small focused fixtures over global mutable state.
- Mock dependencies, not the behavior under test.
- Skip optional dependencies with explicit reasons.
- New GPU/render tests must be marked `gpu`; full flows must also be marked `e2e`.
- Benchmark tests belong under `tests/benchmarks/` and should not run in default pytest.

## Where tests belong

- Pure library behavior: `tests/test_<module_or_feature>.py`
- Writer/backend end-to-end output checks: existing writer/golden test files unless a new format needs its own file.
- Skills/docs examples: `tests/docs/python/` with snippet markers plus `tests/test_skill_snippets.py` coverage.
- Benchmarks: `tests/benchmarks/`.

## Validation commands

Run the smallest relevant subset first, then the default suite:

```bash
python -m pytest <changed-test-file> -q
python -m pytest -q
ruff check source/ tests/ scripts/
ruff format --check source/ tests/ scripts/
```

For GPU/e2e work, also collect or run the selected lane on a suitable host:

```bash
python -m pytest -m "e2e and gpu" --collect-only -q
python -m pytest -m "e2e and gpu" -q
```
