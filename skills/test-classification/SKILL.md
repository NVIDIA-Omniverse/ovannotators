---
name: test-classification
description: Determine correct pytest markers for ovannotators tests. Use when adding tests, reviewing marker correctness, splitting CPU/GPU/e2e coverage, or deciding whether a test belongs in the default PR suite, e2e smoke, GPU nightly, quarantine, or benchmarks.
---

# Test Classification

Classify tests using the taxonomy in `pyproject.toml` and `testing.yml`.

## Marker axes

- Type: `unit`, `integration`, `e2e`, `smoke`, `benchmark`
- Resource: `cpu_only`, `gpu`, `multi_gpu`, `network`
- Duration: `duration_1s`, `duration_10s`, `duration_100s`, `duration_1000s`
- Stability: `stable`, `flaky`, `known_failure_rate`, `experimental`

Defaults for unmarked tests are unit-ish, CPU-only, short, and stable. Add explicit markers when selection behavior matters.

## Decision rules

- Use `unit` for isolated utilities, config validation, serializers, and pure writer helper logic.
- Use `integration` when multiple ovannotators subsystems interact, e.g. writer + backend, renderer output extraction + writer, or optional dependency integration with skips.
- Use `e2e` for full user-visible flows: rendered scene → render vars → writer → committed output/golden, README/tutorial execution, or real transfer/inference workflows.
- Add `gpu` to anything requiring CUDA, `ovrtx`, GPU codecs, or real renderer execution. GPU tests are excluded from default PR CI and should run in the nightly/manual GPU e2e lane. A test that imports `ovrtx` but lacks the marker is a classification bug: it silently skips in CI and only runs on a machine with the renderer installed, so nothing gates it. Verify with `skills/gpu-verification/SKILL.md`, which runs both lanes against the real renderer.
- Add `network` for S3/HTTP/external-service tests unless fully faked.
- Add `benchmark` for pytest-benchmark or performance-only assertions; keep benchmarks under `tests/benchmarks/`.
- Add `flaky`, `known_failure_rate`, or `experimental` only with an owner/explanation; these are excluded from gating default runs.

## Validation

After changing markers, run:

```bash
python -m pytest <test-file> --collect-only -q
python -m pytest -m "e2e and gpu" --collect-only -q
python -m pytest -m "e2e and not gpu" --collect-only -q
```

For changed tests, also run the smallest meaningful selected subset before the full default suite.
