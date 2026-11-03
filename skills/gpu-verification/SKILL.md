---
name: gpu-verification
description: Run the GPU test suite locally against the real ovrtx renderer and report the results on the pull request. Use when changing renderer-facing code, bumping ovrtx/ovstage/PyNvVideoCodec or other GPU dependencies, touching render var extraction or golden data, or when asked to verify something CI cannot run.
---

# GPU Verification

Default CI has no GPU and no `ovrtx`, so it cannot execute the renderer, NVENC,
or golden comparison paths. Tests covering them are marked `gpu` and deselected,
and tests that merely `import ovrtx` skip. A green PR therefore says nothing
about whether the renderer integration works. This skill covers closing that
gap locally and recording the result on the PR.

## When to Use

A local GPU run is **required**, and its output belongs in the PR, when a change
touches any of:

- `src/ovannotators/renderer_outputs/` — render var extraction and identity.
- `ovrtx`, `ovstage`, `warp-lang`, `PyNvVideoCodec`, or `nvidia-*` version
  constraints in `pyproject.toml` or `uv.lock`.
- `src/ovannotators/_internal/codecs.py` — NVENC and EXR paths.
- Golden data under `tests/golden/`, or the writers that produce it.

It is optional for pure CPU work: writer formatting logic, config validation,
backends with faked transports, docs, and repo tooling.

## Prerequisites

Check the machine can actually do the work before trusting a run:

```bash
nvidia-smi                       # a CUDA device must be present
uv run python -c "import warp; warp.init()"   # must list a cuda:N device
```

`ovrtx` on PyPI is a `wheel-stub` redirector, not the renderer -- the real
wheels are published on NVIDIA's index. Install the `dev` extra pointing at it:

```bash
uv sync --extra dev \
  --extra-index-url https://pypi.nvidia.com/ \
  --index-strategy unsafe-best-match
```

Then confirm you have the genuine renderer rather than a stub, because every
`gpu` test skips silently without it:

```bash
uv run python -c "
from ovannotators._internal.optional_deps import ovrtx_available
import ovrtx; print(ovrtx_available(), hasattr(ovrtx, 'Renderer'))"
```

Both values must be `True`. `ovrtx` and `ovstage` must come from the same
release train; mixing them mixes their bundled OpenUSD runtimes.

## Instructions

1. Run the GPU lane and keep the JUnit sidecar:

   ```bash
   uv run pytest -m gpu -v --junitxml=_build/gpu-local.xml
   ```

2. Run the default suite too. With `ovrtx` installed it is **not** the suite CI
   ran: tests that skip in CI for want of the renderer now execute, and some are
   not marked `gpu`. This is where marker gaps surface.

   ```bash
   uv run pytest -q
   ```

3. Account for every skip with `-rs` before calling a run clean. Environmental
   skips (`torch not installed`, `readers venv interpreter not found`) are
   expected; a skip reading `ovrtx not installed` means step 2 of
   **Prerequisites** failed and the run proved nothing.

4. Fix what the run finds, then re-run both suites to green. Treat a failure
   that CI could never have caught as the point of the exercise, not as noise.

5. Post the results to the PR as a comment (see **Output Format**).

## Golden Data Discipline

`UPDATE_GOLDEN=1 uv run pytest tests/test_integration_golden.py -m gpu`
rewrites every golden, including images that were passing. Do not commit that
wholesale.

- Regenerate, then read `git diff` and keep only the files whose change you can
  name and justify.
- Goldens are Git LFS objects, so `git show HEAD:<path>` yields a pointer; pipe
  it through `git lfs smudge` to compare real bytes.
- Image goldens are compared on MAE/RMSE/PSNR thresholds, not byte equality.
  If a regenerated image still passes against the committed one with margin,
  the difference is sampling noise -- revert it rather than banking new
  binaries and weakening the baseline.
- `git checkout -- <path>` the files you are reverting.

## Output Format

Comment on the PR with a short table of both suites (counts and durations), the
ovrtx/ovstage versions and GPU/driver the run used, each skip with its reason,
and -- if the run caught anything -- what failed and how it was fixed. State
plainly that CI could not have caught it.

## Scripts

None. `tests/generate_golden.py` regenerates golden fixtures; it needs the real
renderer and the same discipline as above.

## Limitations

- Results are specific to the driver, GPU, and renderer build used; record them
  so a later reader can tell whether a difference is environmental.
- NVENC rejects frames below roughly 160 px wide at initialization
  (`NV_ENC_ERR_INVALID_PARAM`); a too-small test frame fails in the encoder
  rather than in what the test meant to check.
- A warm renderer caches kernels, so a second run is much faster than the
  first. A fast run is not evidence that work was skipped.

## Troubleshooting

- `KeyError: 'LdrColor'` — ovrtx 0.5 keys `FrameOutput.render_vars` by RenderVar
  prim path. Resolve by `sourceName` via `OvrtxOutput`, or, in tests that map
  buffers directly, `tests._ovstage_session.render_vars_by_source_name()`.
- Sensor products fail to map — call `ovrtx.register_schema_paths()` before
  creating or populating an `ovstage.Stage`.
- `TypeError` buffering encoded video — PyNvVideoCodec 2.2 returns packets as a
  list of mappings, 2.1 returns bytes; `NvVideoEncoder` accepts both.

## References

- `skills/test-classification/SKILL.md` — which marker a test should carry, and
  why an `ovrtx`-dependent test needs `gpu`.
- `skills/test-triage/SKILL.md` — categorizing a failure once you have one.
- `AGENTS.md` — changelog rule; a user-visible fix found this way needs an entry.
