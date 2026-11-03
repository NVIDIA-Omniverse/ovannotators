---
name: test-triage
description: Analyze ovannotators test or CI failures, categorize root cause, use JUnit sidecars/failure reports, and decide whether to fix code, adjust tests, quarantine, or escalate.
---

# Test Triage

Use this for failing local tests, PR CI failures, nightly GPU/e2e failures, or quarantine results.

## Gather evidence

- Test command and full failing traceback.
- JUnit XML: `_build/results.xml` or lane-specific XML.
- Sidecar JSON: `_build/results.json` when available.
- Failure report: `_build/failure-analysis.md` when available.
- Changed files for PR failures.

Generate sidecar/report locally:

```bash
python scripts/generate_sidecar.py _build/results.xml --output _build/results.json
python scripts/failure_analysis.py _build/results.json --output _build/failure-analysis.md
```

For PR context:

```bash
git diff --name-only origin/main...HEAD > _build/changed-files.txt
python scripts/failure_analysis.py _build/results.json --diff _build/changed-files.txt --output _build/failure-analysis.md
```

## Categories

Assign one primary category before fixing:

- `PR_CODE`: changed code caused the failure.
- `TEST_BUG`: test expectation/setup is wrong.
- `UPSTREAM`: optional dependency, renderer, codec, or external service changed/broke.
- `INFRASTRUCTURE`: runner, filesystem, network, permissions, timeout, package install.
- `FLAKY`: nondeterministic; rerun evidence or history needed.
- `PRE_EXISTING`: unrelated to the PR.
- `UNKNOWN`: needs manual investigation.

## Rules

- Reproduce the smallest failing test before broad changes.
- Do not mark tests flaky/xfail just to make CI green.
- Do not re-bless golden data unless the behavior change is intended and a human approves. Re-bless only the files whose change you can name; see the golden discipline in `skills/gpu-verification/SKILL.md`.
- A green CI run does not clear renderer, NVENC, or golden paths: CI has no GPU or `ovrtx`. Reproduce those locally per `skills/gpu-verification/SKILL.md` before concluding `PRE_EXISTING` or `INFRASTRUCTURE`.
- If a GPU/e2e failure only reproduces on a specific runner, preserve artifacts and environment details.
- After three unsuccessful fix attempts, stop and escalate with a concise causal-chain summary.

## Report format

Include:

- failing test/lane
- category
- root cause and confidence
- evidence/artifacts
- proposed fix or escalation owner
