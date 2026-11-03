---
name: test-coverage
description: Run and interpret ovannotators coverage checks, coverage PR comments, coverage-gap audits, and changed-file coverage investigations.
---

# Test Coverage

Use this when asked to run coverage, explain a coverage failure, or identify missing test coverage.

## Commands

Default CI-style coverage:

```bash
python -m pytest --cov=src/ovannotators --cov-report=xml --cov-report=term-missing --cov-fail-under=90 -v
```

Local quick report:

```bash
python -m pytest --cov=src/ovannotators --cov-report=term-missing -q
```

HTML report when detailed inspection helps:

```bash
python -m pytest --cov=src/ovannotators --cov-report=html
```

## What to inspect

- Overall line coverage must stay at or above the CI threshold.
- Public writer/backend APIs should have behavior tests, not only import tests.
- Error paths and optional dependency branches should have explicit tests or documented skips.
- GPU-only paths should be covered by marker-selected GPU/e2e tests where possible.
- Golden tests should verify structure and tolerant metrics; do not bless new data without human approval.

## Reporting

When reporting coverage work, include:

- command run
- pass/fail and threshold
- location of generated artifacts (`coverage.xml`, `htmlcov/`, `_build/results.xml`)
- lowest-covered important files or branches
- concrete follow-up tests to add, if any

Do not modify production code during a coverage-only pass unless explicitly asked.
