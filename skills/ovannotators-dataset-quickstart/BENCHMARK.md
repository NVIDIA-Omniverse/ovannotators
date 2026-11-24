# Skill Benchmark: ovannotators-dataset-quickstart

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `ovannotators-dataset-quickstart`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (4 positive, 4 negative)
- Dataset digest: `sha256:9586cbcdd948af308ee82e456e7977f701cade05b062ed65069c1e6fb5bc2875` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| ovannotators-dataset-quickstart | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| ovannotators-dataset-quickstart | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| ovannotators-dataset-quickstart | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| ovannotators-dataset-quickstart | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| ovannotators-dataset-quickstart | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `ovannotators-dataset-quickstart`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75.9% — baseline ran, but no comparable score was available; uplift unavailable | 78.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 31.3% (-18.7 points) | 87.5% → 68.8% (-18.7 points) |
| Correctness | 77.5% → 100.0% (+22.5 points) | 70.0% → 97.5% (+27.5 points) |
| Discoverability | 88.8% — baseline ran, but no comparable score was available; uplift unavailable | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 71.9% → 85.4% (+13.5 points) | 63.5% → 83.3% (+19.8 points) |
| Efficiency | 74.1% — baseline ran, but no comparable score was available; uplift unavailable | 61.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 90.2% → 86.7% (-3.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 78.0% → 80.5% (+2.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 7,408,851 | 4,740,966 | +2,667,885 | +56.27% | skill 8/8; base 8/8 |
| claude-code | ovannotators-dataset-quickstart-cpu-first-run | 750,219 | 874,514 | -124,295 | -14.21% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-gpu-fallback | 720,077 | 1,139,687 | -419,610 | -36.82% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-gpu-usd-scene | 151,382 | 107,668 | +43,714 | +40.60% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-macos | 827,853 | 1,006,847 | -178,994 | -17.78% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-negative-after-first-run | 642,908 | 607,212 | +35,696 | +5.88% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-negative-new-format | 3,407,095 | 179,200 | +3,227,895 | +1801.28% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-negative-newcomer-yolo | 799,808 | 644,610 | +155,198 | +24.08% | skill 1/1; base 1/1 |
| claude-code | ovannotators-dataset-quickstart-negative-train-model | 109,509 | 181,228 | -71,719 | -39.57% | skill 1/1; base 1/1 |
| codex | All cases | 6,362,530 | 3,187,281 | +3,175,249 | +99.62% | skill 8/8; base 8/8 |
| codex | ovannotators-dataset-quickstart-cpu-first-run | 780,696 | 763,552 | +17,144 | +2.25% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-gpu-fallback | 520,617 | 1,038,400 | -517,783 | -49.86% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-gpu-usd-scene | 62,345 | 59,118 | +3,227 | +5.46% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-macos | 524,568 | 512,041 | +12,527 | +2.45% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-negative-after-first-run | 470,764 | 151,561 | +319,203 | +210.61% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-negative-new-format | 3,252,659 | 54,392 | +3,198,267 | +5880.03% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-negative-newcomer-yolo | 694,805 | 566,848 | +127,957 | +22.57% | skill 1/1; base 1/1 |
| codex | ovannotators-dataset-quickstart-negative-train-model | 56,076 | 41,369 | +14,707 | +35.55% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 13,771,381 | 7,928,247 | +5,843,134 | +73.70% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/ovannotators-dataset-quickstart/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/ovannotators-dataset-quickstart/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (625 chars, recommend 50-150) (`skills/ovannotators-dataset-quickstart/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/ovannotators-dataset-quickstart/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/ovannotators-dataset-quickstart/SKILL.md`)
- 2 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
