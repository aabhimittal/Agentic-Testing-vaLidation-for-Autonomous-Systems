"""Render suite results as Markdown (for humans/PRs) or JSON (for machines)."""

from __future__ import annotations

import json

from .runner import SuiteReport


def to_markdown(suite: SuiteReport) -> str:
    lines = ["# ATLAS Suite Report", ""]
    status = "PASSED" if suite.passed else "FAILED"
    lines.append(f"**Status:** {status} — {len(suite.reports)} scenario(s) run, "
                 f"{len(suite.failures)} failed, {len(suite.skipped_scenarios)} skipped")
    if suite.token_limit:
        lines.append(f"**Token budget:** {suite.tokens_spent}/{suite.token_limit} spent")
    lines.append("")
    lines.append("| Scenario | Steps | Result | Tokens | Compression |")
    lines.append("|---|---|---|---|---|")
    for r in suite.reports:
        result = "pass" if r.passed else "FAIL"
        lines.append(
            f"| {r.scenario} | {r.trace_steps} | {result} | "
            f"{r.tokens_charged} | {r.compression_ratio:.1f}x |"
        )
    for name in suite.skipped_scenarios:
        lines.append(f"| {name} | - | skipped (budget) | - | - |")
    for r in suite.failures:
        lines.append("")
        lines.append(f"## Failures in `{r.scenario}`")
        for v in r.validations:
            for violation in v.violations:
                lines.append(f"- `{violation.validator}`: {violation.message}")
        for m in r.metamorphic:
            if not m.passed:
                lines.append(f"- metamorphic `{m.relation}`: {m.detail}")
    return "\n".join(lines) + "\n"


def to_json(suite: SuiteReport) -> str:
    payload = {
        "passed": suite.passed,
        "tokens_spent": suite.tokens_spent,
        "token_limit": suite.token_limit,
        "skipped_scenarios": list(suite.skipped_scenarios),
        "scenarios": [
            {
                "name": r.scenario,
                "steps": r.trace_steps,
                "truncated": r.truncated,
                "passed": r.passed,
                "tokens_charged": r.tokens_charged,
                "compression_ratio": round(r.compression_ratio, 2),
                "violations": [
                    {"validator": v.validator, "timestep": vi.timestep, "message": vi.message}
                    for v in r.validations
                    for vi in v.violations
                ],
                "metamorphic": [
                    {"relation": m.relation, "passed": m.passed, "detail": m.detail}
                    for m in r.metamorphic
                ],
            }
            for r in suite.reports
        ],
    }
    return json.dumps(payload, indent=2)
