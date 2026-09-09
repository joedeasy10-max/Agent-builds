"""Compare an evaluation run against the baseline on main and gate the PR.

Exit codes:
    0 - all gated metrics within tolerance
    1 - at least one gated metric breached a floor or dropped too far
    2 - could not compare (missing baseline, dataset version mismatch)

Design notes worth defending in an interview:

* Retrieval metrics are deterministic, so they gate on tight tolerances.
  Judge metrics come from an LLM grader and move a few percent between
  identical runs, so they gate on a wider band and on a median of N runs.
  Gating noisy metrics tightly produces a pipeline that fails at random,
  which teaches everyone to ignore it.
* A baseline is only meaningful within one golden-dataset version. Changing
  the dataset changes what the number means, so we warn and skip the gate
  rather than compare two incompatible runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text().strip()
    return json.loads(text) if text else {}


def arrow(delta: float, tolerance: float) -> str:
    if abs(delta) < tolerance / 2:
        return "flat"
    return "up" if delta > 0 else "DOWN"


def evaluate_metric(name: str, current: float, baseline: float | None, rules: dict):
    """Return (passed, reason) for one metric."""
    floor = rules.get("absolute_floor")
    max_drop = rules.get("max_relative_drop")

    if floor is not None and current < floor:
        return False, f"below absolute floor {floor:.3f}"

    if baseline is not None and max_drop is not None and baseline > 0:
        relative_drop = (baseline - current) / baseline
        if relative_drop > max_drop:
            return False, (
                f"dropped {relative_drop:.1%} vs baseline "
                f"(tolerance {max_drop:.1%})"
            )

    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.thresholds.read_text())
    current = load_json(args.current)
    baseline = load_json(args.baseline)

    if not current:
        print("No current results found — did the eval step run?", file=sys.stderr)
        return 2

    expected_version = config["dataset"]["version"]
    current_version = current.get("dataset_version")
    baseline_version = baseline.get("dataset_version")

    lines: list[str] = ["## RAG evaluation", ""]
    failures: list[str] = []
    comparable = True

    if current_version != expected_version:
        print(
            f"Results carry dataset {current_version!r}, config expects "
            f"{expected_version!r}.",
            file=sys.stderr,
        )
        return 2

    if not baseline:
        lines.append("No baseline on the target branch — reporting only.")
        lines.append("")
        comparable = False
    elif baseline_version != current_version:
        lines.append(
            f"Golden dataset moved from `{baseline_version}` to "
            f"`{current_version}`. Metrics are not comparable across versions, "
            "so the gate is reporting only on this run."
        )
        lines.append("")
        comparable = False

    lines.append("| Metric | Baseline | This PR | Delta | Gate |")
    lines.append("| --- | ---: | ---: | ---: | --- |")

    for suite_name, suite in config["suites"].items():
        suite_gating = suite.get("gating", True)
        suite_current = current.get(suite_name)
        if not suite_current:
            lines.append(f"| _{suite_name} suite_ | — | not run | — | skipped |")
            continue

        suite_baseline = baseline.get(suite_name, {}) if baseline else {}

        for metric, rules in suite["metrics"].items():
            if metric not in suite_current:
                continue
            value = float(suite_current[metric])
            prior = suite_baseline.get(metric)
            prior = float(prior) if prior is not None else None

            gated = suite_gating and rules.get("gating", True) and comparable
            passed, reason = (
                evaluate_metric(metric, value, prior, rules)
                if gated
                else (True, "")
            )

            if prior is None:
                delta_cell = "—"
            else:
                delta = value - prior
                tolerance = rules.get("max_relative_drop", 0.02)
                delta_cell = f"{delta:+.3f} {arrow(delta, tolerance)}"

            if not gated:
                gate_cell = "report only"
            elif passed:
                gate_cell = "pass"
            else:
                gate_cell = f"**FAIL** — {reason}"
                failures.append(f"{suite_name}.{metric}: {reason}")

            prior_cell = f"{prior:.3f}" if prior is not None else "—"
            lines.append(
                f"| `{metric}` | {prior_cell} | {value:.3f} | {delta_cell} | {gate_cell} |"
            )

    lines.append("")

    regressions = current.get("per_question_regressions", [])
    limit = config["reporting"]["worst_examples_in_comment"]
    if regressions:
        lines.append(f"### Sharpest per-question regressions (top {limit})")
        lines.append("")
        for item in regressions[:limit]:
            lines.append(
                f"- `{item.get('id')}` — {item.get('question', '')[:110]} "
                f"({item.get('delta', 0):+.2f})"
            )
        lines.append("")

    if failures:
        lines.append(f"**Gate failed on {len(failures)} metric(s).**")
        lines.append("")
        lines.append(
            "If this drop is intentional, update the baseline in the same PR "
            "and say why in the description."
        )
    elif comparable:
        lines.append("**Gate passed.**")

    lines.append("")
    lines.append(
        f"<sub>dataset `{current_version}` · "
        f"judge runs: {current.get('judge_runs', 'n/a')} · "
        f"est. cost ${current.get('estimated_cost_usd', 0):.2f}</sub>"
    )

    report = "\n".join(lines)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(report)
    print(report)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
