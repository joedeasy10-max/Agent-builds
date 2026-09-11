"""Tests for scripts/compare.py — the gate. No network, no LLM (BUILD.md).

The gate's exit code is the whole contract with CI:
    0 = within tolerance, 1 = a gated metric breached, 2 = cannot compare.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = ROOT / "configs" / "eval_config.yaml"

# Read from the config rather than pinned here. These tests are about compare.py's
# version logic, not about which version the dataset happens to be on, and a
# hardcoded literal has now broken them on two separate dataset bumps.
import yaml  # noqa: E402

CONFIG_VERSION = yaml.safe_load(THRESHOLDS.read_text())["dataset"]["version"]
OTHER_VERSION = "v0" if CONFIG_VERSION != "v0" else "v99"


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_compare()


def _run(tmp_path, current: dict | None, baseline: dict | None) -> tuple[int, str]:
    cur = tmp_path / "current.json"
    base = tmp_path / "baseline.json"
    md = tmp_path / "report.md"
    cur.write_text(json.dumps(current) if current is not None else "")
    base.write_text(json.dumps(baseline) if baseline is not None else "")
    argv = [
        "compare.py",
        "--current", str(cur),
        "--baseline", str(base),
        "--thresholds", str(THRESHOLDS),
        "--markdown", str(md),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        code = compare.main()
    finally:
        sys.argv = old
    return code, (md.read_text() if md.exists() else "")


def _current(hit_at_5=0.90, version=None):  # defaults to the configured version
    return {
        "dataset_version": CONFIG_VERSION if version is None else version,
        "retrieval": {"hit_at_5": hit_at_5, "mrr": 0.70, "context_recall_at_10": 0.91},
    }


def test_within_tolerance_passes(tmp_path):
    code, report = _run(tmp_path, _current(hit_at_5=0.895), _current(hit_at_5=0.90))
    assert code == 0
    assert "Gate passed" in report


def test_below_absolute_floor_fails(tmp_path):
    code, report = _run(tmp_path, _current(hit_at_5=0.50), _current(hit_at_5=0.90))
    assert code == 1
    assert "below absolute floor" in report
    assert "Gate failed" in report


def test_relative_drop_beyond_tolerance_fails(tmp_path):
    # hit_at_5 floor is 0.82, tolerance 2%. 0.87 clears the floor but is a
    # ~3.3% drop from 0.90 -> should fail on the relative-drop rule.
    code, report = _run(tmp_path, _current(hit_at_5=0.87), _current(hit_at_5=0.90))
    assert code == 1
    assert "dropped" in report


def test_no_baseline_reports_only(tmp_path):
    code, report = _run(tmp_path, _current(), {})
    assert code == 0
    assert "No baseline" in report


def test_dataset_version_mismatch_baseline_vs_current_reports_only(tmp_path):
    # current matches config, baseline is an older version -> not comparable.
    code, report = _run(tmp_path, _current(), _current(version=OTHER_VERSION))
    assert code == 0
    assert "not comparable across versions" in report


def test_current_version_disagrees_with_config_cannot_compare(tmp_path):
    code, _ = _run(tmp_path, _current(version=OTHER_VERSION),
                   _current(version=OTHER_VERSION))
    assert code == 2


def test_missing_current_cannot_compare(tmp_path):
    code, _ = _run(tmp_path, None, _current())
    assert code == 2


# --- judge metrics are only comparable against the SAME grader --------------


def _judge_payload(backend, faithfulness=0.90, version=None):
    import json as _json
    from pathlib import Path as _Path
    return {
        "dataset_version": version or CONFIG_VERSION,
        "retrieval": {"hit_at_5": 0.90, "mrr": 0.72,
                      "context_recall_at_10": 0.93, "served_context_recall": 0.90},
        "judge": {"faithfulness": faithfulness, "answer_relevancy": 0.89,
                  "context_precision": 0.82, "answer_correctness": 0.83},
        "judge_backend": backend,
        "judge_runs": 1,
    }


def _run_compare(tmp_path, current, baseline):
    import json, subprocess, sys
    (tmp_path / "cur.json").write_text(json.dumps(current))
    (tmp_path / "base.json").write_text(json.dumps(baseline))
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare.py"),
         "--current", str(tmp_path / "cur.json"),
         "--baseline", str(tmp_path / "base.json"),
         "--thresholds", str(ROOT / "configs" / "eval_config.yaml"),
         "--markdown", str(out)],
        capture_output=True, text=True, cwd=ROOT,
    )
    return proc, out.read_text() if out.exists() else ""


def test_a_grader_change_reports_instead_of_gating(tmp_path):
    """An NLI score and an LLM's judgement are different scales.

    Gating one against the other would invent a regression out of nothing — a
    faithfulness of 0.82 from entailment is not "worse" than 0.95 from a model.
    """
    current = _judge_payload("nli", faithfulness=0.55)      # far below the floor
    baseline = _judge_payload("ragas", faithfulness=0.95)
    proc, report = _run_compare(tmp_path, current, baseline)
    assert "not comparable" in report
    assert proc.returncode == 0, "a grader change must not fail the gate"


def test_the_same_grader_still_gates(tmp_path):
    current = _judge_payload("nli", faithfulness=0.10)
    baseline = _judge_payload("nli", faithfulness=0.95)
    proc, report = _run_compare(tmp_path, current, baseline)
    assert "not comparable" not in report
    assert proc.returncode == 1, "a real drop under the same grader must fail"


def test_a_baseline_predating_grader_tracking_reports_only(tmp_path):
    current = _judge_payload("nli", faithfulness=0.10)
    baseline = _judge_payload("nli")
    del baseline["judge_backend"]
    proc, report = _run_compare(tmp_path, current, baseline)
    assert "predates grader tracking" in report
    assert proc.returncode == 0


def test_retrieval_still_gates_when_the_grader_changed(tmp_path):
    """Only the judge suite is affected; retrieval is grader-independent."""
    current = _judge_payload("nli")
    current["retrieval"]["hit_at_5"] = 0.10
    baseline = _judge_payload("ragas")
    proc, report = _run_compare(tmp_path, current, baseline)
    assert proc.returncode == 1, "retrieval must still block"
