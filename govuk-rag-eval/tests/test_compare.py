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


def _current(hit_at_5=0.90, version="v1"):  # matches configs/eval_config.yaml
    return {
        "dataset_version": version,
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
    # current matches config (v1), baseline is an older v0 -> not comparable.
    code, report = _run(tmp_path, _current(version="v1"), _current(version="v0"))
    assert code == 0
    assert "not comparable across versions" in report


def test_current_version_disagrees_with_config_cannot_compare(tmp_path):
    code, _ = _run(tmp_path, _current(version="v2"), _current(version="v2"))
    assert code == 2


def test_missing_current_cannot_compare(tmp_path):
    code, _ = _run(tmp_path, None, _current())
    assert code == 2
