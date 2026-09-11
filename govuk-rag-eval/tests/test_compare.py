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


# --- promoting a baseline -----------------------------------------------
# baseline_from_results used to live as a heredoc in the workflow, where it
# drifted out of step with the reader below it and emitted judge metrics with
# no judge_backend. That baseline looked complete and gated nothing. These
# tests hold the producer and the consumer together.


def _full_results():
    """A results payload shaped like the one src.evaluate writes."""
    return {
        "dataset_version": CONFIG_VERSION,
        "retrieval": {
            "hit_at_5": 0.90,
            "mrr": 0.70,
            "context_recall_at_10": 0.91,
            "served_context_recall": 0.90,
        },
        "retrieval_detail": {
            "n_questions": 65,
            "n_answerable": 41,
            "n_negatives": 24,
            "per_question": [{"id": "q1"}],  # detail must NOT reach the baseline
        },
        # Roughly what the local NLI grader returns. Deliberately real-ish:
        # a RAGAS-scale fixture (faithfulness 0.95) would pass these tests while
        # hiding that the floors no longer match the grader in use.
        "judge": {
            "faithfulness": 0.6889,
            "answer_relevancy": 0.8391,
            "context_precision": 0.8093,
            "answer_correctness": 0.7873,
        },
        "judge_runs": 1,
        "judge_backend": "nli",
        "judge_detail": {"spread": {"faithfulness": {"min": 0.70, "max": 0.70}}},
        "estimated_cost_usd": 0.12,
    }


def test_promoted_baseline_carries_every_key_the_gate_reads():
    baseline = compare.baseline_from_results(_full_results())
    missing = [k for k in compare.BASELINE_KEYS_READ if k not in baseline]
    assert not missing, f"baseline_from_results omits {missing}, so the gate cannot use it"


def test_promoted_baseline_gates_rather_than_reporting(tmp_path):
    """The real regression: promote a baseline, then gate the same run against it."""
    results = _full_results()
    baseline = compare.baseline_from_results(results)
    code, report = _run(tmp_path, results, baseline)
    assert code == 0
    assert "reporting only" not in report
    assert "predates grader tracking" not in report
    assert "Gate passed" in report


def test_promoted_baseline_drops_per_question_detail():
    baseline = compare.baseline_from_results(_full_results())
    assert "per_question" not in baseline["retrieval_detail"]
    assert baseline["retrieval_detail"]["n_questions"] == 65


def test_promoting_a_retrieval_only_run_omits_the_judge_block():
    results = _full_results()
    del results["judge"]
    baseline = compare.baseline_from_results(results)
    assert "judge" not in baseline
    assert "judge_backend" not in baseline
    assert baseline["retrieval"]["hit_at_5"] == 0.90


def test_the_committed_baseline_passes_its_own_gate(tmp_path):
    """The committed baseline, gated against itself, must pass.

    This is the check that was missing when the grader changed. The judge
    floors were RAGAS-scale (faithfulness >= 0.85) and the NLI grader scores
    0.689 on the same unregressed system, so the first green run on main would
    have gone red on the floor alone. Anything that moves the baseline or the
    floors out of step with each other fails here instead of in CI.
    """
    baseline = json.loads((ROOT / "results" / "baseline.json").read_text())
    assert baseline.get("judge_backend"), "baseline must record its grader or the judge suite never gates"
    code, report = _run(tmp_path, baseline, baseline)
    assert "reporting only" not in report, report
    assert code == 0, report


# --- tolerance calibration ----------------------------------------------
# The bands in eval_config.yaml are a claim about noise, and the baseline's
# _provenance.cross_run_spread is the measurement behind that claim. Nothing
# stopped the two drifting apart, and they did: faithfulness' band was moved
# twice against a sample RANGE, which is a biased estimator (max minus min can
# only grow with n), before the rule was corrected to 3 sd.


def _judge_config():
    return yaml.safe_load(THRESHOLDS.read_text())["suites"]["judge"]["metrics"]


def _measured():
    baseline = json.loads((ROOT / "results" / "baseline.json").read_text())
    return baseline["_provenance"]["cross_run_spread"]


def test_every_gated_judge_band_clears_three_sigma():
    """A band below 3 sd of the run-to-run mean will fire on noise alone."""
    measured = _measured()
    for metric, rules in _judge_config().items():
        if rules.get("gating") is False:
            continue
        band = rules.get("max_relative_drop")
        stats = measured.get(metric)
        assert isinstance(stats, dict), f"{metric} is gated but has no measurement"
        noise = stats["sd_over_mean"]
        if band is None or not noise:
            continue
        assert band >= 3 * noise, (
            f"{metric}: band {band:.1%} is {band / noise:.1f} sd of measured "
            f"run-to-run noise ({noise:.2%}); needs >= 3 sd or it fails on variance"
        )


def test_no_band_is_tighter_than_the_two_percent_floor():
    """Below 2% the band is inside the measurement error of a handful of runs."""
    for metric, rules in _judge_config().items():
        if rules.get("gating") is False:
            continue
        band = rules.get("max_relative_drop")
        if band is not None:
            assert band >= 0.02, f"{metric}: band {band:.1%} is below the 2% floor"


def test_the_measurement_covers_every_gated_judge_metric():
    """A band with no measurement behind it is a guess wearing a number."""
    measured = _measured()
    for metric, rules in _judge_config().items():
        if rules.get("gating") is False or rules.get("max_relative_drop") is None:
            continue
        assert metric in measured, f"{metric} is gated but has no cross_run_spread entry"
        assert measured[metric].get("sd") is not None, f"{metric} has no measured sd"


def test_the_floor_sits_below_the_band_not_inside_it():
    """The floor is a catastrophe backstop; if it fires first the band is dead code."""
    baseline = json.loads((ROOT / "results" / "baseline.json").read_text())
    for metric, rules in _judge_config().items():
        if rules.get("gating") is False:
            continue
        floor, band = rules.get("absolute_floor"), rules.get("max_relative_drop")
        value = baseline.get("judge", {}).get(metric)
        if None in (floor, band, value):
            continue
        drop_to_floor = (value - floor) / value
        assert drop_to_floor > band, (
            f"{metric}: floor {floor} is only {drop_to_floor:.1%} under the baseline "
            f"{value:.3f}, inside the {band:.0%} band — the floor would fire first"
        )
