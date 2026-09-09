"""End-to-end gate self-test: evaluate -> baseline -> compare.py -> exit code.

This is the whole step-5 gate chain, run deterministically and offline (hashing
embedder, no network, no LLM), over the committed starter golden set. It proves:

  * a run that matches the baseline passes the gate (exit 0);
  * a regressed run below an absolute floor fails the gate (exit 1);
  * with no baseline, the gate reports only (exit 0).

It uses the REAL committed configs/eval_config.yaml as the thresholds, so the
gate's wiring — not just the metric math — is what's under test.
"""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIG = ROOT / "configs" / "eval_config.yaml"
GOLDEN = ROOT / "data" / "golden" / "questions.jsonl"


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_compare()


def _write_config(tmp_path, cfg):
    p = tmp_path / "retrieval.yaml"
    p.write_text(yaml.safe_dump({
        "seed": cfg.seed,
        "chunking": asdict(cfg.chunking) | {"separators": list(cfg.chunking.separators)},
        "embedding": asdict(cfg.embedding),
        "store": asdict(cfg.store),
        "retrieval": asdict(cfg.retrieval),
    }))
    return p


def _evaluate_to(tmp_path, cfg_path, index_dir, out):
    from src.evaluate import main as evaluate_main
    evaluate_main([
        "--suite", "retrieval", "--dataset", str(GOLDEN),
        "--config", str(cfg_path), "--index", str(index_dir), "--out", str(out),
    ])


def _compare(tmp_path, current, baseline):
    md = tmp_path / "report.md"
    argv = [
        "compare.py", "--current", str(current), "--baseline", str(baseline),
        "--thresholds", str(EVAL_CONFIG), "--markdown", str(md),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        code = compare.main()
    finally:
        sys.argv = old
    return code, md.read_text()


def _setup(tmp_path, fixture_pages, test_config):
    from src.ingest import build_index
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    cfg_path = _write_config(tmp_path, test_config)
    return idx, cfg_path


def test_gate_passes_when_run_matches_baseline(tmp_path, fixture_pages, test_config):
    idx, cfg_path = _setup(tmp_path, fixture_pages, test_config)
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _evaluate_to(tmp_path, cfg_path, idx, baseline)
    _evaluate_to(tmp_path, cfg_path, idx, current)  # deterministic -> identical

    code, report = _compare(tmp_path, current, baseline)
    assert code == 0, report
    assert "Gate passed" in report


def test_gate_fails_on_regression_below_floor(tmp_path, fixture_pages, test_config):
    idx, cfg_path = _setup(tmp_path, fixture_pages, test_config)
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _evaluate_to(tmp_path, cfg_path, idx, baseline)
    _evaluate_to(tmp_path, cfg_path, idx, current)

    # Simulate a retrieval regression: drop hit_at_5 below its absolute floor.
    data = json.loads(current.read_text())
    data["retrieval"]["hit_at_5"] = 0.10
    current.write_text(json.dumps(data))

    code, report = _compare(tmp_path, current, baseline)
    assert code == 1, report
    assert "below absolute floor" in report
    assert "Gate failed" in report


def test_gate_reports_only_without_baseline(tmp_path, fixture_pages, test_config):
    idx, cfg_path = _setup(tmp_path, fixture_pages, test_config)
    current = tmp_path / "current.json"
    empty = tmp_path / "baseline.json"
    empty.write_text("{}")
    _evaluate_to(tmp_path, cfg_path, idx, current)

    code, report = _compare(tmp_path, current, empty)
    assert code == 0, report
    assert "No baseline" in report


def test_committed_eval_config_version_matches_golden(tmp_path):
    """The gate can only compare when eval_config.version == the golden version."""
    from src.golden import dataset_version, load_golden

    cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    assert cfg["dataset"]["version"] == dataset_version(load_golden(GOLDEN))
