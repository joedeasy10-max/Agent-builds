"""Step 6 (offline slice): prove the gate BLOCKS a retrieval regression.

The chunk-size regression (512 -> 2000 in the real config) is demonstrable
deterministically: a larger chunk size collapses the fixture pages and
invalidates the golden set's `#chunk-N` references, so retrieval metrics drop
below their floor and `compare.py` fails the gate (exit 1).

The other two demos in BUILD.md need the real corpus/model and are NOT asserted
here:
  * top_k 5 -> 2: on the 3-record starter set every relevant chunk is already at
    rank 1, so lowering top_k doesn't move the metric. Needs the full golden set.
  * embedding swap to a weaker model: needs real embeddings (a key).
"""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from src.config import from_dict
from src.evaluate import main as evaluate_main
from src.ingest import build_index

ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIG = ROOT / "configs" / "eval_config.yaml"
GOLDEN = ROOT / "data" / "golden" / "questions.jsonl"


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


compare = _load_compare()


def _cfg(chunk_size):
    return from_dict({
        "seed": 42,
        "chunking": {"splitter": "simple", "chunk_size": chunk_size, "chunk_overlap": 20},
        "embedding": {"provider": "hashing", "dimensions": 128},
        "store": {"type": "flat"},
        "retrieval": {"retriever": "dense", "top_k": 10},
    })


def _write_cfg(tmp_path, cfg, name):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump({
        "seed": cfg.seed,
        "chunking": asdict(cfg.chunking) | {"separators": list(cfg.chunking.separators)},
        "embedding": asdict(cfg.embedding),
        "store": asdict(cfg.store),
        "retrieval": asdict(cfg.retrieval),
    }))
    return p


def _eval(tmp_path, cfg, name, fixture_pages, out):
    idx = tmp_path / f"idx-{name}"
    build_index(fixture_pages, cfg, idx)
    evaluate_main([
        "--suite", "retrieval", "--dataset", str(GOLDEN),
        "--config", str(_write_cfg(tmp_path, cfg, name)), "--index", str(idx), "--out", str(out),
    ])


def _gate(tmp_path, current, baseline):
    md = tmp_path / "report.md"
    argv = ["compare.py", "--current", str(current), "--baseline", str(baseline),
            "--thresholds", str(EVAL_CONFIG), "--markdown", str(md)]
    old = sys.argv
    sys.argv = argv
    try:
        return compare.main(), md.read_text()
    finally:
        sys.argv = old


def test_chunk_size_regression_is_blocked(tmp_path, fixture_pages):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _eval(tmp_path, _cfg(200), "baseline", fixture_pages, baseline)   # healthy
    _eval(tmp_path, _cfg(2000), "regressed", fixture_pages, current)  # chunk-size regression

    code, report = _gate(tmp_path, current, baseline)
    assert code == 1, report                       # gate blocks the PR
    assert "Gate failed" in report


def test_no_regression_passes(tmp_path, fixture_pages):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _eval(tmp_path, _cfg(200), "baseline", fixture_pages, baseline)
    _eval(tmp_path, _cfg(200), "same", fixture_pages, current)

    code, report = _gate(tmp_path, current, baseline)
    assert code == 0, report
    assert "Gate passed" in report
