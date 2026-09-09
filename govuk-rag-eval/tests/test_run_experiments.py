"""Experiment benchmark runner (build step 7), offline + deterministic.

Runs several hashing-embedder configs over the fixture corpus and the committed
starter golden set. Also demonstrates, deterministically, that a larger chunk
size degrades retrieval here (it collapses pages and invalidates the golden
set's `#chunk-N` references) — the mechanism behind the step-6 chunk-size demo.
"""

import importlib.util
from pathlib import Path

from src.config import from_dict
from src.golden import load_golden

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden" / "questions.jsonl"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_experiments", ROOT / "scripts" / "run_experiments.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rx = _load_module()


def _cfg(chunk_size, retriever="dense"):
    return from_dict({
        "seed": 42,
        "chunking": {"splitter": "simple", "chunk_size": chunk_size, "chunk_overlap": 20},
        "embedding": {"provider": "hashing", "dimensions": 128},
        "store": {"type": "flat"},
        "retrieval": {"retriever": retriever, "top_k": 10},
    })


def test_runs_all_configs_and_picks_winner(tmp_path, fixture_pages):
    records = load_golden(GOLDEN)
    configs = {"small_chunks": _cfg(200), "large_chunks": _cfg(2000)}
    result = rx.run_experiments(configs, fixture_pages, records, "mrr", tmp_path / "exp")

    names = {r["name"] for r in result["rows"]}
    assert names == {"small_chunks", "large_chunks"}
    assert all(r["status"] == "ok" for r in result["rows"])
    assert result["winner"] == "small_chunks"  # large chunks degrade metrics here


def test_large_chunks_degrade_metrics(tmp_path, fixture_pages):
    records = load_golden(GOLDEN)
    configs = {"small_chunks": _cfg(200), "large_chunks": _cfg(2000)}
    result = rx.run_experiments(configs, fixture_pages, records, "mrr", tmp_path / "exp")
    by = {r["name"]: r["metrics"] for r in result["rows"]}
    assert by["small_chunks"]["context_recall_at_10"] > by["large_chunks"]["context_recall_at_10"]


def test_unimplemented_retriever_is_skipped_not_crashed(tmp_path, fixture_pages):
    records = load_golden(GOLDEN)
    configs = {"dense": _cfg(200), "hybrid": _cfg(200, retriever="hybrid")}
    result = rx.run_experiments(configs, fixture_pages, records, "mrr", tmp_path / "exp")
    hybrid = next(r for r in result["rows"] if r["name"] == "hybrid")
    assert hybrid["status"].startswith("skipped")
    assert "NotImplementedError" in hybrid["status"]
    assert result["winner"] == "dense"


def test_render_table_marks_winner(tmp_path, fixture_pages):
    records = load_golden(GOLDEN)
    configs = {"small_chunks": _cfg(200), "large_chunks": _cfg(2000)}
    result = rx.run_experiments(configs, fixture_pages, records, "mrr", tmp_path / "exp")
    table = rx.render_table(result, "v1")
    assert "Experiment benchmark" in table
    assert "🏆" in table
    assert "`small_chunks`" in table and "`large_chunks`" in table
