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
# The gate/experiment logic is what these exercise, so they use a fixture golden
# set matched to tests/fixtures/pages — not the committed one. Pointing them at
# the real dataset coupled them to its contents: promoting the reviewed v2 set
# (67 records over 21 pages, only 2 of which the fixture corpus holds) dropped
# hit@5 to 0.047 and failed tests that have nothing to do with the dataset.
GOLDEN = ROOT / "tests" / "fixtures" / "golden_fixture.jsonl"


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


def test_table_has_a_column_for_every_suite_metric(tmp_path, fixture_pages):
    """The header is derived, not restated.

    It had already drifted: three columns after a fourth metric was added, so
    the benchmark's own table omitted `served_context_recall` — the only metric
    that can see a top_k regression. A hardcoded header would fail this the
    next time a metric lands.
    """
    records = load_golden(GOLDEN)
    configs = {"small_chunks": _cfg(200), "large_chunks": _cfg(2000)}
    result = rx.run_experiments(configs, fixture_pages, records, "mrr", tmp_path / "exp")
    table = rx.render_table(result, "v1")
    header = next(line for line in table.splitlines() if line.startswith("| Config |"))
    for metric in rx._METRICS:
        assert f"`{metric}`" in header, f"{metric} missing from the table header"
    # Every data row must carry one cell per metric, plus Config/chunks/status.
    for line in table.splitlines():
        if line.startswith("| `"):
            assert line.count("|") == len(rx._METRICS) + 4


def test_unloadable_config_is_a_skipped_row_not_a_crash(tmp_path, fixture_pages, monkeypatch):
    """A config declaring a not-yet-implemented section skips one row.

    Loading used to happen outside run_experiments' guard, so `reranked.yaml`
    (which declares `reranker:`, a key Config rejects) took the whole benchmark
    down rather than skipping itself.
    """
    good = tmp_path / "good.yaml"
    good.write_text(
        "seed: 42\n"
        "chunking: {splitter: simple, chunk_size: 200, chunk_overlap: 20}\n"
        "embedding: {provider: hashing, dimensions: 128}\n"
        "store: {type: flat}\n"
        "retrieval: {retriever: dense, top_k: 10}\n"
    )
    bad = tmp_path / "pending_reranker.yaml"
    bad.write_text(good.read_text() + "reranker: {model: x, top_n: 5}\n")

    # main() would otherwise crawl the real corpus; this test is about the row.
    monkeypatch.setattr(rx, "gather_pages", lambda config: fixture_pages)

    rc = rx.main([
        "--dataset", str(GOLDEN),
        "--configs", str(good), str(bad),
        "--out", str(tmp_path / "out"),
        "--workdir", str(tmp_path / "wd"),
    ])
    assert rc == 0
    table = (tmp_path / "out" / "table.md").read_text()
    assert "`pending_reranker`" in table
    assert "skipped: ValueError" in table
    assert "`good`" in table
    # The winner must come from a config that actually ran.
    assert "**good**" in table


def test_all_configs_unloadable_is_an_error_not_a_silent_empty_table(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "gather_pages", lambda config: [])
    bad = tmp_path / "pending.yaml"
    bad.write_text(
        "seed: 42\n"
        "chunking: {splitter: simple, chunk_size: 200, chunk_overlap: 20}\n"
        "embedding: {provider: hashing, dimensions: 128}\n"
        "store: {type: flat}\n"
        "retrieval: {retriever: dense, top_k: 10}\n"
        "reranker: {model: x, top_n: 5}\n"
    )
    rc = rx.main([
        "--dataset", str(GOLDEN),
        "--configs", str(bad),
        "--out", str(tmp_path / "out"),
        "--workdir", str(tmp_path / "wd"),
    ])
    assert rc == 1


def test_shipped_reranked_config_declares_its_pending_dependency():
    """Guards the actual defect: reranked.yaml must not be a copy of baseline.

    With the reranker commented out the two configs were identical in every
    field that affects retrieval, so the benchmark published baseline's numbers
    twice — once under a name for an experiment nobody had run.
    """
    import yaml
    base = yaml.safe_load((ROOT / "configs/experiments/baseline.yaml").read_text())
    rer = yaml.safe_load((ROOT / "configs/experiments/reranked.yaml").read_text())
    assert rer != base, "reranked.yaml is indistinguishable from baseline.yaml"
    assert "reranker" in rer
