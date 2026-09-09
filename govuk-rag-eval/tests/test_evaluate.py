"""End-to-end retrieval-suite eval, fully offline (hashing embedder, flat store).

This wires the real pieces together — ingest -> Retriever -> metrics -> results
JSON — and asserts on values we can compute by hand:

  q1: relevant = the chunk the retriever actually ranks #1 for its query -> hits.
  q2: relevant = a chunk ID that does not exist -> cannot hit.
  q3: a negative -> excluded from the retrieval aggregates.

So over 2 answerable questions (one hitting at rank 1, one missing entirely):
  hit_at_5 = 0.5, mrr = 0.5, context_recall_at_10 = 0.5.
"""

import json

from src.evaluate import main as evaluate_main
from src.ingest import build_index
from src.retrieve import Retriever


def _golden_for(tmp_path, top1_id):
    lines = [
        {"id": "q1", "question": "When do I need to register for Self Assessment?",
         "ground_truth": "", "source_ids": [top1_id],
         "difficulty": "single_hop", "added_in": "v1"},
        {"id": "q2", "question": "How do I renew my passport online?",
         "ground_truth": "", "source_ids": ["gov-uk/nonexistent-page#chunk-9"],
         "difficulty": "single_hop", "added_in": "v1"},
        {"id": "q3", "question": "Can I pay in monthly instalments?",
         "ground_truth": "", "source_ids": [],
         "difficulty": "negative", "added_in": "v1"},
    ]
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return p


def _run(tmp_path, fixture_pages, cfg):
    idx = tmp_path / "idx"
    build_index(fixture_pages, cfg, idx)
    top1 = Retriever(cfg, idx).retrieve(
        "When do I need to register for Self Assessment?"
    )[0].chunk.chunk_id
    golden = _golden_for(tmp_path, top1)
    out = tmp_path / "current.json"

    # evaluate loads config from a file; write the offline test config out.
    import yaml
    from dataclasses import asdict
    cfg_path = tmp_path / "retrieval.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "seed": cfg.seed,
        "chunking": asdict(cfg.chunking) | {"separators": list(cfg.chunking.separators)},
        "embedding": asdict(cfg.embedding),
        "store": asdict(cfg.store),
        "retrieval": asdict(cfg.retrieval),
    }))

    code = evaluate_main([
        "--suite", "retrieval",
        "--dataset", str(golden),
        "--config", str(cfg_path),
        "--index", str(idx),
        "--out", str(out),
    ])
    return code, json.loads(out.read_text())


def test_retrieval_suite_values(tmp_path, fixture_pages, test_config):
    code, results = _run(tmp_path, fixture_pages, test_config)
    assert code == 0
    assert results["dataset_version"] == "v1"
    assert set(results["retrieval"]) == {"hit_at_5", "mrr", "context_recall_at_10"}
    assert results["retrieval"]["hit_at_5"] == 0.5
    assert results["retrieval"]["mrr"] == 0.5
    assert results["retrieval"]["context_recall_at_10"] == 0.5
    assert results["retrieval_detail"]["n_answerable"] == 2
    assert results["retrieval_detail"]["n_negatives"] == 1


def test_results_shape_matches_compare_contract(tmp_path, fixture_pages, test_config):
    """compare.py reads current['dataset_version'] and current['retrieval'][metric]."""
    _, results = _run(tmp_path, fixture_pages, test_config)
    assert "dataset_version" in results
    assert isinstance(results["retrieval"], dict)
    for v in results["retrieval"].values():
        assert 0.0 <= v <= 1.0


def test_deterministic_across_runs(tmp_path, fixture_pages, test_config):
    _, a = _run(tmp_path / "a", fixture_pages, test_config)
    _, b = _run(tmp_path / "b", fixture_pages, test_config)
    assert a["retrieval"] == b["retrieval"]


def test_limit_caps_questions(tmp_path, fixture_pages, test_config):
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    from src.evaluate import run_retrieval_suite
    from src.golden import load_golden
    top1 = Retriever(test_config, idx).retrieve(
        "When do I need to register for Self Assessment?"
    )[0].chunk.chunk_id
    golden = load_golden(_golden_for(tmp_path, top1))
    suite = run_retrieval_suite(golden, test_config, idx, limit=1)
    assert suite["n_answerable"] == 1
