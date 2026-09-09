"""End-to-end step-1 test: ingest fixture pages, then retrieve — fully offline.

Failing cases this guards against:
  * ingest produces no index / retrieve returns nothing;
  * ranking is non-deterministic;
  * a query embedded by a model that disagrees with the index is silently scored;
  * re-ingesting the same corpus changes the chunk IDs.
"""

import numpy as np

from src.ingest import build_index
from src.retrieve import Retriever


def test_ingest_then_retrieve_finds_relevant_page(tmp_path, fixture_pages, test_config):
    build_index(fixture_pages, test_config, tmp_path / "idx")
    retriever = Retriever(test_config, tmp_path / "idx")

    results = retriever.retrieve("When do I need to register for Self Assessment?")
    assert results, "retrieve returned no chunks"
    assert len(results) <= test_config.retrieval.top_k
    # The registration page should surface for a registration question.
    top_paths = {r.chunk.page_path for r in results}
    assert "gov-uk/register-for-self-assessment" in top_paths


def test_ranking_is_deterministic(tmp_path, fixture_pages, test_config):
    build_index(fixture_pages, test_config, tmp_path / "idx")
    retriever = Retriever(test_config, tmp_path / "idx")
    q = "self assessment deadline"
    first = [(r.chunk.chunk_id, round(r.score, 6)) for r in retriever.retrieve(q)]
    second = [(r.chunk.chunk_id, round(r.score, 6)) for r in retriever.retrieve(q)]
    assert first == second


def test_reingest_produces_identical_chunk_ids(tmp_path, fixture_pages, test_config):
    s1 = build_index(fixture_pages, test_config, tmp_path / "a")
    s2 = build_index(fixture_pages, test_config, tmp_path / "b")
    ids1 = [c.chunk_id for c in s1._chunks]
    ids2 = [c.chunk_id for c in s2._chunks]
    assert ids1 == ids2


def test_index_persists_and_reloads(tmp_path, fixture_pages, test_config):
    built = build_index(fixture_pages, test_config, tmp_path / "idx")
    retriever = Retriever(test_config, tmp_path / "idx")
    assert len(retriever.store._chunks) == len(built._chunks)
    assert retriever.store.embedder_id == built.embedder_id


def test_config_fingerprint_mismatch_is_rejected(tmp_path, fixture_pages, test_config):
    from src.config import from_dict

    build_index(fixture_pages, test_config, tmp_path / "idx")
    # Change something that affects index contents (chunk size) -> new fingerprint.
    other = from_dict(
        {
            "seed": 42,
            "chunking": {"splitter": "simple", "chunk_size": 999, "chunk_overlap": 40},
            "embedding": {"provider": "hashing", "dimensions": 128},
            "store": {"type": "flat", "path": ".index"},
            "retrieval": {"retriever": "dense", "top_k": 3},
        }
    )
    try:
        Retriever(other, tmp_path / "idx")
    except ValueError as exc:
        assert "different retrieval config" in str(exc)
    else:
        raise AssertionError("expected a fingerprint-mismatch ValueError")


def test_top_k_override(tmp_path, fixture_pages, test_config):
    build_index(fixture_pages, test_config, tmp_path / "idx")
    retriever = Retriever(test_config, tmp_path / "idx")
    assert len(retriever.retrieve("tax", top_k=1)) == 1


def test_scores_are_descending(tmp_path, fixture_pages, test_config):
    build_index(fixture_pages, test_config, tmp_path / "idx")
    retriever = Retriever(test_config, tmp_path / "idx")
    scores = [r.score for r in retriever.retrieve("self employment records")]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(s) for s in scores)
