"""FlatStore ranking + persistence, with hand-built vectors."""

import numpy as np

from src.chunk import Chunk
from src.store import FlatStore, load_store


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        page_path="gov-uk/p",
        page_url="u",
        title="t",
        text=f"text {cid}",
        content_hash="deadbeef",
        chunk_index=0,
    )


def _store():
    s = FlatStore(dimensions=2, embedder_id="hashing-d2-s42", config_fingerprint="abc123")
    chunks = [_chunk("gov-uk/p#chunk-0"), _chunk("gov-uk/p#chunk-1"), _chunk("gov-uk/p#chunk-2")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    s.add(chunks, vecs)
    return s


def test_search_ranks_by_cosine():
    s = _store()
    results = s.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    # chunk-0 and chunk-2 both align perfectly; ties break by chunk_id.
    assert results[0].chunk.chunk_id == "gov-uk/p#chunk-0"
    assert results[1].chunk.chunk_id == "gov-uk/p#chunk-2"
    assert results[2].chunk.chunk_id == "gov-uk/p#chunk-1"


def test_tie_break_is_deterministic():
    s = _store()
    a = [r.chunk.chunk_id for r in s.search(np.array([1.0, 0.0], np.float32), 3)]
    b = [r.chunk.chunk_id for r in s.search(np.array([1.0, 0.0], np.float32), 3)]
    assert a == b


def test_top_k_limits_results():
    s = _store()
    assert len(s.search(np.array([1.0, 0.0], np.float32), top_k=1)) == 1


def test_save_and_load_roundtrip(tmp_path):
    s = _store()
    s.save(tmp_path / "idx")
    loaded = load_store(tmp_path / "idx")
    assert loaded.dimensions == 2
    assert loaded.embedder_id == "hashing-d2-s42"
    assert loaded.config_fingerprint == "abc123"
    assert [c.chunk_id for c in loaded._chunks] == [c.chunk_id for c in s._chunks]
    r = loaded.search(np.array([0.0, 1.0], np.float32), top_k=1)
    assert r[0].chunk.chunk_id == "gov-uk/p#chunk-1"
