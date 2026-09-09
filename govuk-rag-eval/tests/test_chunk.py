"""Chunking is the contract with the golden set, so its IDs must be stable.

A failing case looks like: the same page content produces different chunk IDs
between two ingests (re-crawling silently invalidates every golden source_id),
or two different chunks collide on one ID.
"""

from src.chunk import (
    Page,
    chunk_page,
    chunk_pages,
    normalise_text,
    page_path_id,
    strip_html,
)
from src.config import ChunkingConfig

CFG = ChunkingConfig(splitter="simple", chunk_size=120, chunk_overlap=20)


def _page(body: str, base_path: str = "/register-for-self-assessment") -> Page:
    return Page(base_path=base_path, title="T", body=body, url="u")


def test_page_path_id_matches_golden_format():
    assert page_path_id("/register-for-self-assessment") == "gov-uk/register-for-self-assessment"
    assert page_path_id("register-for-self-assessment/") == "gov-uk/register-for-self-assessment"


def test_chunk_ids_have_expected_shape_and_order():
    body = "word " * 200
    chunks = chunk_page(_page(body), CFG)
    assert len(chunks) > 1
    for n, c in enumerate(chunks):
        assert c.chunk_id == f"gov-uk/register-for-self-assessment#chunk-{n}"
        assert c.chunk_index == n


def test_chunk_ids_are_stable_across_runs():
    body = "The quick brown fox. " * 50
    a = [c.chunk_id for c in chunk_page(_page(body), CFG)]
    b = [c.chunk_id for c in chunk_page(_page(body), CFG)]
    assert a == b


def test_chunk_ids_are_unique():
    body = "alpha beta gamma delta. " * 80
    ids = [c.chunk_id for c in chunk_page(_page(body), CFG)]
    assert len(ids) == len(set(ids))


def test_content_hash_changes_when_text_drifts():
    """Corpus drift must be visible: same ID, different content hash."""
    original = chunk_page(_page("register by 5 October " * 30), CFG)
    drifted = chunk_page(_page("register by 31 January " * 30), CFG)
    assert original[0].chunk_id == drifted[0].chunk_id
    assert original[0].content_hash != drifted[0].content_hash


def test_strip_html_and_normalise():
    text = strip_html("<h2>Title</h2><p>Line one.</p><p>Line two.</p>")
    normed = normalise_text(text)
    assert "<" not in normed
    assert "Title" in normed and "Line one." in normed


def test_chunk_pages_is_deterministically_ordered():
    p1 = _page("bbb " * 40, "/b-page")
    p2 = _page("aaa " * 40, "/a-page")
    chunks = chunk_pages([p1, p2], CFG)
    # a-page sorts before b-page regardless of input order.
    assert chunks[0].page_path == "gov-uk/a-page"


def test_recursive_splitter_available():
    """The configured default splitter (langchain-text-splitters) works."""
    cfg = ChunkingConfig(splitter="recursive", chunk_size=120, chunk_overlap=20)
    chunks = chunk_page(_page("Sentence one. Sentence two. " * 40), cfg)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)
