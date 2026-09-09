"""Corpus-drift detection logic (build step 8), offline — no crawl, no network.

The crawl itself needs www.gov.uk and isn't unit-tested (like corpus.crawl); the
diff/merge/summary logic that decides whether to open a PR is pure and is tested
here with hand-built manifest page-lists.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "refresh_corpus", ROOT / "scripts" / "refresh_corpus.py"
    )
    m = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


rc = _load()


def _page(bp, h, ts="2026-01-01T00:00:00Z"):
    return {"base_path": bp, "url": f"https://www.gov.uk{bp}", "title": bp,
            "fetched_at": ts, "content_hash": h}


OLD = [_page("/a", "h1"), _page("/b", "h2"), _page("/c", "h3")]


def test_no_drift_when_hashes_match():
    diff = rc.diff_pages(OLD, list(OLD))
    assert not diff.has_drift
    assert diff.total == 0
    assert "No GOV.UK content drift" in rc.render_summary(diff)


def test_detects_changed_added_removed():
    new = [_page("/a", "h1"), _page("/b", "h2-NEW"), _page("/d", "h4")]  # b changed, c removed, d added
    diff = rc.diff_pages(OLD, new)
    assert diff.changed == ("/b",)
    assert diff.added == ("/d",)
    assert diff.removed == ("/c",)
    assert diff.has_drift and diff.total == 3


def test_summary_lists_each_bucket():
    new = [_page("/a", "h1-x"), _page("/e", "h5")]
    summary = rc.render_summary(rc.diff_pages(OLD, new))
    assert "Changed" in summary and "`/a`" in summary
    assert "Added" in summary and "`/e`" in summary
    assert "Removed" in summary and "`/b`" in summary and "`/c`" in summary


def test_merge_preserves_timestamps_for_unchanged_pages():
    new = [
        _page("/a", "h1", ts="2026-09-08T00:00:00Z"),        # unchanged content, new ts
        _page("/b", "h2-NEW", ts="2026-09-08T00:00:00Z"),    # changed content
    ]
    merged = rc.merge_preserving_timestamps(OLD, new)
    by = {p["base_path"]: p for p in merged}
    # unchanged /a keeps the OLD timestamp; changed /b takes the new one.
    assert by["/a"]["fetched_at"] == "2026-01-01T00:00:00Z"
    assert by["/b"]["fetched_at"] == "2026-09-08T00:00:00Z"
    # merged is sorted by base_path
    assert [p["base_path"] for p in merged] == ["/a", "/b"]


def test_load_manifest_pages_missing_file(tmp_path):
    assert rc.load_manifest_pages(tmp_path / "nope.json") == []


def test_write_then_load_roundtrip(tmp_path):
    from src.config import CorpusConfig
    cfg = CorpusConfig(manifest_path=str(tmp_path / "manifest.json"), search_query="q")
    rc.write_manifest(cfg, [_page("/b", "h2"), _page("/a", "h1")])
    pages = rc.load_manifest_pages(cfg.manifest_path)
    assert [p["base_path"] for p in pages] == ["/a", "/b"]  # sorted on write
