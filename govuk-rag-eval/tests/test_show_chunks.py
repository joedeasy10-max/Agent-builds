"""scripts/show_chunks.py — reads chunk text out of a built index.

Offline by construction: it reads a file the ingest step wrote, so these tests
write that file directly rather than building an index.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "show_chunks", ROOT / "scripts" / "show_chunks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


show = _load()


def _index(tmp_path, records):
    d = tmp_path / ".index"
    d.mkdir()
    (d / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return d


def _chunk(cid, page, idx, text="body text"):
    return {
        "chunk_id": cid, "page_path": page, "chunk_index": idx,
        "title": "T", "page_url": "https://www.gov.uk/x", "text": text,
        "content_hash": "abc123",
    }


def test_an_exact_chunk_id_returns_that_chunk(tmp_path):
    d = _index(tmp_path, [_chunk("gov-uk/p#chunk-0", "gov-uk/p", 0, "first")])
    chunks = show.load_chunks(d)
    assert [c["text"] for c in show.resolve(chunks, "gov-uk/p#chunk-0")] == ["first"]


def test_a_page_path_returns_every_chunk_in_order(tmp_path):
    d = _index(tmp_path, [
        _chunk("gov-uk/p#chunk-2", "gov-uk/p", 2, "third"),
        _chunk("gov-uk/p#chunk-0", "gov-uk/p", 0, "first"),
        _chunk("gov-uk/p#chunk-1", "gov-uk/p", 1, "second"),
        _chunk("gov-uk/other#chunk-0", "gov-uk/other", 0, "elsewhere"),
    ])
    chunks = show.load_chunks(d)
    assert [c["text"] for c in show.resolve(chunks, "gov-uk/p")] == ["first", "second", "third"]


def test_an_unknown_id_resolves_to_nothing_rather_than_guessing(tmp_path):
    d = _index(tmp_path, [_chunk("gov-uk/p#chunk-0", "gov-uk/p", 0)])
    chunks = show.load_chunks(d)
    assert show.resolve(chunks, "gov-uk/p#chunk-9") == []
    assert show.resolve(chunks, "gov-uk/nope") == []


def test_a_missing_id_is_reported_and_exits_nonzero(tmp_path, capsys):
    """Silence on a bad id would read as 'this chunk is empty'. It is not."""
    d = _index(tmp_path, [_chunk("gov-uk/p#chunk-0", "gov-uk/p", 0, "real")])
    code = show.main(["gov-uk/p#chunk-0,gov-uk/ghost#chunk-0", "--index", str(d)])
    out = capsys.readouterr()
    assert code == 1
    assert "NOT IN THE INDEX" in out.out
    assert "gov-uk/ghost#chunk-0" in out.err


def test_full_text_is_the_default(tmp_path, capsys):
    """The whole point is NOT truncating — a 300-char excerpt is what failed."""
    long = "x" * 2000
    d = _index(tmp_path, [_chunk("gov-uk/p#chunk-0", "gov-uk/p", 0, long)])
    assert show.main(["gov-uk/p#chunk-0", "--index", str(d)]) == 0
    assert long in capsys.readouterr().out


def test_chars_truncates_and_says_it_did(tmp_path, capsys):
    d = _index(tmp_path, [_chunk("gov-uk/p#chunk-0", "gov-uk/p", 0, "y" * 500)])
    show.main(["gov-uk/p#chunk-0", "--index", str(d), "--chars", "100"])
    out = capsys.readouterr().out
    assert "chars : 500 (showing 100)" in out


def test_a_missing_index_says_how_to_build_one(tmp_path):
    with pytest.raises(SystemExit) as e:
        show.load_chunks(tmp_path / "nope")
    assert "src.ingest" in str(e.value)
