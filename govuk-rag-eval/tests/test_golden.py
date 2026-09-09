"""Golden-set loading + validation + version derivation."""

import pytest

from src.golden import dataset_version, load_golden

VALID = (
    '{"id":"q1","question":"a?","ground_truth":"x","source_ids":["gov-uk/p#chunk-0"],'
    '"difficulty":"single_hop","added_in":"v1"}\n'
    '{"id":"q2","question":"b?","ground_truth":"y","source_ids":[],'
    '"difficulty":"negative","added_in":"v3"}\n'
)


def _write(tmp_path, text):
    p = tmp_path / "g.jsonl"
    p.write_text(text)
    return p


def test_loads_valid_records(tmp_path):
    recs = load_golden(_write(tmp_path, VALID))
    assert [r.id for r in recs] == ["q1", "q2"]
    assert recs[0].is_answerable and not recs[0].is_negative
    assert recs[1].is_negative and not recs[1].is_answerable


def test_version_is_newest_added_in(tmp_path):
    recs = load_golden(_write(tmp_path, VALID))
    assert dataset_version(recs) == "v3"  # max(v1, v3), not lexical


def test_rejects_unknown_difficulty(tmp_path):
    bad = '{"id":"q1","question":"a","source_ids":["x"],"difficulty":"weird","added_in":"v1"}\n'
    with pytest.raises(ValueError, match="difficulty"):
        load_golden(_write(tmp_path, bad))


def test_rejects_negative_with_sources(tmp_path):
    bad = '{"id":"q1","question":"a","source_ids":["x"],"difficulty":"negative","added_in":"v1"}\n'
    with pytest.raises(ValueError, match="negative"):
        load_golden(_write(tmp_path, bad))


def test_rejects_answerable_without_sources(tmp_path):
    bad = '{"id":"q1","question":"a","source_ids":[],"difficulty":"single_hop","added_in":"v1"}\n'
    with pytest.raises(ValueError, match="source_id"):
        load_golden(_write(tmp_path, bad))


def test_rejects_duplicate_ids(tmp_path):
    dup = VALID + '{"id":"q1","question":"c","source_ids":["y"],"difficulty":"single_hop","added_in":"v1"}\n'
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(_write(tmp_path, dup))


def test_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_golden(_write(tmp_path, "\n\n"))


def test_committed_golden_set_is_valid():
    """The starter golden set that ships in the repo must load."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    recs = load_golden(root / "data" / "golden" / "questions.jsonl")
    assert len(recs) >= 1
    assert dataset_version(recs) == "v1"
