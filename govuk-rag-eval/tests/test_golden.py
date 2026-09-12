"""Golden-set loading + validation + version derivation."""

import json
from pathlib import Path

import pytest

from src.golden import dataset_version, load_golden

ROOT = Path(__file__).resolve().parent.parent

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
    """The golden set that ships in the repo must load and be internally sound.

    Asserts properties that stay true as the set grows, not a pinned version
    literal — the literal broke the moment the reviewed v2 set was promoted, which
    is a change to the data, not a defect. The version is checked for agreement
    with eval_config instead: a baseline is only comparable within one version, so
    a set promoted without bumping the config is the bug worth catching.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent
    recs = load_golden(root / "data" / "golden" / "questions.jsonl")
    assert len(recs) >= 1

    cfg = yaml.safe_load((root / "configs" / "eval_config.yaml").read_text())
    assert dataset_version(recs) == cfg["dataset"]["version"]

    assert any(r.is_answerable for r in recs), "no answerable questions"
    assert any(r.is_negative for r in recs), "no negatives — refusal is untested"
    assert all(r.source_ids for r in recs if r.is_answerable)
    assert not any(r.source_ids for r in recs if r.is_negative)
    assert len({r.id for r in recs}) == len(recs), "duplicate ids"


# --- dataset version and the golden set must agree ----------------------
# `promote` appends records stamped `added_in: v3` and then PRINTS a reminder to
# bump dataset.version in eval_config.yaml by hand. A printed reminder is not a
# mechanism — it is the same "value restated in a second place" that has now
# produced six bugs here. This turns the reminder into a failing test.


def _config_dataset_version():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "eval_config.yaml").read_text())
    return cfg["dataset"]["version"]


def _golden_rows():
    path = ROOT / "configs" / "eval_config.yaml"
    import yaml

    rel = yaml.safe_load(path.read_text())["dataset"]["path"]
    return [
        json.loads(line)
        for line in (ROOT / rel).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _version_number(v):
    return int(str(v).lstrip("v") or 0)


def test_no_golden_record_is_newer_than_the_configured_dataset_version():
    """A record stamped v3 while the config still says v2 means the bump was missed.

    That combination is silently wrong rather than loudly broken: compare.py
    would gate the enlarged set against a v2 baseline measured on fewer
    questions, and call the difference a regression.
    """
    configured = _config_dataset_version()
    newest = max(
        (r.get("added_in", "v0") for r in _golden_rows()), key=_version_number
    )
    assert _version_number(newest) <= _version_number(configured), (
        f"golden set contains records added_in {newest!r} but "
        f"configs/eval_config.yaml says dataset.version is {configured!r} — "
        "promote appended records without the version being bumped"
    )


def test_the_configured_version_actually_exists_in_the_golden_set():
    """The mirror failure: version bumped, data never promoted."""
    configured = _config_dataset_version()
    stamps = {r.get("added_in", "v0") for r in _golden_rows()}
    assert configured in stamps, (
        f"dataset.version is {configured!r} but no golden record is stamped "
        f"added_in={configured!r} (found {sorted(stamps)}) — the version was "
        "bumped without promoting anything into it"
    )
