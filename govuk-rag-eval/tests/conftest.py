"""Shared fixtures: load the offline GOV.UK fixture pages as Page objects.

These exercise the real corpus-cleaning path (corpus._extract_body + HTML strip)
without any network call.
"""

import json
from pathlib import Path

import pytest

from src.chunk import Page, normalise_text, strip_html
from src.corpus import _extract_body

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"


def _load_page(path: Path) -> Page:
    data = json.loads(path.read_text())
    return Page(
        base_path=data["base_path"],
        title=data.get("title", ""),
        body=normalise_text(strip_html(_extract_body(data))),
        url=f"https://www.gov.uk{data['base_path']}",
    )


@pytest.fixture
def fixture_pages() -> list[Page]:
    return [_load_page(p) for p in sorted(FIXTURE_DIR.glob("*.json"))]


@pytest.fixture
def test_config():
    """A retrieval config that runs fully offline: hashing embedder, flat store."""
    from src.config import from_dict

    return from_dict(
        {
            "seed": 42,
            "chunking": {
                "splitter": "simple",
                "chunk_size": 200,
                "chunk_overlap": 40,
            },
            "embedding": {"provider": "hashing", "dimensions": 128},
            "store": {"type": "flat", "path": ".index"},
            "retrieval": {"retriever": "dense", "top_k": 3, "normalize": True},
        }
    )


@pytest.fixture
def fixture_thresholds(tmp_path):
    """The real floors and tolerances, re-stamped with the FIXTURE golden set's
    dataset version.

    Several gate tests deliberately use `tests/fixtures/golden_fixture.jsonl`
    rather than the committed golden set, so they exercise the gate rather than
    the dataset. They were still handing compare.py the real
    `configs/eval_config.yaml`, version field included — and compare.py refuses
    to compare across dataset versions. The moment the real set was promoted to
    v3, the fixture's v2 results stopped comparing: exit 2, no report written,
    and four tests failing for a reason unrelated to what they test.

    This keeps the part those tests care about (the real floors and tolerances)
    and overrides the one field they do not (the version). Third time a test has
    broken on a dataset bump for this reason, hence one shared fixture rather
    than a third private copy.
    """
    import json

    import yaml

    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "configs" / "eval_config.yaml").read_text())
    golden = root / "tests" / "fixtures" / "golden_fixture.jsonl"
    versions = {
        json.loads(line)["added_in"]
        for line in golden.read_text().splitlines()
        if line.strip()
    }
    assert len(versions) == 1, f"fixture golden set spans versions {versions}"
    config["dataset"]["version"] = versions.pop()
    path = tmp_path / "fixture_thresholds.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
