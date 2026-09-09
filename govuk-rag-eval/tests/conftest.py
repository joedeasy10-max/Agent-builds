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
