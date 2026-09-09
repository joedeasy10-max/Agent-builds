"""Load and validate the golden set (data/golden/questions.jsonl).

One JSON object per line:

    {"id": "q_0142", "question": "...", "ground_truth": "...",
     "source_ids": ["gov-uk/register-for-self-assessment#chunk-3"],
     "difficulty": "multi_hop", "added_in": "v2"}

`source_ids` are the relevant chunk IDs (empty for a `negative` — a question the
corpus cannot answer). The dataset *version* is derived from the newest
`added_in` present, so results can be gated only against a baseline built from
the same version (see scripts/compare.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_DIFFICULTIES = {"single_hop", "multi_hop", "negative"}
_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class GoldenRecord:
    id: str
    question: str
    ground_truth: str
    source_ids: tuple[str, ...]
    difficulty: str
    added_in: str

    @property
    def is_negative(self) -> bool:
        return self.difficulty == "negative"

    @property
    def is_answerable(self) -> bool:
        return not self.is_negative and len(self.source_ids) > 0


def _validate(obj: dict, line_no: int) -> GoldenRecord:
    missing = {"id", "question", "source_ids", "difficulty", "added_in"} - set(obj)
    if missing:
        raise ValueError(f"golden line {line_no}: missing keys {sorted(missing)}")
    if obj["difficulty"] not in _DIFFICULTIES:
        raise ValueError(
            f"golden line {line_no}: difficulty {obj['difficulty']!r} "
            f"not in {sorted(_DIFFICULTIES)}"
        )
    if not isinstance(obj["source_ids"], list):
        raise ValueError(f"golden line {line_no}: source_ids must be a list")
    if not _VERSION_RE.match(str(obj["added_in"])):
        raise ValueError(f"golden line {line_no}: added_in must look like 'v3'")
    if obj["difficulty"] == "negative" and obj["source_ids"]:
        raise ValueError(f"golden line {line_no}: a negative must have no source_ids")
    if obj["difficulty"] != "negative" and not obj["source_ids"]:
        raise ValueError(
            f"golden line {line_no}: answerable question needs at least one source_id"
        )
    return GoldenRecord(
        id=str(obj["id"]),
        question=str(obj["question"]),
        ground_truth=str(obj.get("ground_truth", "")),
        source_ids=tuple(obj["source_ids"]),
        difficulty=str(obj["difficulty"]),
        added_in=str(obj["added_in"]),
    )


def load_golden(path: str | Path) -> list[GoldenRecord]:
    records: list[GoldenRecord] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = _validate(json.loads(line), line_no)
        if record.id in seen_ids:
            raise ValueError(f"golden line {line_no}: duplicate id {record.id!r}")
        seen_ids.add(record.id)
        records.append(record)
    if not records:
        raise ValueError(f"golden set {path} is empty")
    return records


def dataset_version(records: list[GoldenRecord]) -> str:
    """Newest `added_in` across the set, e.g. 'v3'. Deterministic."""
    newest = max(int(_VERSION_RE.match(r.added_in).group(1)) for r in records)
    return f"v{newest}"
