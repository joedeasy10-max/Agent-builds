"""Print the full text of chunks by id, straight from a built index.

Why this exists: the source excerpts that reach a human reviewer are truncated —
the review page caps them at ~300 characters — and that is not enough to check
whether a hand-written ground_truth is actually supported. The full corpus is
not reachable from every environment (artefact downloads and www.gov.uk are
both blocked on some networks), but the built index carries every chunk's text
in `chunks.jsonl`, and CI always has one.

Deliberately reads the index rather than the raw corpus: the index is what the
retriever actually searches, so what this prints is what the system really has.
A chunk absent here is absent from retrieval too, which is itself the answer to
"why does this question never hit".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_chunks(index_dir: Path) -> dict[str, dict]:
    """chunk_id -> chunk record, from a built index."""
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(
            f"No chunks.jsonl under {index_dir} — build the index first "
            "(python -m src.ingest --config configs/retrieval.yaml --out .index/)."
        )
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec
    return out


def resolve(chunks: dict[str, dict], wanted: str) -> list[dict]:
    """Exact chunk id, or every chunk of a page when no '#chunk-N' is given."""
    if wanted in chunks:
        return [chunks[wanted]]
    if "#" not in wanted:
        page = [c for c in chunks.values() if c["page_path"] == wanted]
        return sorted(page, key=lambda c: c["chunk_index"])
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", help="comma-separated chunk ids, or page paths for every chunk of a page")
    ap.add_argument("--index", type=Path, default=Path(".index/"))
    ap.add_argument("--chars", type=int, default=0, help="truncate each chunk (0 = full text)")
    args = ap.parse_args(argv)

    chunks = load_chunks(args.index)
    wanted = [w.strip() for w in args.ids.split(",") if w.strip()]
    missing: list[str] = []

    print(f"index {args.index} holds {len(chunks)} chunks\n")
    for w in wanted:
        found = resolve(chunks, w)
        if not found:
            missing.append(w)
            print(f"=== {w} ===\nNOT IN THE INDEX — bad id, or the page is not in the corpus.\n")
            continue
        for c in found:
            text = c["text"]
            shown = text[: args.chars] if args.chars else text
            print(f"=== {c['chunk_id']} ===")
            print(f"title : {c['title']}")
            print(f"url   : {c['page_url']}")
            print(f"chars : {len(text)}" + (f" (showing {len(shown)})" if len(shown) < len(text) else ""))
            print("-" * 72)
            print(shown)
            print("-" * 72 + "\n")

    if missing:
        print(f"{len(missing)} of {len(wanted)} not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
