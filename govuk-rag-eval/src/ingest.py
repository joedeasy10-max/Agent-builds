"""Ingest: fetch -> clean -> chunk -> embed -> index.

Deterministic given (corpus, retrieval config): same inputs, same index. The
CI cache keys on configs/retrieval.yaml + data/corpus/manifest.json for exactly
this reason (see rag-eval.yml).

CLI:
    python -m src.ingest --config configs/retrieval.yaml --out .index/

`build_index(pages, config, out_dir)` is the seam the tests use — they pass
fixture Page objects directly and never touch the network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunk import Page, chunk_pages
from .config import Config, load_config
from .embed import build_embedder
from .store import build_store


def build_index(pages: list[Page], config: Config, out_dir: str | Path):
    """Chunk, embed and index a set of pages; persist to out_dir."""
    chunks = chunk_pages(pages, config.chunking)
    if not chunks:
        raise ValueError("No chunks produced — corpus is empty or all pages were blank.")

    embedder = build_embedder(config)
    vectors = embedder.embed([c.text for c in chunks])

    store = build_store(
        dimensions=embedder.dimensions,
        embedder_id=embedder.embedder_id,
        config_fingerprint=config.config_fingerprint(),
        store_type=config.store.type,
    )
    store.add(chunks, vectors)
    store.save(out_dir)
    return store


def gather_pages(config: Config) -> list[Page]:
    """Load the corpus for a real run.

    If a manifest already exists, rebuild deterministically from it (fetching
    each page from the raw cache, offline) — this is the path CI takes on an
    index cache miss. Otherwise enumerate + crawl GOV.UK to create one.
    """
    import json

    from .corpus import crawl, fetch_page

    manifest_path = Path(config.corpus.manifest_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        base_paths = [p["base_path"] for p in manifest.get("pages", [])]
        pages = [fetch_page(bp, config.corpus, use_cache=True) for bp in base_paths]
        return [p for p in pages if p.body]
    return crawl(config.corpus, use_cache=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pages = gather_pages(config)
    store = build_index(pages, config, args.out)
    print(
        f"Indexed {len(store._chunks)} chunks from {len(pages)} pages "
        f"-> {args.out} (embedder={store.embedder_id}, fingerprint={store.config_fingerprint})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
