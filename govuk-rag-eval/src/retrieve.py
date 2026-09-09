"""Config-driven retrieval: retrieve(question) -> ranked chunks.

The retriever loads a persisted index, embeds the query with the *same* embedder
that built the index (it refuses to run if the config fingerprint disagrees, so
you can never silently score a query against a mismatched model), and returns
the top_k chunks. Ranking is deterministic (see store.search).

CLI smoke test:
    python -m src.retrieve --config configs/retrieval.yaml --index .index/ \
        --query "When do I need to register for Self Assessment?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, load_config
from .embed import build_embedder
from .store import ScoredChunk, load_store


class Retriever:
    def __init__(self, config: Config, index_dir: str | Path):
        self.config = config
        self.store = load_store(index_dir, config.store.type)
        self.embedder = build_embedder(config)

        expected = config.config_fingerprint()
        if self.store.config_fingerprint != expected:
            raise ValueError(
                "Index was built with a different retrieval config "
                f"(index={self.store.config_fingerprint}, config={expected}). "
                "Rebuild the index with `python -m src.ingest`."
            )
        if self.store.embedder_id != self.embedder.embedder_id:
            raise ValueError(
                f"Embedder mismatch: index={self.store.embedder_id}, "
                f"config={self.embedder.embedder_id}."
            )

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self.config.retrieval.retriever != "dense":
            raise NotImplementedError(
                f"retriever={self.config.retrieval.retriever!r} not implemented in "
                "step 1 (dense only); bm25/hybrid come later."
            )
        k = top_k if top_k is not None else self.config.retrieval.top_k
        query_vec = self.embedder.embed([question])[0]
        return self.store.search(query_vec, k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve chunks for a question.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    retriever = Retriever(config, args.index)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    for rank, sc in enumerate(results, start=1):
        print(f"{rank:2d}. {sc.score:+.4f}  {sc.chunk.chunk_id}")
        print(f"     {sc.chunk.text[:120].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
