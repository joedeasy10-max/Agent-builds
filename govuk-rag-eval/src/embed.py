"""Embedding providers behind one small interface.

The model is a config value, not an import (BUILD.md). Swapping the embedding
model is one of the deliberate regressions the harness is meant to catch, so it
has to be a one-line change in configs/retrieval.yaml.

Providers:
  * hashing - deterministic, offline, zero-cost. A hashed bag-of-tokens vector.
              Not competitive with a real model, but it needs no network or key,
              which makes ingest + retrieve testable in CI without spending money.
              This is what the unit tests use.
  * openai  - text-embedding-3-small (the real default). Needs OPENAI_API_KEY.
  * bge     - local sentence-transformers BAAI/bge-small-en-v1.5 (the swap target).

Note: Anthropic has no embeddings API, so there is no `anthropic` embedder — an
Anthropic-only setup uses the local `bge` embedder here and Claude only for
generation/judge (see src/generate.py).

Every embedder reports a stable `embedder_id` recorded in the index metadata so
a query can never be embedded by a different model than the one that built the
index.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Config, EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimensions: int
    embedder_id: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dimensions) float32 array."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic hashed bag-of-tokens embedder. No network, no cost."""

    def __init__(self, dimensions: int = 256, seed: int = 42, normalize: bool = True):
        self.dimensions = dimensions
        self.seed = seed
        self.normalize = normalize
        self.embedder_id = f"hashing-d{dimensions}-s{seed}"

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(
            f"{self.seed}:{token}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                vecs[row, self._bucket(token)] += 1.0
        return _l2_normalize(vecs) if self.normalize else vecs


class OpenAIEmbedder:
    """text-embedding-3-small (or any OpenAI embedding model)."""

    def __init__(self, model: str, dimensions: int, batch_size: int = 64, normalize: bool = True):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.normalize = normalize
        self.embedder_id = f"openai-{model}-d{dimensions}"

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI  # lazy: only needed on the real path

        client = OpenAI()
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dimensions
            )
            rows.extend(d.embedding for d in resp.data)
        matrix = np.asarray(rows, dtype=np.float32)
        return _l2_normalize(matrix) if self.normalize else matrix


class BgeEmbedder:
    """Local BAAI/bge-small-en-v1.5 via sentence-transformers (the swap target)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", normalize: bool = True):
        from sentence_transformers import SentenceTransformer  # lazy, heavy

        self._model = SentenceTransformer(model)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.normalize = normalize
        self.embedder_id = f"bge-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.asarray(
            self._model.encode(texts, normalize_embeddings=False), dtype=np.float32
        )
        return _l2_normalize(matrix) if self.normalize else matrix


def build_embedder(config: Config) -> Embedder:
    emb: EmbeddingConfig = config.embedding
    normalize = config.retrieval.normalize
    if emb.provider == "hashing":
        return HashingEmbedder(
            dimensions=emb.dimensions, seed=config.seed, normalize=normalize
        )
    if emb.provider == "openai":
        return OpenAIEmbedder(
            model=emb.model,
            dimensions=emb.dimensions,
            batch_size=emb.batch_size,
            normalize=normalize,
        )
    if emb.provider == "bge":
        return BgeEmbedder(model=emb.model, normalize=normalize)
    raise ValueError(f"Unknown embedding.provider: {emb.provider!r}")
