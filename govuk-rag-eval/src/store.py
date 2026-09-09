"""Vector store behind a tiny interface. The store type is a config value.

`flat` is the default: a file-backed numpy store. Vectors are L2-normalised at
ingest, so cosine similarity is a dot product. It has no server, installs
nothing beyond numpy, and the on-disk index is a plain directory that the CI
cache can key on — which is all step 1 needs. `lancedb` is left as a config
option to wire in later (BUILD.md lists it as the eventual store); the point of
the interface is that swapping stores never touches retrieve.py.

Search is deterministic: ties are broken by chunk_id so identical input always
produces identical ranking — the whole gate depends on that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunk import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class FlatStore:
    """File-backed, brute-force cosine store."""

    def __init__(self, dimensions: int, embedder_id: str, config_fingerprint: str):
        self.dimensions = dimensions
        self.embedder_id = embedder_id
        self.config_fingerprint = config_fingerprint
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape != (len(chunks), self.dimensions):
            raise ValueError(
                f"vectors shape {vectors.shape} != ({len(chunks)}, {self.dimensions})"
            )
        self._chunks = list(chunks)
        self._vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        q = query_vector.reshape(-1).astype(np.float32)
        scores = self._vectors @ q  # cosine sim (all vectors normalised)
        # Deterministic order: highest score first, chunk_id breaks ties.
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id),
        )
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in order[:top_k]]

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self._vectors if self._vectors is not None else np.zeros((0, self.dimensions), np.float32))
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "store": "flat",
                    "dimensions": self.dimensions,
                    "embedder_id": self.embedder_id,
                    "config_fingerprint": self.config_fingerprint,
                    "count": len(self._chunks),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlatStore":
        src = Path(path)
        meta = json.loads((src / "meta.json").read_text())
        store = cls(
            dimensions=meta["dimensions"],
            embedder_id=meta["embedder_id"],
            config_fingerprint=meta["config_fingerprint"],
        )
        chunks: list[Chunk] = []
        with (src / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
        vectors = np.load(src / "vectors.npy")
        store._chunks = chunks
        store._vectors = vectors.astype(np.float32)
        return store


def build_store(dimensions: int, embedder_id: str, config_fingerprint: str, store_type: str = "flat"):
    if store_type == "flat":
        return FlatStore(dimensions, embedder_id, config_fingerprint)
    if store_type == "lancedb":
        raise NotImplementedError(
            "lancedb store is planned (BUILD.md stack) but not wired up yet; "
            "use store.type: flat for now."
        )
    raise ValueError(f"Unknown store.type: {store_type!r}")


def load_store(path: str | Path, store_type: str = "flat"):
    if store_type == "flat":
        return FlatStore.load(path)
    raise ValueError(f"Unknown or unsupported store.type: {store_type!r}")
