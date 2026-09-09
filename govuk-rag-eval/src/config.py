"""Load and validate configs/retrieval.yaml into typed dataclasses.

The config file is the single source of truth for anything that changes
retrieval quality. We parse it into dataclasses so a typo in a key fails loudly
at load time rather than silently producing different numbers.

A `config_fingerprint` (stable hash of the resolved config) is exposed so the
index and its metadata can record exactly which config produced them; retrieval
refuses to run a query embedder that disagrees with the stored index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CorpusConfig:
    source: str = "govuk"
    search_query: str = ""
    max_pages: int = 300
    request_delay_seconds: float = 1.0
    raw_dir: str = "data/corpus/raw"
    manifest_path: str = "data/corpus/manifest.json"


@dataclass(frozen=True)
class ChunkingConfig:
    splitter: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64


@dataclass(frozen=True)
class StoreConfig:
    type: str = "flat"
    path: str = ".index"


@dataclass(frozen=True)
class RetrievalConfig:
    retriever: str = "dense"
    top_k: int = 5
    normalize: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    # Answer synthesis for the judge suite (build step 4). The model is a config
    # value, not an import. `echo` is a deterministic offline stub for tests.
    # provider is interchangeable: openai | anthropic | echo. Leave `model`
    # blank to use the provider's default (gpt-4o-mini / claude-sonnet-5).
    provider: str = "openai"          # openai | anthropic | echo
    model: str = ""                   # blank = provider default
    temperature: float = 0.0          # applied only where the provider supports it
    max_tokens: int = 512


@dataclass(frozen=True)
class Config:
    seed: int = 42
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def config_fingerprint(self) -> str:
        """Stable short hash of the fields that affect index contents.

        Corpus/store *locations* are excluded — moving the index directory must
        not change the fingerprint. Retriever top_k is excluded too: it changes
        query-time behaviour, not what is stored in the index.
        """
        material = {
            "chunking": asdict(self.chunking),
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
            },
            "seed": self.seed,
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _require_keys(section: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {sorted(unknown)}")


def _build_section(cls, data: dict[str, Any], where: str):
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    _require_keys(data, allowed, where)
    # Preserve tuple typing for separators.
    if "separators" in data and data["separators"] is not None:
        data = {**data, "separators": tuple(data["separators"])}
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Parse a retrieval config file, rejecting unknown keys."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> Config:
    top_allowed = {
        "seed", "corpus", "chunking", "embedding", "store", "retrieval", "generation",
    }
    _require_keys(raw, top_allowed, "retrieval config (top level)")
    return Config(
        seed=int(raw.get("seed", 42)),
        corpus=_build_section(CorpusConfig, raw.get("corpus", {}) or {}, "corpus"),
        chunking=_build_section(ChunkingConfig, raw.get("chunking", {}) or {}, "chunking"),
        embedding=_build_section(EmbeddingConfig, raw.get("embedding", {}) or {}, "embedding"),
        store=_build_section(StoreConfig, raw.get("store", {}) or {}, "store"),
        retrieval=_build_section(RetrievalConfig, raw.get("retrieval", {}) or {}, "retrieval"),
        generation=_build_section(GenerationConfig, raw.get("generation", {}) or {}, "generation"),
    )
