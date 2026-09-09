"""Turn a GOV.UK page into stable, content-derived chunks.

Chunk IDs are the contract between the index and the golden set:

    <page-path>#chunk-<n>        e.g. gov-uk/register-for-self-assessment#chunk-3

The golden set references these IDs, so they must be *stable*: the same page
content must always produce the same IDs. `<n>` is the chunk's position in
document order (deterministic), and each chunk also carries a `content_hash`
(sha256 of its normalised text) so corpus drift is detectable — if a page's
text moves under us, its chunks' hashes change even when the IDs do not.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import ChunkingConfig


@dataclass(frozen=True)
class Page:
    """A fetched, cleaned GOV.UK page."""

    base_path: str          # e.g. "/register-for-self-assessment"
    title: str
    body: str               # plain text (HTML already stripped)
    url: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str           # "<page-path>#chunk-<n>"
    page_path: str          # "gov-uk/register-for-self-assessment"
    page_url: str
    title: str
    text: str
    content_hash: str       # sha256(normalised text)[:16]
    chunk_index: int


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def strip_html(text: str) -> str:
    """Best-effort HTML -> text. GOV.UK content bodies are often HTML."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    # Turn block-ish boundaries into newlines before dropping the tags.
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article)>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    return text


def normalise_text(text: str) -> str:
    """Whitespace normalisation used for both chunking input and hashing.

    Deterministic and idempotent so content hashes are stable across runs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def page_path_id(base_path: str) -> str:
    """`/register-for-self-assessment` -> `gov-uk/register-for-self-assessment`."""
    clean = base_path.strip().strip("/")
    return f"gov-uk/{clean}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split(text: str, cfg: ChunkingConfig) -> list[str]:
    if cfg.splitter == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=list(cfg.separators),
            keep_separator=False,
        )
        return splitter.split_text(text)
    if cfg.splitter == "simple":
        return _simple_split(text, cfg.chunk_size, cfg.chunk_overlap)
    raise ValueError(f"Unknown chunking.splitter: {cfg.splitter!r}")


def _simple_split(text: str, size: int, overlap: int) -> list[str]:
    """Deterministic fixed-window splitter with no external dependency.

    Used as a dependency-light fallback / for tests. Windows step by
    (size - overlap); the final short window is kept.
    """
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    step = size - overlap
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        window = text[i : i + size].strip()
        if window:
            out.append(window)
        i += step
    return out


def chunk_page(page: Page, cfg: ChunkingConfig) -> list[Chunk]:
    """Chunk one page into deterministic, stably-identified pieces."""
    body = normalise_text(strip_html(page.body))
    path_id = page_path_id(page.base_path)
    pieces = _split(body, cfg)

    chunks: list[Chunk] = []
    for n, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{path_id}#chunk-{n}",
                page_path=path_id,
                page_url=page.url,
                title=page.title,
                text=piece,
                content_hash=content_hash(piece),
                chunk_index=n,
            )
        )
    return chunks


def chunk_pages(pages: list[Page], cfg: ChunkingConfig) -> list[Chunk]:
    """Chunk many pages, in a deterministic order (sorted by page path)."""
    out: list[Chunk] = []
    for page in sorted(pages, key=lambda p: p.base_path):
        out.extend(chunk_page(page, cfg))
    return out
