"""Fetch and cache a bounded slice of GOV.UK guidance.

Two open endpoints, no key needed (content is OGL v3):

    Search:   https://www.gov.uk/api/search.json?q=...&count=...
    Content:  https://www.gov.uk/api/content/<path>

We enumerate the slice via search, fetch each page's content JSON, extract the
readable body, and record every URL with a content hash in manifest.json. Raw
API responses are cached under data/corpus/raw/ (gitignored, rebuildable) so
re-running ingest offline is free and deterministic.

The crawl is rate-limited (a small courtesy reviewers notice). None of this is
exercised by the unit tests — those feed Page objects directly — but it is the
real ingest path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .chunk import Page, content_hash, normalise_text, strip_html
from .config import CorpusConfig

SEARCH_URL = "https://www.gov.uk/api/search.json"
CONTENT_URL = "https://www.gov.uk/api/content"
_USER_AGENT = "govuk-rag-eval/0.1 (+https://github.com; polite research crawler)"


@dataclass(frozen=True)
class ManifestEntry:
    base_path: str
    url: str
    title: str
    fetched_at: str
    content_hash: str


def _raw_path(raw_dir: Path, base_path: str) -> Path:
    safe = base_path.strip("/").replace("/", "__") or "_root"
    return raw_dir / f"{safe}.json"


def _extract_body(content_json: dict) -> str:
    """Pull readable text out of a GOV.UK content item."""
    details = content_json.get("details", {}) or {}
    parts: list[str] = []
    if isinstance(details.get("body"), str):
        parts.append(details["body"])
    # Multi-part guides (e.g. Self Assessment) keep sections under `parts`.
    for part in details.get("parts", []) or []:
        if isinstance(part, dict) and isinstance(part.get("body"), str):
            parts.append(part["body"])
    return "\n\n".join(parts)


def enumerate_slice(cfg: CorpusConfig, session=None) -> list[str]:
    """Return base_paths for the configured slice, via the search API."""
    import requests

    session = session or requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    paths: list[str] = []
    per_page = 100
    start = 0
    while len(paths) < cfg.max_pages:
        resp = session.get(
            SEARCH_URL,
            params={
                "q": cfg.search_query,
                "count": min(per_page, cfg.max_pages - len(paths)),
                "start": start,
                "fields": "link",
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for row in results:
            link = row.get("link", "")
            if link.startswith("/"):
                paths.append(link)
        start += per_page
        time.sleep(cfg.request_delay_seconds)
    # De-dupe, keep deterministic order.
    return sorted(dict.fromkeys(paths))[: cfg.max_pages]


def fetch_page(base_path: str, cfg: CorpusConfig, session=None, use_cache: bool = True) -> Page:
    """Fetch (or load from cache) one page and clean its body."""
    import requests

    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _raw_path(raw_dir, base_path)

    if use_cache and cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        session = session or requests.Session()
        session.headers.update({"User-Agent": _USER_AGENT})
        resp = session.get(f"{CONTENT_URL}{base_path}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data, sort_keys=True))
        time.sleep(cfg.request_delay_seconds)

    return Page(
        base_path=data.get("base_path", base_path),
        title=data.get("title", ""),
        body=normalise_text(strip_html(_extract_body(data))),
        url=f"https://www.gov.uk{data.get('base_path', base_path)}",
    )


def crawl(cfg: CorpusConfig, use_cache: bool = True) -> list[Page]:
    """Enumerate + fetch the whole slice and write manifest.json."""
    from datetime import datetime, timezone

    base_paths = enumerate_slice(cfg)
    pages: list[Page] = []
    entries: list[ManifestEntry] = []
    now = datetime.now(timezone.utc).isoformat()
    for base_path in base_paths:
        page = fetch_page(base_path, cfg, use_cache=use_cache)
        if not page.body:
            continue
        pages.append(page)
        entries.append(
            ManifestEntry(
                base_path=page.base_path,
                url=page.url,
                title=page.title,
                fetched_at=now,
                content_hash=content_hash(page.body),
            )
        )
    write_manifest(cfg, entries)
    return pages


def write_manifest(cfg: CorpusConfig, entries: list[ManifestEntry]) -> None:
    manifest_path = Path(cfg.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_query": cfg.search_query,
        "count": len(entries),
        "pages": [
            {
                "base_path": e.base_path,
                "url": e.url,
                "title": e.title,
                "fetched_at": e.fetched_at,
                "content_hash": e.content_hash,
            }
            for e in sorted(entries, key=lambda e: e.base_path)
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
