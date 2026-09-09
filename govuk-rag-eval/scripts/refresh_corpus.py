"""Re-crawl the corpus slice and detect content drift (build step 8).

The sleeper feature: GOV.UK content changes under you over time. This re-fetches
the configured slice, compares each page's content hash against the committed
`data/corpus/manifest.json`, and — if anything drifted — rewrites the manifest so
the eval runs against the new content. The weekly workflow
(.github/workflows/refresh-corpus.yml) runs this and opens a PR when it reports
drift; the RAG-eval gate then evaluates the change automatically.

Timestamps are preserved for unchanged pages, so the manifest only changes where
content actually changed — no spurious PRs from `fetched_at` churn alone.

The crawl needs outbound access to www.gov.uk (available on CI runners; blocked
in some sandboxes). The drift logic itself is pure and unit-tested offline.

CLI:
    python scripts/refresh_corpus.py --config configs/retrieval.yaml \
        --summary-out corpus-drift.md
Writes `changed=<bool>` and `pages_changed=<n>` to $GITHUB_OUTPUT when set.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure `src` resolves when run as a plain script.
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CorpusConfig, load_config  # noqa: E402


@dataclass(frozen=True)
class ManifestDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def has_drift(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)


def load_manifest_pages(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text() or "{}")
    return data.get("pages", [])


def _by_path(pages: list[dict]) -> dict[str, dict]:
    return {p["base_path"]: p for p in pages}


def diff_pages(old_pages: list[dict], new_pages: list[dict]) -> ManifestDiff:
    """Compare two manifest page-lists by content hash. Pure + deterministic."""
    old, new = _by_path(old_pages), _by_path(new_pages)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        bp for bp in set(old) & set(new)
        if old[bp].get("content_hash") != new[bp].get("content_hash")
    )
    return ManifestDiff(tuple(added), tuple(removed), tuple(changed))


def merge_preserving_timestamps(old_pages: list[dict], new_pages: list[dict]) -> list[dict]:
    """New page-list, but unchanged pages keep their old `fetched_at`.

    Keeps the manifest byte-stable where content didn't move, so git (and the
    index cache key) only see a change for pages that actually drifted.
    """
    old = _by_path(old_pages)
    merged: list[dict] = []
    for page in new_pages:
        prior = old.get(page["base_path"])
        if prior and prior.get("content_hash") == page.get("content_hash"):
            page = {**page, "fetched_at": prior.get("fetched_at", page["fetched_at"])}
        merged.append(page)
    return sorted(merged, key=lambda p: p["base_path"])


def render_summary(diff: ManifestDiff) -> str:
    if not diff.has_drift:
        return "No GOV.UK content drift detected."
    lines = [
        "## GOV.UK corpus drift detected",
        "",
        f"- **Changed:** {len(diff.changed)}",
        f"- **Added:** {len(diff.added)}",
        f"- **Removed:** {len(diff.removed)}",
        "",
    ]
    for label, items in (("Changed", diff.changed), ("Added", diff.added), ("Removed", diff.removed)):
        if items:
            lines.append(f"### {label}")
            lines.extend(f"- `{bp}`" for bp in items)
            lines.append("")
    lines.append(
        "The eval will run against the refreshed corpus on this PR. Review the "
        "content changes and, if metrics move, update the baseline accordingly."
    )
    return "\n".join(lines)


def fetch_current_entries(cfg: CorpusConfig) -> list[dict]:
    """Crawl the slice fresh (no cache) and return manifest page dicts."""
    from src.chunk import content_hash
    from src.corpus import enumerate_slice, fetch_page

    now = datetime.now(timezone.utc).isoformat()
    entries: list[dict] = []
    for base_path in enumerate_slice(cfg):
        page = fetch_page(base_path, cfg, use_cache=False)
        if not page.body:
            continue
        entries.append({
            "base_path": page.base_path,
            "url": page.url,
            "title": page.title,
            "fetched_at": now,
            "content_hash": content_hash(page.body),
        })
    return entries


def write_manifest(cfg: CorpusConfig, pages: list[dict]) -> None:
    path = Path(cfg.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_query": cfg.search_query,
        "count": len(pages),
        "pages": sorted(pages, key=lambda p: p["base_path"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _emit_output(changed: bool, n: int) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"changed={'true' if changed else 'false'}\n")
            fh.write(f"pages_changed={n}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-crawl the corpus and detect drift.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config).corpus
    old_pages = load_manifest_pages(cfg.manifest_path)
    new_pages = fetch_current_entries(cfg)
    diff = diff_pages(old_pages, new_pages)

    summary = render_summary(diff)
    if args.summary_out:
        args.summary_out.write_text(summary)

    if diff.has_drift:
        write_manifest(cfg, merge_preserving_timestamps(old_pages, new_pages))
    _emit_output(diff.has_drift, diff.total)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
