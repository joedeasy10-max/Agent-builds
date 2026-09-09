"""Deterministic, LLM-free retrieval metrics — the metrics that carry the gate.

Each function scores ONE question given a ranked list of retrieved chunk IDs and
the set of relevant (ground-truth) chunk IDs. They are pure and deterministic:
identical input yields identical output, which is what lets the gate compare two
runs and trust the delta.

What each measures, and what a failing case looks like:

* hit_at_k        Did any relevant chunk make the top-k? 1.0 / 0.0.
                  Fails when the relevant chunk exists but ranks below k (or is
                  not returned because top_k was lowered).
* reciprocal_rank How high did the FIRST relevant chunk rank? 1/rank, else 0.
                  Fails (drops) when the first relevant hit slips down the list.
* recall_at_k     What fraction of a question's relevant chunks are in top-k?
                  Fails when a multi-hop question's relevant chunks don't all
                  make the cut.

Aggregation is a plain mean over the *answerable* questions (those with at least
one relevant chunk). Negatives are handled by the caller, not here.

IDs are compared as exact strings — the stable `<page-path>#chunk-<n>` chunk IDs
are the contract with the golden set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("hit_at_k is undefined for a question with no relevant IDs")
    return 1.0 if any(cid in relevant for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(
    ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int | None = None
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("reciprocal_rank is undefined for a question with no relevant IDs")
    seq = ranked_ids[:k] if k is not None else ranked_ids
    for rank, cid in enumerate(seq, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("recall_at_k is undefined for a question with no relevant IDs")
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def mean(values: Sequence[float]) -> float:
    """Mean over questions; 0.0 for an empty set (no answerable questions)."""
    return sum(values) / len(values) if values else 0.0
