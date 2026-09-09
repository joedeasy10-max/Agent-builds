"""Retrieval metric math, with hand-built rankings (no ingest, no network).

Each test fixes a ranking and the relevant set and asserts the exact score, so a
change to the metric definition fails loudly.
"""

import pytest

from src.metrics import retrieval as R

# A ranking of chunk IDs, best first.
RANK = ["a", "b", "c", "d", "e"]


# ---- hit_at_k ----------------------------------------------------------------

def test_hit_at_k_true_when_relevant_in_topk():
    assert R.hit_at_k(RANK, {"c"}, k=3) == 1.0


def test_hit_at_k_false_when_relevant_below_k():
    assert R.hit_at_k(RANK, {"d"}, k=3) == 0.0  # d is at rank 4


def test_hit_at_k_true_at_exact_boundary():
    assert R.hit_at_k(RANK, {"c"}, k=3) == 1.0
    assert R.hit_at_k(RANK, {"c"}, k=2) == 0.0


def test_hit_at_k_missing_relevant_is_zero():
    assert R.hit_at_k(RANK, {"zzz"}, k=5) == 0.0


def test_hit_at_k_requires_relevant():
    with pytest.raises(ValueError):
        R.hit_at_k(RANK, set(), k=3)


# ---- reciprocal_rank ---------------------------------------------------------

def test_reciprocal_rank_first_position():
    assert R.reciprocal_rank(RANK, {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert R.reciprocal_rank(RANK, {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant():
    # both b (rank2) and d (rank4) relevant -> first hit is rank 2.
    assert R.reciprocal_rank(RANK, {"b", "d"}) == pytest.approx(1 / 2)


def test_reciprocal_rank_none_relevant_retrieved_is_zero():
    assert R.reciprocal_rank(RANK, {"zzz"}) == 0.0


def test_reciprocal_rank_respects_cutoff():
    # d is rank 4; with cutoff 3 it is not seen -> 0.
    assert R.reciprocal_rank(RANK, {"d"}, k=3) == 0.0


# ---- recall_at_k -------------------------------------------------------------

def test_recall_at_k_partial():
    # relevant {a, d, zzz}; top-3 = a,b,c -> only a recalled -> 1/3.
    assert R.recall_at_k(RANK, {"a", "d", "zzz"}, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_full():
    assert R.recall_at_k(RANK, {"a", "b"}, k=2) == 1.0


def test_recall_at_k_zero_when_none_in_topk():
    assert R.recall_at_k(RANK, {"e"}, k=2) == 0.0


def test_recall_at_k_caps_at_topk_available():
    # relevant has 2 items but only 1 fits in top-1 -> 0.5.
    assert R.recall_at_k(RANK, {"a", "b"}, k=1) == 0.5


# ---- mean --------------------------------------------------------------------

def test_mean_of_empty_is_zero():
    assert R.mean([]) == 0.0


def test_mean_basic():
    assert R.mean([1.0, 0.0]) == 0.5
