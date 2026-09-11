"""Deterministic screener: the no-key path (src/offline_screen.py).

These tests are about what the screener can and cannot know. It is allowed to be
cautious; it is not allowed to approve something it has not actually checked.
"""

import pytest

from src.candidate_review import preceding_peers
from src.offline_screen import (
    MIN_GROUND_TRUTH_SUPPORT,
    OfflineScreener,
    content_tokens,
    overlap,
    unsupported_figures,
)

PASSAGE = (
    "You must register for Self Assessment if you have to send a tax return. "
    "Register by 5 October in your business's second tax year. "
    "The trading allowance is £1,000 for the 2024 to 2025 tax year."
)


class Cand:
    def __init__(self, question="When must you register for Self Assessment?",
                 ground_truth="Register by 5 October in your business's second tax year.",
                 passage=PASSAGE, cid="cand_0001", difficulty="single_hop"):
        self.candidate_id = cid
        self.question = question
        self.ground_truth = ground_truth
        self.source_excerpt = passage
        self.source_ids = ("gov-uk/register-for-self-assessment#chunk-0",)
        self.difficulty = difficulty


def screen(**kw):
    return OfflineScreener().screen(Cand(**kw))


# --- the sharp signal: figures the passage does not contain -----------------


def test_invented_figures_are_caught():
    missing = unsupported_figures("You must register by 31 January and claim £5,000.", PASSAGE)
    assert "31 January" in missing
    assert "£5,000" in missing


def test_figures_are_compared_by_value_not_formatting():
    assert unsupported_figures("The allowance is £1000.", PASSAGE) == []
    assert unsupported_figures("The allowance is £1,000.", PASSAGE) == []


def test_a_ground_truth_citing_a_figure_not_in_the_passage_fails_support():
    ev = screen(ground_truth="Register by 31 January in your first tax year.")
    assert ev.source_support == "fail"
    assert ev.decision == "review"
    assert "does not contain" in ev.reason


def test_no_figures_anywhere_is_not_a_failure():
    """Absence of numbers must not be mistaken for a contradiction."""
    assert unsupported_figures("You must send a tax return.", PASSAGE) == []


def test_empty_passage_cannot_produce_a_figure_failure():
    assert unsupported_figures("Pay £999 by 1 May.", "") == []


# --- what it approves -------------------------------------------------------


def test_a_well_grounded_candidate_is_approved():
    ev = screen()
    assert ev.decision == "approve"
    assert all(getattr(ev, c) == "pass"
               for c in ("relevance", "ground_truth_accuracy",
                         "source_support", "question_quality"))
    assert ev.evaluator == "offline"


def test_self_referential_phrasing_still_routes_to_a_human():
    ev = screen(question="According to the guidance, when must you register?")
    assert ev.question_quality == "fail"
    assert ev.decision == "review"
    assert ev.self_referential is True


def test_an_answer_that_restates_the_question_fails():
    q = "When must you register for Self Assessment in your second tax year?"
    ev = screen(question=q, ground_truth=q)
    assert ev.ground_truth_accuracy == "fail"
    assert ev.decision != "approve"


def test_an_empty_ground_truth_fails():
    ev = screen(ground_truth="")
    assert ev.ground_truth_accuracy == "fail"
    assert ev.decision != "approve"


def test_a_question_about_something_else_fails_relevance():
    ev = screen(question="What is the capital of Peru and its population size?",
                ground_truth="Lima, around 10 million people.")
    assert ev.relevance == "fail" or ev.source_support == "fail"
    assert ev.decision != "approve"


def test_a_statement_is_not_a_question():
    ev = screen(question="Registration deadlines for the second tax year explained.")
    assert ev.question_quality == "fail"


def test_negatives_still_reach_a_human():
    ev = screen(question="How many people did HMRC audit last year?",
                ground_truth="Not answerable from the corpus.",
                passage="", difficulty="negative")
    assert ev.decision == "review"
    assert ev.rule == "negative_needs_corpus_check"


def test_missing_passage_reaches_a_human():
    ev = screen(passage="   ")
    assert ev.decision == "review"
    assert ev.rule == "no_passage"


# --- duplicates and confidence ---------------------------------------------


def test_duplicates_use_the_same_backward_only_rule():
    batch = [
        Cand(cid="c0", question="When must you register for Self Assessment?"),
        Cand(cid="c1", question="When must you register for self assessment?"),
    ]
    s = OfflineScreener()
    first = s.screen(batch[0], others=preceding_peers(batch, 0))
    second = s.screen(batch[1], others=preceding_peers(batch, 1))
    assert first.decision == "approve"
    assert second.decision == "reject" and second.duplicate_of == "c0"


def test_borderline_support_lands_below_the_approval_floor():
    """A knife-edge candidate must reach a human, not be waved through.

    This is how a deterministic screener says "I cannot tell": confidence is a
    margin against its own thresholds, not a pose.
    """
    s = OfflineScreener(min_ground_truth_support=0.99)
    ev = s.screen(Cand())
    assert ev.decision != "approve"


def test_confidence_is_never_outside_zero_to_one():
    for kw in ({}, {"ground_truth": ""}, {"passage": "x"},
               {"question": "According to the guidance, what?"}):
        ev = screen(**kw)
        assert 0.0 <= ev.confidence <= 1.0


def test_it_is_deterministic():
    """The whole point: same input, same verdict, every time."""
    a, b = screen(), screen()
    assert a.to_dict() == b.to_dict()


# --- helpers ----------------------------------------------------------------


def test_overlap_ignores_stopwords():
    assert "the" not in content_tokens("the tax return")
    assert overlap("the tax return", "tax return forms") == 1.0
    assert overlap("capital gains", "beef carcase classification") == 0.0
