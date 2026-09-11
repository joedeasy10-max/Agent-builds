"""Local NLI judge (src/metrics/nli_judge.py), fully offline.

The entailment model and the embedder are fakes. That is deliberate, not a
shortcut: these tests pin MY logic — sentence splitting, the max-over-contexts
aggregation, rank weighting, label resolution — not whether a downloaded
checkpoint agrees with me. The real model binding is proven in CI, because
HuggingFace is not reachable from the dev sandbox.
"""

import pytest

from src.metrics.judge import JUDGE_METRICS, JudgeSample
from src.metrics.nli_judge import (
    CrossEncoderEntailment,
    NLIGrader,
    cosine,
    split_sentences,
    token_f1,
)


class FakeEntailment:
    """Entailment = does the premise contain the hypothesis' first content word."""

    def __init__(self, score_map=None):
        self.score_map = score_map or {}
        self.pairs = []

    def entailment_probs(self, pairs):
        self.pairs = list(pairs)
        out = []
        for premise, hypothesis in pairs:
            if (premise, hypothesis) in self.score_map:
                out.append(self.score_map[(premise, hypothesis)])
                continue
            words = [w for w in hypothesis.lower().split() if len(w) > 3]
            out.append(0.95 if words and words[0] in premise.lower() else 0.05)
        return out


class FakeEmbedder:
    """Bag-of-words vectors over a fixed vocabulary — deterministic cosine."""

    VOCAB = ["register", "october", "tax", "return", "allowance", "peru", "capital"]

    def embed(self, texts):
        return [
            [1.0 if w in t.lower() else 0.0 for w in self.VOCAB] for t in texts
        ]


def sample(**kw):
    base = dict(
        id="q_0001",
        question="When must you register for Self Assessment?",
        answer="Register by 5 October.",
        contexts=("You must register by 5 October in your second tax year.",),
        ground_truth="Register by 5 October in your second tax year.",
    )
    base.update(kw)
    return JudgeSample(**base)


def grader(ent=None, emb=None, **kw):
    return NLIGrader(ent or FakeEntailment(), emb or FakeEmbedder(), **kw)


# --- faithfulness -----------------------------------------------------------


def test_a_supported_answer_scores_high():
    assert grader().faithfulness(sample()) == pytest.approx(0.95)


def test_an_unsupported_sentence_drags_the_score_down():
    """The hallucination case faithfulness exists to catch."""
    two = sample(answer="Register by 5 October. Kangaroos administer the scheme.")
    score = grader().faithfulness(two)
    assert score == pytest.approx((0.95 + 0.05) / 2)
    assert score < grader().faithfulness(sample())


def test_a_sentence_is_scored_against_its_BEST_context_not_all_of_them():
    """A supported answer usually draws on one passage, not every passage."""
    s = sample(contexts=(
        "Completely unrelated text about carcase classification.",
        "You must register by 5 October in your second tax year.",
        "More unrelated text about security vetting.",
    ))
    assert grader().faithfulness(s) == pytest.approx(0.95)


def test_no_contexts_scores_zero_rather_than_crashing():
    assert grader().faithfulness(sample(contexts=())) == 0.0
    assert grader().faithfulness(sample(contexts=("   ",))) == 0.0


def test_empty_answer_scores_zero():
    assert grader().faithfulness(sample(answer="")) == 0.0


def test_context_count_is_capped():
    ent = FakeEntailment()
    g = grader(ent=ent, max_contexts=2)
    g.faithfulness(sample(contexts=("a one", "b two", "c three", "d four")))
    assert len({p[0] for p in ent.pairs}) == 2


def test_pairs_are_premise_then_hypothesis():
    """Order matters for NLI: the context is the premise, the answer the
    hypothesis. Reversing it asks a different and wrong question."""
    ent = FakeEntailment()
    grader(ent=ent).faithfulness(sample())
    premise, hypothesis = ent.pairs[0]
    assert premise.startswith("You must register")
    assert hypothesis.startswith("Register by")


# --- the other metrics ------------------------------------------------------


def test_answer_relevancy_rewards_an_on_topic_answer():
    on = grader().answer_relevancy(sample())
    off = grader().answer_relevancy(sample(answer="The capital of Peru."))
    assert on > off


def test_context_precision_weights_by_rank():
    """The same relevant context is worth more at rank 1 than at rank 3."""
    good = "register october tax"
    noise = "peru capital"
    first = grader().context_precision(sample(contexts=(good, noise, noise)))
    last = grader().context_precision(sample(contexts=(noise, noise, good)))
    assert first > last


def test_answer_correctness_combines_meaning_and_wording():
    exact = grader().answer_correctness(sample(answer="Register by 5 October in your second tax year."))
    wrong = grader().answer_correctness(sample(answer="The capital of Peru is Lima."))
    assert exact > wrong
    assert 0.0 <= wrong <= 1.0


def test_every_metric_stays_in_range():
    g = grader()
    for kw in ({}, {"answer": ""}, {"contexts": ()}, {"ground_truth": ""},
               {"question": ""}, {"answer": "Kangaroos."}):
        for name, value in g.grade([sample(**kw)]).items():
            assert 0.0 <= value <= 1.0, (name, value, kw)


# --- the Grader protocol ----------------------------------------------------


def test_grade_returns_every_judge_metric():
    out = grader().grade([sample(), sample(answer="Kangaroos run it.")])
    assert set(out) == set(JUDGE_METRICS)


def test_grade_is_the_mean_over_samples():
    g = grader()
    a, b = sample(), sample(answer="Kangaroos run the scheme.")
    both = g.grade([a, b])
    assert both["faithfulness"] == pytest.approx(
        (g.faithfulness(a) + g.faithfulness(b)) / 2
    )


def test_empty_batch_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        grader().grade([])


def test_it_is_deterministic():
    """The whole reason for this grader: identical runs, identical numbers."""
    assert grader().grade([sample()]) == grader().grade([sample()])


def test_median_of_n_records_zero_spread():
    from src.metrics.judge import run_judge

    out = run_judge([sample()], grader(), runs=3)
    for m in JUDGE_METRICS:
        assert out["spread"][m]["min"] == out["spread"][m]["max"]


# --- label resolution: the part most likely to be silently wrong ------------


class _Cfg:
    def __init__(self, id2label):
        self.id2label = id2label


class _Inner:
    def __init__(self, id2label):
        self.config = _Cfg(id2label)


class _FakeCE:
    def __init__(self, id2label, scores=None):
        self.model = _Inner(id2label)
        self._scores = scores or []

    def predict(self, pairs, **kw):
        return self._scores


def _entailer(id2label, scores=None):
    obj = CrossEncoderEntailment.__new__(CrossEncoderEntailment)
    obj.model_name = "fake"
    obj.batch_size = 8
    obj._model = _FakeCE(id2label, scores)
    obj._entail_index = obj._resolve_entailment_index()
    return obj


@pytest.mark.parametrize("id2label,expected", [
    ({0: "contradiction", 1: "entailment", 2: "neutral"}, 1),
    ({0: "entailment", 1: "neutral", 2: "contradiction"}, 0),
    ({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, 2),
    ({0: "contradiction", 1: "entailment_label", 2: "neutral"}, 1),
])
def test_entailment_column_is_found_by_name_not_by_position(id2label, expected):
    """Checkpoints disagree about ordering; the wrong column silently inverts
    the metric into something that looks plausible and is wrong."""
    assert _entailer(id2label)._entail_index == expected


def test_a_model_with_no_entailment_label_is_refused():
    with pytest.raises(ValueError, match="no entailment label"):
        _entailer({0: "not_relevant", 1: "relevant"})


def test_a_missing_label_map_is_refused():
    obj = CrossEncoderEntailment.__new__(CrossEncoderEntailment)
    obj.model_name = "fake"
    obj._model = _FakeCE(None)
    with pytest.raises(ValueError, match="id2label"):
        obj._label_map()


def test_too_few_scores_is_refused_rather_than_indexed_blindly():
    ent = _entailer({0: "contradiction", 1: "entailment", 2: "neutral"},
                    scores=[[0.4, 0.6]])
    ent._entail_index = 2
    with pytest.raises(ValueError, match="expected at least"):
        ent.entailment_probs([("a", "b")])


def test_scores_are_read_from_the_resolved_column():
    ent = _entailer({0: "contradiction", 1: "entailment", 2: "neutral"},
                    scores=[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    assert ent.entailment_probs([("a", "b"), ("c", "d")]) == [0.8, 0.2]


def test_no_pairs_means_no_model_call():
    ent = _entailer({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert ent.entailment_probs([]) == []


# --- helpers ----------------------------------------------------------------


def test_sentence_split_handles_currency_and_one_liners():
    assert split_sentences("Pay £500. Then file.") == ["Pay £500.", "Then file."]
    assert split_sentences("No terminator") == ["No terminator"]
    assert split_sentences("") == []


def test_token_f1_edges():
    assert token_f1("a b", "a b") == 1.0
    assert token_f1("", "a") == 0.0
    assert token_f1("a", "b") == 0.0


def test_cosine_handles_zero_vectors():
    assert cosine([0, 0], [1, 1]) == 0.0
