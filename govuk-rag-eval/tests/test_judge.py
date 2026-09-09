"""Judge metrics: deterministic heuristic grader + the median-of-N logic.

No LLM is ever called here. The median-of-N is tested with a scripted fake
grader so we can assert exactly which value is picked and what spread is
recorded — that is the whole point of the median (damp judge variance).
"""

import pytest

from src.metrics import judge as J


def _sample(answer, contexts, ground_truth="", question="q?"):
    return J.JudgeSample(
        id="x", question=question, answer=answer,
        contexts=tuple(contexts), ground_truth=ground_truth,
    )


# ---- HeuristicGrader ---------------------------------------------------------

def test_heuristic_scores_in_range():
    grader = J.HeuristicGrader()
    scores = grader.grade([
        _sample("register by october", ["you must register by october"],
                ground_truth="register by october", question="when to register"),
    ])
    assert set(scores) == set(J.JUDGE_METRICS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_heuristic_faithfulness_high_when_answer_in_context():
    grader = J.HeuristicGrader()
    grounded = grader.grade([_sample("alpha beta", ["alpha beta gamma"])])
    ungrounded = grader.grade([_sample("zulu yankee", ["alpha beta gamma"])])
    assert grounded["faithfulness"] > ungrounded["faithfulness"]
    assert ungrounded["faithfulness"] == 0.0


def test_heuristic_is_deterministic():
    grader = J.HeuristicGrader()
    s = [_sample("a b c", ["a b"], ground_truth="a")]
    assert grader.grade(s) == grader.grade(s)


def test_heuristic_empty_samples():
    scores = J.HeuristicGrader().grade([])
    assert all(v == 0.0 for v in scores.values())


# ---- run_judge median-of-N ---------------------------------------------------

class _FakeGrader:
    """Returns scripted aggregate scores, one dict per successive run."""

    grader_id = "fake"
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness")

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._i = 0

    def grade(self, samples):
        out = self._scripted[self._i]
        self._i += 1
        return out


def test_run_judge_takes_median_over_runs():
    # faithfulness across 3 runs: 0.80, 0.90, 0.85 -> median 0.85.
    scripted = [
        {"faithfulness": 0.80, "answer_relevancy": 0.7, "context_precision": 0.6, "answer_correctness": 0.5},
        {"faithfulness": 0.90, "answer_relevancy": 0.7, "context_precision": 0.6, "answer_correctness": 0.5},
        {"faithfulness": 0.85, "answer_relevancy": 0.7, "context_precision": 0.6, "answer_correctness": 0.5},
    ]
    result = J.run_judge([_sample("a", ["a"])], _FakeGrader(scripted), runs=3)
    assert result["runs"] == 3
    assert result["metrics"]["faithfulness"] == pytest.approx(0.85)
    spread = result["spread"]["faithfulness"]
    assert spread["min"] == pytest.approx(0.80)
    assert spread["max"] == pytest.approx(0.90)
    assert spread["runs"] == [0.80, 0.85, 0.90]  # sorted


def test_run_judge_single_run():
    scripted = [{m: 0.5 for m in _FakeGrader.metrics}]
    result = J.run_judge([_sample("a", ["a"])], _FakeGrader(scripted), runs=1)
    assert result["metrics"]["faithfulness"] == 0.5


def test_run_judge_rejects_zero_runs():
    with pytest.raises(ValueError):
        J.run_judge([], J.HeuristicGrader(), runs=0)


def test_build_grader():
    assert J.build_grader("heuristic").grader_id == "heuristic"
    assert J.build_grader("ragas").grader_id == "ragas"  # constructed, not called
    with pytest.raises(ValueError):
        J.build_grader("bogus")


def test_ragas_grader_provider_selection_without_calling():
    # Construction must not import the provider SDK or call an LLM.
    openai_grader = J.build_grader("ragas", provider="openai")
    anthropic_grader = J.build_grader("ragas", provider="anthropic")
    assert openai_grader.provider == "openai"
    assert openai_grader.model == "gpt-4o-mini"          # provider default
    assert anthropic_grader.provider == "anthropic"
    assert anthropic_grader.model == "claude-sonnet-5"   # provider default
    with pytest.raises(ValueError):
        J.build_grader("ragas", provider="bogus")
