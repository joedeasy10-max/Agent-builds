"""Regression tests for the two defects the first real judge run exposed.

Both were invisible offline, which is exactly why they need tests here:

1. The judge provider defaulted to "openai" regardless of the config, so a repo
   on Anthropic was still graded by OpenAI — and died on an OpenAI credit error.
2. `RagasGrader.grade` did `float(result[m])`, but RAGAS returns a *list* of
   per-sample scores. It raised TypeError on every real run, and would have
   averaged in NaNs for failed samples had it not.

Nothing here calls an LLM: the provider tests are pure resolution, and the
grade() test injects a fake `ragas`/`datasets` into sys.modules.
"""

import math
import sys
import types

import pytest

from src.config import from_dict
from src.metrics import judge as J


# ---- 1. provider comes from the config -------------------------------------

def test_resolves_provider_from_config():
    assert J.resolve_judge_provider("anthropic") == "anthropic"
    assert J.resolve_judge_provider("openai") == "openai"


def test_explicit_override_wins_over_config():
    assert J.resolve_judge_provider("anthropic", "openai") == "openai"


def test_echo_generation_provider_cannot_grade():
    with pytest.raises(ValueError, match="cannot grade"):
        J.resolve_judge_provider("echo")


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="Unknown judge provider"):
        J.resolve_judge_provider("anthropic", "llama-at-home")


def test_anthropic_config_selects_the_claude_judge():
    """The end the bug actually broke: anthropic config -> anthropic grader."""
    cfg = from_dict({"generation": {"provider": "anthropic"}})
    grader = J.build_grader("ragas", provider=J.resolve_judge_provider(cfg.generation.provider))
    assert grader.provider == "anthropic"
    assert grader.model == "claude-sonnet-5"


# ---- 2. per-sample score lists aggregate correctly --------------------------

def test_aggregates_list_of_scores_to_mean():
    assert J.aggregate_metric([1.0, 0.0, 0.5], "faithfulness") == pytest.approx(0.5)


def test_accepts_a_bare_scalar():
    """Forward/backward compatible: a RAGAS build returning a scalar still works."""
    assert J.aggregate_metric(0.75, "faithfulness") == pytest.approx(0.75)


def test_drops_failed_samples_instead_of_poisoning_the_mean():
    """A NaN sample (rate limit, refusal) must not drag the metric to NaN."""
    got = J.aggregate_metric([1.0, float("nan"), 0.0, None], "faithfulness")
    assert got == pytest.approx(0.5)          # mean of the two usable scores
    assert not math.isnan(got)


def test_all_samples_failing_raises_rather_than_scoring_zero():
    """0.0 would look like a quality collapse and trip the gate. It must raise."""
    with pytest.raises(RuntimeError, match="failed to grade"):
        J.aggregate_metric([float("nan"), None], "faithfulness")


def test_empty_result_raises():
    with pytest.raises(RuntimeError, match="empty"):
        J.aggregate_metric([], "faithfulness")


def test_text_result_is_rejected():
    with pytest.raises(TypeError, match="not a score"):
        J.aggregate_metric("0.8", "faithfulness")


# ---- 3. RagasGrader.grade() with a fake RAGAS ------------------------------

class _FakeResult:
    """Stands in for a RAGAS EvaluationResult: per-metric lists of sample scores."""

    def __init__(self, scores):
        self._scores = scores

    def __getitem__(self, metric):
        return self._scores[metric]


def _install_fake_ragas(monkeypatch, result):
    """Inject the modules RagasGrader lazily imports inside grade()."""
    captured = {}

    datasets_mod = types.ModuleType("datasets")

    class _DS:
        @staticmethod
        def from_dict(d):
            captured["dataset"] = d
            return d

    datasets_mod.Dataset = _DS

    ragas_mod = types.ModuleType("ragas")

    def _evaluate(ds, metrics=None, llm=None, **kw):
        captured["llm"] = llm
        return result

    ragas_mod.evaluate = _evaluate

    metrics_mod = types.ModuleType("ragas.metrics")
    for name in J.JUDGE_METRICS:
        setattr(metrics_mod, name, object())

    llms_mod = types.ModuleType("ragas.llms")

    def _wrapper(inner, **kwargs):
        return ("wrapped", inner, kwargs)

    llms_mod.LangchainLLMWrapper = _wrapper

    anthropic_mod = types.ModuleType("langchain_anthropic")
    anthropic_mod.ChatAnthropic = lambda model: ("ChatAnthropic", model)

    for name, mod in [
        ("datasets", datasets_mod),
        ("ragas", ragas_mod),
        ("ragas.metrics", metrics_mod),
        ("ragas.llms", llms_mod),
        ("langchain_anthropic", anthropic_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return captured


def test_grade_reduces_ragas_lists_to_one_score_per_metric(monkeypatch):
    """The exact shape that raised TypeError in CI now produces real numbers."""
    result = _FakeResult({m: [1.0, 0.0] for m in J.JUDGE_METRICS})
    _install_fake_ragas(monkeypatch, result)

    grader = J.RagasGrader(provider="anthropic")
    scores = grader.grade([
        J.JudgeSample(id="q1", question="Q?", answer="A", contexts=("c",), ground_truth="A"),
        J.JudgeSample(id="q2", question="Q2?", answer="B", contexts=("c",), ground_truth="B"),
    ])

    assert set(scores) == set(J.JUDGE_METRICS)
    assert all(isinstance(v, float) for v in scores.values())
    assert scores["faithfulness"] == pytest.approx(0.5)


def test_grade_uses_the_configured_anthropic_model(monkeypatch):
    result = _FakeResult({m: [0.9] for m in J.JUDGE_METRICS})
    captured = _install_fake_ragas(monkeypatch, result)

    J.RagasGrader(provider="anthropic").grade(
        [J.JudgeSample(id="q1", question="Q?", answer="A", contexts=("c",), ground_truth="A")]
    )

    kind, inner, kwargs = captured["llm"]
    assert (kind, inner) == ("wrapped", ("ChatAnthropic", "claude-sonnet-5"))
    # RAGAS mutates llm.temperature before each call and recent Claude models
    # reject the parameter (400: `temperature` is deprecated for this model),
    # which failed every sample of run 34453595063.
    assert kwargs.get("bypass_temperature") is True


# ---- 4. the embeddings adapter exposes what RAGAS actually calls ------------

class _FakeProjectEmbedder:
    """Mimics the project Embedder protocol: .embed(list[str]) -> ndarray."""

    embedder_id = "fake"

    def embed(self, texts):
        import numpy as np

        return np.array([[float(len(t)), 0.5] for t in texts], dtype="float32")


def test_embeddings_adapter_exposes_the_methods_ragas_calls(monkeypatch):
    """Regression: the first adapter satisfied BaseRagasEmbedding's abstract
    methods (embed_text/aembed_text) but RAGAS's metrics call embed_query and
    embed_documents, so every sample died with

        AttributeError: '_ProjectEmbedding' object has no attribute 'embed_query'

    Satisfying an ABC is not the same as satisfying the caller — so this test
    pins the caller's contract, not the ABC's.
    """
    captured = {}

    ragas_mod = types.ModuleType("ragas")
    emb_mod = types.ModuleType("ragas.embeddings")

    def _wrapper(inner):
        captured["inner"] = inner
        return ("wrapped", inner)

    emb_mod.LangchainEmbeddingsWrapper = _wrapper
    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", emb_mod)

    grader = J.RagasGrader(provider="anthropic", embedder=_FakeProjectEmbedder())
    grader._embeddings()

    inner = captured["inner"]
    assert inner.embed_query("hello") == [5.0, 0.5]
    assert inner.embed_documents(["a", "bb"]) == [[1.0, 0.5], [2.0, 0.5]]

    # A real langchain Embeddings, so RAGAS supplies the async variants itself.
    from langchain_core.embeddings import Embeddings

    assert isinstance(inner, Embeddings)


def test_no_embedder_falls_back_to_the_ragas_default():
    assert J.RagasGrader(provider="openai")._embeddings() is None
