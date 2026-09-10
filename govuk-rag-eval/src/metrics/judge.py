"""Judge (generation-quality) metrics — LLM-graded, so noisy and median-damped.

Unlike the retrieval metrics, these come from an LLM grader (RAGAS) and move a
few percent between identical runs. So the suite runs the grader N times and
takes the **median** per metric, and records the spread (min/max/runs) — that
measured variance is what justifies the wide tolerance bands in eval_config.yaml
(build step 4).

What each metric measures, and what a failing case looks like:

* faithfulness       Is every claim in the answer grounded in the retrieved
                     context? Fails when the answer asserts facts the context
                     does not support (hallucination).
* answer_relevancy   Does the answer actually address the question? Fails on
                     evasive or off-topic answers.
* context_precision  Are the retrieved contexts relevant and well-ranked
                     (signal over noise)? Fails when the top contexts are junk.
* answer_correctness Does the answer match the ground truth? Reported but never
                     gated — too noisy to be a build signal (eval_config).

A `Grader` scores a batch of samples and returns one aggregate (mean over
samples) per metric, for a single run. `run_judge` handles the median-of-N.
Two graders ship: `RagasGrader` (real, lazy-imported, needs a key) and
`HeuristicGrader` (deterministic, offline, lexical proxies — for tests and
offline runs; NOT a real quality signal).
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

JUDGE_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "answer_correctness",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class JudgeSample:
    id: str
    question: str
    answer: str
    contexts: tuple[str, ...]
    ground_truth: str


class Grader(Protocol):
    grader_id: str
    metrics: tuple[str, ...]

    def grade(self, samples: Sequence[JudgeSample]) -> dict[str, float]:
        """Aggregate score per metric (mean over samples) for ONE run."""
        ...


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap(a: str, b: str) -> float:
    """Fraction of a's tokens that appear in b. 0.0 when a has no tokens."""
    ta = _tokens(a)
    if not ta:
        return 0.0
    return len(ta & _tokens(b)) / len(ta)


class HeuristicGrader:
    """Deterministic lexical proxies. Offline, zero-cost — for tests, not truth."""

    grader_id = "heuristic"
    metrics = JUDGE_METRICS

    def grade(self, samples: Sequence[JudgeSample]) -> dict[str, float]:
        if not samples:
            return {m: 0.0 for m in self.metrics}
        acc = {m: 0.0 for m in self.metrics}
        for s in samples:
            context = " ".join(s.contexts)
            acc["faithfulness"] += _overlap(s.answer, context)
            acc["answer_relevancy"] += _overlap(s.answer, s.question)
            acc["context_precision"] += _overlap(context, s.ground_truth or s.question)
            acc["answer_correctness"] += _overlap(s.answer, s.ground_truth)
        return {m: acc[m] / len(samples) for m in self.metrics}


_DEFAULT_JUDGE_MODEL = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5"}

#: Providers that can act as a judge LLM. `echo` generates but cannot grade.
JUDGE_PROVIDERS: tuple[str, ...] = tuple(sorted(_DEFAULT_JUDGE_MODEL))


def resolve_judge_provider(generation_provider: str, override: str | None = None) -> str:
    """Which LLM grades the judge suite — the config decides unless overridden.

    The judge provider used to default to "openai" independently of the config,
    so a repo that had switched `generation.provider` to anthropic still judged
    on OpenAI. That is not a preference mismatch, it is a wrong answer: the run
    burns the wrong key and reports metrics from a model nobody selected. Here
    the config is the source of truth; an explicit --judge-provider still wins.
    """
    if override:
        if override not in _DEFAULT_JUDGE_MODEL:
            raise ValueError(f"Unknown judge provider: {override!r}")
        return override
    if generation_provider in _DEFAULT_JUDGE_MODEL:
        return generation_provider
    raise ValueError(
        f"generation.provider is {generation_provider!r}, which cannot grade. "
        f"Set it to one of {list(JUDGE_PROVIDERS)} in the config, or pass "
        "--judge-provider explicitly."
    )


def _as_finite(value: object) -> float | None:
    """Coerce one RAGAS score to a float, or None if it is not a usable number."""
    if isinstance(value, bool):  # bool is an int subclass; never a score
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def aggregate_metric(value: object, metric: str) -> float:
    """Reduce one RAGAS metric to a single mean over samples.

    RAGAS returns a **list** of per-sample scores, not a scalar, so the previous
    `float(result[m])` raised `TypeError: float() argument must be ... not
    'list'` on every real run. Worse, a sample whose grading call failed (rate
    limit, refusal, timeout) comes back as NaN/None *inside* that list instead
    of raising, so a naive mean silently poisons the whole metric.

    Non-finite entries are therefore dropped before the mean. If nothing usable
    survives we raise rather than return 0.0: a zero would read as a genuine
    quality collapse and trip the gate, hiding the fact that grading never
    actually happened.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"Judge metric {metric!r} came back as text, not a score: {value!r}")
    if isinstance(value, (int, float)):
        raw: list[object] = [value]
    else:
        try:
            raw = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                f"Judge metric {metric!r} is neither a number nor a sequence of "
                f"scores (got {type(value).__name__})"
            ) from exc
    if not raw:
        raise RuntimeError(f"Judge metric {metric!r} came back empty — nothing was graded.")
    usable = [f for f in (_as_finite(v) for v in raw) if f is not None]
    if not usable:
        raise RuntimeError(
            f"Judge metric {metric!r}: all {len(raw)} sample score(s) failed to "
            "grade (NaN/None). The judge LLM returned no usable output — check "
            "the provider key and the run log for per-sample errors."
        )
    return sum(usable) / len(usable)


class RagasGrader:
    """Real judge metrics via RAGAS. Lazy-imported; needs a provider key.

    The judge LLM is interchangeable: `provider="openai"` (needs OPENAI_API_KEY)
    or `provider="anthropic"` (needs ANTHROPIC_API_KEY, via langchain-anthropic).
    This is the production grader the nightly / full-eval runs use. Tests never
    call a real LLM; they inject a fake `ragas` module to pin down the response
    handling, which is where both of this class's historical bugs lived.

    `answer_relevancy` and `answer_correctness` additionally need an *embeddings*
    model, and RAGAS falls back to OpenAI embeddings when none is supplied — so
    an Anthropic-only run would still hit OpenAI for half the suite. Pass the
    project's own embedder (`embedder=`) and the judge stays on the configured
    stack end to end; leave it None only to accept the RAGAS default.
    """

    grader_id = "ragas"
    metrics = JUDGE_METRICS

    def __init__(self, provider: str = "openai", model: str = "", embedder=None):
        if provider not in _DEFAULT_JUDGE_MODEL:
            raise ValueError(f"Unknown judge provider: {provider!r}")
        self.provider = provider
        self.model = model or _DEFAULT_JUDGE_MODEL[provider]
        #: Project `Embedder` (see src.embed). None = let RAGAS pick its default.
        self.embedder = embedder

    def _llm(self):
        """Build the RAGAS-wrapped LLM for the configured provider (lazy)."""
        from ragas.llms import LangchainLLMWrapper

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # bypass_temperature: RAGAS *mutates* `llm.temperature` before each
            # call (default 0.01) to vary sampling across n completions. Recent
            # Claude models reject the parameter outright —
            #   400 invalid_request_error: `temperature` is deprecated for this
            #   model
            # — which failed every sample in run 34453595063. ChatAnthropic
            # omits temperature when it is None, so the flag is all that is
            # needed. OpenAI is left alone: temperature works there, and RAGAS
            # varies it deliberately.
            #
            # Note: only ragas' async path (`agenerate_text`) honours this flag;
            # the sync `generate_text` sets the attribute regardless. evaluate()
            # takes the async path, so this holds — but a future ragas that
            # switched to the sync path would reintroduce the 400.
            return LangchainLLMWrapper(
                ChatAnthropic(model=self.model), bypass_temperature=True
            )
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(ChatOpenAI(model=self.model))

    def _embeddings(self):
        """Adapt the project's embedder to the RAGAS embedding interface.

        Without this, RAGAS silently falls back to OpenAI embeddings for the two
        metrics that need them, which defeats the point of choosing a provider
        and fails outright on a key-less or credit-less OpenAI account. Reusing
        the retriever's own embedder also means the judge scores answers in the
        same vector space the retrieval was measured in.
        """
        if self.embedder is None:
            return None
        from ragas.embeddings import BaseRagasEmbedding

        inner = self.embedder

        class _ProjectEmbedding(BaseRagasEmbedding):
            def embed_text(self, text: str, **kwargs) -> list[float]:
                return [float(x) for x in inner.embed([text])[0]]

            async def aembed_text(self, text: str, **kwargs) -> list[float]:
                return self.embed_text(text)

            def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
                # Batched: the whole point of a local model is one forward pass.
                return [[float(x) for x in row] for row in inner.embed(list(texts))]

        return _ProjectEmbedding()

    def grade(self, samples: Sequence[JudgeSample]) -> dict[str, float]:
        from datasets import Dataset  # lazy, heavy
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        ds = Dataset.from_dict(
            {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [list(s.contexts) for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
        )
        kwargs = {}
        embeddings = self._embeddings()
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        result = ragas_evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
            llm=self._llm(),
            **kwargs,
        )
        return {m: aggregate_metric(result[m], m) for m in self.metrics}


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def run_judge(
    samples: Sequence[JudgeSample], grader: Grader, runs: int = 3
) -> dict:
    """Run the grader `runs` times; median per metric + the observed spread."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    per_run = [grader.grade(samples) for _ in range(runs)]
    metrics_out: dict[str, float] = {}
    spread: dict[str, dict] = {}
    for m in grader.metrics:
        vals = sorted(run[m] for run in per_run)
        metrics_out[m] = median(vals)
        spread[m] = {"min": vals[0], "max": vals[-1], "runs": vals}
    return {"metrics": metrics_out, "spread": spread, "runs": runs}


def build_grader(
    backend: str, provider: str = "openai", model: str = "", embedder=None
) -> Grader:
    if backend == "heuristic":
        return HeuristicGrader()
    if backend == "ragas":
        return RagasGrader(provider=provider, model=model, embedder=embedder)
    raise ValueError(f"Unknown judge backend: {backend!r}")
