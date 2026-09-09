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


class RagasGrader:
    """Real judge metrics via RAGAS. Lazy-imported; needs a provider key.

    The judge LLM is interchangeable: `provider="openai"` (needs OPENAI_API_KEY)
    or `provider="anthropic"` (needs ANTHROPIC_API_KEY, via langchain-anthropic).
    Not exercised by tests (they never call an LLM). This is the production
    grader the nightly / full-eval runs use.

    Note: RAGAS `answer_relevancy` also needs an embeddings model. Anthropic has
    no embeddings API, so an Anthropic judge still uses an embeddings backend
    (OpenAI, or a local one) for that one metric — configure it in the RAGAS run
    if you go Anthropic-only.
    """

    grader_id = "ragas"
    metrics = JUDGE_METRICS

    def __init__(self, provider: str = "openai", model: str = ""):
        if provider not in _DEFAULT_JUDGE_MODEL:
            raise ValueError(f"Unknown judge provider: {provider!r}")
        self.provider = provider
        self.model = model or _DEFAULT_JUDGE_MODEL[provider]

    def _llm(self):
        """Build the RAGAS-wrapped LLM for the configured provider (lazy)."""
        from ragas.llms import LangchainLLMWrapper

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return LangchainLLMWrapper(ChatAnthropic(model=self.model))
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(ChatOpenAI(model=self.model))

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
        result = ragas_evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
            llm=self._llm(),
        )
        return {m: float(result[m]) for m in self.metrics}


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


def build_grader(backend: str, provider: str = "openai", model: str = "") -> Grader:
    if backend == "heuristic":
        return HeuristicGrader()
    if backend == "ragas":
        return RagasGrader(provider=provider, model=model)
    raise ValueError(f"Unknown judge backend: {backend!r}")
