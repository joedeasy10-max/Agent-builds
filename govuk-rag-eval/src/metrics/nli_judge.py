"""Local judge metrics via NLI entailment — no API key, no spend, no variance.

The paid judge costs $3.69 a run and, more importantly for a gate, it MOVES:
faithfulness' median shifted 2.5% between two identical runs (0.975 on run
34498691301, 0.950 on 34505214155), which is why its tolerance had to be widened
to 8%. A tolerance that wide cannot see a small real regression — the noise is
using two thirds of the budget.

This grader is deterministic. The same inputs give the same numbers every time,
so its tolerance can be tight, and a move always means the system changed. For a
regression gate that trade is worth taking even though each individual metric is
cruder than an LLM's judgement.

How each metric is computed here, and how that differs from RAGAS:

**Scope: this makes GRADING free, not the whole judge suite.** The suite scores
generated answers, so it still generates them, and that uses
`generation.provider`. Run 34616093028 is the proof: the NLI grader loaded
correctly and the step then died on `import anthropic` inside the generator.
What actually changes is the bulk of the bill — grading was $3.69 a run
(41 questions x 3 runs x $0.03) and is now zero, leaving one generation pass per
question at --runs 1. For a fully key-free suite you would also need a local or
extractive generator, which measures something different and is not claimed here.

* faithfulness      Natural-language inference, the standard pre-LLM approach
                    (SummaC / AlignScore style). Each answer sentence is a
                    hypothesis; each retrieved context is a premise; the score
                    is the mean over sentences of the best entailment
                    probability any context gives it. A sentence no context
                    entails is a hallucination, which is exactly the thing
                    faithfulness is supposed to catch.
* answer_relevancy  Cosine similarity between the question and the answer, using
                    the project's own embedder. RAGAS instead asks an LLM to
                    reverse-engineer questions from the answer; that needs a
                    model, this does not.
* context_precision Mean rank-weighted similarity of the retrieved contexts to
                    the question — a precision-flavoured proxy, not RAGAS's
                    LLM-judged relevance.
* answer_correctness Similarity plus token F1 against the ground truth. Reported,
                    never gated, exactly as before.

**These numbers are NOT comparable to the RAGAS baseline.** Different estimators
measuring similar ideas land on different scales; a faithfulness of 0.82 here
does not mean the system got worse than 0.95 there. Switching grader invalidates
the judge half of results/baseline.json in the same way switching provider does,
and a new baseline has to be measured. compare.py already refuses to gate across
a dataset version change; this is the same class of change and needs the same
care.

What is deliberately NOT claimed: an NLI model is weaker than a large model at
multi-step reasoning, and a sentence that is true but phrased far from the
passage can score low. The mitigation is that this is a *relative* instrument —
it is asked whether quality moved, not what it is in absolute terms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from .judge import JUDGE_METRICS, JudgeSample

#: Sentence split good enough for answer text: a terminator followed by
#: whitespace and a capital or digit. Avoids a dependency on a sentence
#: tokeniser for a job this simple.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9£])")
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Cosine similarity is in [-1, 1] but text embeddings rarely go negative; the
#: metrics are reported in [0, 1], so negatives clamp rather than wrap.
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def split_sentences(text: str) -> list[str]:
    """Answer text into sentences. Always returns at least one item for
    non-empty input, so a one-line answer is still scored."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text]


def token_f1(a: str, b: str) -> float:
    """Harmonic mean of token precision and recall. Order-insensitive."""
    ta = set(_WORD_RE.findall((a or "").lower()))
    tb = set(_WORD_RE.findall((b or "").lower()))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(ta), overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


class Entailment(Protocol):
    """Scores P(premise entails hypothesis) for a batch of pairs."""

    def entailment_probs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[float]: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return num / (na * nb)


class CrossEncoderEntailment:
    """`sentence_transformers.CrossEncoder` wrapped as an `Entailment`.

    The entailment class is resolved BY NAME from the model's own `id2label`,
    never by a hardcoded index. NLI checkpoints disagree about ordering —
    some are (contradiction, entailment, neutral), others (entailment, neutral,
    contradiction) — and picking the wrong column silently inverts the metric
    into something that looks plausible and is wrong. If the label cannot be
    found the constructor raises and names what it did find, because a judge
    that quietly scores the wrong column is worse than one that will not start.

    Lazy-imported: the offline suite must not need torch installed.
    """

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # lazy

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)
        self._entail_index = self._resolve_entailment_index()

    def _resolve_entailment_index(self) -> int:
        labels = self._label_map()
        for idx, name in sorted(labels.items()):
            if str(name).strip().lower().startswith("entail"):
                return int(idx)
        raise ValueError(
            f"{self.model_name!r} exposes no entailment label; found {labels!r}. "
            "This must be a 3-way NLI checkpoint (entailment / neutral / "
            "contradiction), not a relevance reranker."
        )

    def _label_map(self) -> dict:
        # CrossEncoder wraps a HF sequence-classification model; the label map
        # lives on its config. Checked defensively because the attribute path
        # is the part of this integration most likely to move between versions.
        config = getattr(getattr(self._model, "model", None), "config", None)
        labels = getattr(config, "id2label", None)
        if not labels:
            raise ValueError(
                f"Could not read id2label from {self.model_name!r}; without it "
                "the entailment column cannot be identified safely."
            )
        return dict(labels)

    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            [(premise, hypothesis) for premise, hypothesis in pairs],
            batch_size=self.batch_size,
            apply_softmax=True,
        )
        out = []
        for row in scores:
            values = list(row) if hasattr(row, "__len__") else [row]
            if len(values) <= self._entail_index:
                raise ValueError(
                    f"{self.model_name!r} returned {len(values)} score(s); "
                    f"expected at least {self._entail_index + 1} for a 3-way NLI model."
                )
            out.append(_clamp01(values[self._entail_index]))
        return out


class NLIGrader:
    """Deterministic judge metrics. Satisfies the same `Grader` protocol.

    Slots into `run_judge` unchanged, but note that median-of-N is pointless
    here: every run is identical, so `--runs 1` is the right setting and the
    recorded spread will be exactly zero. That zero is the feature.
    """

    grader_id = "nli"
    metrics = JUDGE_METRICS

    def __init__(
        self,
        entailment: Entailment,
        embedder: Embedder,
        *,
        max_contexts: int = 5,
    ):
        self.entailment = entailment
        self.embedder = embedder
        self.max_contexts = max_contexts

    # -- per-sample scores --------------------------------------------------

    def faithfulness(self, sample: JudgeSample) -> float:
        """Mean over answer sentences of the best entailment any context gives.

        "Best any context gives" rather than "the concatenation entails it",
        because a supported answer usually draws on one passage; requiring every
        context to entail every sentence would punish breadth.
        """
        sentences = split_sentences(sample.answer)
        contexts = [c for c in sample.contexts[: self.max_contexts] if c.strip()]
        if not sentences or not contexts:
            return 0.0
        pairs = [(ctx, sent) for sent in sentences for ctx in contexts]
        probs = self.entailment.entailment_probs(pairs)
        per_sentence = [
            max(probs[i * len(contexts) : (i + 1) * len(contexts)])
            for i in range(len(sentences))
        ]
        return sum(per_sentence) / len(per_sentence)

    def _similarity(self, a: str, b: str) -> float:
        if not (a or "").strip() or not (b or "").strip():
            return 0.0
        va, vb = self.embedder.embed([a, b])
        return _clamp01(cosine(va, vb))

    def answer_relevancy(self, sample: JudgeSample) -> float:
        return self._similarity(sample.question, sample.answer)

    def context_precision(self, sample: JudgeSample) -> float:
        """Rank-weighted similarity of retrieved contexts to the question.

        Weighted by 1/rank so a relevant context at rank 1 counts for more than
        the same context at rank 5 — precision-flavoured, which is the property
        the RAGAS metric of this name is after, reached by a different route.
        """
        contexts = [c for c in sample.contexts[: self.max_contexts] if c.strip()]
        if not contexts or not (sample.question or "").strip():
            return 0.0
        vectors = self.embedder.embed([sample.question, *contexts])
        question_vec, context_vecs = vectors[0], vectors[1:]
        weights = [1.0 / (i + 1) for i in range(len(context_vecs))]
        weighted = sum(
            w * _clamp01(cosine(question_vec, cv))
            for w, cv in zip(weights, context_vecs)
        )
        return weighted / sum(weights)

    def answer_correctness(self, sample: JudgeSample) -> float:
        """Half semantic similarity, half token F1 against the ground truth.

        Two views because either alone misleads: similarity alone rewards an
        answer that is on-topic and wrong, F1 alone punishes a correct answer
        that paraphrases. Reported, never gated.
        """
        semantic = self._similarity(sample.answer, sample.ground_truth)
        lexical = token_f1(sample.answer, sample.ground_truth)
        return 0.5 * semantic + 0.5 * lexical

    # -- the Grader protocol ------------------------------------------------

    def grade(self, samples: Sequence[JudgeSample]) -> dict[str, float]:
        if not samples:
            raise ValueError("no samples to grade")
        totals = {m: 0.0 for m in self.metrics}
        for sample in samples:
            totals["faithfulness"] += self.faithfulness(sample)
            totals["answer_relevancy"] += self.answer_relevancy(sample)
            totals["context_precision"] += self.context_precision(sample)
            totals["answer_correctness"] += self.answer_correctness(sample)
        return {m: totals[m] / len(samples) for m in self.metrics}
