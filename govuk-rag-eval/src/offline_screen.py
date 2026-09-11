"""Deterministic candidate screening — no API key, no spend, no variance.

Why this exists, from measured evidence rather than preference. Two paid LLM
screens over the same 134 candidates (runs 34608236507 and 34610387397) decided:

    113  clean_and_confident     model said "all four criteria pass"
     15  quality_fixable         13 of these forced by a REGEX, not the model
      5  source_support_failed   the model's real contribution
      1  clean_but_unsure        the model's real contribution
      0  clear_failure           the model never rejected anything. Twice.

So of 134 decisions the model independently drove 8, and rejected nothing at
all. That is not a failure of the prompt: it follows from how the candidates are
made. The drafter writes each question AND its ground truth FROM the passage, so
"is the question about this passage" and "does the answer answer it" are close to
guaranteed by construction — the two criteria a language model is genuinely
needed for are the two that cannot fail here.

What CAN go wrong in a drafted candidate is mechanical, and mechanical checks
catch mechanical faults better than a model does:

  * phrasing that refers to the passage ("According to the guidance, ...") —
    the model missed 3 of 13; a regex caught 13 of 13;
  * the same question drafted twice — word overlap, exactly;
  * a ground truth asserting a figure the passage does not contain — this is
    string containment, and a model asked to eyeball it will sometimes say yes.

There is a second argument, and for a regression gate it is the stronger one:
**this screen is reproducible and the LLM one is not.** The judge's faithfulness
median moved 2.5% between two identical runs, which is why its tolerance had to
be widened to 8%. A deterministic screen has zero run-to-run variance, so a
change in its output always means the candidates changed.

Measured against the paid screen on the same 134 candidates, full passages,
run 34613876697 vs 34610387397:

    offline   106 approve / 28 review     < 1 second      $0.00
    LLM       113 approve / 21 review     6m 41s          $1.34
    agreement 117/134 = 87%

The 17 disagreements split 12 / 5. Twelve are the offline screener being more
cautious than the model, which costs review time and nothing else. **Five are
the other direction and are the real price of this change**: candidates offline
approves that the model held back —

    2  the model found a source-support problem the lexical check missed
    2  the model disliked phrasing the rules here accept
    1  the model was simply unsure

Two unsupported answers reaching the dataset unreviewed, per 134 candidates, is
the honest cost. Whether that is worth $1.34 and seven minutes a batch is a
judgement, not a fact, and the `llm` backend is still there for anyone who
decides it is — the sensible middle is to screen offline per batch and run the
paid screen once before a promotion.

What this deliberately does NOT claim: it cannot tell you a ground truth is
factually wrong in a way that reuses the passage's own vocabulary, and the two
missed support failures above are exactly that shape. Nothing here pretends
otherwise. Candidates it cannot judge go to a human.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .candidate_review import (
    CRITERIA,
    Evaluation,
    decide,
    find_near_duplicates,
    is_self_referential,
    tokens,
)

#: Content-word overlap floors. Deliberately low: these are sanity checks for
#: "is this about the same thing at all", not similarity scores. A question
#: legitimately shares few words with its passage (it asks; the passage tells).
MIN_QUESTION_OVERLAP = 0.04
MIN_GROUND_TRUTH_SUPPORT = 0.30

#: Words carrying no topic signal, excluded before overlap is measured so that
#: "what is the" matching "the" does not read as relevance.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by
with without about into over under again further is are was were be been being am
do does did doing have has had having i you he she it we they them his her its our
your their what which who whom when where why how all any both each few more most
other some such no nor not only own same so too very can will just should now as
""".split())

#: Figures are where a drafted ground truth goes wrong in a way code can see.
#: Money, percentages, plain numbers, dates and form codes all have to appear in
#: the passage if the answer asserts them.
_FIGURE_RE = re.compile(
    r"(?:£\s?[\d,]+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s?%"
    r"|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\b"
    r"|\b(?:19|20)\d{2}(?:[-/]\d{2})?\b"
    r"|\b[A-Z]{1,3}\d{2,4}[A-Z]?\b"
    r"|\b\d[\d,]*(?:\.\d+)?\b)",
    re.IGNORECASE,
)

# Words that can open a question. Note what is NOT here any more: `if in under
# for by on at to after before during`. Those prepositions were in the list only
# to let adverbial openers past a check anchored at `^`, which is the tell that
# the anchoring was the bug — the fix is `reads_as_question` below, not a longer
# list of first words.
_OPENS_A_QUESTION = re.compile(
    r"^\s*(what|when|where|which|who|whom|whose|why|how|do|does|did|is|are|was|were"
    r"|can|could|should|must|will|would|may|might)\b",
    re.IGNORECASE,
)


def reads_as_question(text: str) -> bool:
    """Is this phrased as a question?

    The previous check was `^(what|when|...)` alone — anchored at the first
    word. Any question opening with an adverbial clause failed it, and seven of
    the 134 drafted candidates went to human review on that basis and nothing
    else:

        "While studying, what is the interest rate on a Plan 2 loan?"
        "From which tax year does cash basis become the default method?"
        "As the intermediary, what do you need to do?"

    Nothing is wrong with any of those. A trailing "?" is the strongest signal
    available and is now checked first; the anchored word list stays as the
    fallback for a question written without the punctuation.

    Deliberately NOT done: searching for a question word anywhere near the
    front. The auxiliaries in that list (is/are/was/were/do/does/did) occur in
    ordinary statements, so "The rules are set out in the guidance below" would
    have scored as a question. Trailing "?" plus first-word inversion covers the
    real cases without that false pass.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return stripped.endswith("?") or bool(_OPENS_A_QUESTION.match(stripped))

MIN_QUESTION_CHARS = 20
MAX_QUESTION_CHARS = 300
MIN_GROUND_TRUTH_CHARS = 3


def content_tokens(text: str) -> set[str]:
    return tokens(text) - _STOPWORDS


def overlap(a: str, b: str) -> float:
    """Fraction of a's content words that also appear in b. Directional."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _normalise_figure(fig: str) -> str:
    """Compare figures by value, not formatting: '£1,000' == '£1000' == '1000'."""
    return re.sub(r"[£,\s]", "", fig).lower().rstrip(".")


def unsupported_figures(ground_truth: str, passage: str) -> list[str]:
    """Figures asserted by the answer that do not appear in the passage.

    The sharpest deterministic signal available here. A drafted answer citing
    "£1,000" when the passage says "£500" is wrong in a way that needs no
    judgement to see, and an LLM asked to eyeball the same comparison will
    sometimes wave it through.
    """
    if not passage.strip():
        return []
    have = {_normalise_figure(f) for f in _FIGURE_RE.findall(passage)}
    missing = []
    for fig in _FIGURE_RE.findall(ground_truth):
        norm = _normalise_figure(fig)
        if norm and norm not in have:
            missing.append(fig.strip())
    return missing


class OfflineScreener:
    """Screens a candidate with no network call. Same verdict shape as the LLM.

    Produces the same four criteria and feeds the same `decide()` rules, so the
    review page, the queue format and the promotion gate are all unchanged — the
    only difference is where the pass/fail values come from.

    Confidence is not a pose. Each criterion is scored with a margin, and the
    reported confidence is driven by the narrowest one: comfortable margins give
    a confident verdict, borderline ones drop below the approval floor so the
    candidate reaches a human. That is how a deterministic screener expresses
    "I cannot tell" without pretending to a judgement it has not made.
    """

    screener_id = "offline"

    def __init__(
        self,
        min_question_overlap: float = MIN_QUESTION_OVERLAP,
        min_ground_truth_support: float = MIN_GROUND_TRUTH_SUPPORT,
    ):
        self.min_question_overlap = min_question_overlap
        self.min_ground_truth_support = min_ground_truth_support

    def screen(self, candidate, others: Sequence[tuple[str, str]] = ()) -> Evaluation:
        question = (candidate.question or "").strip()
        ground_truth = (candidate.ground_truth or "").strip()
        passage = (candidate.source_excerpt or "").strip()
        is_negative = candidate.difficulty == "negative"

        near = find_near_duplicates(question, others)
        duplicate = bool(near)

        notes: list[str] = []
        margins: list[float] = []
        criteria: dict[str, str] = {}

        # --- relevance: is the question about this passage at all? ----------
        q_overlap = overlap(question, passage) if passage else 0.0
        criteria["relevance"] = "pass" if q_overlap >= self.min_question_overlap else "fail"
        if criteria["relevance"] == "fail":
            notes.append(
                f"question shares {q_overlap:.0%} of its content words with the passage"
            )
        margins.append(_margin(q_overlap, self.min_question_overlap, scale=0.15))

        # --- source support: is the answer actually in the passage? ---------
        missing = unsupported_figures(ground_truth, passage)
        gt_support = overlap(ground_truth, passage) if passage else 0.0
        if missing:
            criteria["source_support"] = "fail"
            notes.append(
                "the answer states figures the passage does not contain: "
                + ", ".join(missing[:4])
            )
            margins.append(0.0)          # a concrete contradiction; no margin
        elif gt_support < self.min_ground_truth_support:
            criteria["source_support"] = "fail"
            notes.append(
                f"only {gt_support:.0%} of the answer's content words appear in the passage"
            )
            margins.append(_margin(gt_support, self.min_ground_truth_support, scale=0.25))
        else:
            criteria["source_support"] = "pass"
            margins.append(_margin(gt_support, self.min_ground_truth_support, scale=0.25))

        # --- ground truth accuracy: only the checkable part -----------------
        # Deliberately narrow. Whether an answer is FACTUALLY right, when it
        # reuses the passage's own words, is not decidable here and is not
        # claimed. What is decidable: that an answer exists, says something, and
        # is not simply the question repeated back.
        if len(ground_truth) < MIN_GROUND_TRUTH_CHARS:
            criteria["ground_truth_accuracy"] = "fail"
            notes.append("the ground truth is empty or too short to be an answer")
            margins.append(0.0)
        elif overlap(ground_truth, question) > 0.95 and len(content_tokens(ground_truth)) > 2:
            criteria["ground_truth_accuracy"] = "fail"
            notes.append("the ground truth restates the question instead of answering it")
            margins.append(0.0)
        else:
            criteria["ground_truth_accuracy"] = "pass"
            margins.append(1.0)

        # --- question quality ------------------------------------------------
        quality_problem = None
        if is_self_referential(question):
            quality_problem = (
                "phrasing refers to the source rather than standing alone; a real "
                "user asks the question without naming the guidance"
            )
        elif len(question) < MIN_QUESTION_CHARS:
            quality_problem = f"question is only {len(question)} characters"
        elif len(question) > MAX_QUESTION_CHARS:
            quality_problem = f"question is {len(question)} characters, too long to be natural"
        elif not reads_as_question(question):
            quality_problem = "does not read as a question"
        criteria["question_quality"] = "fail" if quality_problem else "pass"
        if quality_problem:
            notes.append(quality_problem)
        margins.append(0.0 if quality_problem else 1.0)

        confidence = round(min(margins), 2) if margins else 0.0
        # A clean candidate is reported at the approval floor or above; a
        # borderline one falls below it and reaches a human by the same rules
        # the LLM screen used.
        decision, reason, rule = decide(
            criteria,
            confidence=confidence,
            duplicate=duplicate,
            is_negative=is_negative,
            has_passage=bool(passage),
        )
        detail = ("  " + "; ".join(notes)) if notes else ""
        return Evaluation(
            decision=decision,
            confidence=confidence,
            relevance=criteria["relevance"],
            ground_truth_accuracy=criteria["ground_truth_accuracy"],
            source_support=criteria["source_support"],
            question_quality=criteria["question_quality"],
            duplicate=duplicate,
            reason=reason + detail,
            self_referential=is_self_referential(question),
            suggested_question="",
            model_decision="",
            model_reason="",
            duplicate_of=near[0][0] if near else "",
            rule=rule,
            evaluator=self.screener_id,
        )


def _margin(value: float, threshold: float, scale: float) -> float:
    """How comfortably `value` clears `threshold`, as a 0..1 confidence.

    Right at the threshold scores 0.5 — below the approval floor, so a knife-edge
    candidate goes to a human instead of being waved through. `scale` is how far
    past the threshold counts as fully clear.
    """
    if scale <= 0:
        return 1.0 if value >= threshold else 0.0
    return max(0.0, min(1.0, 0.5 + (value - threshold) / (2 * scale)))
