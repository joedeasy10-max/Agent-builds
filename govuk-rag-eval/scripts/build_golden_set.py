"""Draft golden-set candidates with an LLM, then queue them for HUMAN review.

Build step 3. BUILD.md is explicit about the workflow:

    "Draft with an LLM over your chunks, then review every one by hand — that
     review is what makes the numbers mean anything."

So this tool deliberately **never writes to data/golden/questions.jsonl from LLM
output**. It has three subcommands and a one-way flow through a review queue:

    draft   -> data/golden/review_queue.jsonl   (every candidate status=pending)
    <you edit the queue: set status to approved / rejected, fix wording>
    promote -> appends only status=approved candidates to questions.jsonl

Each candidate carries the source chunk's text (`source_excerpt`), so reviewing
is reading one screen — not hunting through the corpus for what the question was
grounded in.

Negatives matter: BUILD.md wants 20-30 questions the corpus genuinely cannot
answer (the right behaviour is refusal). `draft --negatives N` produces those;
they get `difficulty: negative` and empty `source_ids`.

Cost discipline: drafting is one LLM call per chunk. `--limit` caps the chunks
used and `--max-usd` refuses to start if the estimated spend is over budget.

The LLM provider is interchangeable (openai | anthropic), taken from
`generation.provider` in the retrieval config unless `--provider` overrides it.

    python scripts/build_golden_set.py draft \
        --config configs/retrieval.yaml --index .index/ \
        --per-chunk 2 --limit 60 --negatives 25
    python scripts/build_golden_set.py status
    python scripts/build_golden_set.py promote --version v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Run as a plain script: put the repo root on the path so `src` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidate_review import (  # noqa: E402
    FatalEvaluatorError,
    apply_decision,
    evaluate_one,
    preceding_peers,
)
from src.chunk import Chunk  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.golden import load_golden  # noqa: E402
from src.store import load_store  # noqa: E402

DIFFICULTIES = {"single_hop", "multi_hop", "negative"}
STATUSES = {"pending", "approved", "rejected"}

DEFAULT_QUEUE = Path("data/golden/review_queue.jsonl")
DEFAULT_GOLDEN = Path("data/golden/questions.jsonl")

# Rough per-draft-call cost, used only for the pre-flight budget check.
_COST_PER_CALL_USD = 0.01

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    status: str
    question: str
    ground_truth: str
    source_ids: tuple[str, ...]
    difficulty: str
    source_excerpt: str
    notes: str = ""
    #: Automated pre-screen verdict (src/candidate_review.py), or None if this
    #: candidate has not been evaluated. Advisory metadata: `status` above stays
    #: the field that decides promotion, so an evaluation can never promote by
    #: itself.
    evaluation: dict | None = None


# --- pure helpers (unit-tested; no LLM, no network) -------------------------


def strip_fences(text: str) -> str:
    """LLMs like to wrap JSON in ```json fences. Remove them."""
    return _FENCE_RE.sub("", text.strip())


def _warn_dropped(raw: str, why: str) -> None:
    """Report unusable drafting output loudly; empty output is not noteworthy."""
    if not raw.strip():
        return
    tail = raw.strip()[-160:].replace("\n", " ")
    print(
        f"WARNING: dropped a drafting response ({why}); "
        f"{len(raw)} chars, ends: …{tail}",
        file=sys.stderr,
    )


def parse_candidates(
    raw: str, chunk: Chunk | None, start_index: int, negative: bool = False
) -> list[Candidate]:
    """Parse an LLM JSON array into Candidates. Skips malformed entries.

    Drafting output is untrusted: anything missing a question, or carrying a
    difficulty we don't recognise, is dropped rather than trusted.
    """
    try:
        rows = json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        # Dropping malformed output is correct — it is untrusted — but doing it
        # silently is how a run asked for 25 negatives, got 0, and reported
        # success. The usual cause is a response truncated at max_tokens, so the
        # tail of the raw text is the useful diagnostic.
        _warn_dropped(raw, "not valid JSON")
        return []
    if not isinstance(rows, list):
        _warn_dropped(raw, f"top level is {type(rows).__name__}, expected a list")
        return []

    out: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        if negative:
            difficulty = "negative"
            source_ids: tuple[str, ...] = ()
            excerpt = ""
        else:
            difficulty = str(row.get("difficulty", "single_hop")).strip()
            if difficulty not in DIFFICULTIES or difficulty == "negative":
                difficulty = "single_hop"
            if chunk is None:
                continue
            source_ids = (chunk.chunk_id,)
            excerpt = chunk.text
        out.append(
            Candidate(
                candidate_id=f"cand_{start_index + len(out):04d}",
                status="pending",
                question=question,
                ground_truth=str(row.get("ground_truth", "")).strip(),
                source_ids=source_ids,
                difficulty=difficulty,
                source_excerpt=excerpt,
            )
        )
    return out


def write_queue(path: str | Path, candidates: list[Candidate]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in candidates:
            row = asdict(c)
            row["source_ids"] = list(c.source_ids)
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def load_queue(path: str | Path) -> list[Candidate]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[Candidate] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        status = str(row.get("status", "pending"))
        if status not in STATUSES:
            raise ValueError(
                f"{p}: candidate {row.get('candidate_id')!r} has status "
                f"{status!r}; expected one of {sorted(STATUSES)}"
            )
        out.append(
            Candidate(
                candidate_id=str(row["candidate_id"]),
                status=status,
                question=str(row.get("question", "")),
                ground_truth=str(row.get("ground_truth", "")),
                source_ids=tuple(row.get("source_ids", []) or []),
                difficulty=str(row.get("difficulty", "single_hop")),
                source_excerpt=str(row.get("source_excerpt", "")),
                notes=str(row.get("notes", "")),
                evaluation=row.get("evaluation") or None,
            )
        )
    return out


def _next_id_number(existing_ids: list[str]) -> int:
    highest = 0
    for qid in existing_ids:
        m = re.match(r"^q_(\d+)$", qid)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def promote(
    candidates: list[Candidate], existing: list[dict], version: str
) -> tuple[list[dict], list[str]]:
    """Turn approved candidates into golden records. Returns (new, skipped).

    * only `approved` candidates are promoted — pending/rejected are left alone;
    * a candidate whose question text already exists is skipped (no duplicates);
    * ids continue from the highest existing q_NNNN, so promoting twice never
      renumbers or collides with what the golden set already references.
    """
    if not re.match(r"^v\d+$", version):
        raise ValueError(f"version must look like 'v2', got {version!r}")

    seen_questions = {r.get("question", "").strip().lower() for r in existing}
    next_n = _next_id_number([str(r.get("id", "")) for r in existing])

    new: list[dict] = []
    skipped: list[str] = []
    for c in candidates:
        if c.status != "approved":
            continue
        key = c.question.strip().lower()
        if key in seen_questions:
            skipped.append(c.candidate_id)
            continue
        if c.difficulty == "negative":
            if c.source_ids:
                skipped.append(c.candidate_id)
                continue
        elif not c.source_ids:
            skipped.append(c.candidate_id)
            continue
        seen_questions.add(key)
        new.append(
            {
                "id": f"q_{next_n:04d}",
                "question": c.question.strip(),
                "ground_truth": c.ground_truth.strip(),
                "source_ids": list(c.source_ids),
                "difficulty": c.difficulty,
                "added_in": version,
            }
        )
        next_n += 1
    return new, skipped


def summarise(candidates: list[Candidate]) -> dict:
    """Counts by human status, by difficulty, and by automated decision."""
    by_status = {s: 0 for s in sorted(STATUSES)}
    by_difficulty: dict[str, int] = {}
    by_auto: dict[str, int] = {"unevaluated": 0}
    for c in candidates:
        by_status[c.status] = by_status.get(c.status, 0) + 1
        by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
        decision = (c.evaluation or {}).get("decision")
        key = decision if decision else "unevaluated"
        by_auto[key] = by_auto.get(key, 0) + 1
    return {
        "total": len(candidates),
        "by_status": by_status,
        "by_difficulty": by_difficulty,
        "by_auto": by_auto,
    }


# --- LLM drafters (lazy; never exercised by tests) --------------------------

_DRAFT_SYSTEM = (
    "You write evaluation questions for a retrieval system over GOV.UK guidance. "
    "Given ONE passage, write questions that are answerable ONLY from that passage. "
    "Ground every answer in the passage's own wording — never add outside facts. "
    "Return ONLY a JSON array, each item: "
    '{"question": str, "ground_truth": str, "difficulty": "single_hop"|"multi_hop"}. '
    "Use multi_hop only when answering needs two or more separate facts from the passage."
)

_NEGATIVE_SYSTEM = (
    "You write NEGATIVE evaluation questions for a retrieval system over a bounded "
    "slice of GOV.UK guidance. Write plausible questions a real user might ask on this "
    "topic that the given material genuinely CANNOT answer, so the right behaviour is "
    "refusal. Do not write questions the passages clearly answer. "
    "Return ONLY a JSON array, each item: "
    '{"question": str, "ground_truth": "Not answerable from the corpus."}'
)

#: Appended when earlier batches already produced negatives. Each batch is an
#: independent call with an identical prompt, so without this the model happily
#: writes the same question again — one real run produced two verbatim copies of
#: "What percentage of self-employed people were audited by HMRC last year?"
_AVOID_TEMPLATE = (
    "\n\nThese questions have already been written. Do not repeat them, and do not "
    "write minor variations of them (swapping a country, a year, or a figure is a "
    "repeat):\n{listing}"
)


#: Conditions no amount of retrying will fix, so a run should stop rather than
#: burn 150 sequential calls discovering the same thing 150 times. Matched on
#: the message text because provider SDKs surface these as differently-shaped
#: exceptions. Same list the judge pre-flight uses (src/metrics/judge.py).
_FATAL_MARKERS = (
    "credit balance", "insufficient_quota", "quota",
    "authentication", "invalid api key", "invalid x-api-key",
    "permission", "not_found_error", "model not found",
    # A missing SDK is not a provider problem but it is just as unrecoverable,
    # and it used to slip through: the pre-flight treated "No module named
    # 'anthropic'" as a non-fatal warning and let the run continue, so
    # run 34605958838 made 134 calls that could never have worked.
    "no module named",
)


def is_fatal_provider_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _FATAL_MARKERS)


class AnthropicDrafter:
    drafter_id = "anthropic"

    def __init__(self, model: str, max_tokens: int = 1024):
        self.model = model if model.startswith("claude") else "claude-sonnet-5"
        self.max_tokens = max_tokens

    def _call(self, system: str, user: str) -> str:
        import anthropic  # lazy

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete(self, system: str, user: str) -> str:
        """Public one-shot completion — the seam the candidate evaluator uses.

        Exists so evaluation reuses this class's provider selection, model
        config and pre-flight instead of opening a second route to an LLM.
        """
        return self._call(system, user)

    def preflight(self) -> None:
        """One tiny call, so a dead key costs 2 seconds instead of a traceback.

        Drafting is a long sequential loop of paid calls. Without this, a run
        with no credit dies on call 1 with a raw provider traceback (run
        34508149133 did exactly that), and a run whose credit runs out mid-way
        dies at call N with the same traceback. The judge suite already works
        this way; the drafter did not, which is the asymmetry this closes.
        """
        self._call("Reply with the single word: ok", "ping")

    def draft(self, chunk: Chunk, n: int) -> str:
        return self._call(
            _DRAFT_SYSTEM,
            f"Write {n} question(s) from this passage.\n\n"
            f"Title: {chunk.title}\nPassage:\n{chunk.text}",
        )

    def draft_negatives(self, topic: str, n: int, avoid: Sequence[str] = ()) -> str:
        return self._call(_NEGATIVE_SYSTEM, _negative_prompt(topic, n, avoid))


class OpenAIDrafter:
    drafter_id = "openai"

    def __init__(self, model: str, max_tokens: int = 1024):
        self.model = model or "gpt-4o-mini"
        self.max_tokens = max_tokens

    def _call(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str) -> str:
        """See AnthropicDrafter.complete — same seam, other provider."""
        return self._call(system, user)

    def preflight(self) -> None:
        """See AnthropicDrafter.preflight — same contract, other provider."""
        self._call("Reply with the single word: ok", "ping")

    def draft(self, chunk: Chunk, n: int) -> str:
        return self._call(
            _DRAFT_SYSTEM,
            f"Write {n} question(s) from this passage.\n\n"
            f"Title: {chunk.title}\nPassage:\n{chunk.text}",
        )

    def draft_negatives(self, topic: str, n: int, avoid: Sequence[str] = ()) -> str:
        return self._call(_NEGATIVE_SYSTEM, _negative_prompt(topic, n, avoid))


def build_drafter(config: Config, provider: str | None = None):
    provider = provider or config.generation.provider
    if provider == "anthropic":
        return AnthropicDrafter(config.generation.model)
    if provider == "openai":
        return OpenAIDrafter(config.generation.model)
    raise ValueError(
        f"Unsupported drafting provider: {provider!r} (use openai or anthropic; "
        "`echo` cannot draft questions)"
    )


def _negative_prompt(topic: str, n: int, avoid: Sequence[str] = ()) -> str:
    """User message for a negatives batch, naming what not to repeat."""
    prompt = f"Topic of the corpus: {topic}\nWrite {n} questions."
    if avoid:
        listing = "\n".join(f"- {q}" for q in avoid)
        prompt += _AVOID_TEMPLATE.format(listing=listing)
    return prompt


def normalise_question(question: str) -> str:
    """Key for duplicate detection: case, spacing and trailing punctuation only."""
    return re.sub(r"\s+", " ", question).strip().strip("?.!").casefold()


class PartialDraft(Exception):
    """Raised when drafting stops on an unrecoverable provider error.

    Carries the candidates drafted before the failure so the caller can write
    them out. Losing paid work to an exception is the thing this exists to stop.
    """

    def __init__(self, candidates: list[Candidate], cause: Exception):
        super().__init__(str(cause))
        self.candidates = candidates
        self.cause = cause


def draft_candidates(
    chunks: list[Chunk],
    drafter,
    per_chunk: int,
    negatives: int,
    topic: str,
    seed: Sequence[Candidate] = (),
) -> list[Candidate]:
    """Draft over chunks (+ optional negatives). `drafter` is injected for tests.

    `seed` carries candidates forward from an existing queue, so negatives can be
    redrafted without paying to redraft — and re-review — grounded questions that
    were already fine. Seeded questions also count for duplicate detection, so a
    new negative cannot restate a question the queue already holds.
    """
    # Renumbered so ids stay contiguous even if the seed had gaps.
    out: list[Candidate] = [
        replace(c, candidate_id=f"cand_{i:04d}") for i, c in enumerate(seed)
    ]
    # A fatal provider error mid-loop must not throw away the calls already
    # paid for. Drafting 150 chunks is 150 sequential paid calls; dying at 140
    # and losing all 140 is the failure mode this guards. `partial` is raised to
    # the caller so it can still write the queue, then report honestly.
    for chunk in chunks:
        try:
            out.extend(parse_candidates(drafter.draft(chunk, per_chunk), chunk, len(out)))
        except Exception as exc:  # noqa: BLE001 - provider errors are untyped
            if is_fatal_provider_error(exc):
                raise PartialDraft(out, exc) from exc
            # Anything transient costs this chunk, not the run.
            print(f"  ! chunk {chunk.chunk_id} failed: {exc}", file=sys.stderr)
    # Negatives are requested in batches. A single call for all of them
    # overruns the drafter's max_tokens (25 questions of JSON did exactly that),
    # the array truncates mid-entry, and the whole batch is dropped as malformed
    # — silently, because dropping bad output is the documented behaviour. Small
    # batches keep every response inside the token budget, and a batch that does
    # fail now costs a few candidates instead of all of them.
    #
    # Each batch is an independent call with an identical prompt, so the model
    # repeats itself across batches unless told what it has already written —
    # one run produced two verbatim copies of the same question. Earlier
    # questions are fed back in, and an exact-duplicate check backs that up,
    # because a prompt instruction is guidance and this needs a guarantee.
    seen = {normalise_question(c.question) for c in out}
    # Seeded negatives go into the avoid list too. Without this a top-up spends
    # its whole budget regenerating questions the queue already has, and the
    # duplicate check silently throws them all away.
    asked: list[str] = [c.question for c in out if c.difficulty == "negative"]
    remaining = negatives
    while remaining > 0:
        batch = min(NEGATIVES_PER_CALL, remaining)
        drafted = parse_candidates(
            drafter.draft_negatives(topic, batch, tuple(asked)), None, len(out), negative=True
        )
        kept = 0
        for cand in drafted:
            key = normalise_question(cand.question)
            if key in seen:
                continue
            seen.add(key)
            asked.append(cand.question)
            # Renumber: ids must stay contiguous once duplicates are dropped.
            out.append(replace(cand, candidate_id=f"cand_{len(out):04d}"))
            kept += 1
        if kept < batch:
            print(
                f"WARNING: asked for {batch} negatives, kept {kept} "
                f"({len(drafted) - kept} duplicate(s) dropped).",
                file=sys.stderr,
            )
        remaining -= batch
    return out


#: Chunks shorter than this are navigation/boilerplate ("Log in and file your
#: Self Assessment tax return"), and a question drafted from one is worthless as
#: ground truth — it tests nothing a retriever could get wrong.
MIN_CHUNK_CHARS = 200

#: Negatives per LLM call — small enough that the JSON array cannot truncate.
NEGATIVES_PER_CALL = 8


def select_chunks(
    chunks: list,
    limit: int | None,
    min_chars: int = MIN_CHUNK_CHARS,
    per_page: int = 1,
) -> list:
    """Pick `limit` chunks spread across pages, not the first N alphabetically.

    Chunk ids are `<page-path>#chunk-<n>`, so sorting by id groups every chunk
    of a page together — taking a head slice would draw the whole golden set
    from the handful of alphabetically-first pages. A golden set that only
    covers 4 of 292 pages measures almost nothing. So: stride across pages,
    then round-robin by chunk index. Deterministic (pages and chunks sorted).

    `per_page` is how many chunks to take from each selected page, and it
    exists because the stride has a consequence that is easy to miss. The
    stride picks about `limit` pages, so a budget of `limit` chunks is spent
    almost entirely on ONE chunk from each — depth is never reached. At
    limit=40 over 292 pages the stride selects 37 pages, and the round-robin
    yields 37 chunk-0s and 3 chunk-1s before truncating.

    That is not a hypothetical: it is exactly the shape of the v2 golden set,
    whose 41 answerable questions are 37 `#chunk-0` and 4 `#chunk-1`. A set
    like that tests whether retrieval finds the right *document* and barely
    tests whether it finds the right *passage within* one.

    per_page > 1 selects proportionally fewer pages (`limit / per_page`) and
    takes the first `per_page` chunks of each — trading page coverage for
    depth coverage, which is a real trade and should be a deliberate one.
    per_page=1 reproduces the previous selection exactly.
    """
    by_page: dict[str, list] = {}
    for c in chunks:
        if len(c.text.strip()) < min_chars:
            continue
        by_page.setdefault(c.page_path, []).append(c)
    for page in by_page.values():
        page.sort(key=lambda c: c.chunk_index)

    all_pages = [by_page[k] for k in sorted(by_page)]
    if per_page < 1:
        raise ValueError("per_page must be >= 1")
    if limit is None:
        return _round_robin(all_pages, start=0, stop=None)
    if limit <= 0:
        return []

    wanted_pages = max(1, math.ceil(limit / per_page))
    pages = all_pages
    if wanted_pages < len(all_pages):
        pages = all_pages[:: math.ceil(len(all_pages) / wanted_pages)]

    ordered = _round_robin(pages, start=0, stop=per_page)

    # Short of budget? Go deeper into the pages already selected before
    # widening, so the depth the caller asked for is honoured first.
    if len(ordered) < limit:
        ordered += _round_robin(pages, start=per_page, stop=None)

    # Still short — the selected pages simply do not hold enough chunks. Widen
    # to the rest of the corpus rather than silently under-delivering a budget
    # the caller is paying an LLM for.
    if len(ordered) < limit:
        seen = {id(c) for c in ordered}
        for c in _round_robin(all_pages, start=0, stop=None):
            if id(c) not in seen:
                ordered.append(c)

    return ordered[:limit]


def _round_robin(pages: list[list], start: int, stop: int | None) -> list:
    """Chunk `start` of every page, then chunk `start+1`, ... up to `stop`."""
    depth = max((len(pg) for pg in pages), default=0)
    upper = depth if stop is None else min(depth, stop)
    return [pg[i] for i in range(start, upper) for pg in pages if i < len(pg)]


# --- CLI --------------------------------------------------------------------


def _cmd_draft(args) -> int:
    config = load_config(args.config)
    index_dir = args.index or Path(config.store.path)
    store = load_store(index_dir, config.store.type)
    chunks = select_chunks(list(store.chunks), args.limit, per_page=args.per_page)

    # --merge-queue keeps the grounded questions from an earlier queue and
    # redrafts only the negatives, so a fix to negative drafting does not force
    # a re-review of work that was already good.
    seed: list[Candidate] = []
    if args.merge_queue is not None:
        existing = load_queue(args.merge_queue)
        if not existing:
            raise SystemExit(f"--merge-queue {args.merge_queue} is empty or missing.")
        if args.keep_negatives:
            # Top up: everything survives and the existing negatives seed the
            # duplicate check, so new ones must actually be new.
            seed = list(existing)
            kept_neg = sum(c.difficulty == "negative" for c in seed)
            print(
                f"Merging (top-up): kept all {len(seed)} candidate(s) from "
                f"{args.merge_queue}, including {kept_neg} negative(s); drafting "
                f"{args.negatives} more.",
                file=sys.stderr,
            )
        else:
            seed = [c for c in existing if c.difficulty != "negative"]
            print(
                f"Merging (redraft): kept {len(seed)} grounded candidate(s) from "
                f"{args.merge_queue}, dropped {len(existing) - len(seed)} negative(s) "
                "to redraft.",
                file=sys.stderr,
            )

    calls = len(chunks) + (1 if args.negatives > 0 else 0)
    estimated = calls * _COST_PER_CALL_USD
    if estimated > args.max_usd:
        raise SystemExit(
            f"Drafting {len(chunks)} chunks would cost ~${estimated:.2f}, over the "
            f"${args.max_usd:.2f} cap. Lower --limit or raise --max-usd."
        )

    drafter = build_drafter(config, args.provider)

    # One cheap call before committing to ~len(chunks) paid ones. A dead key or
    # an empty account is then a two-second message instead of a traceback on
    # call 1 (run 34508149133) — and the estimate above is only meaningful if
    # the account can actually pay it.
    preflight = getattr(drafter, "preflight", None)
    if preflight is not None:
        try:
            preflight()
        except Exception as exc:  # noqa: BLE001 - provider errors are untyped
            if is_fatal_provider_error(exc):
                raise SystemExit(
                    f"Draft pre-flight failed against {drafter.drafter_id}: {exc}\n"
                    f"Nothing was drafted and nothing was spent. {args.out} is "
                    "unchanged. Fix the provider account or key and re-run."
                ) from exc
            print(f"Pre-flight warning (continuing): {exc}", file=sys.stderr)

    print(
        f"Drafting from {len(chunks)} chunks x {args.per_chunk} "
        f"(+{args.negatives} negatives) via {drafter.drafter_id}, est ${estimated:.2f}…",
        file=sys.stderr,
    )
    try:
        candidates = draft_candidates(
            chunks, drafter, args.per_chunk, args.negatives, args.topic, seed=seed
        )
    except PartialDraft as partial:
        # Keep what was paid for. Writing the queue and THEN failing means the
        # work is recoverable with --merge-queue instead of re-bought.
        write_queue(args.out, partial.candidates)
        raise SystemExit(
            f"Drafting stopped early: {partial.cause}\n"
            f"Kept {len(partial.candidates)} candidate(s) already drafted -> "
            f"{args.out}. Nothing is lost: fix the provider account, then re-run "
            f"with --merge-queue {args.out} --keep-negatives to top up rather "
            "than redraft."
        ) from partial

    write_queue(args.out, candidates)

    s = summarise(candidates)
    print(
        f"Wrote {s['total']} candidates to {args.out} (all status=pending).\n"
        f"  by difficulty: {s['by_difficulty']}\n\n"
        "NEXT: review every one by hand — set \"status\" to \"approved\" or "
        '"rejected" (fix wording/ground_truth as you go), then run:\n'
        f"  python scripts/build_golden_set.py promote --version <vN>"
    )
    return 0


def _cmd_evaluate(args) -> int:
    """Screen a review queue with the automated evaluator (build step 3).

    Annotates every candidate with a verdict and, unless --annotate-only, sets
    the status the verdict implies. `review` maps to `pending`, which promote()
    already refuses — so nothing uncertain can reach the golden set this way.
    """
    candidates = load_queue(args.queue)
    if not candidates:
        raise SystemExit(f"No review queue at {args.queue} — run `draft` first.")

    config = load_config(args.config)
    completer = build_drafter(config, args.provider)

    # Same reasoning as the drafter: one cheap call before N paid ones.
    preflight = getattr(completer, "preflight", None)
    if preflight is not None:
        try:
            preflight()
        except Exception as exc:  # noqa: BLE001 - provider errors are untyped
            if is_fatal_provider_error(exc):
                raise SystemExit(
                    f"Evaluator pre-flight failed against {completer.drafter_id}: "
                    f"{exc}\nNothing was evaluated and {args.queue} is unchanged."
                ) from exc

    targets = [
        c for c in candidates
        if args.reevaluate or not c.evaluation
    ]
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        print("Every candidate already has a verdict. Use --reevaluate to redo them.")
        return 0

    est = len(targets) * _COST_PER_CALL_USD
    if est > args.max_usd:
        raise SystemExit(
            f"Evaluating {len(targets)} candidates would cost ~${est:.2f}, over the "
            f"${args.max_usd:.2f} cap. Lower --limit or raise --max-usd."
        )
    print(
        f"Evaluating {len(targets)} of {len(candidates)} candidates via "
        f"{completer.drafter_id}, est ${est:.2f}…",
        file=sys.stderr,
    )

    by_id = {c.candidate_id: c for c in candidates}
    position = {c.candidate_id: i for i, c in enumerate(candidates)}
    # Duplicate rejection has to look BACKWARDS only. Comparing each candidate
    # against the whole batch makes two copies of a question reject each other
    # and lose it entirely — found by an end-to-end run, not by unit tests.
    dropped: set[str] = set()
    done = 0
    unreadable = 0
    fatal: Exception | None = None
    try:
        for c in targets:
            peers = preceding_peers(candidates, position[c.candidate_id], dropped)
            try:
                ev = evaluate_one(c, completer, others=peers)
            except FatalEvaluatorError as exc:
                # The evaluator cannot work at all. Stop rather than make the
                # same doomed call once per remaining candidate.
                fatal = exc
                break
            if ev.rule == "unparseable":
                unreadable += 1
            if ev.decision == "reject":
                dropped.add(c.candidate_id)
            status = (
                c.status if args.annotate_only
                else apply_decision(c.status, ev.decision,
                                    respect_human=not args.override_human)
            )
            by_id[c.candidate_id] = replace(
                c, evaluation=ev.to_dict(), status=status
            )
            done += 1
    finally:
        # Write whatever was evaluated, even on interruption: these are paid
        # calls and losing them is the mistake PR #27 was about.
        write_queue(args.out or args.queue, [by_id[c.candidate_id] for c in candidates])

    final = load_queue(args.out or args.queue)
    s = summarise(final)
    print(
        f"Evaluated {done} candidate(s) -> {args.out or args.queue}\n"
        f"  automated: {s['by_auto']}\n"
        f"  status:    {s['by_status']}"
    )

    if fatal is not None:
        raise SystemExit(
            f"\nEvaluator stopped: {fatal}\n"
            f"{done} of {len(targets)} candidate(s) were screened and saved; the "
            "rest are untouched. Re-run once fixed and only the unscreened ones "
            "are charged for."
        )

    # A screen that screened nothing must not be a green build. Run 34605958838
    # turned every one of 134 candidates into `review` because the anthropic SDK
    # was missing, and still exited 0 — the verdicts were individually correct
    # (a failure never becomes an approval) and the run was still worthless.
    if done and unreadable == done:
        raise SystemExit(
            f"\nEvery one of {done} evaluations was unreadable, so nothing was "
            "actually screened. That is an evaluator or environment fault, not a "
            "verdict on these candidates — check the reasons above. Statuses were "
            "left as they were."
        )
    if unreadable:
        print(
            f"\nWARNING: {unreadable} of {done} evaluations were unreadable and "
            "defaulted to human review. Check the reasons before trusting the split."
        )

    print(
        "\nNothing uncertain was promoted: `review` leaves status=pending, and "
        "promote only takes approved.\n"
        "NEXT: open the review page (or `status`) and work the review pile."
    )
    return 0


def _cmd_status(args) -> int:
    candidates = load_queue(args.queue)
    if not candidates:
        print(f"No review queue at {args.queue} — run `draft` first.")
        return 0
    s = summarise(candidates)
    print(f"{args.queue}: {s['total']} candidates")
    print(f"  status:     {s['by_status']}")
    print(f"  difficulty: {s['by_difficulty']}")
    negatives = s["by_difficulty"].get("negative", 0)
    if negatives < 20:
        print(f"  note: {negatives} negatives — BUILD.md targets 20-30.")
    return 0


def _cmd_promote(args) -> int:
    candidates = load_queue(args.queue)
    if not candidates:
        raise SystemExit(f"No review queue at {args.queue} — nothing to promote.")

    golden_path = Path(args.golden)
    existing_rows: list[dict] = []
    if golden_path.exists():
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_rows.append(json.loads(line))

    new, skipped = promote(candidates, existing_rows, args.version)
    if not new:
        print(f"Nothing to promote (approved: 0 new). Skipped {len(skipped)}.")
        return 0

    if args.dry_run:
        print(f"[dry-run] would append {len(new)} records to {golden_path}:")
        for r in new[:5]:
            print(f"  {r['id']}  {r['question'][:80]}")
        return 0

    with golden_path.open("a", encoding="utf-8") as fh:
        for row in new:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # Fail loudly if we just wrote something the loader would reject.
    records = load_golden(golden_path)
    print(
        f"Appended {len(new)} records to {golden_path} (skipped {len(skipped)}). "
        f"Golden set now {len(records)} records.\n"
        f"Remember: bump `dataset.version` in configs/eval_config.yaml to "
        f"{args.version} and regenerate results/baseline.json."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft + review the golden set.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft", help="LLM-draft candidates into the review queue")
    d.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    d.add_argument("--index", type=Path, default=None)
    d.add_argument("--out", type=Path, default=DEFAULT_QUEUE)
    d.add_argument("--per-chunk", type=int, default=2)
    d.add_argument(
        "--per-page", type=int, default=1,
        help=(
            "chunks to draft from each selected page (default 1). Round-robin "
            "exhausts chunk-0 of every page before reaching chunk-1, so at "
            "per-page=1 a limit below the page count yields only #chunk-0 — "
            "which is how the v2 set ended up 37/41 chunk-0. Raise this to "
            "buy depth coverage at the cost of page coverage."
        ),
    )
    d.add_argument(
        "--limit", type=int, default=None,
        help="cap chunks, spread across pages (cost discipline)",
    )
    d.add_argument("--negatives", type=int, default=25)
    d.add_argument("--topic", type=str, default="Self Assessment and self-employment tax guidance")
    d.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    d.add_argument(
        "--merge-queue", type=Path, default=None,
        help="keep grounded candidates from this queue and redraft only the negatives",
    )
    d.add_argument(
        "--keep-negatives", action="store_true",
        help="with --merge-queue: keep the existing negatives too and top up, "
             "rather than replacing them",
    )
    d.add_argument("--max-usd", type=float, default=5.0)
    d.set_defaults(func=_cmd_draft)

    e = sub.add_parser(
        "evaluate",
        help="automated pre-screen of a review queue (approve / reject / review)",
    )
    e.add_argument("--queue", type=Path, default=Path("data/golden/review_queue.jsonl"))
    e.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    e.add_argument(
        "--out", type=Path, default=None,
        help="write here instead of in place (the queue is rewritten by default)",
    )
    e.add_argument(
        "--provider", default=None,
        help="override generation.provider from the config (openai | anthropic)",
    )
    e.add_argument("--limit", type=int, default=0, help="evaluate at most N candidates")
    e.add_argument("--max-usd", type=float, default=4.00)
    e.add_argument(
        "--reevaluate", action="store_true",
        help="re-score candidates that already carry a verdict",
    )
    e.add_argument(
        "--annotate-only", action="store_true",
        help="record verdicts without changing any candidate's status",
    )
    e.add_argument(
        "--override-human", action="store_true",
        help=(
            "let a verdict overwrite a status a person already set. Off by "
            "default so re-running never discards human review."
        ),
    )
    e.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("status", help="summarise the review queue")
    s.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    s.set_defaults(func=_cmd_status)

    p = sub.add_parser("promote", help="append approved candidates to the golden set")
    p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--version", type=str, required=True, help="e.g. v2")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_promote)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
