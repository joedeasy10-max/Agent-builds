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
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Run as a plain script: put the repo root on the path so `src` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# --- pure helpers (unit-tested; no LLM, no network) -------------------------


def strip_fences(text: str) -> str:
    """LLMs like to wrap JSON in ```json fences. Remove them."""
    return _FENCE_RE.sub("", text.strip())


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
        return []
    if not isinstance(rows, list):
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
    by_status = {s: 0 for s in sorted(STATUSES)}
    by_difficulty = {d: 0 for d in sorted(DIFFICULTIES)}
    for c in candidates:
        by_status[c.status] = by_status.get(c.status, 0) + 1
        by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
    return {"total": len(candidates), "by_status": by_status, "by_difficulty": by_difficulty}


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

    def draft(self, chunk: Chunk, n: int) -> str:
        return self._call(
            _DRAFT_SYSTEM,
            f"Write {n} question(s) from this passage.\n\n"
            f"Title: {chunk.title}\nPassage:\n{chunk.text}",
        )

    def draft_negatives(self, topic: str, n: int) -> str:
        return self._call(_NEGATIVE_SYSTEM, f"Topic of the corpus: {topic}\nWrite {n} questions.")


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

    def draft(self, chunk: Chunk, n: int) -> str:
        return self._call(
            _DRAFT_SYSTEM,
            f"Write {n} question(s) from this passage.\n\n"
            f"Title: {chunk.title}\nPassage:\n{chunk.text}",
        )

    def draft_negatives(self, topic: str, n: int) -> str:
        return self._call(_NEGATIVE_SYSTEM, f"Topic of the corpus: {topic}\nWrite {n} questions.")


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


def draft_candidates(
    chunks: list[Chunk], drafter, per_chunk: int, negatives: int, topic: str
) -> list[Candidate]:
    """Draft over chunks (+ optional negatives). `drafter` is injected for tests."""
    out: list[Candidate] = []
    for chunk in chunks:
        out.extend(parse_candidates(drafter.draft(chunk, per_chunk), chunk, len(out)))
    if negatives > 0:
        out.extend(
            parse_candidates(
                drafter.draft_negatives(topic, negatives), None, len(out), negative=True
            )
        )
    return out


# --- CLI --------------------------------------------------------------------


def _cmd_draft(args) -> int:
    config = load_config(args.config)
    index_dir = args.index or Path(config.store.path)
    store = load_store(index_dir, config.store.type)
    chunks = sorted(store.chunks, key=lambda c: c.chunk_id)
    if args.limit is not None:
        chunks = chunks[: args.limit]

    calls = len(chunks) + (1 if args.negatives > 0 else 0)
    estimated = calls * _COST_PER_CALL_USD
    if estimated > args.max_usd:
        raise SystemExit(
            f"Drafting {len(chunks)} chunks would cost ~${estimated:.2f}, over the "
            f"${args.max_usd:.2f} cap. Lower --limit or raise --max-usd."
        )

    drafter = build_drafter(config, args.provider)
    print(
        f"Drafting from {len(chunks)} chunks x {args.per_chunk} "
        f"(+{args.negatives} negatives) via {drafter.drafter_id}, est ${estimated:.2f}…",
        file=sys.stderr,
    )
    candidates = draft_candidates(chunks, drafter, args.per_chunk, args.negatives, args.topic)
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
    d.add_argument("--limit", type=int, default=None, help="cap chunks (cost discipline)")
    d.add_argument("--negatives", type=int, default=25)
    d.add_argument("--topic", type=str, default="Self Assessment and self-employment tax guidance")
    d.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    d.add_argument("--max-usd", type=float, default=5.0)
    d.set_defaults(func=_cmd_draft)

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
