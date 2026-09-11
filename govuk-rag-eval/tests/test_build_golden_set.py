"""Golden-set drafting tool (build step 3), offline — no LLM, no network.

The LLM call is injected, so these tests cover the parts that must not go wrong:
parsing untrusted model output, the review-queue round trip, and `promote` —
which is the only thing that writes to the golden set and therefore must never
promote unreviewed work, duplicate a question, or renumber existing ids.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.chunk import Chunk

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_golden_set", ROOT / "scripts" / "build_golden_set.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # so dataclass annotations resolve
    spec.loader.exec_module(m)
    return m


bgs = _load()


def _chunk(cid="gov-uk/p#chunk-0", text="Register by 5 October."):
    return Chunk(
        chunk_id=cid, page_path="gov-uk/p", page_url="u", title="T",
        text=text, content_hash="abc123", chunk_index=0,
    )


# ---- parsing untrusted LLM output ------------------------------------------

def test_parses_plain_json_array():
    raw = json.dumps([
        {"question": "When do I register?", "ground_truth": "By 5 October.", "difficulty": "single_hop"},
    ])
    got = bgs.parse_candidates(raw, _chunk(), 0)
    assert len(got) == 1
    c = got[0]
    assert c.candidate_id == "cand_0000"
    assert c.status == "pending"                       # never auto-approved
    assert c.source_ids == ("gov-uk/p#chunk-0",)
    assert c.source_excerpt == "Register by 5 October."  # review needs no lookup


def test_strips_code_fences():
    raw = '```json\n[{"question": "Q?", "ground_truth": "A"}]\n```'
    assert len(bgs.parse_candidates(raw, _chunk(), 0)) == 1


def test_malformed_json_yields_nothing():
    assert bgs.parse_candidates("not json at all", _chunk(), 0) == []
    assert bgs.parse_candidates('{"not": "a list"}', _chunk(), 0) == []


def test_drops_entries_without_a_question():
    raw = json.dumps([{"ground_truth": "A"}, {"question": "  ", "ground_truth": "B"},
                      {"question": "Real?", "ground_truth": "C"}])
    got = bgs.parse_candidates(raw, _chunk(), 0)
    assert [c.question for c in got] == ["Real?"]


def test_unknown_difficulty_falls_back_to_single_hop():
    raw = json.dumps([{"question": "Q?", "difficulty": "wildly_invented"}])
    assert bgs.parse_candidates(raw, _chunk(), 0)[0].difficulty == "single_hop"


def test_negatives_get_no_source_ids():
    raw = json.dumps([{"question": "Unanswerable?", "ground_truth": "N/A"}])
    c = bgs.parse_candidates(raw, None, 0, negative=True)[0]
    assert c.difficulty == "negative"
    assert c.source_ids == ()


def test_candidate_ids_continue_from_start_index():
    raw = json.dumps([{"question": "A?"}, {"question": "B?"}])
    got = bgs.parse_candidates(raw, _chunk(), 7)
    assert [c.candidate_id for c in got] == ["cand_0007", "cand_0008"]


# ---- queue round trip ------------------------------------------------------

def test_queue_roundtrip(tmp_path):
    cands = bgs.parse_candidates(json.dumps([{"question": "Q?", "ground_truth": "A"}]), _chunk(), 0)
    q = tmp_path / "review_queue.jsonl"
    bgs.write_queue(q, cands)
    back = bgs.load_queue(q)
    assert back == cands


def test_load_queue_rejects_bad_status(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"candidate_id": "cand_0000", "status": "maybe", "question": "Q?"}) + "\n")
    with pytest.raises(ValueError, match="status"):
        bgs.load_queue(q)


def test_load_queue_missing_file_is_empty(tmp_path):
    assert bgs.load_queue(tmp_path / "nope.jsonl") == []


# ---- draft_candidates with an injected fake drafter ------------------------

class _FakeDrafter:
    drafter_id = "fake"

    def draft(self, chunk, n):
        return json.dumps([{"question": f"About {chunk.chunk_id}?", "ground_truth": "A"}])

    def draft_negatives(self, topic, n, avoid=()):
        # Distinct per call, so the dedup check does not collapse the batch.
        start = len(avoid)
        return json.dumps(
            [{"question": f"Unanswerable {start + i} about {topic}?"} for i in range(n)]
        )


def test_draft_candidates_covers_chunks_and_negatives():
    chunks = [_chunk("gov-uk/a#chunk-0"), _chunk("gov-uk/b#chunk-0")]
    got = bgs.draft_candidates(chunks, _FakeDrafter(), per_chunk=1, negatives=2, topic="tax")
    assert len(got) == 4                                  # 2 chunk-grounded + 2 negatives
    assert sum(c.difficulty == "negative" for c in got) == 2
    assert all(c.status == "pending" for c in got)        # nothing auto-approved
    assert len({c.candidate_id for c in got}) == 4        # ids unique across batches


# ---- promote: the only path that writes to the golden set ------------------

def _cand(cid, question, status="approved", difficulty="single_hop", source_ids=("gov-uk/p#chunk-0",)):
    return bgs.Candidate(
        candidate_id=cid, status=status, question=question, ground_truth="A",
        source_ids=tuple(source_ids), difficulty=difficulty, source_excerpt="x",
    )


def test_promote_only_takes_approved():
    cands = [_cand("c1", "Approved?"), _cand("c2", "Pending?", status="pending"),
             _cand("c3", "Rejected?", status="rejected")]
    new, _ = bgs.promote(cands, [], "v2")
    assert [r["question"] for r in new] == ["Approved?"]


def test_promote_assigns_ids_continuing_from_existing():
    existing = [{"id": "q_0001", "question": "Old one"}, {"id": "q_0007", "question": "Another"}]
    new, _ = bgs.promote([_cand("c1", "New?")], existing, "v2")
    assert new[0]["id"] == "q_0008"      # continues past the highest, never renumbers
    assert new[0]["added_in"] == "v2"


def test_promote_skips_duplicate_questions():
    existing = [{"id": "q_0001", "question": "Already asked?"}]
    new, skipped = bgs.promote([_cand("c1", "  already asked?  ")], existing, "v2")
    assert new == [] and skipped == ["c1"]


def test_promote_skips_answerable_without_sources():
    new, skipped = bgs.promote([_cand("c1", "Q?", source_ids=())], [], "v2")
    assert new == [] and skipped == ["c1"]


def test_promote_skips_negative_that_has_sources():
    new, skipped = bgs.promote(
        [_cand("c1", "Q?", difficulty="negative", source_ids=("gov-uk/p#chunk-0",))], [], "v2"
    )
    assert new == [] and skipped == ["c1"]


def test_promote_accepts_valid_negative():
    new, _ = bgs.promote([_cand("c1", "Q?", difficulty="negative", source_ids=())], [], "v2")
    assert new[0]["difficulty"] == "negative" and new[0]["source_ids"] == []


def test_promote_rejects_bad_version():
    with pytest.raises(ValueError, match="version"):
        bgs.promote([_cand("c1", "Q?")], [], "version-two")


def test_promoted_records_load_as_valid_golden(tmp_path):
    """What promote writes must satisfy the real golden-set loader."""
    from src.golden import load_golden

    new, _ = bgs.promote(
        [_cand("c1", "Answerable?"), _cand("c2", "Unanswerable?", difficulty="negative", source_ids=())],
        [], "v2",
    )
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in new) + "\n")
    records = load_golden(p)
    assert [r.id for r in records] == ["q_0001", "q_0002"]
    assert records[1].is_negative


# ---- summarise -------------------------------------------------------------

def test_summarise_counts():
    cands = [_cand("c1", "A?"), _cand("c2", "B?", status="pending"),
             _cand("c3", "C?", difficulty="negative", source_ids=())]
    s = bgs.summarise(cands)
    assert s["total"] == 3
    assert s["by_status"]["approved"] == 2 and s["by_status"]["pending"] == 1
    assert s["by_difficulty"]["negative"] == 1


# ---- drafter selection (constructed, never called) -------------------------

def test_build_drafter_selects_provider():
    from src.config import from_dict

    anth = bgs.build_drafter(from_dict({"generation": {"provider": "anthropic"}}))
    openai = bgs.build_drafter(from_dict({"generation": {"provider": "openai"}}))
    assert anth.drafter_id == "anthropic" and anth.model == "claude-sonnet-5"
    assert openai.drafter_id == "openai" and openai.model == "gpt-4o-mini"


def test_build_drafter_rejects_echo():
    from src.config import from_dict

    with pytest.raises(ValueError, match="cannot draft"):
        bgs.build_drafter(from_dict({"generation": {"provider": "echo"}}))


# ---- chunk selection: coverage across pages, not a head slice --------------

_SUBSTANTIAL = "x" * (bgs.MIN_CHUNK_CHARS + 50)


def _c(page, idx, text=None):
    return Chunk(
        chunk_id=f"{page}#chunk-{idx}", page_path=page, page_url="u", title="T",
        text=_SUBSTANTIAL if text is None else text, content_hash="h", chunk_index=idx,
    )


def test_select_chunks_spreads_across_pages():
    """A head slice of id-sorted chunks would draw only from page 'a'."""
    chunks = [_c("a", i) for i in range(5)] + [_c("b", i) for i in range(5)] \
        + [_c("c", i) for i in range(5)]
    got = bgs.select_chunks(chunks, 3)
    assert [c.page_path for c in got] == ["a", "b", "c"]      # one per page first
    assert all(c.chunk_index == 0 for c in got)


def test_select_chunks_second_pass_takes_deeper_chunks():
    chunks = [_c("a", i) for i in range(2)] + [_c("b", i) for i in range(2)]
    got = bgs.select_chunks(chunks, 4)
    assert [(c.page_path, c.chunk_index) for c in got] == [
        ("a", 0), ("b", 0), ("a", 1), ("b", 1),
    ]


def test_select_chunks_handles_uneven_pages():
    chunks = [_c("a", 0), _c("b", 0), _c("b", 1), _c("b", 2)]
    got = bgs.select_chunks(chunks, None)
    assert [(c.page_path, c.chunk_index) for c in got] == [
        ("a", 0), ("b", 0), ("b", 1), ("b", 2),
    ]


def test_select_chunks_is_deterministic():
    chunks = [_c(p, i) for p in ("c", "a", "b") for i in range(3)]
    assert bgs.select_chunks(chunks, 5) == bgs.select_chunks(list(reversed(chunks)), 5)


def test_select_chunks_no_limit_returns_everything():
    chunks = [_c("a", 0), _c("b", 0)]
    assert len(bgs.select_chunks(chunks, None)) == 2


def test_select_chunks_skips_boilerplate():
    """One-line nav chunks make worthless ground truth, so they never draft."""
    chunks = [_c("a", 0, "Log in and file your Self Assessment tax return"), _c("b", 0)]
    got = bgs.select_chunks(chunks, None)
    assert [c.page_path for c in got] == ["b"]


def test_select_chunks_strides_when_budget_is_smaller_than_the_corpus():
    """A budget of 3 over 9 pages must span the corpus, not take the first 3.

    Round-robin alone walks pages alphabetically, so a small budget sampled only
    the alphabetically-first pages — which is how a real draft came back
    dominated by one section of GOV.UK.
    """
    chunks = [_c(f"p{i:02d}", 0) for i in range(9)]
    got = bgs.select_chunks(chunks, 3)
    assert [c.page_path for c in got] == ["p00", "p03", "p06"]


def test_select_chunks_limit_at_or_above_page_count_does_not_stride():
    chunks = [_c(f"p{i}", 0) for i in range(3)]
    assert [c.page_path for c in bgs.select_chunks(chunks, 3)] == ["p0", "p1", "p2"]


def test_draft_candidates_batches_negatives(monkeypatch):
    """25 negatives in one call truncated at max_tokens and silently yielded 0."""
    calls = []

    class _Counting(_FakeDrafter):
        def draft_negatives(self, topic, n, avoid=()):
            calls.append(n)
            start = len(avoid)
            return json.dumps([{"question": f"Q{start + i}?"} for i in range(n)])

    got = bgs.draft_candidates([], _Counting(), per_chunk=1, negatives=25, topic="tax")
    assert calls == [8, 8, 8, 1]                      # batched, never 25 at once
    assert all(n <= bgs.NEGATIVES_PER_CALL for n in calls)
    assert sum(c.difficulty == "negative" for c in got) == 25


def test_parse_candidates_warns_when_it_drops_everything(capsys):
    """Silent drops are how a run reported success having produced nothing."""
    bgs.parse_candidates('[{"question": "truncated...', _chunk(), 0)
    assert "WARNING" in capsys.readouterr().err


def test_parse_candidates_stays_quiet_on_empty_output(capsys):
    bgs.parse_candidates("   ", _chunk(), 0)
    assert capsys.readouterr().err == ""


# ---- negatives must not repeat across batches ------------------------------

class _RepeatingDrafter(_FakeDrafter):
    """Worst case: every batch returns exactly the same questions."""

    def __init__(self):
        self.seen_avoid = []

    def draft_negatives(self, topic, n, avoid=()):
        self.seen_avoid.append(tuple(avoid))
        return json.dumps([{"question": f"Fixed question {i}?"} for i in range(n)])


def test_duplicate_negatives_are_dropped_across_batches():
    """A real run emitted the same negative twice; the prompt alone can't prevent it."""
    d = _RepeatingDrafter()
    got = bgs.draft_candidates([], d, per_chunk=1, negatives=24, topic="tax")
    questions = [c.question for c in got]
    assert len(questions) == len(set(questions))          # no repeats survive
    assert len(questions) == bgs.NEGATIVES_PER_CALL       # only the first batch is new


def test_later_batches_are_told_what_was_already_asked():
    d = _RepeatingDrafter()
    bgs.draft_candidates([], d, per_chunk=1, negatives=24, topic="tax")
    assert d.seen_avoid[0] == ()                          # nothing to avoid yet
    assert len(d.seen_avoid[1]) == bgs.NEGATIVES_PER_CALL  # batch 2 sees batch 1


def test_candidate_ids_stay_contiguous_after_duplicates_are_dropped():
    got = bgs.draft_candidates([], _RepeatingDrafter(), per_chunk=1, negatives=24, topic="tax")
    assert [c.candidate_id for c in got] == [f"cand_{i:04d}" for i in range(len(got))]


def test_normalise_question_ignores_case_spacing_and_punctuation():
    assert bgs.normalise_question("  What IS   this? ") == bgs.normalise_question("what is this")
    assert bgs.normalise_question("A?") != bgs.normalise_question("B?")


# ---- seeding: redraft negatives without redrafting grounded questions ------

def test_seed_is_carried_through_and_renumbered():
    seed = [_cand("old_a", "Grounded one?"), _cand("old_b", "Grounded two?")]
    got = bgs.draft_candidates([], _FakeDrafter(), per_chunk=1, negatives=2, topic="tax", seed=seed)
    assert [c.question for c in got[:2]] == ["Grounded one?", "Grounded two?"]
    assert [c.candidate_id for c in got] == [f"cand_{i:04d}" for i in range(len(got))]
    assert sum(c.difficulty == "negative" for c in got) == 2


def test_new_negatives_cannot_duplicate_a_seeded_question():
    """The seed counts for dedup, or a redraft could restate a kept question."""

    class _EchoesSeed(_FakeDrafter):
        def draft_negatives(self, topic, n, avoid=()):
            return json.dumps([{"question": "Grounded one?"} for _ in range(n)])

    seed = [_cand("old_a", "Grounded one?")]
    got = bgs.draft_candidates([], _EchoesSeed(), per_chunk=1, negatives=4, topic="tax", seed=seed)
    assert len(got) == 1                                  # nothing new survived
    assert got[0].question == "Grounded one?"


def test_select_chunks_limit_zero_selects_nothing():
    """`--limit 0` is how a negatives-only redraft skips chunk drafting."""
    assert bgs.select_chunks([_c("a", 0), _c("b", 0)], 0) == []


def test_topup_tells_the_drafter_about_existing_negatives():
    """Otherwise a top-up regenerates what the queue already has and dedup bins it."""
    d = _RepeatingDrafter()
    seed = [
        _cand("old_a", "Grounded?"),
        _cand("old_n", "Existing negative?", difficulty="negative", source_ids=()),
    ]
    bgs.draft_candidates([], d, per_chunk=1, negatives=8, topic="tax", seed=seed)
    assert d.seen_avoid[0] == ("Existing negative?",)   # not the grounded one


def test_topup_keeps_seeded_negatives_and_adds_new_ones():
    seed = [_cand("old_n", "Existing negative?", difficulty="negative", source_ids=())]
    got = bgs.draft_candidates([], _FakeDrafter(), per_chunk=1, negatives=3, topic="tax", seed=seed)
    negatives = [c.question for c in got if c.difficulty == "negative"]
    assert "Existing negative?" in negatives      # the old one survives
    assert len(negatives) == 4                    # 1 kept + 3 new


def _fake_chunks(n_pages: int, per_page: int):
    """Synthetic chunks: n_pages pages, each with `per_page` chunks."""
    from dataclasses import dataclass

    @dataclass
    class C:
        chunk_id: str
        page_path: str
        chunk_index: int
        text: str

    return [
        C(f"page-{p:03d}#chunk-{i}", f"page-{p:03d}", i, "x" * 400)
        for p in range(n_pages)
        for i in range(per_page)
    ]


def test_per_page_1_reproduces_the_v2_selection_shape():
    """Documents the failure mode rather than asserting it away.

    The stride selects about `limit` pages, so a budget of `limit` chunks buys
    roughly one chunk from each and never reaches depth. At limit=40 over 292
    pages that is 37 pages selected -> 37 chunk-0 and 3 chunk-1, which is
    within one question of the real v2 golden set (37 chunk-0, 4 chunk-1).
    """
    chunks = bgs.select_chunks(_fake_chunks(292, 8), limit=40, per_page=1)
    assert len(chunks) == 40
    from collections import Counter
    depths = Counter(c.chunk_index for c in chunks)
    assert depths[0] == 37 and depths[1] == 3, depths
    # It does spread across the corpus — that part works as intended.
    pages = {c.page_path for c in chunks}
    assert len(pages) == 37
    # Spread across the whole corpus, not the alphabetically-first pages.
    # (Checked on the page set, not the last element: chunk-1s come after all
    # chunk-0s, so the final item is a low-numbered page by construction.)
    assert max(pages) > "page-250"


def test_per_page_above_1_buys_depth_coverage():
    chunks = bgs.select_chunks(_fake_chunks(292, 8), limit=120, per_page=3)
    assert len(chunks) == 120
    from collections import Counter
    depths = Counter(c.chunk_index for c in chunks)
    # The point of the option: chunk-0 stops dominating.
    assert depths[0] / len(chunks) < 0.4, depths
    assert {0, 1, 2} <= set(depths)
    # Depth is bought with page coverage: ~37 pages deep, not 120 pages shallow.
    counts = Counter(c.page_path for c in chunks)
    assert 30 <= len(counts) <= 45
    assert min(counts.values()) >= 3, "every selected page contributes its depth"


def test_widens_beyond_selected_pages_rather_than_under_delivering():
    """A budget paid for in LLM calls must not quietly come back short."""
    # 10 pages x 2 chunks = 20 available; per_page=5 would strand it at 4 pages.
    chunks = bgs.select_chunks(_fake_chunks(10, 2), limit=20, per_page=5)
    assert len(chunks) == 20, "should widen to the rest of the corpus"
    assert len({c.chunk_id for c in chunks}) == 20, "no duplicates when widening"


def test_cannot_return_more_than_the_corpus_holds():
    chunks = bgs.select_chunks(_fake_chunks(3, 2), limit=99, per_page=4)
    assert len(chunks) == 6
    assert len({c.chunk_id for c in chunks}) == 6


def test_per_page_1_is_unchanged_from_the_old_behaviour():
    """per_page defaults to 1, so existing callers get identical selection."""
    chunks = _fake_chunks(50, 4)
    assert [c.chunk_id for c in bgs.select_chunks(chunks, limit=25)] == [
        c.chunk_id for c in bgs.select_chunks(chunks, limit=25, per_page=1)
    ]


def test_per_page_rejects_nonsense():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        bgs.select_chunks(_fake_chunks(5, 2), limit=4, per_page=0)


# --- fail-fast + don't-lose-paid-work (run 34508149133) ---------------------

_CREDIT_ERROR = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
)


def test_recognises_the_errors_that_retrying_cannot_fix():
    assert bgs.is_fatal_provider_error(Exception(_CREDIT_ERROR))
    assert bgs.is_fatal_provider_error(Exception("invalid x-api-key"))
    assert bgs.is_fatal_provider_error(Exception("insufficient_quota"))
    assert bgs.is_fatal_provider_error(Exception("model not found"))
    # Transient things must NOT abort a 150-call run.
    assert not bgs.is_fatal_provider_error(Exception("overloaded_error"))
    assert not bgs.is_fatal_provider_error(Exception("Connection reset by peer"))
    assert not bgs.is_fatal_provider_error(Exception("rate_limit_error: slow down"))


class _DrafterDyingAfter:
    """Drafts fine for `n` chunks, then raises the credit error for good."""

    drafter_id = "fake"

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def draft(self, chunk, per_chunk):
        self.calls += 1
        if self.calls > self.n:
            raise Exception(_CREDIT_ERROR)
        return json.dumps(
            [{"question": f"Q{self.calls}?", "ground_truth": "GT", "difficulty": "single_hop"}]
        )

    def draft_negatives(self, topic, n, avoid=()):
        return "[]"


def test_fatal_error_midway_preserves_the_calls_already_paid_for():
    """The failure this guards: dying at call 140 of 150 and losing all 140."""
    chunks = [_chunk(cid=f"gov-uk/p{i}#chunk-0", text="Register by 5 October." * 12)
              for i in range(10)]
    drafter = _DrafterDyingAfter(6)
    with pytest.raises(bgs.PartialDraft) as caught:
        bgs.draft_candidates(chunks, drafter, per_chunk=1, negatives=0, topic="t")
    assert len(caught.value.candidates) == 6, "must carry the 6 successful drafts"
    assert "credit balance" in str(caught.value)


def test_transient_error_costs_one_chunk_not_the_run(capsys):
    class _Flaky:
        drafter_id = "fake"

        def __init__(self):
            self.calls = 0

        def draft(self, chunk, per_chunk):
            self.calls += 1
            if self.calls == 2:
                raise Exception("overloaded_error: try again")
            return json.dumps(
                [{"question": f"Q{self.calls}?", "ground_truth": "GT",
                  "difficulty": "single_hop"}]
            )

        def draft_negatives(self, topic, n, avoid=()):
            return "[]"

    chunks = [_chunk(cid=f"gov-uk/p{i}#chunk-0", text="Register by 5 October." * 12)
              for i in range(4)]
    out = bgs.draft_candidates(chunks, _Flaky(), per_chunk=1, negatives=0, topic="t")
    assert len(out) == 3, "3 of 4 chunks drafted; the run continued"
    assert "overloaded_error" in capsys.readouterr().err


def test_drafters_expose_a_preflight():
    """Both providers must offer it, or the fail-fast path silently no-ops."""
    for cls in (bgs.AnthropicDrafter, bgs.OpenAIDrafter):
        assert callable(getattr(cls("", 16), "preflight", None)), cls.__name__


def test_preflight_makes_exactly_one_cheap_call():
    calls = []
    d = bgs.AnthropicDrafter("claude-sonnet-5")
    d._call = lambda system, user: calls.append((system, user)) or "ok"
    d.preflight()
    assert len(calls) == 1


# --- evaluate subcommand: queue round-trip + promotion safety ----------------


def test_evaluation_round_trips_through_the_queue(tmp_path):
    q = tmp_path / "queue.jsonl"
    c = bgs.Candidate(
        candidate_id="cand_0000", status="pending", question="Q?",
        ground_truth="GT", source_ids=("gov-uk/p#chunk-0",),
        difficulty="single_hop", source_excerpt="passage",
        evaluation={"decision": "review", "confidence": 0.4, "rule": "clean_but_unsure"},
    )
    bgs.write_queue(q, [c])
    back = bgs.load_queue(q)[0]
    assert back.evaluation["decision"] == "review"
    assert back.evaluation["rule"] == "clean_but_unsure"
    # A candidate written before this feature existed still loads.
    q.write_text('{"candidate_id":"cand_0001","status":"pending","question":"Q?",'
                 '"ground_truth":"GT","source_ids":["gov-uk/p#chunk-0"],'
                 '"difficulty":"single_hop","source_excerpt":"p","notes":""}\n')
    assert bgs.load_queue(q)[0].evaluation is None


def test_summarise_reports_the_automated_split():
    def mk(cid, status, decision):
        return bgs.Candidate(
            candidate_id=cid, status=status, question=cid, ground_truth="GT",
            source_ids=("gov-uk/p#chunk-0",), difficulty="single_hop",
            source_excerpt="p",
            evaluation={"decision": decision} if decision else None,
        )
    s = bgs.summarise([
        mk("a", "approved", "approve"), mk("b", "rejected", "reject"),
        mk("c", "pending", "review"), mk("d", "pending", None),
    ])
    assert s["by_auto"] == {"unevaluated": 1, "approve": 1, "reject": 1, "review": 1}
    assert s["by_status"]["approved"] == 1


def test_review_verdict_cannot_promote_itself(tmp_path):
    """The safety property: `review` -> pending, and promote ignores pending."""
    cands = [
        bgs.Candidate(candidate_id="cand_0000", status="pending", question="Uncertain?",
                      ground_truth="GT", source_ids=("gov-uk/p#chunk-0",),
                      difficulty="single_hop", source_excerpt="p",
                      evaluation={"decision": "review", "confidence": 0.3}),
        bgs.Candidate(candidate_id="cand_0001", status="rejected", question="Bad?",
                      ground_truth="GT", source_ids=("gov-uk/p#chunk-0",),
                      difficulty="single_hop", source_excerpt="p",
                      evaluation={"decision": "reject", "confidence": 0.95}),
        bgs.Candidate(candidate_id="cand_0002", status="approved", question="Good?",
                      ground_truth="GT", source_ids=("gov-uk/p#chunk-0",),
                      difficulty="single_hop", source_excerpt="p",
                      evaluation={"decision": "approve", "confidence": 0.95}),
    ]
    new, skipped = bgs.promote(cands, [], "v3")
    assert [r["question"] for r in new] == ["Good?"], "only the approved one"
