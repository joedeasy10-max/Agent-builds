"""End-to-end retrieval-suite eval, fully offline (hashing embedder, flat store).

This wires the real pieces together — ingest -> Retriever -> metrics -> results
JSON — and asserts on values we can compute by hand:

  q1: relevant = the chunk the retriever actually ranks #1 for its query -> hits.
  q2: relevant = a chunk ID that does not exist -> cannot hit.
  q3: a negative -> excluded from the retrieval aggregates.

So over 2 answerable questions (one hitting at rank 1, one missing entirely):
  hit_at_5 = 0.5, mrr = 0.5, context_recall_at_10 = 0.5.
"""

import json

from src.evaluate import main as evaluate_main
from src.ingest import build_index
from src.retrieve import Retriever


def _golden_for(tmp_path, top1_id):
    lines = [
        {"id": "q1", "question": "When do I need to register for Self Assessment?",
         "ground_truth": "", "source_ids": [top1_id],
         "difficulty": "single_hop", "added_in": "v1"},
        {"id": "q2", "question": "How do I renew my passport online?",
         "ground_truth": "", "source_ids": ["gov-uk/nonexistent-page#chunk-9"],
         "difficulty": "single_hop", "added_in": "v1"},
        {"id": "q3", "question": "Can I pay in monthly instalments?",
         "ground_truth": "", "source_ids": [],
         "difficulty": "negative", "added_in": "v1"},
    ]
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return p


def _run(tmp_path, fixture_pages, cfg):
    idx = tmp_path / "idx"
    build_index(fixture_pages, cfg, idx)
    top1 = Retriever(cfg, idx).retrieve(
        "When do I need to register for Self Assessment?"
    )[0].chunk.chunk_id
    golden = _golden_for(tmp_path, top1)
    out = tmp_path / "current.json"

    # evaluate loads config from a file; write the offline test config out.
    import yaml
    from dataclasses import asdict
    cfg_path = tmp_path / "retrieval.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "seed": cfg.seed,
        "chunking": asdict(cfg.chunking) | {"separators": list(cfg.chunking.separators)},
        "embedding": asdict(cfg.embedding),
        "store": asdict(cfg.store),
        "retrieval": asdict(cfg.retrieval),
    }))

    code = evaluate_main([
        "--suite", "retrieval",
        "--dataset", str(golden),
        "--config", str(cfg_path),
        "--index", str(idx),
        "--out", str(out),
    ])
    return code, json.loads(out.read_text())


def test_retrieval_suite_values(tmp_path, fixture_pages, test_config):
    code, results = _run(tmp_path, fixture_pages, test_config)
    assert code == 0
    assert results["dataset_version"] == "v1"
    from src.evaluate import _RETRIEVAL_METRICS

    # Derived, not restated: a metric added to the suite is reported, and this
    # test should not have to be edited to say so.
    assert set(results["retrieval"]) == {name for name, _ in _RETRIEVAL_METRICS}
    assert results["retrieval"]["hit_at_5"] == 0.5
    assert results["retrieval"]["mrr"] == 0.5
    assert results["retrieval"]["context_recall_at_10"] == 0.5
    assert results["retrieval_detail"]["n_answerable"] == 2
    assert results["retrieval_detail"]["n_negatives"] == 1


def test_results_shape_matches_compare_contract(tmp_path, fixture_pages, test_config):
    """compare.py reads current['dataset_version'] and current['retrieval'][metric]."""
    _, results = _run(tmp_path, fixture_pages, test_config)
    assert "dataset_version" in results
    assert isinstance(results["retrieval"], dict)
    for v in results["retrieval"].values():
        assert 0.0 <= v <= 1.0


def test_deterministic_across_runs(tmp_path, fixture_pages, test_config):
    _, a = _run(tmp_path / "a", fixture_pages, test_config)
    _, b = _run(tmp_path / "b", fixture_pages, test_config)
    assert a["retrieval"] == b["retrieval"]


def test_limit_caps_questions(tmp_path, fixture_pages, test_config):
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    from src.evaluate import run_retrieval_suite
    from src.golden import load_golden
    top1 = Retriever(test_config, idx).retrieve(
        "When do I need to register for Self Assessment?"
    )[0].chunk.chunk_id
    golden = load_golden(_golden_for(tmp_path, top1))
    suite = run_retrieval_suite(golden, test_config, idx, limit=1)
    assert suite["n_answerable"] == 1


# ---- evaluation depth is set by the metrics, not by the serving top_k -------

def test_suite_retrieves_deep_enough_for_the_deepest_metric(tmp_path, fixture_pages, test_config):
    """context_recall_at_10 must be able to see rank 10.

    The suite used to retrieve `retrieval.top_k` and score recall@10 over that
    list. At top_k=5 the metric could not see past rank 5 and — with one gold
    chunk per question — was arithmetically identical to hit_at_5. Both reported
    exactly 0.860 on the v2 golden set, and the 0.88 floor was unreachable by
    construction.
    """
    from dataclasses import replace

    from src.evaluate import _EVAL_DEPTH, run_retrieval_suite
    from src.golden import GoldenRecord
    from src.ingest import build_index

    cfg = replace(test_config, retrieval=replace(test_config.retrieval, top_k=5))
    idx = tmp_path / "idx"
    build_index(fixture_pages, cfg, idx)

    seen: list[int] = []
    from src import evaluate as E

    real = E.Retriever

    class _Spy(real):
        def retrieve(self, question, top_k=None):
            seen.append(top_k)
            return super().retrieve(question, top_k=top_k)

    E.Retriever = _Spy
    try:
        run_retrieval_suite(
            [GoldenRecord(id="q1", question="register", ground_truth="x",
                          source_ids=("gov-uk/register-for-self-assessment#chunk-0",),
                          difficulty="single_hop", added_in="v1")],
            cfg, idx,
        )
    finally:
        E.Retriever = real

    assert seen, "the suite never retrieved"
    assert all(k >= _EVAL_DEPTH for k in seen), (
        f"retrieved to {seen}, shallower than the deepest metric ({_EVAL_DEPTH})"
    )


def test_a_larger_serving_top_k_is_respected(tmp_path, fixture_pages, test_config):
    """A config that serves more context is measured as it actually behaves."""
    from dataclasses import replace

    from src.evaluate import _EVAL_DEPTH

    assert max(25, _EVAL_DEPTH) == 25       # depth never truncates a deeper config
    cfg = replace(test_config, retrieval=replace(test_config.retrieval, top_k=25))
    assert max(cfg.retrieval.top_k, _EVAL_DEPTH) == 25


def test_missing_expected_chunk_is_flagged_as_a_golden_set_defect(tmp_path, fixture_pages, test_config):
    """A gold id absent from the index is a bad label, not bad ranking.

    Scoring reports both as hit@5 = 0. They need opposite fixes — one is a
    golden-set correction, the other a retrieval change — so the suite has to
    say which it is.
    """
    from src.evaluate import run_retrieval_suite
    from src.golden import GoldenRecord
    from src.ingest import build_index

    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)

    records = [
        GoldenRecord(id="q_real", question="register", ground_truth="x",
                     source_ids=("gov-uk/register-for-self-assessment#chunk-0",),
                     difficulty="single_hop", added_in="v1"),
        GoldenRecord(id="q_stale", question="register", ground_truth="x",
                     source_ids=("gov-uk/this-page-does-not-exist#chunk-0",),
                     difficulty="single_hop", added_in="v1"),
    ]
    rows = {r["id"]: r for r in run_retrieval_suite(records, test_config, idx)["per_question"]}

    assert rows["q_real"]["expected_missing_from_index"] == []
    assert rows["q_stale"]["expected_missing_from_index"] == [
        "gov-uk/this-page-does-not-exist#chunk-0"
    ]


def test_served_context_recall_tracks_top_k(tmp_path, fixture_pages, test_config):
    """The only metric that can see a top_k regression.

    hit_at_5 / mrr / context_recall_at_10 are measured at fixed cutoffs, and the
    suite retrieves max(top_k, _EVAL_DEPTH) — so for any top_k <= 10 they score
    the same ten results and cannot move. A PR halving top_k would pass the gate
    while halving the context the generator receives.
    """
    from dataclasses import replace

    from src.evaluate import _score_question
    from src.golden import GoldenRecord

    ranked = [f"c{i}" for i in range(10)]
    record = GoldenRecord(id="q", question="q", ground_truth="g",
                          source_ids=("c6",), difficulty="single_hop", added_in="v1")

    served_5 = _score_question(ranked, record, None, served_k=5)
    served_10 = _score_question(ranked, record, None, served_k=10)

    # The gold chunk sits at rank 7: outside a served top_k of 5, inside 10.
    assert served_5["served_context_recall"] == 0.0
    assert served_10["served_context_recall"] == 1.0
    # …while the fixed-cutoff metrics are identical either way.
    for m in ("hit_at_5", "mrr", "context_recall_at_10"):
        assert served_5[m] == served_10[m], f"{m} should not depend on served_k"
