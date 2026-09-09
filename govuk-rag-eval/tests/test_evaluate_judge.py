"""End-to-end judge suite, fully offline: echo generator + heuristic grader.

No LLM, no network, no key. Proves the pipeline (retrieve -> generate -> judge
-> median -> results JSON) and the cost-cap pre-flight.
"""

import json
from dataclasses import asdict

import pytest
import yaml

from src.evaluate import main as evaluate_main
from src.ingest import build_index


def _write_config(tmp_path, cfg, provider="echo"):
    p = tmp_path / "retrieval.yaml"
    p.write_text(yaml.safe_dump({
        "seed": cfg.seed,
        "chunking": asdict(cfg.chunking) | {"separators": list(cfg.chunking.separators)},
        "embedding": asdict(cfg.embedding),
        "store": asdict(cfg.store),
        "retrieval": asdict(cfg.retrieval),
        "generation": {"provider": provider, "model": "gpt-4o-mini"},
    }))
    return p


def _golden(tmp_path):
    lines = [
        {"id": "q1", "question": "When do I register for Self Assessment?",
         "ground_truth": "Register by 5 October.", "source_ids": ["gov-uk/register-for-self-assessment#chunk-0"],
         "difficulty": "single_hop", "added_in": "v1"},
        {"id": "q2", "question": "What is the online tax return deadline?",
         "ground_truth": "31 January.", "source_ids": ["gov-uk/self-assessment-tax-returns#chunk-1"],
         "difficulty": "single_hop", "added_in": "v1"},
    ]
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return p


def test_judge_suite_end_to_end(tmp_path, fixture_pages, test_config):
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    cfg_path = _write_config(tmp_path, test_config, provider="echo")
    out = tmp_path / "current.json"

    code = evaluate_main([
        "--suite", "judge",
        "--dataset", str(_golden(tmp_path)),
        "--config", str(cfg_path),
        "--index", str(idx),
        "--judge-backend", "heuristic",
        "--runs", "3",
        "--out", str(out),
    ])
    assert code == 0
    results = json.loads(out.read_text())
    assert results["dataset_version"] == "v1"
    assert set(results["judge"]) == {
        "faithfulness", "answer_relevancy", "context_precision", "answer_correctness",
    }
    assert all(0.0 <= v <= 1.0 for v in results["judge"].values())
    assert results["judge_runs"] == 3
    assert results["estimated_cost_usd"] == 0.0  # heuristic grader is free
    assert results["judge_detail"]["backend"] == "heuristic"
    assert results["judge_detail"]["generator"] == "echo"


def test_judge_merges_into_existing_retrieval_results(tmp_path, fixture_pages, test_config):
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    cfg_path = _write_config(tmp_path, test_config, provider="echo")
    golden = _golden(tmp_path)
    current = tmp_path / "current.json"

    # retrieval first, then judge merged into the same file.
    evaluate_main([
        "--suite", "retrieval", "--dataset", str(golden),
        "--config", str(cfg_path), "--index", str(idx), "--out", str(current),
    ])
    evaluate_main([
        "--suite", "judge", "--dataset", str(golden), "--config", str(cfg_path),
        "--index", str(idx), "--judge-backend", "heuristic", "--runs", "1",
        "--merge-into", str(current),
    ])
    results = json.loads(current.read_text())
    assert "retrieval" in results and "judge" in results  # both suites present


def test_cost_cap_aborts_before_grading(tmp_path, fixture_pages, test_config):
    idx = tmp_path / "idx"
    build_index(fixture_pages, test_config, idx)
    cfg_path = _write_config(tmp_path, test_config, provider="echo")
    # A cap that the ragas backend's estimated cost (0.03/q/run) will exceed.
    eval_cfg = tmp_path / "eval_config.yaml"
    eval_cfg.write_text(yaml.safe_dump({"cost": {"max_judge_questions_per_run": 120, "max_usd_per_run": 0.01}}))

    with pytest.raises(SystemExit, match="over the"):
        evaluate_main([
            "--suite", "judge", "--dataset", str(_golden(tmp_path)),
            "--config", str(cfg_path), "--index", str(idx),
            "--judge-backend", "ragas", "--runs", "3",
            "--eval-config", str(eval_cfg), "--out", str(tmp_path / "c.json"),
        ])


def test_question_cap_limits_judged(tmp_path, fixture_pages, test_config):
    from dataclasses import replace

    from src.config import GenerationConfig
    from src.evaluate import run_judge_suite
    from src.golden import load_golden

    cfg = replace(test_config, generation=GenerationConfig(provider="echo"))
    idx = tmp_path / "idx"
    build_index(fixture_pages, cfg, idx)
    records = load_golden(_golden(tmp_path))
    caps = {"max_judge_questions_per_run": 1, "max_usd_per_run": 4.0}
    suite = run_judge_suite(records, cfg, idx, runs=1, backend="heuristic", caps=caps)
    assert suite["n_judged"] == 1
