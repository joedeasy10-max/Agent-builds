"""Evaluation runner: dataset x config -> results.json.

Step 2 implements the deterministic `retrieval` suite (hit@5, MRR,
context_recall@10). The `judge` suite (RAGAS) lands in build step 4.

The retriever loads the index built by `python -m src.ingest` (defaulting to the
config's store path), retrieves each question, and scores the returned ranking
against the golden set's `source_ids`. Metrics are computed over what the system
actually returns (config.retrieval.top_k), so lowering top_k shows up here — the
"top_k 5 -> 2" regression demo is meant to trip this suite.

CLI (matches .github/workflows/rag-eval.yml):
    python -m src.evaluate --suite retrieval \
        --dataset data/golden/questions.jsonl \
        --config configs/retrieval.yaml \
        --out results/current.json

    python -m src.evaluate --suite judge ... --runs 3 --merge-into results/current.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config, load_config
from .generate import build_generator
from .golden import GoldenRecord, dataset_version, load_golden
from .metrics import judge as J
from .metrics import retrieval as R
from .retrieve import Retriever

# The retrieval metrics this suite reports. Names match configs/eval_config.yaml.
# Each is (metric_name, cutoff-or-None). cutoff None => whole returned ranking.
_RETRIEVAL_METRICS: list[tuple[str, int | None]] = [
    ("hit_at_5", 5),
    ("mrr", None),
    ("context_recall_at_10", 10),
]


def _score_question(ranked_ids: list[str], record: GoldenRecord) -> dict:
    relevant = record.source_ids
    first_rank = next(
        (i for i, cid in enumerate(ranked_ids, start=1) if cid in set(relevant)),
        None,
    )
    return {
        "id": record.id,
        "difficulty": record.difficulty,
        "hit_at_5": R.hit_at_k(ranked_ids, relevant, 5),
        "mrr": R.reciprocal_rank(ranked_ids, relevant),
        "context_recall_at_10": R.recall_at_k(ranked_ids, relevant, 10),
        "first_relevant_rank": first_rank,
    }


def run_retrieval_suite(
    records: list[GoldenRecord],
    config: Config,
    index_dir: str | Path,
    limit: int | None = None,
) -> dict:
    retriever = Retriever(config, index_dir)
    answerable = [r for r in records if r.is_answerable]
    if limit is not None:
        answerable = answerable[:limit]

    per_question: list[dict] = []
    for record in answerable:
        results = retriever.retrieve(record.question)
        ranked_ids = [sc.chunk.chunk_id for sc in results]
        per_question.append(_score_question(ranked_ids, record))

    aggregate = {
        name: R.mean([q[name] for q in per_question]) for name, _ in _RETRIEVAL_METRICS
    }
    return {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "n_negatives": sum(1 for r in records if r.is_negative),
        "metrics": aggregate,
        "per_question": per_question,
    }


# Rough per-question cost of one judge run, by backend. The heuristic grader is
# free (no LLM); the ragas figure is a deliberately conservative guess used only
# for the pre-flight cap check, not for billing.
_JUDGE_COST_PER_QUESTION_USD = {"heuristic": 0.0, "ragas": 0.03}


def _load_cost_caps(eval_config_path: Path | None) -> dict:
    """Read cost guardrails from eval_config.yaml (best-effort, safe defaults)."""
    defaults = {"max_judge_questions_per_run": 120, "max_usd_per_run": 4.00}
    if eval_config_path is None or not eval_config_path.exists():
        return defaults
    import yaml

    cost = (yaml.safe_load(eval_config_path.read_text()) or {}).get("cost", {}) or {}
    return {**defaults, **{k: cost[k] for k in defaults if k in cost}}


def run_judge_suite(
    records: list[GoldenRecord],
    config: Config,
    index_dir: str | Path,
    runs: int,
    backend: str,
    caps: dict,
    limit: int | None = None,
    judge_provider: str = "openai",
) -> dict:
    """Generate answers over the golden questions and grade them (median-of-N).

    Enforces the cost caps BEFORE any grading: caps the question count and
    refuses to start if the estimated spend exceeds max_usd_per_run.
    """
    retriever = Retriever(config, index_dir)
    generator = build_generator(config)
    grader = J.build_grader(backend, provider=judge_provider)

    answerable = [r for r in records if r.is_answerable]
    max_q = caps["max_judge_questions_per_run"]
    ceiling = min(x for x in (limit, max_q) if x is not None)
    answerable = answerable[:ceiling]

    per_q_cost = _JUDGE_COST_PER_QUESTION_USD.get(backend, 0.0)
    estimated_cost = len(answerable) * runs * per_q_cost
    if estimated_cost > caps["max_usd_per_run"]:
        raise SystemExit(
            f"Judge run would cost ~${estimated_cost:.2f}, over the "
            f"${caps['max_usd_per_run']:.2f} cap. Lower --limit, --runs, or the "
            "backend cost."
        )

    samples = []
    for record in answerable:
        results = retriever.retrieve(record.question)
        contexts = tuple(sc.chunk.text for sc in results)
        answer = generator.generate(record.question, list(contexts))
        samples.append(
            J.JudgeSample(
                id=record.id,
                question=record.question,
                answer=answer,
                contexts=contexts,
                ground_truth=record.ground_truth,
            )
        )

    judged = J.run_judge(samples, grader, runs=runs)
    return {
        "n_judged": len(samples),
        "runs": runs,
        "backend": grader.grader_id,
        "generator": generator.generator_id,
        "metrics": judged["metrics"],
        "spread": judged["spread"],
        "estimated_cost_usd": round(estimated_cost, 4),
    }


def _write_results(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an evaluation suite.")
    parser.add_argument("--suite", choices=["retrieval", "judge"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=None, help="index dir (default: store.path)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--merge-into", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap questions (cost discipline)")
    parser.add_argument("--runs", type=int, default=1, help="judge suite: median-of-N runs")
    parser.add_argument(
        "--judge-backend", choices=["ragas", "heuristic"], default="ragas",
        help="judge grader: ragas (real, needs a key) or heuristic (offline stub)",
    )
    parser.add_argument(
        "--judge-provider", choices=["openai", "anthropic"], default="openai",
        help="LLM provider for the ragas judge",
    )
    parser.add_argument(
        "--eval-config", type=Path, default=Path("configs/eval_config.yaml"),
        help="source of judge cost caps",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    records = load_golden(args.dataset)
    version = dataset_version(records)
    index_dir = args.index if args.index is not None else Path(config.store.path)

    # Assemble results in the shape scripts/compare.py expects: dataset_version
    # at the top level, and each suite's aggregate metrics under its suite name.
    if args.merge_into is not None and args.merge_into.exists():
        payload = json.loads(args.merge_into.read_text())
    else:
        payload = {}
    payload["dataset_version"] = version

    if args.suite == "judge":
        caps = _load_cost_caps(args.eval_config)
        suite = run_judge_suite(
            records, config, index_dir, runs=args.runs,
            backend=args.judge_backend, caps=caps, limit=args.limit,
            judge_provider=args.judge_provider,
        )
        payload["judge"] = suite["metrics"]
        payload["judge_runs"] = suite["runs"]
        payload["estimated_cost_usd"] = suite["estimated_cost_usd"]
        payload["judge_detail"] = {
            "n_judged": suite["n_judged"],
            "backend": suite["backend"],
            "generator": suite["generator"],
            "spread": suite["spread"],
        }
        summary = (
            f"[judge] dataset {version} · {suite['n_judged']} judged × "
            f"{suite['runs']} runs ({suite['backend']}) · "
            f"faithfulness={suite['metrics']['faithfulness']:.3f} "
            f"answer_relevancy={suite['metrics']['answer_relevancy']:.3f} · "
            f"est ${suite['estimated_cost_usd']:.2f}"
        )
    else:
        suite = run_retrieval_suite(records, config, index_dir, limit=args.limit)
        payload["retrieval"] = suite["metrics"]
        payload["retrieval_detail"] = {
            "n_questions": suite["n_questions"],
            "n_answerable": suite["n_answerable"],
            "n_negatives": suite["n_negatives"],
            "per_question": suite["per_question"],
        }
        m = suite["metrics"]
        summary = (
            f"[retrieval] dataset {version} · {suite['n_answerable']} answerable "
            f"(+{suite['n_negatives']} negatives) · "
            f"hit@5={m['hit_at_5']:.3f} mrr={m['mrr']:.3f} "
            f"recall@10={m['context_recall_at_10']:.3f}"
        )

    target = args.out or args.merge_into
    if target is None:
        raise SystemExit("Provide --out or --merge-into.")
    _write_results(target, payload)
    print(f"{summary} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
