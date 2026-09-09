# results/experiments/

Comparison tables + per-config metrics from `scripts/run_experiments.py`
(build step 7). Generated files (`*.json`, `table.md`) are gitignored; this
directory holds the published benchmark once real numbers exist.

## Producing the benchmark

```bash
cd govuk-rag-eval
python scripts/run_experiments.py \
    --dataset data/golden/questions.jsonl \
    --configs configs/experiments/baseline.yaml \
              configs/experiments/small_chunks.yaml \
              configs/experiments/hybrid_bm25.yaml \
              configs/experiments/reranked.yaml \
    --primary-metric mrr --out results/experiments
```

The corpus is held fixed across experiments — only the retrieval config varies —
so any difference in the table is caused by the config change.

## Status

The four named configs use the real (`openai`) embedder, and `hybrid_bm25` /
`reranked` need retriever/reranker support that lands in later work, so a
meaningful published table needs an `OPENAI_API_KEY` and those retrievers. Until
then the runner is exercised offline with deterministic (`hashing`) configs in
`tests/test_run_experiments.py`, which also demonstrates that a larger chunk size
degrades retrieval on the fixtures.
