# Agent builds

Self-contained builds, each in its own directory.

## [`govuk-rag-eval/`](./govuk-rag-eval)

A retrieval-augmented QA system over GOV.UK guidance, wrapped in an evaluation
harness that blocks pull requests when retrieval or generation quality drops.
Plain Python (no orchestration framework), deterministic retrieval metrics,
LLM-judge metrics via RAGAS with median-of-N, and a CI gate driven by
`scripts/compare.py`. The LLM provider is interchangeable (OpenAI or Anthropic).

See [`govuk-rag-eval/README.md`](./govuk-rag-eval/README.md) for build status and
[`govuk-rag-eval/BUILD.md`](./govuk-rag-eval/BUILD.md) for the full plan.

CI: `.github/workflows/rag-eval.yml` (PR gate + nightly) and
`.github/workflows/refresh-corpus.yml` (weekly corpus re-crawl).
