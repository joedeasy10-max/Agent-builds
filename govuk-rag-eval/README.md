# govuk-rag-eval

A RAG system over GOV.UK guidance, wrapped in an evaluation harness that blocks
PRs when retrieval or generation quality regresses.

Full spec and build order: [`BUILD.md`](./BUILD.md).

> **Corpus licence:** GOV.UK content is published under the
> [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
> The crawler is rate-limited to be a polite API citizen.

## Build status

Steps are defined in `BUILD.md` → "Build order". There are **8**.

| # | Step | State |
|---|------|-------|
| 1 | **Ingest + retrieve** — crawl, chunk, index, `retrieve(question)` | ✅ done |
| 2 | **Retrieval metrics** (hit@5, MRR, recall@10) + `src/evaluate.py` | ✅ done |
| 3 | Golden set to full size (150–300, hand-reviewed) | ⬜ blocked: needs GOV.UK crawl (sandbox egress denies gov.uk) |
| 4 | **Generation + RAGAS judge metrics** (median-of-N, measure variance) | 🟡 scaffolded — pipeline + median-of-N done; real RAGAS grader lazy-wired |
| 5 | **CI gate** — wire `rag-eval.yml` + `compare.py`, commit a baseline | 🟡 wired + proven end-to-end; activates with `OPENAI_API_KEY` + a real baseline |
| 6 | Regression demos — 3 blocked PRs | 🟡 chunk-size demo proven offline (gate blocks); top_k/embedding demos need real data |
| 7 | Experiment benchmark — `run_experiments.py` + table | 🟡 runner done + demoed; real table needs embeddings |
| 8 | **Corpus-drift workflow** (`refresh-corpus.yml`) | 🟡 scaffolded — workflow + `refresh_corpus.py` + drift logic tested; runs once egress exists |

The gate lives at the **repo root** workflow `.github/workflows/rag-eval.yml`
(GitHub only runs workflows from the root; the steps `cd` into `govuk-rag-eval/`).
`evaluate` + `compare.py` are wired and the full pass/fail chain is proven
offline in `tests/test_gate.py` (a run matching baseline passes; a regression
below a floor fails; no baseline reports-only). The `eval_config` dataset version
is aligned to the shipped golden set (`v1`), and `results/baseline.json` is an
intentional `{}` placeholder (reporting-only). The workflow is **guarded on
`OPENAI_API_KEY`**: without the secret every real step skips and the job stays
green, so it is never a red check; it activates once the secret is set and a real
corpus + golden set (step 3) + committed baseline exist — see
`results/README.md`. If this subproject is ever split into its own repo, move the
root workflow to that repo's `.github/workflows/` and drop the `govuk-rag-eval/`
path prefixes.

## Everything is a config value

Nothing that affects retrieval quality is hardcoded — it's all in
[`configs/retrieval.yaml`](./configs/retrieval.yaml): chunk size, overlap,
splitter, embedding provider/model, vector store, retriever, top_k. A PR changes
retrieval behaviour by editing that file, and the CI index cache rebuilds
because its key hashes the file. Swappable backends live behind small
interfaces (`src/embed.py`, `src/store.py`), so changing the embedder or store
is a one-line config edit, not a code change.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
pytest                      # deterministic, offline — no network, no LLM

# Build the index and query it (real path needs OPENAI_API_KEY for the
# openai embedder; set embedding.provider: hashing in the config to run offline)
python -m src.ingest   --config configs/retrieval.yaml --out .index/
python -m src.retrieve --config configs/retrieval.yaml --index .index/ \
    --query "When do I need to register for Self Assessment?"
```

### Running offline / in tests

The tests use a deterministic `hashing` embedder and a file-backed numpy
(`flat`) store, so the whole ingest → retrieve loop runs with no API key and no
network. Point any config at `embedding.provider: hashing` to do the same by
hand.

## Determinism & stable IDs

- Fixed seed, sorted iteration, tie-broken ranking — identical input yields
  identical output, or the gate is worthless.
- Chunk IDs are content-derived and stable: `<page-path>#chunk-<n>` (e.g.
  `gov-uk/register-for-self-assessment#chunk-0`), each carrying a content hash so
  corpus drift is detectable. The golden set references these IDs; re-ingesting
  the same content does not invalidate them.

## Layout

See `BUILD.md` → "Repo structure". This tree implements steps 1, 2 and 4:
step 1 (`src/config.py`, `corpus.py`, `chunk.py`, `embed.py`, `store.py`,
`ingest.py`, `retrieve.py`), step 2 (`src/golden.py`, `src/evaluate.py`,
`src/metrics/retrieval.py`) and step 4, scaffolded (`src/generate.py`,
`src/metrics/judge.py` + the judge suite in `src/evaluate.py`), plus the step-5
gate files.

### Evaluate

```bash
python -m src.ingest    --config configs/retrieval.yaml --out .index/
python -m src.evaluate  --suite retrieval \
    --dataset data/golden/questions.jsonl \
    --config configs/retrieval.yaml --index .index/ --out results/current.json
```

Retrieval metrics are deterministic and LLM-free, computed over the ranking the
system actually returns (`retrieval.top_k`) — so a `top_k` regression shows up in
`hit@5`/`recall@10`. Negatives (unanswerable questions) are excluded from these
aggregates; refusal behaviour is a generation concern (step 4). Use `--limit` to
cap questions in a dev loop.

### Judge suite (step 4, scaffolded)

```bash
# Offline / deterministic (no key, no cost) — what the tests use:
python -m src.evaluate --suite judge \
    --dataset data/golden/questions.jsonl --config configs/retrieval.yaml \
    --index .index/ --judge-backend heuristic --runs 3 \
    --merge-into results/current.json

# Real judge metrics (RAGAS) — needs a provider key and the judge extras:
pip install -r requirements-judge.lock
python -m src.evaluate --suite judge ... --judge-backend ragas --judge-provider openai --runs 3 ...
python -m src.evaluate --suite judge ... --judge-backend ragas --judge-provider anthropic --runs 3 ...
```

**Provider is interchangeable — OpenAI or Anthropic.** Generation
(`generation.provider`: `openai` | `anthropic` | `echo`) and the RAGAS judge
(`--judge-provider`) both swap with a one-line change; leave `generation.model`
blank to take the provider default (`gpt-4o-mini` / `claude-sonnet-5`). The CI
gate activates on **either** `OPENAI_API_KEY` **or** `ANTHROPIC_API_KEY`. Caveat:
**Anthropic has no embeddings API**, so retrieval embeddings stay OpenAI or the
local `bge` — an Anthropic-only run pairs `embedding.provider: bge` (local, no
key) with `generation.provider: anthropic`. (RAGAS `answer_relevancy` also needs
an embeddings model, so a fully Anthropic judge still uses an embeddings backend
for that one metric.)

Judge metrics are LLM-graded and noisy, so the suite runs the grader N times and
takes the **median** per metric, recording the spread (`judge_detail.spread`) —
that measured variance is what justifies the wide tolerance bands in
`eval_config.yaml`. Cost guardrails (`cost.max_judge_questions_per_run`,
`max_usd_per_run`) are enforced **before** any grading. Generation is config-driven
(`generation.provider`: `openai` | `echo`); the grader backend is `ragas` (real,
lazy-imported) or `heuristic` (deterministic offline stub, **not** a real quality
signal). The real RAGAS grader is wired but not exercised by tests — no test calls
an LLM.

### Regression demos (step 6)

Three deliberate regressions the gate is meant to catch, each a one-line edit to
`configs/retrieval.yaml`:

| Demo | Edit | Offline-provable here? |
| --- | --- | --- |
| Chunking | `chunk_size: 512 → 2000` | ✅ yes — `tests/test_regression_demo.py` shows the gate failing (exit 1) |
| Retrieval depth | `top_k: 5 → 2` | ⬜ needs the full golden set (on the 3-record starter every relevant chunk is already rank 1) |
| Embedding | `text-embedding-3-small → a weaker model` | ⬜ needs real embeddings (a key) |

The chunk-size demo is proven deterministically: a bigger chunk size collapses
pages and invalidates the golden set's `#chunk-N` references, so metrics fall
below their floor and `compare.py` blocks the PR. The other two behave the same
way once the real corpus/model and full golden set are in place.

### Experiment benchmark (step 7)

`scripts/run_experiments.py` runs several configs over one shared corpus + golden
set and emits a comparison table (winner = best primary metric); a config whose
retriever/embedder can't run (e.g. `hybrid`, or `openai`/`bge` without a key) is
reported as skipped, not a crash.

```bash
python scripts/run_experiments.py \
    --dataset data/golden/questions.jsonl \
    --configs configs/experiments/*.yaml \
    --primary-metric mrr --out results/experiments
```

Real numbers for the four named experiments (`baseline`, `small_chunks`,
`hybrid_bm25`, `reranked`) go here once embeddings are available; see
`results/experiments/README.md`. The runner and its table are proven offline in
`tests/test_run_experiments.py` (including the chunk-size effect).

### Corpus-drift workflow (step 8)

`.github/workflows/refresh-corpus.yml` re-crawls the slice weekly. If any page's
content hash drifted, `scripts/refresh_corpus.py` rewrites
`data/corpus/manifest.json` (preserving `fetched_at` for unchanged pages, so
there are no spurious PRs) and the workflow opens a PR labelled `corpus-drift`.
The RAG-eval gate's index cache keys on the manifest, so the eval then runs
against the refreshed corpus automatically — catching quality drops caused by
the *source content* moving, not just our own code.

```bash
python scripts/refresh_corpus.py --config configs/retrieval.yaml --summary-out corpus-drift.md
```

The crawl needs outbound `www.gov.uk` (fine on CI runners; no LLM key needed).
The drift diff/merge/summary logic is pure and unit-tested offline
(`tests/test_refresh_corpus.py`).
