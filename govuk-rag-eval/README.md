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
| 2 | **Retrieval metrics** (hit@5, MRR, recall@10, served recall) + `src/evaluate.py` | ✅ done |
| 3 | Golden set to full size (150–300, hand-reviewed) | 🟡 **65 reviewed records** (41 answerable, 24 negatives) — real and gating, but short of the 150–300 target, and only 4 multi-hop |
| 4 | **Generation + RAGAS judge metrics** (median-of-N, measure variance) | ✅ done — spread measured, tolerances derived from it (table below) |
| 5 | **CI gate** — wire `rag-eval.yml` + `compare.py`, commit a baseline | ✅ done — active, with a committed v2 baseline; blocks PRs (see step 6) |
| 6 | Regression demos — 3 blocked PRs | ✅ done — PRs #18, #19, #20 are open and red; each fails on real numbers |
| 7 | Experiment benchmark — `run_experiments.py` + table | ✅ done — table below |
| 8 | **Corpus-drift workflow** (`refresh-corpus.yml`) | ✅ done — `refresh_corpus.py` + drift logic tested; runs on schedule |

The gate lives at the **repo root** workflow `.github/workflows/rag-eval.yml`
(GitHub only runs workflows from the root; the steps `cd` into `govuk-rag-eval/`).
It is **live**: `eval_config` is on dataset `v2`, `results/baseline.json` holds a
real measured baseline, and three regression PRs are sitting red against it.

The workflow is **guarded on a provider key** (`OPENAI_API_KEY` *or*
`ANTHROPIC_API_KEY` — they are interchangeable): without either, every real step
skips and the job stays green, so it is never a red check on a fork that has not
opted in. Judge metrics are extra-gated on `RUN_JUDGE`, which is true nightly, on
manual dispatch, and on a PR labelled `full-eval` — retrieval alone runs on every
PR, because it is deterministic and free.

If this subproject is ever split into its own repo, move the root workflow to
that repo's `.github/workflows/` and drop the `govuk-rag-eval/` path prefixes.

### Current baseline (dataset `v2`, 65 records)

Deterministic retrieval, measured on
[run 34500807132](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34500807132):

| Metric | Value | Floor |
| --- | ---: | ---: |
| `hit_at_5` | 0.902 | 0.82 |
| `mrr` | 0.715 | 0.65 |
| `context_recall_at_10` | 0.927 | 0.88 |
| `served_context_recall` | 0.902 | 0.82 |

### Judge metrics and their measured noise

RAGAS, graded by `anthropic/claude-sonnet-5`, 41 questions × 3 runs, ~$3.69 per
run, from
[run 34505214155](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34505214155).
The gate uses the **median**; the spread is what calibrates the tolerances.

| Metric | Median | Run values | Spread | Relative | Floor | Tolerance |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `faithfulness` | 0.950 | 0.944 / 0.950 / 0.976 | 0.032 | 3.35% | 0.85 | 8% |
| `answer_relevancy` | 0.899 | 0.898 / 0.899 / 0.900 | 0.001 | 0.14% | 0.80 | 2% |
| `context_precision` | 0.820 | 0.809 / 0.820 / 0.827 | 0.019 | 2.28% | 0.70 | 5% |
| `answer_correctness` | 0.831 | 0.822 / 0.831 / 0.834 | 0.012 | 1.43% | — | ungated |

**The tolerances were guesses (0.05 / 0.05 / 0.06) and are now derived from
this.** Two of the three moved, in opposite directions:

- **`answer_relevancy` was far too loose.** At 0.14% spread it is the most
  stable metric here by a factor of ten, and a 5% band could not have caught
  any realistic regression. Tightened to 2% — still ~14× the observed noise,
  deliberately conservative because three samples is a thin basis for cutting
  close to the measurement.
- **`faithfulness` was thin, though not breached.** Worth being exact: nothing
  observed would have failed at 5%. The worst plausible median-to-median drop
  is (0.9756 − 0.9438) / 0.9756 = **3.26%**, which passes 5% while consuming
  65% of the budget. Its median also moved 2.5% between two independent 3-run
  measurements (0.975 → 0.950), so a genuine regression had only about a third
  of the band left to show up in. Widened to 8%, taking budget usage to 41%.
  The better fix is more runs — median-of-3 is thin for this metric — but at
  ~$1.20 per extra run, widening is a deliberate cost trade.
- **`answer_correctness` stays ungated, for a corrected reason.** It was
  described as "too noisy to be a build signal"; the measurement doesn't
  support that — at 1.43% it is *less* noisy than two metrics that do gate. The
  real reason is that it scores answers against one hand-written
  `ground_truth` string, so it measures agreement with a phrasing rather than
  correctness.

### Where retrieval still misses

4 of 41 answerable questions miss at rank 5. None of them is a stale label —
every gold chunk id is present in the index:

| Question | What happened |
| --- | --- |
| `q_0006` | gold chunk found at **rank 6** — a near miss, recoverable by depth |
| `q_0015` | ranks 1 and 2 are `#chunk-1` and `#chunk-3` of **the correct page**; the gold label names `#chunk-0`. Retrieval found the right document and the metric scores it zero |
| `q_0001` | gold chunk absent from the top 10; nothing on-topic retrieved |
| `q_0042` | gold chunk absent from the top 10; nothing on-topic retrieved |

`q_0015` is the interesting one: it argues for a page-level companion to the
chunk-level metric, since "right page, adjacent chunk" is scored identically to
"completely wrong".

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

All three are open and red against the committed `v2` baseline. Numbers are
from each PR's own gate run, not a simulation:

| PR | Edit | `hit_at_5` | `mrr` | `recall@10` | `served_recall` | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| — | *baseline* | 0.902 | 0.715 | 0.927 | 0.902 | — |
| [#18](https://github.com/joedeasy10-max/Agent-builds/pull/18) | `chunk_size: 512 → 2000` | **0.805** ❌ | 0.735 ⬆ | 0.902 ⬆ | **0.805** ❌ | failed on 2 metrics |
| [#19](https://github.com/joedeasy10-max/Agent-builds/pull/19) | `top_k: 5 → 2` | 0.902 ✅ | 0.715 ✅ | 0.927 ✅ | **0.732** ❌ | failed on 1 metric |
| [#20](https://github.com/joedeasy10-max/Agent-builds/pull/20) | weaker embedding model | **0.537** ❌ | **0.418** ❌ | **0.659** ❌ | **0.537** ❌ | failed on 4 metrics |

Two of these are worth reading closely.

**#18 shows why a single metric is not a gate.** Bigger chunks *improved* `mrr`
and `context_recall_at_10` — a fatter chunk is more likely to contain the gold
text — while `hit_at_5` fell through the floor. A gate watching only the
average, or only recall, would have waved this through.

**#19 is the reason `served_context_recall` exists.** Cutting `top_k` from 5 to
2 left `hit_at_5`, `mrr` and `context_recall_at_10` *completely unchanged*, since
all three are measured at fixed cutoffs and cannot see how much context is
actually served. Only `served_context_recall` moved, and it alone blocked the
PR. Before that metric was added (PR #17) this regression would have passed the
gate untouched.

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

Measured on the real corpus (292 GOV.UK pages) and the `v2` golden set, on
[run 34501424650](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34501424650).
Each config gets its own index — a shared one would score three configs against
vectors built for the fourth.

| Config | `hit_at_5` | `mrr` | `recall@10` | `served_recall` | chunks | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` 🏆 | **0.902** | **0.715** | **0.927** | **0.902** | 6103 | ok |
| `small_chunks` (256) | 0.512 | 0.457 | 0.610 | 0.512 | 13834 | ok |
| `hybrid_bm25` | — | — | — | — | — | skipped: `hybrid` retriever not implemented |
| `reranked` | — | — | — | — | — | skipped: `reranker` section not implemented |

**Winner: `baseline`** — the shipped `configs/retrieval.yaml`, on the primary
metric `mrr`. It is not a close call: halving the chunk size costs 39 points of
`hit_at_5`.

Two of the four are honestly *pending*, not run. `hybrid_bm25` needs a retriever
that does not exist yet; `reranked` declares a `reranker:` section that
`Config` rejects, so it fails to load and says so. That second row used to
report numbers — it carried the reranker commented out, which made the file
identical to `baseline.yaml`, and the benchmark dutifully published baseline's
results under the name of an experiment that had never been performed. A
pending experiment should look pending.

> **Open question on `small_chunks`.** A 39-point drop from halving the chunk
> size is larger than a chunking change alone would suggest, and the cause is
> not yet established. The obvious suspect — that `#chunk-N` gold labels
> renumber when chunk size changes — was checked and **does not hold**: 37 of
> the 41 answerable questions reference `#chunk-0`, which is the first chunk of
> its page at any chunk size. Whether the remainder is a genuine retrieval
> effect (256-char chunks being too small to carry enough context to match a
> question) or a subtler labelling artefact needs the real corpus to settle,
> and is not claimed either way here.

The runner and its table are proven offline in `tests/test_run_experiments.py`,
including that the table has a column for every metric in the suite and that an
unloadable config becomes a skipped row rather than a crashed run.

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

### Building the golden set (step 3)

`scripts/build_golden_set.py` drafts candidates with an LLM and queues them for
**human review** — it deliberately never writes LLM output straight into
`data/golden/questions.jsonl`, because that hand review is what makes the numbers
mean anything (BUILD.md).

```bash
# 1. draft over the indexed chunks (+ negatives) into a review queue
python scripts/build_golden_set.py draft \
    --config configs/retrieval.yaml --index .index/ \
    --per-chunk 2 --limit 60 --negatives 25

# 2. review data/golden/review_queue.jsonl BY HAND: set each "status" to
#    "approved" or "rejected", fixing wording/ground_truth as you go.
python scripts/build_golden_set.py status          # progress + negative count

# 3. append only the approved ones to the golden set
python scripts/build_golden_set.py promote --version v2 --dry-run
python scripts/build_golden_set.py promote --version v2
```

Each candidate carries the source chunk's text (`source_excerpt`), so reviewing
is reading one screen rather than hunting through the corpus. `promote` continues
ids from the highest existing `q_NNNN` (never renumbering what the golden set
already references), skips duplicate questions, and validates the result with the
real loader. Drafting is provider-interchangeable (`--provider openai|anthropic`,
defaulting to `generation.provider`), capped by `--limit` and `--max-usd`.

After promoting: bump `dataset.version` in `configs/eval_config.yaml` to match,
then regenerate `results/baseline.json` (see `results/README.md`).

### API keys

Set **`ANTHROPIC_API_KEY`** or **`OPENAI_API_KEY`** — the providers are
interchangeable:

* **locally** — export it in your shell, or put it in a `.env` (gitignored);
* **in CI** — add it as a GitHub Actions repository secret under the exact same
  name. `rag-eval.yml` activates on either key and green-skips without them.

Never commit a key.
