# GOV.UK RAG Regression Lab — build plan

A retrieval-augmented QA system over GOV.UK guidance, wrapped in an evaluation
harness that blocks pull requests when retrieval or generation quality drops.

> CV line: Built a CI-gated RAG evaluation framework using RAGAS and GitHub
> Actions, automatically detecting retrieval and generation regressions across
> versioned golden datasets.

## Stack

Chosen for CI reproducibility over feature surface. Every dependency here has
to install cleanly on a cold runner in under a minute and behave identically
run to run.

| Layer | Choice | Why |
| --- | --- | --- |
| Corpus | GOV.UK Content API | Real, public, Open Government Licence, and it changes over time — which makes the "did content drift break us?" story real. |
| Chunking | `langchain-text-splitters` | Tiny package, no framework lock-in. |
| Embeddings | OpenAI `text-embedding-3-small`, with a local `bge-small` config as the swap target | The swap is one of your deliberate regressions. |
| Vector store | LanceDB (embedded, file-backed) | No server in CI, index caches as a directory. |
| Generation | One provider behind a thin interface | Model is a config value, not an import. |
| Judge metrics | RAGAS (pin the minor version) | The API has broken between minors before. |
| Retrieval metrics | Hand-written, ~80 lines | Deterministic and free — these do the actual gating. |
| Orchestration | Plain Python | No LangChain chains, no LlamaIndex. A framework in the middle makes "which change caused the regression?" much harder to answer, which is the entire point of the project. |

Pin everything into `requirements.lock`. An eval harness whose numbers move
because a transitive dependency bumped is not an eval harness.

## Repo structure

```
govuk-rag-eval/
├── .github/workflows/
│   ├── rag-eval.yml           # PR gate + nightly full run
│   └── refresh-corpus.yml     # weekly re-crawl, opens a PR if content drifted
├── configs/
│   ├── retrieval.yaml         # the active config — this is what PRs edit
│   ├── eval_config.yaml       # thresholds, tolerances, cost caps
│   └── experiments/           # the 3-4 named configs you benchmark
│       ├── baseline.yaml
│       ├── small_chunks.yaml
│       ├── hybrid_bm25.yaml
│       └── reranked.yaml
├── data/
│   ├── corpus/
│   │   ├── manifest.json      # URLs, fetch dates, content hashes
│   │   └── raw/               # cached API responses (gitignored, rebuildable)
│   └── golden/
│       ├── questions.jsonl    # the golden set, versioned
│       └── CHANGELOG.md       # why each version changed
├── src/
│   ├── ingest.py              # fetch → clean → chunk → embed → index
│   ├── retrieve.py            # config-driven retrieval
│   ├── generate.py            # answer synthesis
│   ├── metrics/
│   │   ├── retrieval.py       # hit@k, MRR, recall@k — deterministic
│   │   └── judge.py           # RAGAS wrapper, median-of-N
│   └── evaluate.py            # runner: dataset × config → results.json
├── scripts/
│   ├── compare.py             # the gate
│   ├── build_golden_set.py    # LLM draft → human review queue
│   └── run_experiments.py     # benchmark all configs, write the report
├── results/
│   ├── baseline.json          # committed on main, the reference point
│   └── experiments/           # benchmark tables for the README
└── requirements.lock
```

## The GOV.UK corpus

Two endpoints, both open, no key needed:

- Search: `https://www.gov.uk/api/search.json?q=...&count=...`
- Content: `https://www.gov.uk/api/content/<path>`

Pick a bounded slice with real depth — self-employment and Self Assessment tax
guidance is a good one: roughly 200–400 pages, heavy cross-referencing,
genuinely hard questions, and content people actually get wrong. Enumerate it
via the search API, fetch each page's content JSON, and record every URL with a
content hash in `manifest.json`.

Note the licence (OGL v3) in your README, and rate-limit the crawl politely.
Small courtesy, but reviewers notice it.

The `refresh-corpus.yml` workflow is the sleeper feature: re-crawl weekly, and
if any page's hash changed, open a PR with the updated corpus. The eval then
runs against it automatically. Now your harness catches quality drops caused by
the source content moving underneath you, not just by your own code. Very few
portfolio projects demonstrate that, and it's the failure mode real production
RAG systems actually suffer from.

## Golden dataset

Target 150–300 records:

```json
{
  "id": "q_0142",
  "question": "When do I need to register for Self Assessment?",
  "ground_truth": "...",
  "source_ids": ["gov-uk/register-for-self-assessment#chunk-3"],
  "difficulty": "multi_hop",
  "added_in": "v2"
}
```

Draft with an LLM over your chunks, then review every one by hand — that review
is what makes the numbers mean anything. Tag difficulty (`single_hop`,
`multi_hop`, `negative`) so you can report per-slice.

Include 20–30 negatives: questions the corpus genuinely cannot answer. The right
behaviour is refusal, and a system that silently invents an answer to those is
broken in the way that gets an organisation in trouble. Most golden sets skip
this. Yours shouldn't.

## Build order

1. **Ingest + retrieve.** Crawl the slice, chunk, index, and get a working
   `retrieve(question)`. No eval yet.
2. **Retrieval metrics on 30 questions.** Hand-write hit@k and MRR. Small set,
   fast loop, no LLM cost. You'll find retrieval bugs here.
3. **Golden set to full size.** The slowest step. Budget real hours.
4. **Generation + RAGAS.** Add answer synthesis and the judge metrics. Run the
   same config three times and record the spread — that measured variance is
   what justifies your tolerance bands, and it's the single best thing to have a
   number for when someone asks how you set the thresholds.
5. **CI gate.** Wire up `rag-eval.yml` and `compare.py`. Commit a baseline.
6. **The regression demos.** Three PRs, each blocked, each screenshotted:
   chunk 512 → 2000; embedding model swapped for a weaker one; top_k 5 → 2.
7. **Experiment benchmark.** Run all four configs, publish the comparison table
   in the README with the winner and the reasoning.
8. **Corpus drift workflow.** The weekly re-crawl.

Steps 1–6 make the claim on your CV true. Steps 7–8 are what make an interviewer
stop skim-reading.

## Two things that will bite

**Judge variance.** LLM-graded metrics move 3–5% between identical runs. If you
gate tightly on them you get a pipeline that fails at random, and anyone
technical will spot immediately that you never ran it twice. Median of three,
wide tolerance band, and let the deterministic retrieval metrics carry the gate.

**Cost.** Full RAGAS across 300 questions on every push adds up quickly.
Retrieval metrics on every PR, judge metrics nightly or behind a `full-eval`
label, and a hard cost cap in the config.
