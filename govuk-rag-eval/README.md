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
| 3 | Golden set to full size (150–300, hand-reviewed) | 🟡 **65 reviewed records** (41 answerable, 24 negatives). 134 further candidates are drafted and screened — 112 auto-approved, 22 awaiting a human — which would take the set to ~199, inside the target. Promotion is wired: dispatch `Promote golden set`. |
| 4 | **Generation + judge metrics** (local NLI by default; RAGAS behind a label) | ✅ done — grader swapped to a free deterministic one, cross-run spread measured, floors and tolerances re-derived for it (table below) |
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

The default grader is **local NLI entailment** (`cross-encoder/nli-deberta-v3-small`),
not an LLM. It needs no API key, costs nothing to grade, and returns the same
number every time for the same answers. RAGAS is still wired and still runs,
behind a `ragas-judge` label on the PR.

Baseline from
[run 34619048854](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34619048854)
on commit `ae7dfa2`, golden set v2, 41 answerable questions.

| Metric | NLI | Floor | Tolerance | Cross-run spread |
| --- | ---: | ---: | ---: | ---: |
| `faithfulness` | 0.689 | 0.60 | 7% | 3.41% |
| `answer_relevancy` | 0.839 | 0.78 | 3% | 1.11% |
| `context_precision` | 0.809 | 0.70 | 2% | 0.04% |
| `answer_correctness` | 0.787 | — | ungated | 2.72% |

**These numbers are not comparable to the RAGAS ones they replace.** The same
unregressed system scores `faithfulness` 0.950 under RAGAS and 0.689 under NLI —
entailment and an LLM's judgement measure similar ideas on different scales.
Results therefore record `judge_backend`, and `compare.py` refuses to gate the
judge suite when the baseline used a different grader, reporting instead. That
guard is what stopped the switch from being reported as a 26-point regression.
Retrieval gates either way, being grader-independent.

The previous RAGAS numbers, for the record: faithfulness 0.950, answer_relevancy
0.899, context_precision 0.820, answer_correctness 0.831, median of 3 runs at
~$3.69 a run
([run 34505214155](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34505214155)).

#### Why the grader changed

Cost was the smaller reason. The real one is that **a gate wants
reproducibility more than it wants absolute accuracy.** RAGAS `faithfulness`
moved 2.5% between two identical 3-run measurements (0.975 → 0.950), which is
why its band had to be 8% wide — and an 8% band cannot see a 3% regression.
The local grader removes the *grader* half of that noise entirely. Generation
noise remains (3.41% on faithfulness), so the band is 7% rather than 8% — a
smaller win than "deterministic grader" suggests, and worth stating plainly.
What it does buy is that the grader contributes nothing, so a move is either a
real change or the generator, never the scorer disagreeing with itself. Grading also went from $3.69 a run to
$0, which is why judge metrics now run on **every** PR instead of nightly.

#### How the tolerances were calibrated, and what nearly went wrong

Two traps here, both worth stating because both are easy to fall into.

**The grader is deterministic; the generator is not.** `generation.temperature: 0.0`
is honoured by OpenAI and ignored by the Anthropic models in use, so answers
differ between runs and the judge metrics move with them. Setting a ~1% band on
"deterministic grader ⇒ zero variance" would have produced a gate that fails
constantly. So the bands come from repeated **whole runs** on one commit
(34618486278 and 34619048854 on `ae7dfa2`, plus 34616914158 and 34620099721,
which differ only in cost reporting and in config/docs — neither touches the
generator or the grader). Band = 2–3× the widest observed spread, never below 2%.

**Three samples was not enough, and this is worth keeping visible.**
`faithfulness`'s band was set to 5% on the first three runs, which spread 2.32%.
The fourth run returned 0.712 — above the entire prior range — taking the spread
to 3.41% and 5%'s headroom to 1.47×, under the rule just stated. It was widened
to 7% before merging rather than after the first spurious red build. This metric
is the noisiest of the four and its band should be revisited as runs accumulate.

**The in-run `judge_spread` cannot see this noise.** `--runs N` repeats the
*grader* over answers generated once, so under a deterministic grader it reports
0.0000 by construction — which reads like "no noise, tighten freely" and is
wrong. `runs` is now 1 (raising it buys nothing but grading time) and the CI log
says explicitly what that zero does and does not mean.

One tolerance moved the *wrong* way as a result: **`answer_relevancy` loosened
from 2% to 3%.** Its 2% band was calibrated against RAGAS's 0.14% noise; under
this grader the generation-driven spread is 1.11%, eight times larger, leaving
2% with only 1.8× headroom. `context_precision` tightened hard, 5% → 2%, on a
spread of 0.04%. `faithfulness` barely moved at all, 8% → 7%.

The floors moved too, and had to: `faithfulness`'s floor of 0.85 was a
RAGAS-scale number that the NLI grader cannot reach on a healthy system, so
carrying it across would have failed the first green run on `main`.
`answer_relevancy`'s floor of 0.80 sat only 4.6% under the observed 0.839 —
closer than the band above it, so the floor would have fired first, inverting
the intended band-then-backstop order. A test now promotes the committed
baseline, gates it against itself and requires a pass, so floors and baseline
can no longer drift apart silently.

`answer_correctness` stays ungated, for the same corrected reason as before: not
noise (2.71% is ordinary here) but that it scores answers against one
hand-written `ground_truth` string, measuring agreement with a phrasing rather
than correctness.

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

### Screening candidates without an API key

`scripts/build_golden_set.py evaluate` defaults to `--backend offline`:
deterministic checks, no key, no spend, no run-to-run variance.

```bash
# free, deterministic, ~1 second for 134 candidates
python scripts/build_golden_set.py evaluate --queue data/golden/review_queue.jsonl

# ask a model as well — slower, costs money, varies between runs
python scripts/build_golden_set.py evaluate --backend llm
```

Measured on the same 134 candidates with full passages
([offline](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34613876697)
vs [LLM](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34610387397)):

| | approve | review | time | cost |
| --- | ---: | ---: | ---: | ---: |
| `offline` | 106 | 28 | **< 1 s** | **$0.00** |
| `llm` | 113 | 21 | 6m 41s | $1.34 |

**They agree on 117 of 134 (87%).** Of the 17 disagreements, 12 are offline
being more cautious — which costs review time and nothing else. Five go the
other way and are the real price: 2 source-support problems the model caught and
the lexical check missed, 2 phrasing calls, 1 the model was unsure about.

Why offline is the default despite that:

- Across two paid screens the model produced **zero rejections** and drove **8
  of 134 decisions** independently. The rest were already decided by
  deterministic checks. Candidates are drafted *from* their passages, so
  relevance and ground-truth accuracy are near-guaranteed by construction — the
  criteria a model is needed for are the ones that cannot fail here.
- What *does* go wrong is mechanical, and code is better at mechanical: on
  self-referential phrasing the model caught 10 of 13, a regex caught 13 of 13.
- **Reproducibility.** The judge's faithfulness median moved 2.5% between two
  identical runs, which is why its tolerance sits at 8%. This screen has zero
  variance, so a change in its output always means the candidates changed.

The sharpest deterministic check is figure support: every money amount,
percentage, date, year and form code asserted by an answer must appear in the
passage, compared by value so `£1,000` matches `£1000`. A drafted answer citing
a number the passage does not contain is wrong in a way that needs no judgement
to see.

The sensible middle, if you want both: screen offline per batch, and run
`--backend llm` once before promoting.

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
