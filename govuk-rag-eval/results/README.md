# results/

- **`baseline.json`** — the reference metrics on `main` that a PR is gated
  against. It is **populated and the gate is live**: golden set v2, retrieval
  plus the judge suite, measured on commit `ae7dfa2`
  ([run 34619048854](https://github.com/joedeasy10-max/Agent-builds/actions/runs/34619048854)).
  (This file previously said the baseline "currently holds `{}`". It has not
  since PR #24; the sentence outlived the fact.)
- Other files here (`current.json`, `report.md`) are produced per-run by the CI
  workflow and are gitignored.

## What a baseline must contain

`compare.py` will not gate a metric it cannot compare like for like, and it
fails **open** — reporting instead of gating — rather than inventing a
regression. Two fields decide that, and a baseline missing either is a baseline
that quietly gates nothing:

| Field | Missing ⇒ |
| --- | --- |
| `dataset_version` | judge *and* retrieval report only — no comparison at all |
| `judge_backend` | judge metrics report only ("baseline predates grader tracking") |
| `judge_n_judged` | no guard against comparing aggregates over different numbers of questions |

`compare.BASELINE_KEYS_READ` is the authoritative list, and
`compare.baseline_from_results()` is the only thing that should build a baseline
document. They live in the same module deliberately: the projection used to be a
copy inside `rag-eval.yml`, it drifted, and it emitted judge metrics with no
`judge_backend` — producing a baseline that looked complete and gated nothing,
permanently and silently.

## Promoting a baseline

Easiest path, and the one to prefer: take it from a green run on `main`. Every
run prints the exact document to commit, between
`=== BASELINE-READY JSON ===` markers, built by `baseline_from_results()`. Copy
it into `results/baseline.json`, add a `_provenance` block naming the commit and
run URL, and commit on a branch.

It is printed to the log on purpose — artefact downloads from
`blob.core.windows.net` are blocked on some networks, which once made the
measurement effectively unreadable.

To regenerate locally instead, run **both** suites into one file. Running only
the retrieval line below gives a baseline with no judge block, which disables
judge gating without saying so:

```bash
cd govuk-rag-eval
python -m src.ingest   --config configs/retrieval.yaml --out .index/
python -m src.evaluate --suite retrieval \
    --dataset data/golden/questions.jsonl \
    --config configs/retrieval.yaml --index .index/ --out results/current.json
python -m src.evaluate --suite judge \
    --dataset data/golden/questions.jsonl \
    --config configs/retrieval.yaml --index .index/ \
    --judge-backend nli --runs 1 --merge-into results/current.json
python -c "import sys,json; sys.path.insert(0,'scripts'); from compare import baseline_from_results; \
    json.dump(baseline_from_results(json.load(open('results/current.json'))), \
              open('results/baseline.json','w'), indent=2, sort_keys=True)"
```

Grading is free; **generation is not** — the judge scores answers, so it has to
produce them first, and that uses `generation.provider`. Budget ~$0.12 a run at
41 answerable questions.

## Promoting candidates into a new dataset version

Use the **Promote golden set** workflow (`.github/workflows/promote-golden-set.yml`),
dispatched with the run id of the `Evaluate candidates` run whose screened queue
you want. It runs in Actions rather than locally because the screened queue only
exists as an artefact of that run, and artefact downloads redirect to
`blob.core.windows.net`, which some networks block with a 403.

It leaves `dry_run` on by default: the first dispatch prints what it would
promote and changes nothing. Turn it off to have it push a branch and open a PR.

Three things it does that are easy to get wrong by hand:

- **Refuses a version that already exists** in the golden set or in the config.
- **Bumps `dataset.version` in the same commit as the data.** `promote` only
  prints a reminder to do this, and a printed reminder is not a mechanism.
  `tests/test_golden.py` fails in both directions — records stamped `added_in:
  v3` while the config says `v2`, or a config on `v3` with nothing promoted into
  it — so the bump cannot be forgotten or arrive early.
- **Warns if the new set exceeds `cost.max_judge_questions_per_run`**, which
  would otherwise truncate every future judge run.

It deliberately does **not** update `results/baseline.json`. A baseline is only
valid within one dataset version, so the PR it opens is expected to show the
judge metrics reporting rather than gating until a fresh baseline is measured on
that branch and committed to it. The PR body spells out those steps.

## Changing the grader

Switching `--judge-backend` invalidates the judge half of the baseline and its
floors together. NLI `faithfulness` reads 0.689 where RAGAS reads 0.950 on the
same unregressed system, so the old floor of 0.85 would fail the first green run
on `main`. Re-measure the baseline **and** re-derive the floors and tolerances in
`configs/eval_config.yaml` in the same PR;
`tests/test_compare.py::test_the_committed_baseline_passes_its_own_gate` catches
the case where only one of the two moves.

Tolerances need repeated **whole runs** on one commit, not `--runs N`. `--runs N`
repeats the grader over answers generated once, so with a deterministic grader
the in-run spread is zero by construction and tells you nothing about the noise
the gate actually sees. See `_provenance.cross_run_spread` in `baseline.json`.

Regenerate the baseline whenever `configs/retrieval.yaml` changes on `main` in a
way that intentionally moves the metrics, or when the golden-set version bumps
(the old baseline is not comparable across dataset versions). Explain the change
in the PR description — never lower a threshold in `configs/eval_config.yaml`
just to make a build pass.

That last rule is worth separating from what this PR did, since they look alike.
Re-deriving floors for a **different grader**, from a measurement, in the same PR
that swaps the grader, is calibration: the old numbers describe a scale that is
no longer in use. Widening a band on the **same** grader because a build went red
is the thing the rule forbids. The test to apply is whether you can point at the
measurement that produced the number — `_provenance.cross_run_spread` and the run
URLs behind it — or only at the failing build.
