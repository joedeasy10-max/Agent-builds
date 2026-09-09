# results/

- **`baseline.json`** — the reference metrics on `main` that a PR is gated
  against. It currently holds `{}` (an intentional placeholder): with no
  baseline, `scripts/compare.py` runs in **reporting-only** mode and the gate
  passes. The gate's pass/fail machinery is proven end-to-end in
  `tests/test_gate.py` (deterministic, offline).
- Other files here (`current.json`, `report.md`) are produced per-run by the CI
  workflow and are gitignored.

## Populating a real baseline (activating the gate)

The gate goes live once the real pipeline can run (build step 3 delivers the
full golden set; an `OPENAI_API_KEY` repo secret enables the real embedder and
the RAGAS judge). Then, on `main`:

```bash
cd govuk-rag-eval
python -m src.ingest   --config configs/retrieval.yaml --out .index/
python -m src.evaluate --suite retrieval \
    --dataset data/golden/questions.jsonl \
    --config configs/retrieval.yaml --index .index/ --out results/baseline.json
git add results/baseline.json && git commit -m "Set RAG eval baseline (<dataset version>)"
```

Regenerate the baseline whenever `configs/retrieval.yaml` changes on `main` in a
way that intentionally moves the metrics, or when the golden-set version bumps
(the old baseline is not comparable across dataset versions). Explain the change
in the PR description — never lower a threshold in `configs/eval_config.yaml`
just to make a build pass.
