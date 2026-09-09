# Golden set changelog

The golden set is versioned. `compare.py` only gates a PR against a baseline of
the **same** version — changing the dataset changes what the numbers mean, so a
version bump makes the gate report-only until a new baseline is committed.

Bump `dataset.version` in `configs/eval_config.yaml` on every change here, and
record what changed and why below.

## v1 — starter (build step 1)
- 3 seed records tied to the fixture pages, to shape the schema and exercise
  retrieval end-to-end. Includes one `negative` (unanswerable) question.
- **Not the real golden set.** Build step 3 grows this to 150–300 hand-reviewed
  records (drafted with an LLM over the chunks, then reviewed one by one),
  tagged `single_hop` / `multi_hop` / `negative`, with 20–30 negatives.
