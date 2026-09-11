# Review page

`review_page.html` is the published candidate-review UI (an Artifact). It is
kept here so the UI is version-controlled rather than living only as a published
page, and so `decision_rules.js` has a home that tests can reach.

* **`decision_rules.js`** — the decision-rule *interpreter*, embedded verbatim in
  the page. The rules **table** is not here: it is exported from
  `src/candidate_review.py` (`DECISION_RULES` → `rules_table_json()`) and
  embedded into the page as the `#rules` payload, so thresholds, ordering and
  reasons exist in exactly one place.
* **`tests/test_rules_parity.py`** runs this interpreter against the Python
  implementation over the full cross-product of inputs (896 combinations) and
  fails on any disagreement. That test is what makes two implementations of the
  same logic safe to keep.

The page holds no API key. Automated evaluation runs through the Artifact
`sample` capability, which asks Claude with the *viewer's* credentials — so
nothing secret is ever present in the HTML. Where that capability is
unavailable, the button disables itself and points at the CLI instead.

Verdicts and decisions persist through the `db` capability, with `localStorage`
as a per-viewer fallback.

Regenerating the page after changing the rules or the candidate set means
re-embedding both payloads (`#rules`, `#data`) and republishing to the same URL.
