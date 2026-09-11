"""The page's JS matcher must agree with Python on every possible input.

The review page cannot import Python, so `web/decision_rules.js` transliterates
`_rule_matches` / `decide` from src/candidate_review.py. Two implementations of
the same logic is exactly the drift that has already produced four bugs in this
project, so it is pinned here rather than trusted: the rules TABLE is exported
from Python (one source), and this test proves the interpreter agrees over the
full cross-product of inputs.

Skipped when node is unavailable, because the Python suite must stay runnable
offline with no extra toolchain.
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.candidate_review import CRITERIA, DECISION_RULES, decide, rules_table_json

ROOT = Path(__file__).resolve().parent.parent
MATCHER = ROOT / "web" / "decision_rules.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for the JS parity check"
)


def _python_grid() -> list[dict]:
    rows = []
    for combo in itertools.product(["pass", "fail"], repeat=len(CRITERIA)):
        crit = dict(zip(CRITERIA, combo))
        for conf in (0.0, 0.5, 0.69, 0.70, 0.79, 0.80, 1.0):
            for dup in (False, True):
                for neg in (False, True):
                    for has_passage in (False, True):
                        d, _, rule = decide(
                            crit, confidence=conf, duplicate=dup,
                            is_negative=neg, has_passage=has_passage,
                        )
                        rows.append({
                            "crit": crit, "conf": conf, "dup": dup, "neg": neg,
                            "hp": has_passage, "decision": d, "rule": rule,
                        })
    return rows


def test_js_matcher_agrees_with_python_on_every_combination(tmp_path):
    assert MATCHER.exists(), f"missing {MATCHER}"
    grid = _python_grid()
    assert len(grid) > 500, "the grid should be a real cross-product"

    (tmp_path / "rules.json").write_text(rules_table_json())
    (tmp_path / "grid.json").write_text(json.dumps(grid))
    runner = tmp_path / "run.js"
    runner.write_text(f"""
const fs = require("fs");
const {{makeDecider}} = require({str(MATCHER)!r});
const RULES = JSON.parse(fs.readFileSync({str(tmp_path / "rules.json")!r}, "utf8"));
const grid = JSON.parse(fs.readFileSync({str(tmp_path / "grid.json")!r}, "utf8"));
const decide = makeDecider(RULES);
const bad = [];
for (const row of grid) {{
  const got = decide(row.crit, row.conf, row.dup, row.neg, row.hp);
  if (got.decision !== row.decision || got.rule !== row.rule) {{
    bad.push({{row, got}});
  }}
}}
console.log(JSON.stringify({{compared: grid.length, mismatches: bad.slice(0, 5),
                            total: bad.length}}));
""")
    out = subprocess.run(
        ["node", str(runner)], capture_output=True, text=True, timeout=120, check=True
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["compared"] == len(grid)
    assert result["total"] == 0, f"JS and Python disagree: {result['mismatches']}"


def test_exported_table_matches_the_module():
    """The export the page embeds must be the table the module decides with."""
    exported = json.loads(rules_table_json())
    assert exported["criteria"] == list(CRITERIA)
    assert [r["id"] for r in exported["rules"]] == [r["id"] for r in DECISION_RULES]
    for got, want in zip(exported["rules"], DECISION_RULES):
        assert got["decision"] == want["decision"]
        assert got["when"] == want["when"]
