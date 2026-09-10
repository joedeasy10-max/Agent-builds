"""Benchmark several retrieval configs against one corpus + golden set.

Build step 7: run each named config in configs/experiments/ over the SAME corpus
and golden set, score the deterministic retrieval suite, and emit a comparison
table (the winner is the config with the best primary metric). The corpus is
held fixed across experiments — only the retrieval config varies — so a
difference in the table is caused by the config change, which is the whole point.

A config whose retriever/embedder can't run in this environment (e.g. `hybrid`
retriever not implemented yet, or an `openai`/`bge` embedder with no key) is
reported as skipped with the reason, not a crash.

CLI:
    python scripts/run_experiments.py \
        --dataset data/golden/questions.jsonl \
        --configs configs/experiments/*.yaml \
        --primary-metric mrr \
        --out results/experiments

`run_experiments(...)` is the seam tests use — it takes pages + records directly,
so it never touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run as a plain script (python scripts/run_experiments.py): put the repo root
# on the path so `src` imports resolve regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import Page
from src.config import Config, load_config
from src.evaluate import _RETRIEVAL_METRICS, run_retrieval_suite
from src.golden import GoldenRecord, dataset_version, load_golden
from src.ingest import build_index, gather_pages

# Derived from the suite rather than restated. This tuple had already drifted:
# it still listed three metrics after a fourth was added, so the benchmark would
# have silently omitted it from every comparison table.
_METRICS = tuple(name for name, _ in _RETRIEVAL_METRICS)


def run_experiments(
    named_configs: dict[str, Config],
    pages: list[Page],
    records: list[GoldenRecord],
    primary_metric: str = "mrr",
    workdir: str | Path = ".experiments",
) -> dict:
    """Run each config over the shared corpus; return rows + the winner.

    Iterates configs in sorted name order for deterministic output.
    """
    workdir = Path(workdir)
    rows: list[dict] = []
    for name in sorted(named_configs):
        config = named_configs[name]
        row: dict = {"name": name}
        try:
            store = build_index(pages, config, workdir / name)
            suite = run_retrieval_suite(records, config, workdir / name)
            row["metrics"] = suite["metrics"]
            row["n_chunks"] = len(store._chunks)
            row["n_answerable"] = suite["n_answerable"]
            row["status"] = "ok"
        except (NotImplementedError, ValueError, RuntimeError, ImportError, OSError) as exc:
            row["status"] = f"skipped: {type(exc).__name__}: {exc}"
        rows.append(row)

    ranked = [r for r in rows if r["status"] == "ok"]
    ranked.sort(key=lambda r: (-r["metrics"][primary_metric], r["name"]))
    winner = ranked[0]["name"] if ranked else None
    return {"rows": rows, "winner": winner, "primary_metric": primary_metric}


def render_table(result: dict, version: str) -> str:
    """Render the comparison table, one column per suite metric.

    The columns are derived from `_METRICS` rather than written out. The
    hand-written header had already drifted once: it listed three metrics after
    a fourth (`served_context_recall`) was added, so the benchmark's own table
    silently omitted the only metric that can see a `top_k` regression.
    """
    header = "| Config | " + " | ".join(f"`{m}`" for m in _METRICS) + " | chunks | status |"
    rule = "| --- | " + " | ".join("---:" for _ in _METRICS) + " | ---: | --- |"
    lines = [
        f"## Experiment benchmark (dataset `{version}`)",
        "",
        f"Primary metric: `{result['primary_metric']}`. "
        f"Winner: **{result['winner'] or '—'}**.",
        "",
        header,
        rule,
    ]
    for row in result["rows"]:
        name = row["name"]
        mark = " 🏆" if name == result["winner"] else ""
        if row["status"] == "ok":
            m = row["metrics"]
            cells = " | ".join(f"{m[metric]:.3f}" for metric in _METRICS)
            lines.append(f"| `{name}`{mark} | {cells} | {row['n_chunks']} | ok |")
        else:
            blanks = " | ".join("—" for _ in _METRICS)
            lines.append(f"| `{name}` | {blanks} | — | {row['status']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval configs.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-metric", default="mrr", choices=_METRICS)
    parser.add_argument("--out", type=Path, default=Path("results/experiments"))
    parser.add_argument("--workdir", type=Path, default=Path(".experiments"))
    args = parser.parse_args(argv)

    # A config that cannot even be parsed must be reported, not fatal. The
    # module contract is that an experiment which can't run in this environment
    # is skipped with a reason — but loading happened outside that guard, so a
    # config declaring a not-yet-implemented section (`reranker:`) took the
    # whole benchmark down instead of skipping one row.
    named_configs: dict[str, Config] = {}
    unloadable: dict[str, str] = {}
    for path in args.configs:
        try:
            named_configs[path.stem] = load_config(path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            unloadable[path.stem] = f"skipped: {type(exc).__name__}: {exc}"

    if not named_configs:
        for name, reason in sorted(unloadable.items()):
            print(f"{name}: {reason}", file=sys.stderr)
        print("no runnable experiment configs", file=sys.stderr)
        return 1

    records = load_golden(args.dataset)
    version = dataset_version(records)
    # Corpus is shared across experiments — load it once from the first config.
    pages = gather_pages(next(iter(named_configs.values())))

    result = run_experiments(named_configs, pages, records, args.primary_metric, args.workdir)
    for name, reason in unloadable.items():
        result["rows"].append({"name": name, "status": reason})
    result["rows"].sort(key=lambda r: r["name"])

    args.out.mkdir(parents=True, exist_ok=True)
    for row in result["rows"]:
        (args.out / f"{row['name']}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    table = render_table(result, version)
    (args.out / "table.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
