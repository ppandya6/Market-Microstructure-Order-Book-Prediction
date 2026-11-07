"""Write a Markdown research readout from a generated artifact."""

import argparse
from pathlib import Path

import pandas as pd

from orderbook.execution import simulate, summary
from orderbook.research import load_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="artifacts/research.json.gz")
    parser.add_argument("--output", default="artifacts/readout.md")
    args = parser.parse_args()
    a = load_artifact(args.artifact)
    lines = [
        "# Research readout",
        "",
        f"Source: **{a['provenance']['kind']}** — {a['provenance']['session']}",
        f"Generated: {a['created_utc']}",
        "",
        "Exploratory results; synthetic data is not market evidence.",
        "",
        "| Horizon | Model | Validation log loss | Test log loss | Balanced accuracy | Net USD* | Trades* |",
        "|---|---|---|---|---|---|---|",
    ]
    book = pd.DataFrame(a["test_book"])
    for h, r in a["results"].items():
        for name, m in r["models"].items():
            s = summary(simulate(book, r["test_indices"], m["probabilities"], int(h), 1, 0.5, 0.6))
            lines.append(
                f"| {h} | {name} | {m['validation']['log_loss']:.4f} | {m['test']['log_loss']:.4f} | {m['test']['balanced_accuracy']:.1%} | {s['net']:.2f} | {s['trades']} |"
            )
    lines += [
        "",
        "*Execution: 1 event delay, 0.5 bps fee each side, 60% directional threshold, one share, nonoverlapping trades. These settings are illustrative, not selected for performance.",
        "",
        "Reserve additional dates for confirmatory evaluation. Quote-based execution omits market impact, borrow constraints, and passive fills.",
    ]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
