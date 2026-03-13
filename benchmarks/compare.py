#!/usr/bin/env python3
"""Compare two benchmark result files and print a delta table.

Useful for measuring the impact of optimization work: run benchmarks/run.py
before and after your changes, then compare the two result files.

Usage:
    python benchmarks/compare.py benchmarks/results/baseline.json benchmarks/results/new.json

Output columns:
    <metric>: new_mean_ms (Δ% vs baseline)
    Positive Δ% = slower, negative Δ% = faster.
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_delta(base_s: float, new_s: float) -> str:
    pct = (new_s / base_s - 1.0) * 100
    sign = "+" if pct >= 0 else ""
    return f"{new_s * 1000:.1f} ms ({sign}{pct:.0f}%)"


def _compare_section(
    baseline_results: list,
    new_results: list,
    metrics: list[str],
    section_title: str,
) -> None:
    if not baseline_results and not new_results:
        return
    base_by_label = {r["label"]: r for r in baseline_results}
    new_by_label = {r["label"]: r for r in new_results}
    common_labels = sorted(set(base_by_label) & set(new_by_label))
    if not common_labels:
        return

    print(f"\n### {section_title}\n")
    cols = ["input"] + metrics
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")

    for label in common_labels:
        base_r = base_by_label[label]
        new_r = new_by_label[label]
        row = [label]
        for m in metrics:
            bt = base_r["timings"].get(m, {})
            nt = new_r["timings"].get(m, {})
            if "mean_s" in bt and "mean_s" in nt:
                row.append(fmt_delta(bt["mean_s"], nt["mean_s"]))
            elif "skipped" in nt:
                row.append(nt.get("skipped", "skipped"))
            elif "error" in nt:
                row.append("error")
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")


def compare(baseline: dict, new: dict) -> None:
    print(f"Baseline : {baseline['timestamp']}")
    print(f"New      : {new['timestamp']}")

    _compare_section(
        baseline.get("static_results", []),
        new.get("static_results", []),
        baseline.get("static_metrics", []),
        "Static structure metrics",
    )
    _compare_section(
        baseline.get("trajectory_results", []),
        new.get("trajectory_results", []),
        baseline.get("trajectory_metrics", []),
        "Trajectory metrics",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two BindingMetrics benchmark result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("baseline", type=Path, help="Baseline JSON result file")
    parser.add_argument("new", type=Path, help="New JSON result file to compare against baseline")
    args = parser.parse_args()

    compare(load(args.baseline), load(args.new))


if __name__ == "__main__":
    main()
