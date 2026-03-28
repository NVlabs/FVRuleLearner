#!/usr/bin/env python3
"""Analyze QTree RAG summary CSV output.

This utility mirrors the aggregate reporting performed inside
`test_qtree_with_rag.py` without re-running any LLM inference.  It loads a
`qtree_rag_test_summary.csv` file, computes success/failure breakdowns, and can
optionally regenerate summary plots.
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "num_suggestions",
    "new_syntax",
    "new_pec",
    "new_relax_pec",
    "bleu",
    "rouge",
    "exact_match",
]


def load_summary(csv_path: Path) -> pd.DataFrame:
    """Load summary CSV and normalize column types."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize boolean column stored as strings
    df["fixed"] = (
        df.get("fixed", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "num_suggestions" in df.columns:
        df["num_suggestions"] = df["num_suggestions"].fillna(0).astype(int)

    return df


def compute_overall_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute high-level aggregate metrics."""

    total_rows = int(len(df))
    success_rows = int(df["fixed"].sum()) if total_rows else 0
    stats = {
        "total_rows": total_rows,
        "success_rows": success_rows,
        "success_rate": success_rows / total_rows if total_rows else 0.0,
        "unique_tests": int(df["test_id"].nunique()) if "test_id" in df else 0,
        "unique_designs": int(df["design_name"].nunique())
        if "design_name" in df
        else 0,
    }

    for metric in ["bleu", "rouge", "new_pec", "new_relax_pec", "new_syntax"]:
        if metric in df.columns:
            stats[f"mean_{metric}"] = float(df[metric].mean(skipna=True))

    if "num_suggestions" in df.columns:
        stats["mean_num_suggestions"] = float(df["num_suggestions"].mean())
        stats["max_num_suggestions"] = int(df["num_suggestions"].max())

    return stats


def compute_success_breakdown(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return per-value success breakdown for a categorical column."""

    if column not in df.columns:
        return pd.DataFrame(columns=[column, "total", "success", "success_rate"])

    grouped = (
        df.groupby(column)["fixed"].agg(total="size", success="sum")
    )
    grouped["success_rate"] = grouped["success"] / grouped["total"]
    grouped = grouped.sort_values(
        by=["success", "success_rate", "total"], ascending=[False, False, False]
    )
    grouped = grouped.reset_index()
    return grouped


def clean_list_field(value: object) -> str:
    """Remove trailing "+N more" markers and normalize whitespace."""

    if pd.isna(value):
        return ""
    cleaned = re.sub(r"\s*\(\+\d+\s+more\)\s*", "", str(value))
    return cleaned.strip()


def iterate_list_entries(series: Iterable, separators: Tuple[str, ...]) -> Iterable[str]:
    """Yield individual entries from delimited string fields."""

    for raw in series:
        cleaned = clean_list_field(raw)
        if not cleaned:
            continue
        items = [cleaned]
        for sep in separators:
            next_items: List[str] = []
            for item in items:
                next_items.extend(item.split(sep))
            items = next_items
        for item in items:
            token = item.strip()
            if token:
                yield token


def aggregate_list_field(
    df: pd.DataFrame, column: str, separators: Tuple[str, ...]
) -> pd.DataFrame:
    """Aggregate total/success counts for list-like columns."""

    if column not in df.columns:
        return pd.DataFrame(columns=[column, "total", "success", "success_rate"])

    stats_map: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})

    for value, fixed in zip(df[column], df["fixed"]):
        entries = list(iterate_list_entries([value], separators))
        if not entries:
            continue
        for entry in entries:
            stats_map[entry]["total"] += 1
            if fixed:
                stats_map[entry]["success"] += 1

    rows = []
    for name, counts in stats_map.items():
        total = counts["total"]
        success = counts["success"]
        success_rate = success / total if total else np.nan
        rows.append((name, total, success, success_rate))

    result = pd.DataFrame(rows, columns=[column, "total", "success", "success_rate"])
    if not result.empty:
        result = result.sort_values(
            by=["success", "success_rate", "total"], ascending=[False, False, False]
        )
    return result


def build_report(
    stats: Dict[str, float],
    breakdowns: Dict[str, pd.DataFrame],
    top_sources: pd.DataFrame,
    top_branches: pd.DataFrame,
    top_limit: int,
) -> str:
    """Construct a human-readable text report."""

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("QTree RAG Summary Analysis")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Overall Metrics")
    lines.append("- Total rows:          {total_rows}".format(**stats))
    lines.append("- Successful rows:     {success_rows}".format(**stats))
    lines.append(
        "- Success rate:        {0:.2%}".format(stats.get("success_rate", 0.0))
    )
    lines.append("- Unique test IDs:     {unique_tests}".format(**stats))
    lines.append("- Unique designs:      {unique_designs}".format(**stats))

    if "mean_num_suggestions" in stats:
        lines.append(
            "- Avg. num suggestions: {0:.2f}".format(stats["mean_num_suggestions"])
        )
        lines.append("- Max num suggestions: {0}".format(stats.get("max_num_suggestions", 0)))

    for metric in ["bleu", "rouge", "new_pec", "new_relax_pec", "new_syntax"]:
        key = f"mean_{metric}"
        if key in stats:
            lines.append(f"- Mean {metric}:         {stats[key]:.3f}")

    lines.append("")

    for label, df_section in breakdowns.items():
        if df_section.empty:
            continue
        lines.append("=" * 40)
        lines.append(label)
        lines.append("=" * 40)
        lines.append(
            df_section.head(top_limit).to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        lines.append("")

    if not top_sources.empty:
        lines.append("Top Source QTrees")
        lines.append("=" * 40)
        lines.append(
            top_sources.head(top_limit).to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        lines.append("")

    if not top_branches.empty:
        lines.append("Top Branch Patterns")
        lines.append("=" * 40)
        lines.append(
            top_branches.head(top_limit).to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        lines.append("")

    return "\n".join(lines)


def ensure_output_dir(path: Path) -> Path:
    if path is None:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_report(report: str, output_dir: Path, filename: str) -> Path:
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    report_path = output_dir / filename
    report_path.write_text(report)
    return report_path


def generate_plots(df: pd.DataFrame, output_dir: Path, top_sources: pd.DataFrame) -> List[Path]:
    """Generate key plots; returns list of generated file paths."""

    generated: List[Path] = []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return generated

    if output_dir is None:
        print("No output directory specified; skipping plots.")
        return generated

    # Success rate by number of suggestions
    if "num_suggestions" in df.columns:
        by_suggestions = compute_success_breakdown(df, "num_suggestions")
        if not by_suggestions.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(by_suggestions["num_suggestions"], by_suggestions["success_rate"], color="steelblue")
            ax.set_xlabel("Number of Suggestions")
            ax.set_ylabel("Success Rate")
            ax.set_title("Success Rate by Suggestions Used")
            ax.set_ylim(0, 1.05)
            for _, row in by_suggestions.iterrows():
                ax.text(
                    row["num_suggestions"],
                    row["success_rate"] + 0.02,
                    f"n={int(row['total'])}",
                    ha="center",
                    fontsize=8,
                )
            path = output_dir / "plot_success_rate_by_suggestions.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(path)

    # Success rate by rule type
    if "rule_types" in df.columns:
        by_rule_type = compute_success_breakdown(df, "rule_types")
        if not by_rule_type.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(by_rule_type["rule_types"], by_rule_type["success_rate"], color="seagreen")
            ax.set_xlabel("Rule Type")
            ax.set_ylabel("Success Rate")
            ax.set_title("Success Rate by Rule Type")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=30)
            path = output_dir / "plot_success_rate_by_rule_type.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(path)

    # Success vs total usage for top sources
    if not top_sources.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        display_df = top_sources.head(20)
        indices = np.arange(len(display_df))
        width = 0.35

        totals = display_df["total"].to_numpy()
        successes = display_df["success"].to_numpy()

        bars_total = ax.bar(indices - width / 2, totals, width, label="Total uses", color="lightgray")
        bars_success = ax.bar(indices + width / 2, successes, width, label="Successful uses", color="slategray")

        ax.set_xlabel("Source QTree")
        ax.set_ylabel("Count")
        ax.set_title("Top Source QTrees: Total vs Successful Uses")
        ax.set_xticks(indices)
        ax.set_xticklabels(display_df["source_qtrees"], rotation=45, ha="right")
        ax.legend()

        # Annotate bars with counts for clarity
        for bar in list(bars_total) + list(bars_success):
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        path = output_dir / "plot_top_source_qtrees.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    return generated


def main() -> None:
    # parser = argparse.ArgumentParser(description="Analyze qtree RAG summary CSV")
    # args = parser.parse_args()

    # df = load_summary(args.csv)
    # output_dir = Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-25T12-52-04.970384_pdx-container-xterm-064.prd.it.nvidia.com_liwan")
    output_dir = Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-25T09-17-51.824162_pdx-container-xterm-064.prd.it.nvidia.com_liwan")
    df = load_summary(os.path.join(output_dir, "qtree_rag_test_summary.csv"))
    report_name = "qtree_summary_analysis_v2.txt"
    plots = True
    top_k = 20


    stats = compute_overall_stats(df)

    breakdowns: Dict[str, pd.DataFrame] = {}
    breakdowns["Success by Number of Suggestions"] = compute_success_breakdown(
        df, "num_suggestions"
    )
    breakdowns["Success by Rule Type"] = compute_success_breakdown(df, "rule_types")
    breakdowns["Success by Design Name"] = compute_success_breakdown(df, "design_name")
    breakdowns["Success by Test ID"] = compute_success_breakdown(df, "test_id")

    top_sources = aggregate_list_field(df, "source_qtrees", separators=(",",))

    top_branches = aggregate_list_field(df, "branches", separators=("|", ","))

    report = build_report(stats, breakdowns, top_sources, top_branches, top_k)

    print(report)

    output_dir_path = ensure_output_dir(output_dir)
    if output_dir_path is not None:
        report_path = save_report(report, output_dir_path, report_name)
        if report_path:
            print(f"Report written to {report_path}")

    if plots:
        if output_dir_path is None:
            print("Cannot generate plots without --output-dir")
        else:
            generated = generate_plots(df, output_dir_path, top_sources)
            for path in generated:
                print(f"Plot saved: {path}")


if __name__ == "__main__":
    main()

