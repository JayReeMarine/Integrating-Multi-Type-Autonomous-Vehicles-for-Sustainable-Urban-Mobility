"""
Algorithm Comparison Module: Greedy vs Hungarian

This module generates comparison tables and visualizations between
Greedy and Hungarian algorithms with Hungarian as baseline (100%).

Usage:
    python -m visualization.compare_algorithms

    or from project root:
    python visualization/compare_algorithms.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Configuration
# ============================================================================

# Data paths
GREEDY_DIR = PROJECT_ROOT / "data" / "results" / "greedy"
HUNGARIAN_DIR = PROJECT_ROOT / "data" / "results" / "hungarian"

# Output paths
TABLE_OUTPUT_DIR = PROJECT_ROOT / "visualization" / "figures" / "comparison" / "tables"
PLOT_OUTPUT_DIR = PROJECT_ROOT / "visualization" / "figures" / "comparison" / "plots"

# Metrics to compare (Hungarian = 100% baseline)
COMPARISON_METRICS = [
    "runtime_sec",
    "total_saving",
    "matched_ratio",
    "avg_saving_per_pv",
    "saving_percent",
]

# Human-readable metric names
METRIC_LABELS = {
    "runtime_sec": "Runtime (sec)",
    "total_saving": "Total Saving",
    "matched_ratio": "Match Ratio",
    "avg_saving_per_pv": "Avg Saving/PV",
    "saving_percent": "Saving %",
}

# Scenario types
SCENARIOS = ["capacity_sweep", "length_sweep", "pv_sweep", "av_sweep"]

SCENARIO_LABELS = {
    "capacity_sweep": "Capacity Sweep",
    "length_sweep": "Length Sweep",
    "pv_sweep": "PV Sweep (Fixed AV)",
    "av_sweep": "AV Sweep (Fixed PV)",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_algorithm_data(algorithm: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files for a given algorithm."""
    data_dir = GREEDY_DIR if algorithm == "greedy" else HUNGARIAN_DIR

    data = {}
    for scenario in ["capacity_sweep", "length_sweep", "pv_av_sweep"]:
        csv_path = data_dir / f"{scenario}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[scenario] = df
        else:
            print(f"Warning: {csv_path} not found")

    return data


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load data for both algorithms."""
    greedy_data = load_algorithm_data("greedy")
    hungarian_data = load_algorithm_data("hungarian")
    return greedy_data, hungarian_data


# ============================================================================
# Data Processing
# ============================================================================

def aggregate_by_scenario(
    df: pd.DataFrame,
    scenario_type: str,
    group_cols: List[str]
) -> pd.DataFrame:
    """Aggregate data by scenario parameters, averaging across seeds."""

    scenario_df = df[df["scenario_type"] == scenario_type].copy()
    if scenario_df.empty:
        return pd.DataFrame()

    # Group and aggregate
    agg_df = scenario_df.groupby(group_cols, as_index=False).agg({
        metric: ["mean", "std"] for metric in COMPARISON_METRICS
    })

    # Flatten column names
    agg_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg_df.columns
    ]

    return agg_df


def compute_relative_performance(
    greedy_df: pd.DataFrame,
    hungarian_df: pd.DataFrame,
    key_cols: List[str]
) -> pd.DataFrame:
    """
    Compute Greedy performance relative to Hungarian (baseline = 100%).

    Returns DataFrame with:
    - Greedy absolute values
    - Hungarian absolute values
    - Relative performance (Greedy/Hungarian * 100)
    """

    if greedy_df.empty or hungarian_df.empty:
        return pd.DataFrame()

    # Merge on key columns
    merged = pd.merge(
        greedy_df,
        hungarian_df,
        on=key_cols,
        suffixes=("_greedy", "_hungarian")
    )

    # Compute relative performance for each metric
    result = merged[key_cols].copy()

    for metric in COMPARISON_METRICS:
        greedy_col = f"{metric}_mean_greedy"
        hungarian_col = f"{metric}_mean_hungarian"

        if greedy_col in merged.columns and hungarian_col in merged.columns:
            result[f"{metric}_greedy"] = merged[greedy_col]
            result[f"{metric}_hungarian"] = merged[hungarian_col]

            # Relative performance (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                relative = np.where(
                    merged[hungarian_col] != 0,
                    (merged[greedy_col] / merged[hungarian_col]) * 100,
                    np.nan
                )
            result[f"{metric}_relative"] = relative

    return result


# ============================================================================
# Comparison Table Generation
# ============================================================================

def generate_capacity_sweep_comparison(
    greedy_data: Dict[str, pd.DataFrame],
    hungarian_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Generate comparison table for capacity sweep scenario."""

    greedy_df = greedy_data.get("capacity_sweep", pd.DataFrame())
    hungarian_df = hungarian_data.get("capacity_sweep", pd.DataFrame())

    if greedy_df.empty or hungarian_df.empty:
        print("No capacity sweep data available")
        return pd.DataFrame()

    # Aggregate by capacity_max
    greedy_agg = aggregate_by_scenario(
        greedy_df, "capacity_sweep", ["capacity_max"]
    )
    hungarian_agg = aggregate_by_scenario(
        hungarian_df, "capacity_sweep", ["capacity_max"]
    )

    return compute_relative_performance(
        greedy_agg, hungarian_agg, ["capacity_max"]
    )


def generate_length_sweep_comparison(
    greedy_data: Dict[str, pd.DataFrame],
    hungarian_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Generate comparison table for length sweep scenario."""

    greedy_df = greedy_data.get("length_sweep", pd.DataFrame())
    hungarian_df = hungarian_data.get("length_sweep", pd.DataFrame())

    if greedy_df.empty or hungarian_df.empty:
        print("No length sweep data available")
        return pd.DataFrame()

    # Aggregate by highway_length
    greedy_agg = aggregate_by_scenario(
        greedy_df, "length_sweep", ["highway_length"]
    )
    hungarian_agg = aggregate_by_scenario(
        hungarian_df, "length_sweep", ["highway_length"]
    )

    return compute_relative_performance(
        greedy_agg, hungarian_agg, ["highway_length"]
    )


def generate_pv_sweep_comparison(
    greedy_data: Dict[str, pd.DataFrame],
    hungarian_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Generate comparison table for PV sweep scenario (fixed AV, varying PV)."""

    greedy_df = greedy_data.get("pv_av_sweep", pd.DataFrame())
    hungarian_df = hungarian_data.get("pv_av_sweep", pd.DataFrame())

    if greedy_df.empty or hungarian_df.empty:
        print("No PV/AV sweep data available")
        return pd.DataFrame()

    # Filter for pv_sweep and aggregate by num_av and num_pv
    greedy_agg = aggregate_by_scenario(
        greedy_df, "pv_sweep", ["num_av", "num_pv"]
    )
    hungarian_agg = aggregate_by_scenario(
        hungarian_df, "pv_sweep", ["num_av", "num_pv"]
    )

    return compute_relative_performance(
        greedy_agg, hungarian_agg, ["num_av", "num_pv"]
    )


def generate_av_sweep_comparison(
    greedy_data: Dict[str, pd.DataFrame],
    hungarian_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Generate comparison table for AV sweep scenario (fixed PV, varying AV)."""

    greedy_df = greedy_data.get("pv_av_sweep", pd.DataFrame())
    hungarian_df = hungarian_data.get("pv_av_sweep", pd.DataFrame())

    if greedy_df.empty or hungarian_df.empty:
        print("No PV/AV sweep data available")
        return pd.DataFrame()

    # Filter for av_sweep and aggregate by num_av and num_pv
    greedy_agg = aggregate_by_scenario(
        greedy_df, "av_sweep", ["num_av", "num_pv"]
    )
    hungarian_agg = aggregate_by_scenario(
        hungarian_df, "av_sweep", ["num_av", "num_pv"]
    )

    return compute_relative_performance(
        greedy_agg, hungarian_agg, ["num_av", "num_pv"]
    )


# ============================================================================
# Table Formatting & Output
# ============================================================================

def format_comparison_table(
    df: pd.DataFrame,
    scenario_name: str,
    key_col: str
) -> str:
    """Format comparison DataFrame as a readable markdown table."""

    if df.empty:
        return f"No data available for {scenario_name}\n"

    lines = []
    lines.append(f"\n## {SCENARIO_LABELS.get(scenario_name, scenario_name)}")
    lines.append(f"\n**Hungarian = 100% (Baseline)**\n")

    # Build header
    header_parts = [key_col]
    for metric in COMPARISON_METRICS:
        label = METRIC_LABELS.get(metric, metric)
        header_parts.extend([
            f"{label} (G)",
            f"{label} (H)",
            f"{label} (%)"
        ])

    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

    # Build rows
    for _, row in df.iterrows():
        row_parts = [str(int(row[key_col]) if pd.notna(row[key_col]) else "")]

        for metric in COMPARISON_METRICS:
            greedy_val = row.get(f"{metric}_greedy", np.nan)
            hungarian_val = row.get(f"{metric}_hungarian", np.nan)
            relative_val = row.get(f"{metric}_relative", np.nan)

            # Format based on metric type
            if metric == "runtime_sec":
                g_str = f"{greedy_val:.4f}" if pd.notna(greedy_val) else "-"
                h_str = f"{hungarian_val:.4f}" if pd.notna(hungarian_val) else "-"
            elif metric == "matched_ratio":
                g_str = f"{greedy_val:.3f}" if pd.notna(greedy_val) else "-"
                h_str = f"{hungarian_val:.3f}" if pd.notna(hungarian_val) else "-"
            else:
                g_str = f"{greedy_val:.2f}" if pd.notna(greedy_val) else "-"
                h_str = f"{hungarian_val:.2f}" if pd.notna(hungarian_val) else "-"

            rel_str = f"{relative_val:.1f}%" if pd.notna(relative_val) else "-"

            row_parts.extend([g_str, h_str, rel_str])

        lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines)


def generate_summary_statistics(comparisons: Dict[str, pd.DataFrame]) -> str:
    """Generate overall summary statistics across all scenarios."""

    lines = []
    lines.append("\n# Summary Statistics")
    lines.append("\n## Average Relative Performance (Greedy vs Hungarian)")
    lines.append("\n**Interpretation**: Values < 100% mean Greedy underperforms; > 100% means Greedy outperforms\n")

    # Collect all relative values
    all_metrics = {metric: [] for metric in COMPARISON_METRICS}

    for scenario_name, df in comparisons.items():
        if df.empty:
            continue
        for metric in COMPARISON_METRICS:
            rel_col = f"{metric}_relative"
            if rel_col in df.columns:
                values = df[rel_col].dropna().tolist()
                all_metrics[metric].extend(values)

    # Compute summary
    lines.append("| Metric | Mean (%) | Std (%) | Min (%) | Max (%) |")
    lines.append("| --- | --- | --- | --- | --- |")

    for metric in COMPARISON_METRICS:
        values = all_metrics[metric]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            label = METRIC_LABELS.get(metric, metric)
            lines.append(
                f"| {label} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |"
            )

    return "\n".join(lines)


def save_comparison_tables(comparisons: Dict[str, pd.DataFrame]) -> None:
    """Save all comparison tables to files."""

    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual CSVs
    for scenario_name, df in comparisons.items():
        if not df.empty:
            csv_path = TABLE_OUTPUT_DIR / f"{scenario_name}_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    # Generate and save markdown report
    report_lines = []
    report_lines.append("# Greedy vs Hungarian Algorithm Comparison")
    report_lines.append("\n**Baseline**: Hungarian Algorithm = 100%\n")
    report_lines.append("**Interpretation**: Relative % shows Greedy performance compared to Hungarian")
    report_lines.append("- Runtime: Lower is better (< 100% means Greedy is faster)")
    report_lines.append("- Other metrics: Higher is better (> 100% means Greedy outperforms)")

    # Key columns for each scenario
    key_cols = {
        "capacity_sweep": "capacity_max",
        "length_sweep": "highway_length",
        "pv_sweep": "num_pv",  # Will need special handling
        "av_sweep": "num_av",  # Will need special handling
    }

    for scenario_name, df in comparisons.items():
        key_col = key_cols.get(scenario_name, "fixed_value")
        if scenario_name in ["pv_sweep", "av_sweep"]:
            # For these, we'll format differently
            report_lines.append(format_pv_av_comparison_table(df, scenario_name))
        else:
            report_lines.append(format_comparison_table(df, scenario_name, key_col))

    # Add summary
    report_lines.append(generate_summary_statistics(comparisons))

    # Save report
    report_path = TABLE_OUTPUT_DIR / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")


def format_pv_av_comparison_table(df: pd.DataFrame, scenario_name: str) -> str:
    """Format PV/AV sweep comparison with two key columns."""

    if df.empty:
        return f"\nNo data available for {scenario_name}\n"

    lines = []
    lines.append(f"\n## {SCENARIO_LABELS.get(scenario_name, scenario_name)}")
    lines.append(f"\n**Hungarian = 100% (Baseline)**\n")

    # Determine grouping
    if scenario_name == "pv_sweep":
        group_col, vary_col = "num_av", "num_pv"
        group_label, vary_label = "Fixed AV", "PV Count"
    else:
        group_col, vary_col = "num_pv", "num_av"
        group_label, vary_label = "Fixed PV", "AV Count"

    # Group by the fixed column
    for group_val in sorted(df[group_col].unique()):
        sub_df = df[df[group_col] == group_val].sort_values(vary_col)

        lines.append(f"\n### {group_label} = {int(group_val)}\n")

        # Header for key metrics only (simplified)
        header = [vary_label, "Match% (G)", "Match% (H)", "Rel%",
                  "Saving% (G)", "Saving% (H)", "Rel%",
                  "Runtime G", "Runtime H", "Speedup"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for _, row in sub_df.iterrows():
            vary_val = int(row[vary_col])

            # Match ratio
            mr_g = row.get("matched_ratio_greedy", np.nan)
            mr_h = row.get("matched_ratio_hungarian", np.nan)
            mr_rel = row.get("matched_ratio_relative", np.nan)

            # Saving percent
            sp_g = row.get("saving_percent_greedy", np.nan)
            sp_h = row.get("saving_percent_hungarian", np.nan)
            sp_rel = row.get("saving_percent_relative", np.nan)

            # Runtime
            rt_g = row.get("runtime_sec_greedy", np.nan)
            rt_h = row.get("runtime_sec_hungarian", np.nan)
            speedup = rt_h / rt_g if rt_g > 0 else np.nan

            row_data = [
                str(vary_val),
                f"{mr_g:.3f}" if pd.notna(mr_g) else "-",
                f"{mr_h:.3f}" if pd.notna(mr_h) else "-",
                f"{mr_rel:.1f}%" if pd.notna(mr_rel) else "-",
                f"{sp_g:.2f}" if pd.notna(sp_g) else "-",
                f"{sp_h:.2f}" if pd.notna(sp_h) else "-",
                f"{sp_rel:.1f}%" if pd.notna(sp_rel) else "-",
                f"{rt_g:.4f}" if pd.notna(rt_g) else "-",
                f"{rt_h:.2f}" if pd.notna(rt_h) else "-",
                f"{speedup:.0f}x" if pd.notna(speedup) else "-",
            ]

            lines.append("| " + " | ".join(row_data) + " |")

    return "\n".join(lines)


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    title: str,
    output_path: Path
) -> None:
    """Create a grouped bar chart comparing Greedy vs Hungarian."""

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    greedy_vals = df[f"{metric}_greedy"].values
    hungarian_vals = df[f"{metric}_hungarian"].values

    bars1 = ax.bar(x - width/2, greedy_vals, width, label='Greedy', color='#2ecc71')
    bars2 = ax.bar(x + width/2, hungarian_vals, width, label='Hungarian', color='#3498db')

    ax.set_xlabel(x_col)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in df[x_col].values])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_relative_performance(
    df: pd.DataFrame,
    x_col: str,
    metrics: List[str],
    title: str,
    output_path: Path
) -> None:
    """Plot relative performance (Greedy/Hungarian * 100) for multiple metrics."""

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = df[x_col].values

    for metric in metrics:
        rel_col = f"{metric}_relative"
        if rel_col in df.columns:
            label = METRIC_LABELS.get(metric, metric)
            ax.plot(x, df[rel_col].values, marker='o', label=label)

    # Add baseline at 100%
    ax.axhline(y=100, color='red', linestyle='--', label='Baseline (100%)')

    ax.set_xlabel(x_col)
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(comparisons: Dict[str, pd.DataFrame]) -> None:
    """Generate all comparison plots."""

    # Capacity sweep plots
    if "capacity_sweep" in comparisons and not comparisons["capacity_sweep"].empty:
        df = comparisons["capacity_sweep"]

        # Bar chart for each metric
        for metric in ["total_saving", "matched_ratio", "saving_percent"]:
            plot_comparison_bar_chart(
                df, "capacity_max", metric,
                f"Capacity Sweep: {METRIC_LABELS[metric]}",
                PLOT_OUTPUT_DIR / "capacity" / f"capacity_{metric}_comparison.png"
            )

        # Relative performance plot
        plot_relative_performance(
            df, "capacity_max",
            ["total_saving", "matched_ratio", "saving_percent"],
            "Capacity Sweep: Relative Performance (Greedy vs Hungarian)",
            PLOT_OUTPUT_DIR / "capacity" / "capacity_relative_performance.png"
        )

    # Length sweep plots
    if "length_sweep" in comparisons and not comparisons["length_sweep"].empty:
        df = comparisons["length_sweep"]

        for metric in ["total_saving", "matched_ratio", "saving_percent"]:
            plot_comparison_bar_chart(
                df, "highway_length", metric,
                f"Length Sweep: {METRIC_LABELS[metric]}",
                PLOT_OUTPUT_DIR / "length" / f"length_{metric}_comparison.png"
            )

        plot_relative_performance(
            df, "highway_length",
            ["total_saving", "matched_ratio", "saving_percent"],
            "Length Sweep: Relative Performance (Greedy vs Hungarian)",
            PLOT_OUTPUT_DIR / "length" / "length_relative_performance.png"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main function to run all comparisons."""

    print("=" * 60)
    print(" Algorithm Comparison: Greedy vs Hungarian")
    print("=" * 60)
    print()

    # Load data
    print("[1/4] Loading data...")
    greedy_data, hungarian_data = load_all_data()

    # Generate comparisons
    print("[2/4] Generating comparison tables...")
    comparisons = {
        "capacity_sweep": generate_capacity_sweep_comparison(greedy_data, hungarian_data),
        "length_sweep": generate_length_sweep_comparison(greedy_data, hungarian_data),
        "pv_sweep": generate_pv_sweep_comparison(greedy_data, hungarian_data),
        "av_sweep": generate_av_sweep_comparison(greedy_data, hungarian_data),
    }

    # Save tables
    print("[3/4] Saving comparison tables...")
    save_comparison_tables(comparisons)

    # Generate plots
    print("[4/4] Generating comparison plots...")
    generate_all_plots(comparisons)

    print()
    print("=" * 60)
    print(" Comparison complete!")
    print("=" * 60)
    print(f"\nTables saved to: {TABLE_OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")
    print("\nTo view the report:")
    print(f"  cat {TABLE_OUTPUT_DIR / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
