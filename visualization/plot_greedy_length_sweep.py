import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/greedy/length_sweep.csv"
FIG_DIR = "visualization/figures/greedy/length"


def load_results(csv_file: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(csv_file)


def _save_plot(filename: str) -> None:
    """Save current matplotlib figure into FIG_DIR."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")


def plot_length_sweep(df: pd.DataFrame) -> None:
    """
    Length sweep plots.
    X-axis: highway_length
    Y-axis: metrics (mean over seeds)
    """
    length_df = df[df["scenario_type"] == "length_sweep"].copy()
    if length_df.empty:
        print("No length_sweep data found in CSV.")
        return

    # group by highway_length and take mean across seeds
    length_df = length_df.sort_values(["highway_length", "seed"])
    length_df = length_df.groupby(["highway_length"], as_index=False).mean(numeric_only=True)

    # 1) Length vs Runtime
    plt.figure()
    plt.plot(length_df["highway_length"], length_df["runtime_sec"], marker="o")
    plt.xlabel("Highway Length")
    plt.ylabel("Runtime (seconds)")
    plt.title("Length Sweep: Runtime vs Highway Length")
    plt.grid(True)
    _save_plot("length_sweep_runtime.png")
    plt.close()

    # 2) Length vs Total Saving
    plt.figure()
    plt.plot(length_df["highway_length"], length_df["total_saving"], marker="o")
    plt.xlabel("Highway Length")
    plt.ylabel("Total Saving Distance")
    plt.title("Length Sweep: Total Saving vs Highway Length")
    plt.grid(True)
    _save_plot("length_sweep_total_saving.png")
    plt.close()

    # 3) Length vs Matched Ratio
    plt.figure()
    plt.plot(length_df["highway_length"], length_df["matched_ratio"], marker="o")
    plt.xlabel("Highway Length")
    plt.ylabel("Matched PV Ratio")
    plt.title("Length Sweep: Matched Ratio vs Highway Length")
    plt.grid(True)
    _save_plot("length_sweep_matched_ratio.png")
    plt.close()

    # 4) Length vs Avg Saving Per Matched PV
    plt.figure()
    plt.plot(length_df["highway_length"], length_df["avg_saving_per_pv"], marker="o")
    plt.xlabel("Highway Length")
    plt.ylabel("Avg Saving per Matched PV")
    plt.title("Length Sweep: Avg Saving per Matched PV vs Highway Length")
    plt.grid(True)
    _save_plot("length_sweep_avg_saving_per_pv.png")
    plt.close()

    # 5) Length vs Saving Percent
    if "saving_percent" not in length_df.columns:
        length_df["saving_percent"] = length_df["total_saving"] / length_df["baseline_total_distance"] * 100.0

    plt.figure()
    plt.plot(length_df["highway_length"], length_df["saving_percent"], marker="o")
    plt.xlabel("Highway Length")
    plt.ylabel("Total Saving Percent (%)")
    plt.title("Length Sweep: Saving Percent vs Highway Length")
    plt.grid(True)
    _save_plot("length_sweep_saving_percent.png")
    plt.close()


def main() -> None:
    df = load_results(CSV_FILE)
    print("Loaded length sweep results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_length_sweep(df)


if __name__ == "__main__":
    main()
