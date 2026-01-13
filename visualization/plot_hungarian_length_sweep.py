import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/hungarian/length_sweep.csv"
FIG_DIR = "visualization/figures/hungarian/length"


def load_results(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)


def _save_plot(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")


def plot_length_sweep(df: pd.DataFrame) -> None:
    length_df = df[df["scenario_type"] == "length_sweep"].copy()
    if length_df.empty:
        print("No length_sweep data found in CSV.")
        return

    # highway_length 기준 seed 평균
    length_df = length_df.sort_values(["highway_length", "seed"])
    length_df = length_df.groupby(["highway_length"], as_index=False).mean(numeric_only=True)

    if "saving_percent" not in length_df.columns and "baseline_total_distance" in length_df.columns:
        length_df["saving_percent"] = length_df["total_saving"] / length_df["baseline_total_distance"] * 100.0

    def plot_metric(metric: str, ylabel: str, title: str, out: str) -> None:
        plt.figure()
        plt.plot(length_df["highway_length"], length_df[metric], marker="o")
        plt.xlabel("Highway Length")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        _save_plot(out)
        plt.close()

    plot_metric("runtime_sec", "Runtime (seconds)",
                "Hungarian Length Sweep: Runtime vs Highway Length", "length_sweep_runtime.png")

    plot_metric("total_saving", "Total Saving Distance",
                "Hungarian Length Sweep: Total Saving vs Highway Length", "length_sweep_total_saving.png")

    plot_metric("matched_ratio", "Matched PV Ratio",
                "Hungarian Length Sweep: Matched Ratio vs Highway Length", "length_sweep_matched_ratio.png")

    plot_metric("avg_saving_per_pv", "Avg Saving per Matched PV",
                "Hungarian Length Sweep: Avg Saving per Matched PV vs Highway Length", "length_sweep_avg_saving_per_pv.png")

    if "saving_percent" in length_df.columns:
        plot_metric("saving_percent", "Total Saving Percent (%)",
                    "Hungarian Length Sweep: Saving Percent vs Highway Length", "length_sweep_saving_percent.png")


def main() -> None:
    df = load_results(CSV_FILE)
    print("Loaded Hungarian length sweep results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_length_sweep(df)


if __name__ == "__main__":
    main()
