import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/greedy/capacity_sweep.csv"
FIG_DIR = "visualization/figures/greedy/capacity"


def load_results(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)


def _save_plot(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")


def plot_capacity_sweep(df: pd.DataFrame) -> None:
    cap_df = df[df["scenario_type"] == "capacity_sweep"].copy()
    if cap_df.empty:
        print("No capacity_sweep data found in CSV.")
        return

    # We'll use capacity_max as the x-axis (since cap_min is fixed at 1)
    cap_df = cap_df.sort_values(["capacity_max", "seed"])

    metric_cols = [
        "runtime_sec",
        "baseline_total_distance",
        "greedy_total_distance",
        "total_saving",
        "matched_pv",
        "matched_ratio",
        "avg_saving_per_pv",
        "saving_percent",
    ]

    cap_df = (
        cap_df[["capacity_min", "capacity_max"] + metric_cols]
        .groupby(["capacity_min", "capacity_max"])
        .agg(["mean", "std"])
        .reset_index()
    )

    # Helper
    def plot_metric(metric: str, xlabel: str, ylabel: str, title: str, out: str) -> None:
        plt.figure()
        plt.plot(
            cap_df["capacity_max"],
            cap_df[(metric, "mean")],
            marker="o",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        _save_plot(out)
        plt.close()

    plot_metric(
        "runtime_sec",
        "AV capacity max (cap_min=1)",
        "Runtime (seconds)",
        "Capacity Sweep: Runtime vs AV Capacity Range",
        "capacity_sweep_runtime.png",
    )

    plot_metric(
        "total_saving",
        "AV capacity max (cap_min=1)",
        "Total Saving Distance",
        "Capacity Sweep: Total Saving vs AV Capacity Range",
        "capacity_sweep_total_saving.png",
    )

    plot_metric(
        "matched_ratio",
        "AV capacity max (cap_min=1)",
        "Matched PV Ratio",
        "Capacity Sweep: Matched Ratio vs AV Capacity Range",
        "capacity_sweep_matched_ratio.png",
    )

    plot_metric(
        "avg_saving_per_pv",
        "AV capacity max (cap_min=1)",
        "Avg Saving per Matched PV",
        "Capacity Sweep: Avg Saving per Matched PV vs AV Capacity Range",
        "capacity_sweep_avg_saving_per_pv.png",
    )

    plot_metric(
        "saving_percent",
        "AV capacity max (cap_min=1)",
        "Total Saving Percent (%)",
        "Capacity Sweep: Saving Percent vs AV Capacity Range",
        "capacity_sweep_saving_percent.png",
    )


def main() -> None:
    df = load_results(CSV_FILE)
    print("Loaded capacity sweep results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_capacity_sweep(df)


if __name__ == "__main__":
    main()
