import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/hungarian/capacity_sweep.csv"
FIG_DIR = "visualization/figures/hungarian/capacity"


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

    # Group by capacity_max and calculate mean across seeds (cap_min is usually fixed to 1)
    cap_df = cap_df.sort_values(["capacity_max", "seed"])
    cap_df = cap_df.groupby(["capacity_min", "capacity_max"], as_index=False).mean(numeric_only=True)

    if "saving_percent" not in cap_df.columns and "baseline_total_distance" in cap_df.columns:
        cap_df["saving_percent"] = cap_df["total_saving"] / cap_df["baseline_total_distance"] * 100.0

    def plot_metric(metric: str, ylabel: str, title: str, out: str) -> None:
        plt.figure()
        plt.plot(cap_df["capacity_max"], cap_df[metric], marker="o")
        plt.xlabel("AV capacity max (cap_min fixed)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        _save_plot(out)
        plt.close()

    plot_metric("runtime_sec", "Runtime (seconds)",
                "Hungarian Capacity Sweep: Runtime vs AV Capacity Range", "capacity_sweep_runtime.png")

    plot_metric("total_saving", "Total Saving Distance",
                "Hungarian Capacity Sweep: Total Saving vs AV Capacity Range", "capacity_sweep_total_saving.png")

    plot_metric("matched_ratio", "Matched PV Ratio",
                "Hungarian Capacity Sweep: Matched Ratio vs AV Capacity Range", "capacity_sweep_matched_ratio.png")

    plot_metric("avg_saving_per_pv", "Avg Saving per Matched PV",
                "Hungarian Capacity Sweep: Avg Saving per Matched PV vs AV Capacity Range", "capacity_sweep_avg_saving_per_pv.png")

    if "saving_percent" in cap_df.columns:
        plot_metric("saving_percent", "Total Saving Percent (%)",
                    "Hungarian Capacity Sweep: Saving Percent vs AV Capacity Range", "capacity_sweep_saving_percent.png")


def main() -> None:
    df = load_results(CSV_FILE)
    print("Loaded Hungarian capacity sweep results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_capacity_sweep(df)


if __name__ == "__main__":
    main()
