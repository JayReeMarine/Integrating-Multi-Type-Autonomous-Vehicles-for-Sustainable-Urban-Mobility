import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/hungarian/pv_av_sweep.csv"
FIG_DIR = "visualization/figures/hungarian/pv_av"


def load_results(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)


def _save_plot(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")


def plot_pv_sweep(df: pd.DataFrame) -> None:
    """PV sweep plots (Fix AV, increase PV). Lines = fixed AV."""
    pv_df = df[df["scenario_type"] == "pv_sweep"].copy()
    if pv_df.empty:
        print("No pv_sweep data found in CSV.")
        return

    # Calculate mean across seeds only
    pv_df = pv_df.sort_values(["fixed_value", "num_pv"])
    pv_df = pv_df.groupby(["fixed_value", "num_pv"], as_index=False).mean(numeric_only=True)

    # Calculate saving_percent if not present (unit: %)
    if "saving_percent" not in pv_df.columns and "baseline_total_distance" in pv_df.columns:
        pv_df["saving_percent"] = pv_df["total_saving"] / pv_df["baseline_total_distance"] * 100.0

    def plot_metric(metric: str, xlabel: str, ylabel: str, title: str, out: str) -> None:
        plt.figure()
        for fixed_av, group in pv_df.groupby("fixed_value"):
            plt.plot(group["num_pv"], group[metric], marker="o", label=f"AV={int(fixed_av)}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        _save_plot(out)
        plt.close()

    plot_metric("runtime_sec", "Number of PVs", "Runtime (seconds)",
                "Hungarian PV Sweep: Runtime vs PV (lines = fixed AV)", "pv_sweep_runtime.png")

    plot_metric("total_saving", "Number of PVs", "Total Saving Distance",
                "Hungarian PV Sweep: Total Saving vs PV (lines = fixed AV)", "pv_sweep_total_saving.png")

    plot_metric("matched_ratio", "Number of PVs", "Matched PV Ratio",
                "Hungarian PV Sweep: Matched Ratio vs PV (lines = fixed AV)", "pv_sweep_matched_ratio.png")

    plot_metric("avg_saving_per_pv", "Number of PVs", "Avg Saving per Matched PV",
                "Hungarian PV Sweep: Avg Saving per Matched PV vs PV (lines = fixed AV)", "pv_sweep_avg_saving_per_pv.png")

    if "saving_percent" in pv_df.columns:
        plot_metric("saving_percent", "Number of PVs", "Total Saving Percent (%)",
                    "Hungarian PV Sweep: Saving Percent vs PV (lines = fixed AV)", "pv_sweep_saving_percent.png")


def plot_av_sweep(df: pd.DataFrame) -> None:
    """AV sweep plots (Fix PV, increase AV). Lines = fixed PV."""
    av_df = df[df["scenario_type"] == "av_sweep"].copy()
    if av_df.empty:
        print("No av_sweep data found in CSV.")
        return

    av_df = av_df.sort_values(["fixed_value", "num_av"])
    av_df = av_df.groupby(["fixed_value", "num_av"], as_index=False).mean(numeric_only=True)

    if "saving_percent" not in av_df.columns and "baseline_total_distance" in av_df.columns:
        av_df["saving_percent"] = av_df["total_saving"] / av_df["baseline_total_distance"] * 100.0

    def plot_metric(metric: str, xlabel: str, ylabel: str, title: str, out: str) -> None:
        plt.figure()
        for fixed_pv, group in av_df.groupby("fixed_value"):
            plt.plot(group["num_av"], group[metric], marker="o", label=f"PV={int(fixed_pv)}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        _save_plot(out)
        plt.close()

    plot_metric("runtime_sec", "Number of AVs", "Runtime (seconds)",
                "Hungarian AV Sweep: Runtime vs AV (lines = fixed PV)", "av_sweep_runtime.png")

    plot_metric("total_saving", "Number of AVs", "Total Saving Distance",
                "Hungarian AV Sweep: Total Saving vs AV (lines = fixed PV)", "av_sweep_total_saving.png")

    plot_metric("matched_ratio", "Number of AVs", "Matched PV Ratio",
                "Hungarian AV Sweep: Matched Ratio vs AV (lines = fixed PV)", "av_sweep_matched_ratio.png")

    plot_metric("avg_saving_per_pv", "Number of AVs", "Avg Saving per Matched PV",
                "Hungarian AV Sweep: Avg Saving per Matched PV vs AV (lines = fixed PV)", "av_sweep_avg_saving_per_pv.png")

    if "saving_percent" in av_df.columns:
        plot_metric("saving_percent", "Number of AVs", "Total Saving Percent (%)",
                    "Hungarian AV Sweep: Saving Percent vs AV (lines = fixed PV)", "av_sweep_saving_percent.png")


def main() -> None:
    df = load_results(CSV_FILE)
    print("Loaded Hungarian PV/AV sweep results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_pv_sweep(df)
    plot_av_sweep(df)


if __name__ == "__main__":
    main()
