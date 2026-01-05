# --------------------------------------------------
# Plot experiment results from experiment_results.csv
# --------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/results/greedy.csv" 
FIG_DIR = "visualization/figures/greedy"  


def load_results(csv_file: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(csv_file)


def _save_plot(filename: str) -> None:
    """Save current matplotlib figure into visualization/figures/greedy/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")


# --------------------------------------------------
# PV sweep plots (Fix AV, increase PV)
# --------------------------------------------------

def plot_pv_sweep(df: pd.DataFrame) -> None:
    """Create multi-line plots for PV sweep (line=Fixed AV)."""
    pv_df = df[df["scenario_type"] == "pv_sweep"].copy()
    if pv_df.empty:
        print("No pv_sweep data found in CSV.")
        return

    pv_df = pv_df.sort_values(["fixed_value", "num_pv"])

    # 1) PV vs Runtime
    plt.figure()
    for fixed_av, group in pv_df.groupby("fixed_value"):
        plt.plot(group["num_pv"], group["runtime_sec"], marker="o", label=f"AV={int(fixed_av)}")
    plt.xlabel("Number of PVs")
    plt.ylabel("Runtime (seconds)")
    plt.title("PV Sweep: Runtime vs PV (lines = fixed AV)")
    plt.grid(True)
    plt.legend()
    _save_plot("pv_sweep_runtime.png")
    plt.close()

    # 2) PV vs Total Saving
    plt.figure()
    for fixed_av, group in pv_df.groupby("fixed_value"):
        plt.plot(group["num_pv"], group["total_saving"], marker="o", label=f"AV={int(fixed_av)}")
    plt.xlabel("Number of PVs")
    plt.ylabel("Total Saving Distance")
    plt.title("PV Sweep: Total Saving vs PV (lines = fixed AV)")
    plt.grid(True)
    plt.legend()
    _save_plot("pv_sweep_total_saving.png")
    plt.close()

    # 3) PV vs Matched Ratio
    plt.figure()
    for fixed_av, group in pv_df.groupby("fixed_value"):
        plt.plot(group["num_pv"], group["matched_ratio"], marker="o", label=f"AV={int(fixed_av)}")
    plt.xlabel("Number of PVs")
    plt.ylabel("Matched PV Ratio")
    plt.title("PV Sweep: Matched Ratio vs PV (lines = fixed AV)")
    plt.grid(True)
    plt.legend()
    _save_plot("pv_sweep_matched_ratio.png")
    plt.close()

    # ✅ 4) PV vs Avg Saving Per Matched PV
    plt.figure()
    for fixed_av, group in pv_df.groupby("fixed_value"):
        plt.plot(group["num_pv"], group["avg_saving_per_pv"], marker="o", label=f"AV={int(fixed_av)}")
    plt.xlabel("Number of PVs")
    plt.ylabel("Avg Saving per Matched PV")
    plt.title("PV Sweep: Avg Saving per Matched PV vs PV (lines = fixed AV)")
    plt.grid(True)
    plt.legend()
    _save_plot("pv_sweep_avg_saving_per_pv.png")
    plt.close()

    # ✅ 5) PV vs Saving Percent (= total_saving / baseline_total_distance)
    # If you already store saving_percent in CSV, use df["saving_percent"] directly.
    # Here we compute it safely in case it doesn't exist.
    if "saving_percent" not in pv_df.columns:
        pv_df["saving_percent"] = pv_df["total_saving"] / pv_df["baseline_total_distance"]

    plt.figure()
    for fixed_av, group in pv_df.groupby("fixed_value"):
        plt.plot(group["num_pv"], group["saving_percent"], marker="o", label=f"AV={int(fixed_av)}")
    plt.xlabel("Number of PVs")
    plt.ylabel("Total Saving Percent")
    plt.title("PV Sweep: Saving Percent vs PV (lines = fixed AV)")
    plt.grid(True)
    plt.legend()
    _save_plot("pv_sweep_saving_percent.png")
    plt.close()


# --------------------------------------------------
# AV sweep plots (Fix PV, increase AV)
# --------------------------------------------------

def plot_av_sweep(df: pd.DataFrame) -> None:
    """Create multi-line plots for AV sweep (line=Fixed PV)."""
    av_df = df[df["scenario_type"] == "av_sweep"].copy()
    if av_df.empty:
        print("No av_sweep data found in CSV.")
        return

    av_df = av_df.sort_values(["fixed_value", "num_av"])

    # 1) AV vs Runtime
    plt.figure()
    for fixed_pv, group in av_df.groupby("fixed_value"):
        plt.plot(group["num_av"], group["runtime_sec"], marker="o", label=f"PV={int(fixed_pv)}")
    plt.xlabel("Number of AVs")
    plt.ylabel("Runtime (seconds)")
    plt.title("AV Sweep: Runtime vs AV (lines = fixed PV)")
    plt.grid(True)
    plt.legend()
    _save_plot("av_sweep_runtime.png")
    plt.close()

    # 2) AV vs Total Saving
    plt.figure()
    for fixed_pv, group in av_df.groupby("fixed_value"):
        plt.plot(group["num_av"], group["total_saving"], marker="o", label=f"PV={int(fixed_pv)}")
    plt.xlabel("Number of AVs")
    plt.ylabel("Total Saving Distance")
    plt.title("AV Sweep: Total Saving vs AV (lines = fixed PV)")
    plt.grid(True)
    plt.legend()
    _save_plot("av_sweep_total_saving.png")
    plt.close()

    # 3) AV vs Matched Ratio
    plt.figure()
    for fixed_pv, group in av_df.groupby("fixed_value"):
        plt.plot(group["num_av"], group["matched_ratio"], marker="o", label=f"PV={int(fixed_pv)}")
    plt.xlabel("Number of AVs")
    plt.ylabel("Matched PV Ratio")
    plt.title("AV Sweep: Matched Ratio vs AV (lines = fixed PV)")
    plt.grid(True)
    plt.legend()
    _save_plot("av_sweep_matched_ratio.png")
    plt.close()

    # ✅ 4) AV vs Avg Saving Per Matched PV
    plt.figure()
    for fixed_pv, group in av_df.groupby("fixed_value"):
        plt.plot(group["num_av"], group["avg_saving_per_pv"], marker="o", label=f"PV={int(fixed_pv)}")
    plt.xlabel("Number of AVs")
    plt.ylabel("Avg Saving per Matched PV")
    plt.title("AV Sweep: Avg Saving per Matched PV vs AV (lines = fixed PV)")
    plt.grid(True)
    plt.legend()
    _save_plot("av_sweep_avg_saving_per_pv.png")
    plt.close()

    # ✅ 5) AV vs Saving Percent
    if "saving_percent" not in av_df.columns:
        av_df["saving_percent"] = av_df["total_saving"] / av_df["baseline_total_distance"]

    plt.figure()
    for fixed_pv, group in av_df.groupby("fixed_value"):
        plt.plot(group["num_av"], group["saving_percent"], marker="o", label=f"PV={int(fixed_pv)}")
    plt.xlabel("Number of AVs")
    plt.ylabel("Total Saving Percent")
    plt.title("AV Sweep: Saving Percent vs AV (lines = fixed PV)")
    plt.grid(True)
    plt.legend()
    _save_plot("av_sweep_saving_percent.png")
    plt.close()


def main():
    df = load_results(CSV_FILE)

    print("Loaded experiment results:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")

    plot_pv_sweep(df)
    plot_av_sweep(df)


if __name__ == "__main__":
    main()
