# plot.py
# --------------------------------------------------
# Plot experiment results from experiment_results.csv
#
# Graphs:
# 1) PV count vs Runtime
# 2) PV count vs Total Saving
# 3) PV count vs Matched PV Ratio
#
# This script is used for Task 3 analysis and
# scalability evaluation of the greedy algorithm.
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "data/experiment_results.csv"


def load_results(csv_file: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(csv_file)


def plot_pv_vs_runtime(df: pd.DataFrame) -> None:
    """Plot Number of PVs vs Runtime (seconds)."""
    plt.figure()
    plt.plot(df["num_pv"], df["runtime_sec"], marker="o")
    plt.xlabel("Number of PVs")
    plt.ylabel("Runtime (seconds)")
    plt.title("Greedy Runtime vs Number of PVs")
    plt.grid(True)
    plt.show()


def plot_pv_vs_total_saving(df: pd.DataFrame) -> None:
    """Plot Number of PVs vs Total Saving."""
    plt.figure()
    plt.plot(df["num_pv"], df["total_saving"], marker="o")
    plt.xlabel("Number of PVs")
    plt.ylabel("Total Saving Distance")
    plt.title("Total Saving vs Number of PVs")
    plt.grid(True)
    plt.show()


def plot_pv_vs_matched_ratio(df: pd.DataFrame) -> None:
    """Plot Number of PVs vs Matched PV Ratio."""
    plt.figure()
    plt.plot(df["num_pv"], df["matched_ratio"], marker="o")
    plt.xlabel("Number of PVs")
    plt.ylabel("Matched PV Ratio")
    plt.title("Matched PV Ratio vs Number of PVs")
    plt.grid(True)
    plt.show()


def main():
    df = load_results(CSV_FILE)

    print("Loaded experiment results:")
    print(df)

    plot_pv_vs_runtime(df)
    plot_pv_vs_total_saving(df)
    plot_pv_vs_matched_ratio(df)


if __name__ == "__main__":
    main()
