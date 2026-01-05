import os
import time
import csv

from core.data import generate_mock_data
from core.greedy import greedy_platoon_matching
from core.metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from core.analysis import (
    analyze_trip_distribution,
    analyze_feasible_pairs,
)

# -----------------------
# CSV configuration
# -----------------------

CSV_FILE = "data/results/greedy.csv"

CSV_FIELDS = [
    # Scenario identifiers
    "scenario_type",   # "pv_sweep" or "av_sweep"
    "fixed_value",     # fixed AV (for pv_sweep) or fixed PV (for av_sweep)
    "seed",            # random seed used

    # Experiment sizes
    "num_av",
    "num_pv",

    # Metrics
    "runtime_sec",
    "baseline_total_distance",
    "greedy_total_distance",
    "total_saving",
    "matched_pv",
    "matched_ratio",
    "avg_saving_per_pv",
    "saving_percent",
]

# -----------------------
# Experiment sweeps
# -----------------------
# PV sweep: fix AV count, increase PV count
PV_SWEEP_FIXED_AVS = [5, 20, 50, 160, 500]
PV_SWEEP_PVS = [20, 50, 100, 200, 400, 800, 1600, 2000]

# AV sweep: fix PV count, increase AV count
AV_SWEEP_FIXED_PVS = [50, 200, 800, 2000]
AV_SWEEP_AVS = [5, 10, 20, 50, 80, 160, 320, 500]


def run_one_scenario(
    *,
    num_av: int,
    num_pv: int,
    highway_length: int,
    av_capacity_range: tuple[int, int],
    min_trip_length: int,
    seed: int,
    run_task2_checks: bool,
) -> dict:
    """Run one experiment scenario and return a flat dict for CSV writing."""
    # Generate data
    avs, pvs, l_min = generate_mock_data(
        num_av=num_av,
        num_pv=num_pv,
        highway_length=highway_length,
        av_capacity_range=av_capacity_range,
        min_trip_length=min_trip_length,
        seed=seed,
    )

    # Task 2 sanity checks (only for the first scenario)
    if run_task2_checks:
        analyze_trip_distribution(avs, highway_length, label="AV")
        analyze_trip_distribution(pvs, highway_length, label="PV")
        analyze_feasible_pairs(avs, pvs, l_min)

    # Baseline
    baseline_total = baseline_total_powered_distance(avs, pvs)

    # Greedy + runtime
    start_time = time.perf_counter()
    assignments, _ = greedy_platoon_matching(avs, pvs, l_min)
    runtime_sec = time.perf_counter() - start_time

    greedy_total = greedy_total_powered_distance(baseline_total, assignments)

    # Metrics (your compute_extended_metrics must return the snake_case keys)
    metrics = compute_extended_metrics(
        avs=avs,
        pvs=pvs,
        assignments=assignments,
        baseline_total=baseline_total,
        greedy_total=greedy_total,
        runtime_sec=runtime_sec,
    )

    # Flatten output for CSV
    row = {
        "num_av": num_av,
        "num_pv": num_pv,
        "runtime_sec": runtime_sec,
        "baseline_total_distance": baseline_total,
        "greedy_total_distance": greedy_total,
        "total_saving": metrics["total_saving"],
        "matched_pv": metrics["matched_pv"],
        "matched_ratio": metrics["matched_ratio"],
        "avg_saving_per_pv": metrics["avg_saving_per_pv"],
        "saving_percent": metrics["saving_percent"],
    }
    return row


def main():
    # -----------------------
    # Global experiment parameters
    # -----------------------
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEEDS = [42, 43, 44, 45, 46]

    # Ensure output folder exists
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    # We only want Task 2 prints once for the very first run
    did_task2_once = False

    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        # =======================
        # 1) PV SWEEP experiments
        # =======================
        print("\n=======================")
        print("Running PV sweep")
        print("(Fix AV, increase PV)")
        print("=======================")

        for fixed_av in PV_SWEEP_FIXED_AVS:
            for num_pv in PV_SWEEP_PVS:
                num_av = fixed_av

                for seed in SEEDS: 
                    print(f"\n[pv_sweep] fixed AV={fixed_av} | AV={num_av}, PV={num_pv} | seed={seed}")

                    row = run_one_scenario(
                        num_av=num_av,
                        num_pv=num_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
                        min_trip_length=MIN_TRIP_LENGTH,
                        seed=seed,
                        run_task2_checks=(not did_task2_once),
                    )
                    did_task2_once = True

                    row["scenario_type"] = "pv_sweep"
                    row["fixed_value"] = fixed_av
                    row["seed"] = seed 

                    writer.writerow(row)

                # Console summary
                print(f"Runtime (sec): {row['runtime_sec']:.4f}")
                print(f"Total saving : {row['total_saving']:.2f}")
                print(f"Matched PV   : {row['matched_pv']} ({row['matched_ratio']:.2f})")

        # =======================
        # 2) AV SWEEP experiments
        # =======================
        print("\n=======================")
        print("Running AV sweep")
        print("(Fix PV, increase AV)")
        print("=======================")

        for fixed_pv in AV_SWEEP_FIXED_PVS:
            for num_av in AV_SWEEP_AVS:
                num_pv = fixed_pv

                for seed in SEEDS:
                    print(f"\n[av_sweep] fixed PV={fixed_pv} | AV={num_av}, PV={num_pv} | seed={seed}")

                    row = run_one_scenario(
                        num_av=num_av,
                        num_pv=num_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
                        min_trip_length=MIN_TRIP_LENGTH,
                        seed=seed, 
                        run_task2_checks=False,
                    )

                    row["scenario_type"] = "av_sweep"
                    row["fixed_value"] = fixed_pv
                    row["seed"] = seed

                    writer.writerow(row)

                # Console summary
                print(f"Runtime (sec): {row['runtime_sec']:.4f}")
                print(f"Total saving : {row['total_saving']:.2f}")
                print(f"Matched PV   : {row['matched_pv']} ({row['matched_ratio']:.2f})")

    print(f"\nAll experiments completed. Results saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
