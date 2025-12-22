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

CSV_FILE = "data/experiment_results.csv"

CSV_FIELDS = [
    "num_av",
    "num_pv",
    "runtime_sec",
    "baseline_total_distance",
    "greedy_total_distance",
    "total_saving",
    "matched_pv",
    "matched_ratio",
    "avg_saving_per_pv",
]

# -----------------------
# Experiment scenarios
# -----------------------

SCENARIOS = [
    {"num_av": 5, "num_pv": 20},
    {"num_av": 50, "num_pv": 200},
    {"num_av": 500, "num_pv": 2000},
]


def main():
    # -----------------------
    # Global experiment parameters
    # -----------------------
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEED = 42

    # -----------------------
    # Prepare CSV file
    # -----------------------
    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        # -----------------------
        # Run scenarios
        # -----------------------
        for idx, scenario in enumerate(SCENARIOS):
            num_av = scenario["num_av"]
            num_pv = scenario["num_pv"]

            print(f"\n=== Running Scenario {idx + 1} ===")
            print(f"AV = {num_av}, PV = {num_pv}")

            # -----------------------
            # Data generation
            # -----------------------
            avs, pvs, l_min = generate_mock_data(
                num_av=num_av,
                num_pv=num_pv,
                highway_length=HIGHWAY_LENGTH,
                av_capacity_range=AV_CAPACITY_RANGE,
                min_trip_length=MIN_TRIP_LENGTH,
                seed=SEED,
            )

            # -----------------------
            # Data sanity check (Task 2)
            # Only run for the smallest scenario
            # -----------------------
            if idx == 0:
                analyze_trip_distribution(avs, HIGHWAY_LENGTH, label="AV")
                analyze_trip_distribution(pvs, HIGHWAY_LENGTH, label="PV")
                analyze_feasible_pairs(avs, pvs, l_min)

            # -----------------------
            # Baseline computation
            # -----------------------
            baseline_total = baseline_total_powered_distance(avs, pvs)

            # -----------------------
            # Greedy algorithm + runtime
            # -----------------------
            start_time = time.perf_counter()
            assignments, _ = greedy_platoon_matching(avs, pvs, l_min)
            runtime = time.perf_counter() - start_time

            greedy_total = greedy_total_powered_distance(
                avs, pvs, assignments
            )

            # -----------------------
            # Metrics
            # -----------------------
            metrics = compute_extended_metrics(
                avs=avs,
                pvs=pvs,
                assignments=assignments,
                baseline_total=baseline_total,
                greedy_total=greedy_total,
                runtime_sec=runtime,
            )

            # -----------------------
            # Save results to CSV
            # -----------------------
            writer.writerow({
                "num_av": num_av,
                "num_pv": num_pv,
                "runtime_sec": runtime,
                "baseline_total_distance": baseline_total,
                "greedy_total_distance": greedy_total,
                "total_saving": metrics["total_saving"],
                "matched_pv": metrics["matched_pv"],
                "matched_ratio": metrics["matched_ratio"],
                "avg_saving_per_pv": metrics["avg_saving_per_pv"],
            })

            # -----------------------
            # Console summary
            # -----------------------
            print(f"Runtime (sec): {runtime:.4f}")
            print(f"Total saving : {metrics['total_saving']:.2f}")
            print(f"Matched PV   : {metrics['matched_pv']} "
                  f"({metrics['matched_ratio']:.2f})")

    print(f"\nAll experiments completed. Results saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
