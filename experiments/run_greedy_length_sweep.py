import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.greedy_multi import greedy_multi_av_matching  # Changed: single -> multi AV matching


def run_length_sweep(*, output_csv: str) -> None:
    # Fixed AV/PV values for this sweep (Reduced for performance)
    FIXED_NUM_AV = 80   # Reduced: 160 -> 80
    FIXED_NUM_PV = 400  # Reduced: 800 -> 400

    # Highway length sweep parameters (Reduced: removed 1600, 3200)
    HIGHWAY_LENGTHS = [50, 100, 200, 400, 800]

    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEEDS = [42, 43, 44]  # Reduced: 5 -> 3 seeds

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n=======================")
        print("Running length sweep (Greedy)")
        print("=======================")
        print(f"Fixed AV={FIXED_NUM_AV}, PV={FIXED_NUM_PV}")

        for length in HIGHWAY_LENGTHS:
            for seed in SEEDS:
                params = ScenarioParams(
                    num_av=FIXED_NUM_AV,
                    num_pv=FIXED_NUM_PV,
                    highway_length=length,
                    av_capacity_range=AV_CAPACITY_RANGE,
                    min_trip_length=MIN_TRIP_LENGTH,
                    seed=seed,
                )

                row = run_one_scenario(
                    params=params,
                    matcher=greedy_multi_av_matching,  # Changed: single -> multi
                    run_task2_checks=False,
                )

                row["algorithm"] = "greedy"
                row["scenario_type"] = "length_sweep"
                row["fixed_value"] = length  # Using fixed_value as length for this sweep
                row["seed"] = seed
                writer.writerow(row)

    print(f"\n Length sweep done. Saved to: {output_csv}")


def main() -> None:
    run_length_sweep(output_csv="data/results/greedy/length_sweep.csv")


if __name__ == "__main__":
    main()
