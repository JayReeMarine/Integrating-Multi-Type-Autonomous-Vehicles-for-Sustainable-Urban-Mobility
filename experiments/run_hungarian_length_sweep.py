import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian_multi import hungarian_multi_av_matching  # Changed: single -> multi AV matching


def run_length_sweep(*, output_csv: str) -> None:
    # Original values (for later use)
    # FIXED_NUM_AV = 160
    # FIXED_NUM_PV = 800
    # HIGHWAY_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200]
    # SEEDS = [42, 43, 44, 45, 46]

    # Reduced values (same as Greedy for fair comparison)
    FIXED_NUM_AV = 80   # Reduced: 160 -> 80
    FIXED_NUM_PV = 400  # Reduced: 800 -> 400
    HIGHWAY_LENGTHS = [50, 100, 200, 400, 800]  # Removed 1600, 3200
    SEEDS = [42, 43, 44]  # Reduced: 5 -> 3 seeds

    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n===========================")
        print("Running length sweep (Hungarian)")
        print("===========================")

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
                    matcher=hungarian_multi_av_matching,  # Changed: single -> multi
                    run_task2_checks=False,
                )

                row["algorithm"] = "hungarian"
                row["scenario_type"] = "length_sweep"
                row["fixed_value"] = length
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n✅ Length sweep (Hungarian) done. Saved to: {output_csv}")


def main() -> None:
    run_length_sweep(output_csv="data/results/hungarian/length_sweep.csv")


if __name__ == "__main__":
    main()
