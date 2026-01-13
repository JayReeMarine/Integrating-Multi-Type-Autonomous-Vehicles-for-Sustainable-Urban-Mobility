import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian import hungarian_platoon_matching


def run_length_sweep(*, output_csv: str) -> None:
    # Original values (for later use - some too large for Hungarian)
    FIXED_NUM_AV = 160
    FIXED_NUM_PV = 800
    HIGHWAY_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200]
    # SEEDS = [42, 43, 44, 45, 46]
    
    # Medium scale values (optimized for Hungarian algorithm)
    # FIXED_NUM_AV = 80  # Reduced from 160
    # FIXED_NUM_PV = 200  # Reduced from 800
    # HIGHWAY_LENGTHS = [50, 100, 200, 400]  # Removed very large values
    SEEDS = [42, 43]  # Only 2 seeds instead of 5

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
                    matcher=hungarian_platoon_matching,
                    run_task2_checks=False,
                )

                row["algorithm"] = "hungarian"
                row["scenario_type"] = "length_sweep"
                row["fixed_value"] = length
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n✅ Length sweep (Hungarian) done. Saved to: {output_csv}")
