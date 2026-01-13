import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian import hungarian_platoon_matching

# Original values (for later use - some too large for Hungarian)
# AV_CAPACITY_RANGES = [(1, 2), (1, 4), (1, 8), (1, 16), (1, 32)]

# Optimized ranges (focus on meaningful comparisons)
AV_CAPACITY_RANGES = [(1, 2), (1, 4), (1, 8), (1, 16)]  # Removed very large capacities


def run_capacity_sweep(*, output_csv: str) -> None:
    HIGHWAY_LENGTH = 100
    MIN_TRIP_LENGTH = 10
    
    # Moderate scale values (optimized for Hungarian algorithm)
    FIXED_NUM_AV = 50  # Keep original - manageable size
    FIXED_NUM_PV = 200  # Keep original - reasonable for comparison
    SEEDS = [42, 43]  # Only 2 seeds instead of 5

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n===========================")
        print("Running capacity sweep (Hungarian)")
        print("===========================")

        for cap_min, cap_max in AV_CAPACITY_RANGES:
            for seed in SEEDS:
                params = ScenarioParams(
                    num_av=FIXED_NUM_AV,
                    num_pv=FIXED_NUM_PV,
                    highway_length=HIGHWAY_LENGTH,
                    av_capacity_range=(cap_min, cap_max),
                    min_trip_length=MIN_TRIP_LENGTH,
                    seed=seed,
                )

                row = run_one_scenario(
                    params=params,
                    matcher=hungarian_platoon_matching,
                    run_task2_checks=(not did_task2_once),
                )
                did_task2_once = True

                row["algorithm"] = "hungarian"
                row["scenario_type"] = "capacity_sweep"
                row["fixed_value"] = cap_max
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n✅ Capacity sweep (Hungarian) done. Saved to: {output_csv}")
