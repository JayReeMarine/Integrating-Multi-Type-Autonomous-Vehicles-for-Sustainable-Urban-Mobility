import os
import csv
from typing import Optional, Tuple

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian_multi import hungarian_multi_av_matching, DEFAULT_TIME_TOLERANCE

# Original values (for comprehensive comparison with Greedy)
# AV_CAPACITY_RANGES = [(1, 2), (1, 4), (1, 8), (1, 16), (1, 32)]
# SEEDS = [42, 43, 44, 45, 46]

# Reduced values (same as Greedy for fair comparison)
AV_CAPACITY_RANGES = [(1, 2), (1, 4), (1, 8), (1, 16)]  # Removed (1, 32)


def run_capacity_sweep(
    *,
    output_csv: str,
    # Step 4: Time constraint parameters
    enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
    time_window: float = 100.0,
    av_speed_range: Optional[Tuple[float, float]] = None,
    pv_speed_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Run capacity sweep experiment (Hungarian).

    Step 4 Enhancement:
    - enable_time_constraints: If True, uses time-based matching
    - time_tolerance: Max time difference for coupling (default: 5.0)
    - time_window: Time span for vehicle entry (default: 100.0)
    - av_speed_range: Speed range for AVs (default: (1.0, 1.0))
    - pv_speed_range: Speed range for PVs (default: (1.0, 1.0))
    """
    HIGHWAY_LENGTH = 100
    MIN_TRIP_LENGTH = 10
    FIXED_NUM_AV = 50
    FIXED_NUM_PV = 200
    SEEDS = [42, 43, 44, 45]  # Reduced: 5 -> 4 seeds (same as Greedy)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        time_mode = "WITH" if enable_time_constraints else "WITHOUT"
        print("\n===========================")
        print(f"Running capacity sweep (Hungarian) {time_mode} time constraints")
        print("===========================")
        if enable_time_constraints:
            print(f"  Time tolerance: {time_tolerance}")
            print(f"  Time window: {time_window}")
            print(f"  AV speed range: {av_speed_range}")
            print(f"  PV speed range: {pv_speed_range}")

        for cap_min, cap_max in AV_CAPACITY_RANGES:
            for seed in SEEDS:
                params = ScenarioParams(
                    num_av=FIXED_NUM_AV,
                    num_pv=FIXED_NUM_PV,
                    highway_length=HIGHWAY_LENGTH,
                    av_capacity_range=(cap_min, cap_max),
                    min_trip_length=MIN_TRIP_LENGTH,
                    seed=seed,
                    # Step 4: Time constraint parameters
                    enable_time_constraints=enable_time_constraints,
                    time_tolerance=time_tolerance,
                    time_window=time_window,
                    av_speed_range=av_speed_range,
                    pv_speed_range=pv_speed_range,
                )

                row = run_one_scenario(
                    params=params,
                    matcher=hungarian_multi_av_matching,
                    run_task2_checks=(not did_task2_once),
                )
                did_task2_once = True

                row["algorithm"] = "hungarian"
                row["scenario_type"] = "capacity_sweep"
                row["fixed_value"] = cap_max
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n Capacity sweep (Hungarian) done. Saved to: {output_csv}")


def main() -> None:
    # Run with time constraints (Step 4)
    run_capacity_sweep(
        output_csv="data/results/hungarian/capacity_sweep.csv",
        enable_time_constraints=True,
        time_tolerance=5.0,
        time_window=100.0,
        av_speed_range=(0.8, 1.2),  # AV speed varies ±20%
        pv_speed_range=(0.8, 1.2),  # PV speed varies ±20%
    )


if __name__ == "__main__":
    main()
