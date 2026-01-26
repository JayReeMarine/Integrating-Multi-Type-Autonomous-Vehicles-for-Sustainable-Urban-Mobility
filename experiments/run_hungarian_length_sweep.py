import os
import csv
from typing import Optional, Tuple

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian_multi import hungarian_multi_av_matching, DEFAULT_TIME_TOLERANCE


def run_length_sweep(
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
    Run highway length sweep experiment (Hungarian).

    Step 4 Enhancement:
    - enable_time_constraints: If True, uses time-based matching
    - time_tolerance: Max time difference for coupling (default: 5.0)
    - time_window: Time span for vehicle entry (default: 100.0)
    - av_speed_range: Speed range for AVs (default: (1.0, 1.0))
    - pv_speed_range: Speed range for PVs (default: (1.0, 1.0))

    Note: time_window scales with highway_length to maintain realistic scenarios
    """
    # Original values (for later use)
    # FIXED_NUM_AV = 160
    # FIXED_NUM_PV = 800
    # HIGHWAY_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200]
    # SEEDS = [42, 43, 44, 45, 46]

    # Reduced values (same as Greedy for fair comparison)
    FIXED_NUM_AV = 80   # Reduced: 160 -> 80
    FIXED_NUM_PV = 400  # Reduced: 800 -> 400
    HIGHWAY_LENGTHS = [50, 100, 200, 400, 800, 1600]  # Removed 3200
    SEEDS = [42, 43, 44, 45]  # Reduced: 5 -> 4 seeds

    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        time_mode = "WITH" if enable_time_constraints else "WITHOUT"
        print("\n===========================")
        print(f"Running length sweep (Hungarian) {time_mode} time constraints")
        print("===========================")
        print(f"Fixed AV={FIXED_NUM_AV}, PV={FIXED_NUM_PV}")
        if enable_time_constraints:
            print(f"  Time tolerance: {time_tolerance}")
            print(f"  Base time window: {time_window}")
            print(f"  AV speed range: {av_speed_range}")
            print(f"  PV speed range: {pv_speed_range}")

        for length in HIGHWAY_LENGTHS:
            # Step 4: Scale time_window with highway length
            # Longer highways need longer time windows for realistic scenarios
            scaled_time_window = time_window * (length / 100.0)

            for seed in SEEDS:
                params = ScenarioParams(
                    num_av=FIXED_NUM_AV,
                    num_pv=FIXED_NUM_PV,
                    highway_length=length,
                    av_capacity_range=AV_CAPACITY_RANGE,
                    min_trip_length=MIN_TRIP_LENGTH,
                    seed=seed,
                    # Step 4: Time constraint parameters
                    enable_time_constraints=enable_time_constraints,
                    time_tolerance=time_tolerance,
                    time_window=scaled_time_window,
                    av_speed_range=av_speed_range,
                    pv_speed_range=pv_speed_range,
                )

                row = run_one_scenario(
                    params=params,
                    matcher=hungarian_multi_av_matching,
                    run_task2_checks=False,
                )

                row["algorithm"] = "hungarian"
                row["scenario_type"] = "length_sweep"
                row["fixed_value"] = length
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n Length sweep (Hungarian) done. Saved to: {output_csv}")


def main() -> None:
    # Run with time constraints (Step 4)
    run_length_sweep(
        output_csv="data/results/hungarian/length_sweep.csv",
        enable_time_constraints=True,
        time_tolerance=5.0,
        time_window=100.0,  # Base time window (scaled with highway length)
        av_speed_range=(0.8, 1.2),  # AV speed varies ±20%
        pv_speed_range=(0.8, 1.2),  # PV speed varies ±20%
    )


if __name__ == "__main__":
    main()
