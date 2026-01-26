import os
import csv
from typing import Optional, Tuple

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian_multi import hungarian_multi_av_matching, DEFAULT_TIME_TOLERANCE

# Original values (for later use)
# PV_SWEEP_FIXED_AVS = [20, 80, 320, 640]
# PV_SWEEP_PVS = [50, 100, 200, 400, 800, 1600]
# AV_SWEEP_FIXED_PVS = [200, 400, 800, 1600]
# AV_SWEEP_AVS = [10, 20, 40, 80, 160, 320]
# SEEDS = [42, 43, 44, 45, 46]

# Reduced values (same as Greedy for fair comparison)
PV_SWEEP_FIXED_AVS = [20, 80, 160, 320]       # Removed 320, 640
PV_SWEEP_PVS = [50, 100, 200, 400, 800]       # Removed 800, 1600
AV_SWEEP_FIXED_PVS = [200, 400, 800, 1600]     # Removed 1600
AV_SWEEP_AVS = [10, 20, 40, 80, 160, 320]     # Removed 320


def run_pv_av_sweep(
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
    Run PV/AV count sweep experiment (Hungarian).

    Step 4 Enhancement:
    - enable_time_constraints: If True, uses time-based matching
    - time_tolerance: Max time difference for coupling (default: 5.0)
    - time_window: Time span for vehicle entry (default: 100.0)
    - av_speed_range: Speed range for AVs (default: (1.0, 1.0))
    - pv_speed_range: Speed range for PVs (default: (1.0, 1.0))
    """
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEEDS = [42, 43, 44, 45]  # Reduced: 5 -> 3 seeds

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        time_mode = "WITH" if enable_time_constraints else "WITHOUT"
        print("\n===========================")
        print(f"Running PV/AV sweep (Hungarian) {time_mode} time constraints")
        print("===========================")
        if enable_time_constraints:
            print(f"  Time tolerance: {time_tolerance}")
            print(f"  Time window: {time_window}")
            print(f"  AV speed range: {av_speed_range}")
            print(f"  PV speed range: {pv_speed_range}")

        # PV sweep
        print("\n--- PV sweep (Fix AV, increase PV) ---")
        for fixed_av in PV_SWEEP_FIXED_AVS:
            for num_pv in PV_SWEEP_PVS:
                for seed in SEEDS:
                    params = ScenarioParams(
                        num_av=fixed_av,
                        num_pv=num_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
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
                    row["scenario_type"] = "pv_sweep"
                    row["fixed_value"] = fixed_av
                    row["seed"] = seed

                    writer.writerow(row)

        # AV sweep
        print("\n--- AV sweep (Fix PV, increase AV) ---")
        for fixed_pv in AV_SWEEP_FIXED_PVS:
            for num_av in AV_SWEEP_AVS:
                for seed in SEEDS:
                    params = ScenarioParams(
                        num_av=num_av,
                        num_pv=fixed_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
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
                        run_task2_checks=False,
                    )

                    row["algorithm"] = "hungarian"
                    row["scenario_type"] = "av_sweep"
                    row["fixed_value"] = fixed_pv
                    row["seed"] = seed

                    writer.writerow(row)

    print(f"\n PV/AV sweep (Hungarian) done. Saved to: {output_csv}")


def main() -> None:
    # Run with time constraints (Step 4)
    run_pv_av_sweep(
        output_csv="data/results/hungarian/pv_av_sweep.csv",
        enable_time_constraints=True,
        time_tolerance=5.0,
        time_window=100.0,
        av_speed_range=(0.8, 1.2),  # AV speed varies ±20%
        pv_speed_range=(0.8, 1.2),  # PV speed varies ±20%
    )


if __name__ == "__main__":
    main()
