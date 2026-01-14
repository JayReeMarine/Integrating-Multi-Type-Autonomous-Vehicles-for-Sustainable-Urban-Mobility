import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.hungarian import hungarian_platoon_matching

# Original values (for later use - too large for Hungarian)
PV_SWEEP_FIXED_AVS = [20, 80, 320, 640]
PV_SWEEP_PVS = [50, 100, 200, 400, 800, 1600]
AV_SWEEP_FIXED_PVS = [200, 400, 800, 1600]
AV_SWEEP_AVS = [10, 20, 40, 80, 160, 320]

# Medium scale values (optimized for Hungarian algorithm)
# PV_SWEEP_FIXED_AVS = [20, 80, 320]  # Reduced from 4 to 2 values
# PV_SWEEP_PVS = [50, 100, 200, 400]  # Reduced from 6 to 3 values
# AV_SWEEP_FIXED_PVS = [200, 400, 800]  # Reduced from 4 to 2 values  
# AV_SWEEP_AVS = [10, 20, 40, 80]  # Reduced from 6 to 4 values


def run_pv_av_sweep(*, output_csv: str) -> None:
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    
    # Reduced seeds for faster testing
    # SEEDS = [42, 43]  # Only 2 seeds instead of 5
    SEEDS = [42, 43, 44, 45, 46]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n===========================")
        print("Running PV/AV sweep (Hungarian)")
        print("===========================")

        # PV sweep
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
                    )

                    row = run_one_scenario(
                        params=params,
                        matcher=hungarian_platoon_matching,
                        run_task2_checks=(not did_task2_once),
                    )
                    did_task2_once = True

                    row["algorithm"] = "hungarian"
                    row["scenario_type"] = "pv_sweep"
                    row["fixed_value"] = fixed_av
                    row["seed"] = seed

                    writer.writerow(row)

        # AV sweep
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
                    )

                    row = run_one_scenario(
                        params=params,
                        matcher=hungarian_platoon_matching,
                        run_task2_checks=False,
                    )

                    row["algorithm"] = "hungarian"
                    row["scenario_type"] = "av_sweep"
                    row["fixed_value"] = fixed_pv
                    row["seed"] = seed

                    writer.writerow(row)

    print(f"\n✅ PV/AV sweep (Hungarian) done. Saved to: {output_csv}")


if __name__ == "__main__":
    run_pv_av_sweep(output_csv="data/results/hungarian/pv_av_sweep.csv")
