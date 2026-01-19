import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.greedy_multi import greedy_multi_av_matching  # Changed: single -> multi AV matching

# Reduced for performance: removed 320, 640 from AVs; removed 800, 1600 from PVs
PV_SWEEP_FIXED_AVS = [20, 80, 160]
PV_SWEEP_PVS = [50, 100, 200, 400]

# Reduced: removed 1600 from PVs; removed 320 from AVs
AV_SWEEP_FIXED_PVS = [200, 400, 800]
AV_SWEEP_AVS = [10, 20, 40, 80, 160]


def run_pv_av_sweep(*, output_csv: str) -> None:
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEEDS = [42, 43, 44]  # Reduced: 5 -> 3 seeds

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n=======================")
        print("Running PV/AV sweep (Greedy)")
        print("=======================")

        # 1) PV sweep: fix AV, increase PV
        print("\n--- PV sweep (Fix AV, increase PV) ---")
        for fixed_av in PV_SWEEP_FIXED_AVS:
            for num_pv in PV_SWEEP_PVS:
                num_av = fixed_av

                for seed in SEEDS:
                    params = ScenarioParams(
                        num_av=num_av,
                        num_pv=num_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
                        min_trip_length=MIN_TRIP_LENGTH,
                        seed=seed,
                    )

                    row = run_one_scenario(
                        params=params,
                        matcher=greedy_multi_av_matching,
                        run_task2_checks=(not did_task2_once),
                    )
                    did_task2_once = True

                    row["algorithm"] = "greedy"
                    row["scenario_type"] = "pv_sweep"
                    row["fixed_value"] = fixed_av
                    row["seed"] = seed
                    writer.writerow(row)

        # 2) AV sweep: fix PV, increase AV
        print("\n--- AV sweep (Fix PV, increase AV) ---")
        for fixed_pv in AV_SWEEP_FIXED_PVS:
            for num_av in AV_SWEEP_AVS:
                num_pv = fixed_pv

                for seed in SEEDS:
                    params = ScenarioParams(
                        num_av=num_av,
                        num_pv=num_pv,
                        highway_length=HIGHWAY_LENGTH,
                        av_capacity_range=AV_CAPACITY_RANGE,
                        min_trip_length=MIN_TRIP_LENGTH,
                        seed=seed,
                    )

                    row = run_one_scenario(
                        params=params,
                        matcher=greedy_multi_av_matching,
                        run_task2_checks=False,
                    )

                    row["algorithm"] = "greedy"
                    row["scenario_type"] = "av_sweep"
                    row["fixed_value"] = fixed_pv
                    row["seed"] = seed
                    writer.writerow(row)

    print(f"\n✅ PV/AV sweep done. Saved to: {output_csv}")


def main() -> None:
    run_pv_av_sweep(output_csv="data/results/greedy/pv_av_sweep.csv")


if __name__ == "__main__":
    main()
