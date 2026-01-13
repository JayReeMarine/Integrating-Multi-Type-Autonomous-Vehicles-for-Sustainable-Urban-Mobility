import os
import csv

from experiments.common import CSV_FIELDS, ScenarioParams, run_one_scenario
from core.greedy import greedy_platoon_matching

AV_CAPACITY_RANGES = [(1, 2), (1, 4), (1, 8), (1, 16), (1, 32)]


def run_capacity_sweep(*, output_csv: str) -> None:
    HIGHWAY_LENGTH = 100
    MIN_TRIP_LENGTH = 10
    FIXED_NUM_AV = 50
    FIXED_NUM_PV = 200
    SEEDS = [42, 43, 44, 45, 46]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        print("\n===========================")
        print("Running capacity sweep (Greedy)")
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
                    matcher=greedy_platoon_matching,
                    run_task2_checks=(not did_task2_once),
                )
                did_task2_once = True

                row["algorithm"] = "greedy"
                row["scenario_type"] = "capacity_sweep"
                row["fixed_value"] = cap_max  # x축으로 쓰기 편함
                row["seed"] = seed

                writer.writerow(row)

    print(f"\n✅ Capacity sweep (Greedy) done. Saved to: {output_csv}")


def main() -> None:
    run_capacity_sweep(output_csv="data/results/greedy/capacity_sweep.csv")


if __name__ == "__main__":
    main()
