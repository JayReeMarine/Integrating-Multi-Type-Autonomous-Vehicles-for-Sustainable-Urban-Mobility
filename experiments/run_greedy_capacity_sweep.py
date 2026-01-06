import os
import time
import csv

from core.data import generate_mock_data
from core.greedy import greedy_platoon_matching
from core.metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from core.analysis import (
    analyze_trip_distribution,
    analyze_feasible_pairs,
)

CSV_FIELDS = [
    "scenario_type",
    "capacity_min",
    "capacity_max",
    "seed",
    "highway_length",
    "min_trip_length",
    "num_av",
    "num_pv",
    "runtime_sec",
    "baseline_total_distance",
    "greedy_total_distance",
    "total_saving",
    "matched_pv",
    "matched_ratio",
    "avg_saving_per_pv",
    "saving_percent",
]

AV_CAPACITY_RANGES = [(1, 3), (1, 5), (1, 10), (1, 20)]


def run_one_scenario(
    *,
    num_av: int,
    num_pv: int,
    highway_length: int,
    av_capacity_range: tuple[int, int],
    min_trip_length: int,
    seed: int,
    run_task2_checks: bool,
) -> dict:
    capacity_min, capacity_max = av_capacity_range

    avs, pvs, l_min = generate_mock_data(
        num_av=num_av,
        num_pv=num_pv,
        highway_length=highway_length,
        av_capacity_range=av_capacity_range,
        min_trip_length=min_trip_length,
        seed=seed,
    )

    if run_task2_checks:
        analyze_trip_distribution(avs, highway_length, label="AV")
        analyze_trip_distribution(pvs, highway_length, label="PV")
        analyze_feasible_pairs(avs, pvs, l_min)

    baseline_total = baseline_total_powered_distance(avs, pvs)

    start_time = time.perf_counter()
    assignments, _ = greedy_platoon_matching(avs, pvs, l_min)
    runtime_sec = time.perf_counter() - start_time

    greedy_total = greedy_total_powered_distance(baseline_total, assignments)

    metrics = compute_extended_metrics(
        avs=avs,
        pvs=pvs,
        assignments=assignments,
        baseline_total=baseline_total,
        greedy_total=greedy_total,
        runtime_sec=runtime_sec,
    )

    return {
        "scenario_type": "capacity_sweep",
        "capacity_min": capacity_min,
        "capacity_max": capacity_max,
        "seed": seed,
        "highway_length": highway_length,
        "min_trip_length": min_trip_length,
        "num_av": num_av,
        "num_pv": num_pv,
        "runtime_sec": runtime_sec,
        "baseline_total_distance": baseline_total,
        "greedy_total_distance": greedy_total,
        "total_saving": metrics["total_saving"],
        "matched_pv": metrics["matched_pv"],
        "matched_ratio": metrics["matched_ratio"],
        "avg_saving_per_pv": metrics["avg_saving_per_pv"],
        "saving_percent": metrics["saving_percent"],
    }


def run_capacity_sweep(*, output_csv: str) -> None:
    HIGHWAY_LENGTH = 100
    MIN_TRIP_LENGTH = 10  # must stay fixed
    FIXED_NUM_AV = 50
    FIXED_NUM_PV = 200
    SEEDS = [42, 43, 44, 45, 46]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    did_task2_once = False

    print("\n===============================")
    print("Running capacity sweep")
    print(f"(Fixed AV={FIXED_NUM_AV}, PV={FIXED_NUM_PV}, length={HIGHWAY_LENGTH}, min_trip_length={MIN_TRIP_LENGTH})")
    print("===============================")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for cap_min, cap_max in AV_CAPACITY_RANGES:
            for seed in SEEDS:
                print(f"\n[capacity_sweep] cap=[{cap_min},{cap_max}] | seed={seed}")

                row = run_one_scenario(
                    num_av=FIXED_NUM_AV,
                    num_pv=FIXED_NUM_PV,
                    highway_length=HIGHWAY_LENGTH,
                    av_capacity_range=(cap_min, cap_max),
                    min_trip_length=MIN_TRIP_LENGTH,
                    seed=seed,
                    run_task2_checks=(not did_task2_once),
                )
                did_task2_once = True
                writer.writerow(row)

            print(f"Runtime (sec): {row['runtime_sec']:.4f}")
            print(f"Total saving : {row['total_saving']:.2f}")
            print(f"Matched PV   : {row['matched_pv']} ({row['matched_ratio']:.2f})")

    print(f"\nCapacity sweep completed. Results saved to {output_csv}")


def main() -> None:
    run_capacity_sweep(output_csv="data/results/greedy/capacity_sweep.csv")


if __name__ == "__main__":
    main()
