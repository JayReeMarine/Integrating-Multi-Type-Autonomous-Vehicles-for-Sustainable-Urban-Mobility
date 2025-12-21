import time

from data import generate_mock_data
from greedy import greedy_platoon_matching
from metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from analysis import (
    analyze_trip_distribution,
    analyze_feasible_pairs,
)


def print_metrics_table(metrics: dict):
    print("\n=== Experiment Results ===")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<25}: {value:>10.2f}")
        else:
            print(f"{key:<25}: {value}")
    print("-" * 50)


def main():
    # -----------------------
    # Experiment parameters
    # -----------------------
    NUM_AV = 5
    NUM_PV = 20
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEED = 42

    # Generate data
    avs, pvs, l_min = generate_mock_data(
        num_av=NUM_AV,
        num_pv=NUM_PV,
        highway_length=HIGHWAY_LENGTH,
        av_capacity_range=AV_CAPACITY_RANGE,
        min_trip_length=MIN_TRIP_LENGTH,
        seed=SEED,
    )

    # Data distribution analysis
    analyze_trip_distribution(avs, HIGHWAY_LENGTH, label="AV")
    analyze_trip_distribution(pvs, HIGHWAY_LENGTH, label="PV")
    analyze_feasible_pairs(avs, pvs, l_min)

    # Baseline
    baseline_total = baseline_total_powered_distance(avs, pvs)

    # Greedy runtime measurement
    start_time = time.perf_counter()
    assignments, _ = greedy_platoon_matching(avs, pvs, l_min)
    end_time = time.perf_counter()

    runtime = end_time - start_time

    # Greedy powered distance
    greedy_total = greedy_total_powered_distance(avs, pvs, assignments)

    # Metrics
    metrics = compute_extended_metrics(
        avs=avs,
        pvs=pvs,
        assignments=assignments,
        baseline_total=baseline_total,
        greedy_total=greedy_total,
        runtime_sec=runtime,
    )

    # Print table
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
