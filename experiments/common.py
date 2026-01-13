import time
from typing import Dict, Tuple, Literal

from core.data import generate_mock_data
from core.greedy import greedy_platoon_matching
from core.metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from core.analysis import analyze_trip_distribution, analyze_feasible_pairs

from core.hungarian import hungarian_platoon_matching

Algorithm = Literal["greedy", "hungarian"]

CSV_FIELDS = [
    # Scenario identifiers
    "scenario_type",     # pv_sweep / av_sweep / length_sweep / capacity_sweep ...
    "fixed_value",       # depends on scenario (e.g., fixed AV or fixed PV)
    "seed",

    # Parameters (for scaling future experiments)
    "highway_length",
    "capacity_min",
    "capacity_max",

    # Experiment sizes
    "num_av",
    "num_pv",

    # Metrics
    "runtime_sec",
    "baseline_total_distance",
    "greedy_total_distance",
    "total_saving",
    "matched_pv",
    "matched_ratio",
    "avg_saving_per_pv",
    "saving_percent",
]

def run_one_scenario(
    *,
    num_av: int,
    num_pv: int,
    highway_length: int,
    av_capacity_range: Tuple[int, int],
    min_trip_length: int,
    seed: int,
    run_task2_checks: bool,
    algorithm: Algorithm = "greedy",
) -> Dict:
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

    if algorithm == "greedy":
        assignments, _ = greedy_platoon_matching(avs, pvs, l_min)
    elif algorithm == "hungarian":
        assignments, _ = hungarian_platoon_matching(avs, pvs, l_min)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    runtime_sec = time.perf_counter() - start_time

    total_after = greedy_total_powered_distance(baseline_total, assignments)

    metrics = compute_extended_metrics(
        avs=avs,
        pvs=pvs,
        assignments=assignments,
        baseline_total=baseline_total,
        greedy_total=total_after,
        runtime_sec=runtime_sec,
    )

    cap_min, cap_max = av_capacity_range

    return {
        "num_av": num_av,
        "num_pv": num_pv,
        "highway_length": highway_length,
        "capacity_min": cap_min,
        "capacity_max": cap_max,
        "runtime_sec": runtime_sec,
        "baseline_total_distance": baseline_total,
        "greedy_total_distance": total_after, 
        "total_saving": metrics["total_saving"],
        "matched_pv": metrics["matched_pv"],
        "matched_ratio": metrics["matched_ratio"],
        "avg_saving_per_pv": metrics["avg_saving_per_pv"],
        "saving_percent": metrics["saving_percent"],
        # "algorithm": algorithm,
    }