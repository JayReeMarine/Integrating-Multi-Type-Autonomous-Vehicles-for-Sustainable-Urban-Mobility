# experiments/common.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Union

from core.data import generate_mock_data
from core.models import ActiveVehicle, PassiveVehicle
from core.greedy import Assignment
from core.metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from core.analysis import analyze_trip_distribution, analyze_feasible_pairs


Number = Union[int, float]

MatcherFn = Callable[
    [List[ActiveVehicle], List[PassiveVehicle], int],
    Tuple[List[Assignment], Number],
]


CSV_FIELDS = [
    "algorithm",          # NEW: greedy / hungarian
    "scenario_type",      # pv_sweep / av_sweep / length_sweep / capacity_sweep ...
    "fixed_value",        # depends on scenario
    "seed",

    "highway_length",
    "min_trip_length",
    "capacity_min",
    "capacity_max",

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


@dataclass(frozen=True)
class ScenarioParams:
    num_av: int
    num_pv: int
    highway_length: int
    av_capacity_range: Tuple[int, int]
    min_trip_length: int
    seed: int


def run_one_scenario(
    *,
    params: ScenarioParams,
    matcher: MatcherFn,
    run_task2_checks: bool,
) -> Dict[str, Number]:
    """
    Algorithm-agnostic experiment runner.
    Only difference between greedy/hungarian is which matcher function we pass in.
    """
    avs, pvs, l_min = generate_mock_data(
        num_av=params.num_av,
        num_pv=params.num_pv,
        highway_length=params.highway_length,
        av_capacity_range=params.av_capacity_range,
        min_trip_length=params.min_trip_length,
        seed=params.seed,
    )

    if run_task2_checks:
        analyze_trip_distribution(avs, params.highway_length, label="AV")
        analyze_trip_distribution(pvs, params.highway_length, label="PV")
        analyze_feasible_pairs(avs, pvs, l_min)

    baseline_total = baseline_total_powered_distance(avs, pvs)

    start_time = time.perf_counter()
    assignments, _ = matcher(avs, pvs, l_min)
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

    cap_min, cap_max = params.av_capacity_range

    return {
        "highway_length": params.highway_length,
        "min_trip_length": params.min_trip_length,
        "capacity_min": cap_min,
        "capacity_max": cap_max,
        "num_av": params.num_av,
        "num_pv": params.num_pv,
        "runtime_sec": runtime_sec,
        "baseline_total_distance": baseline_total,
        "greedy_total_distance": greedy_total,
        "total_saving": metrics["total_saving"],
        "matched_pv": metrics["matched_pv"],
        "matched_ratio": metrics["matched_ratio"],
        "avg_saving_per_pv": metrics["avg_saving_per_pv"],
        "saving_percent": metrics["saving_percent"],
    }
