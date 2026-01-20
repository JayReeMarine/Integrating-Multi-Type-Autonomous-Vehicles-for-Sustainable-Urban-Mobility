# experiments/common.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Union, Optional

from core.data import generate_mock_data
from core.greedy_multi import SegmentAssignment, DEFAULT_TIME_TOLERANCE
from core.metrics import (
    baseline_total_powered_distance,
    greedy_total_powered_distance,
    compute_extended_metrics,
)
from core.analysis import analyze_trip_distribution, analyze_feasible_pairs


Number = Union[int, float]

# Changed: Matcher now returns 3 values (assignments, total_saving, pv_assignments_dict)
# Step 4: Matcher can accept keyword arguments for time constraints
MatcherFn = Callable[..., Tuple[List[SegmentAssignment], Number, Dict]]


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

    # Step 4: Time constraint fields
    "enable_time_constraints",
    "time_tolerance",
    "time_window",

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
    """
    Parameters for a single experiment scenario.

    Step 4 Enhancement:
    - enable_time_constraints: If True, vehicles have time-based entry/exit
    - time_tolerance: Maximum time difference allowed for AV-PV coupling
    - time_window: Time span during which vehicles can enter highway
    - av_speed_range: Speed range for AVs (min, max)
    - pv_speed_range: Speed range for PVs (min, max)
    """
    num_av: int
    num_pv: int
    highway_length: int
    av_capacity_range: Tuple[int, int]
    min_trip_length: int
    seed: int

    # Step 4: Time constraint parameters
    enable_time_constraints: bool = False
    time_tolerance: float = DEFAULT_TIME_TOLERANCE
    time_window: float = 100.0
    av_speed_range: Optional[Tuple[float, float]] = None
    pv_speed_range: Optional[Tuple[float, float]] = None


def run_one_scenario(
    *,
    params: ScenarioParams,
    matcher: MatcherFn,
    run_task2_checks: bool,
) -> Dict[str, Number]:
    """
    Algorithm-agnostic experiment runner.
    Only difference between greedy/hungarian is which matcher function we pass in.

    Step 4 Enhancement:
    - Supports time constraint parameters
    - Passes enable_time_constraints and time_tolerance to matcher
    """
    avs, pvs, l_min = generate_mock_data(
        num_av=params.num_av,
        num_pv=params.num_pv,
        highway_length=params.highway_length,
        av_capacity_range=params.av_capacity_range,
        min_trip_length=params.min_trip_length,
        seed=params.seed,
        # Step 4: Time parameters
        enable_time_constraints=params.enable_time_constraints,
        av_speed_range=params.av_speed_range,
        pv_speed_range=params.pv_speed_range,
        time_window=params.time_window,
    )

    if run_task2_checks:
        analyze_trip_distribution(avs, params.highway_length, label="AV")
        analyze_trip_distribution(pvs, params.highway_length, label="PV")
        analyze_feasible_pairs(avs, pvs, l_min)

    baseline_total = baseline_total_powered_distance(avs, pvs)

    start_time = time.perf_counter()
    # Step 4: Pass time constraint parameters to matcher
    assignments, _, _ = matcher(
        avs, pvs, l_min,
        enable_time_constraints=params.enable_time_constraints,
        time_tolerance=params.time_tolerance
    )
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
        # Step 4: Time constraint fields
        "enable_time_constraints": params.enable_time_constraints,
        "time_tolerance": params.time_tolerance,
        "time_window": params.time_window,
        # Metrics
        "runtime_sec": runtime_sec,
        "baseline_total_distance": baseline_total,
        "greedy_total_distance": greedy_total,
        "total_saving": metrics["total_saving"],
        "matched_pv": metrics["matched_pv"],
        "matched_ratio": metrics["matched_ratio"],
        "avg_saving_per_pv": metrics["avg_saving_per_pv"],
        "saving_percent": metrics["saving_percent"],
    }
