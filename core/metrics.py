from __future__ import annotations

from typing import List, Tuple, Dict

from core.models import ActiveVehicle, PassiveVehicle
from core.greedy import Assignment


def trip_length(entry: int, exit: int) -> int:
    """Total travel distance for a vehicle on 1D highway (=exit-entry)."""
    return exit - entry


def baseline_total_powered_distance(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
) -> int:
    """
    Baseline: No platooning.
    All vehicles move from entry->exit using their own propulsion.
    """
    total_av = sum(trip_length(av.entry_point, av.exit_point) for av in avs)
    total_pv = sum(trip_length(pv.entry_point, pv.exit_point) for pv in pvs)
    return total_av + total_pv


def greedy_total_powered_distance(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    assignments: List[Assignment],
) -> int:
    """
    Greedy: PVs turn off propulsion when being towed in segment (cp->dp).
    Therefore, PV's 'self-propelled distance' is reduced by the towing distance.
    (AV's additional resistance/fuel efficiency degradation is ignored in simplified model)
    """
    baseline = baseline_total_powered_distance(avs, pvs)
    towed_total = sum(a.saved_distance for a in assignments)
    return baseline - towed_total


def compute_saving_stats(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    assignments: List[Assignment],
) -> Tuple[int, int, float]:
    """
    Returns:
        baseline_total (int)
        greedy_total (int)
        saving_percent (float)  # in percentage
    """
    baseline_total = baseline_total_powered_distance(avs, pvs)
    greedy_total = greedy_total_powered_distance(avs, pvs, assignments)

    saved = baseline_total - greedy_total  # == sum(saved_distance)
    saving_percent = (saved / baseline_total * 100.0) if baseline_total > 0 else 0.0

    return baseline_total, greedy_total, saving_percent


def compute_extended_metrics(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    assignments: List[Assignment],
    baseline_total: int,
    greedy_total: int,
    runtime_sec: float,
) -> Dict[str, float]:
    """
    Compute extended evaluation metrics.
    Keys are standardized for CSV / plotting.
    """
    matched_pvs = len(assignments)
    total_pvs = len(pvs)

    total_saving = baseline_total - greedy_total
    matched_ratio = matched_pvs / total_pvs if total_pvs > 0 else 0.0
    avg_saving_per_pv = (
        total_saving / matched_pvs if matched_pvs > 0 else 0.0
    )

    return {
        "baseline_total": baseline_total,
        "greedy_total": greedy_total,
        "total_saving": total_saving,
        "saving_percent": (
            total_saving / baseline_total * 100.0
            if baseline_total > 0 else 0.0
        ),
        "matched_pv": matched_pvs,
        "matched_ratio": matched_ratio,
        "avg_saving_per_pv": avg_saving_per_pv,
        "runtime_sec": runtime_sec,
    }

