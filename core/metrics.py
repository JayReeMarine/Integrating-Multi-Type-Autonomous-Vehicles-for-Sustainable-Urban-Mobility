from __future__ import annotations

from typing import List, Tuple, Dict, Union

from core.models import ActiveVehicle, PassiveVehicle
from core.greedy_multi import SegmentAssignment  # Changed: Assignment -> SegmentAssignment


Number = Union[int, float]


def trip_length(entry: int, exit: int) -> int:
    """Total travel distance for a vehicle on 1D highway (= exit - entry)."""
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
    baseline_total: int,
    assignments: List[SegmentAssignment],
) -> int:
    """
    Greedy: PVs turn off propulsion when being towed in segment (cp->dp),
    so PV's self-propelled distance is reduced by the towing distance.

    Note: We ignore AV extra resistance/fuel efficiency degradation in this simplified model.
    """
    towed_total = sum(a.saved_distance for a in assignments)
    return baseline_total - towed_total


def compute_saving_stats(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    assignments: List[SegmentAssignment],
) -> Tuple[int, int, float]:
    """
    Returns:
        baseline_total (int)
        greedy_total (int)
        saving_percent (float)  # percentage
    """
    baseline_total = baseline_total_powered_distance(avs, pvs)
    greedy_total = greedy_total_powered_distance(baseline_total, assignments)

    saved = baseline_total - greedy_total  # == sum(saved_distance)
    saving_percent = (saved / baseline_total * 100.0) if baseline_total > 0 else 0.0

    return baseline_total, greedy_total, saving_percent


def compute_extended_metrics(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    assignments: List[SegmentAssignment],
    baseline_total: int,
    greedy_total: int,
    runtime_sec: float,
) -> Dict[str, Number]:
    """
    Compute extended evaluation metrics.
    Keys are standardized for CSV / plotting.

    Expected keys used by experiments/main.py:
      - total_saving
      - matched_pv
      - matched_ratio
      - avg_saving_per_pv
      - (optional) runtime_sec, saving_percent, baseline_total, greedy_total
    """
    # Changed: Count unique PVs (one PV can have multiple segment assignments)
    matched_pvs = len(set(a.pv.id for a in assignments))
    total_pvs = len(pvs)

    total_saving = baseline_total - greedy_total

    matched_ratio = matched_pvs / total_pvs if total_pvs > 0 else 0.0
    avg_saving_per_pv = total_saving / matched_pvs if matched_pvs > 0 else 0.0

    saving_percent = (
        total_saving / baseline_total * 100.0
        if baseline_total > 0 else 0.0
    )

    return {
        "baseline_total": baseline_total,
        "greedy_total": greedy_total,
        "total_saving": total_saving,
        "saving_percent": saving_percent,
        "matched_pv": matched_pvs,
        "matched_ratio": matched_ratio,
        "avg_saving_per_pv": avg_saving_per_pv,
        "runtime_sec": runtime_sec,
    }
