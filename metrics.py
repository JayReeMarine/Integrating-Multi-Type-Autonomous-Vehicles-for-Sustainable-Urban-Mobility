from __future__ import annotations

from typing import List, Tuple

from models import ActiveVehicle, PassiveVehicle
from greedy import Assignment


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
