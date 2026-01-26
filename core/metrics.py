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


def compute_av_utilization(
    avs: List[ActiveVehicle],
    assignments: List[SegmentAssignment],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute AV capacity utilization metrics.

    Utilization = (total PV-distance carried) / (AV capacity × AV route length)

    This measures how efficiently AVs are using their capacity.
    - 100% = AV always carrying at full capacity
    - 50% = On average, half the capacity is used
    - 0% = AV carrying no PVs

    Returns:
        avg_utilization: Average utilization across all AVs (%)
        weighted_utilization: Weighted by AV route length (%)
        per_av_utilization: Dict mapping AV id to its utilization (%)
    """
    if not avs:
        return 0.0, 0.0, {}

    # Group assignments by AV
    av_assignments: Dict[str, List[SegmentAssignment]] = {av.id: [] for av in avs}
    for assignment in assignments:
        av_assignments[assignment.av.id].append(assignment)

    per_av_utilization: Dict[str, float] = {}
    total_weighted_util = 0.0
    total_av_distance = 0.0

    for av in avs:
        av_route_length = av.exit_point - av.entry_point
        if av_route_length <= 0 or av.capacity <= 0:
            per_av_utilization[av.id] = 0.0
            continue

        # Calculate total PV-distance for this AV
        # PV-distance = sum of (dp - cp) for all assigned segments
        pv_distance = sum(a.saved_distance for a in av_assignments[av.id])

        # Maximum possible = capacity × route_length
        max_pv_distance = av.capacity * av_route_length

        # Utilization for this AV
        util = (pv_distance / max_pv_distance * 100.0) if max_pv_distance > 0 else 0.0
        per_av_utilization[av.id] = util

        # For weighted average
        total_weighted_util += util * av_route_length
        total_av_distance += av_route_length

    # Calculate averages
    avg_utilization = sum(per_av_utilization.values()) / len(avs) if avs else 0.0
    weighted_utilization = (
        total_weighted_util / total_av_distance
        if total_av_distance > 0 else 0.0
    )

    return avg_utilization, weighted_utilization, per_av_utilization


def compute_pv_coverage(
    pvs: List[PassiveVehicle],
    assignments: List[SegmentAssignment],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute PV route coverage metrics.

    Coverage = (matched distance) / (total PV route distance)

    This measures what fraction of PV routes are covered by AV matching.

    Returns:
        avg_coverage: Average coverage across all PVs (%)
        weighted_coverage: Weighted by PV route length (%)
        per_pv_coverage: Dict mapping PV id to its coverage (%)
    """
    if not pvs:
        return 0.0, 0.0, {}

    # Group assignments by PV
    pv_assignments: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}
    for assignment in assignments:
        pv_assignments[assignment.pv.id].append(assignment)

    per_pv_coverage: Dict[str, float] = {}
    total_weighted_cov = 0.0
    total_pv_distance = 0.0

    for pv in pvs:
        pv_route_length = pv.exit_point - pv.entry_point
        if pv_route_length <= 0:
            per_pv_coverage[pv.id] = 0.0
            continue

        # Calculate matched distance for this PV
        matched_distance = sum(a.saved_distance for a in pv_assignments[pv.id])

        # Coverage for this PV
        cov = (matched_distance / pv_route_length * 100.0) if pv_route_length > 0 else 0.0
        per_pv_coverage[pv.id] = cov

        # For weighted average
        total_weighted_cov += cov * pv_route_length
        total_pv_distance += pv_route_length

    # Calculate averages
    avg_coverage = sum(per_pv_coverage.values()) / len(pvs) if pvs else 0.0
    weighted_coverage = (
        total_weighted_cov / total_pv_distance
        if total_pv_distance > 0 else 0.0
    )

    return avg_coverage, weighted_coverage, per_pv_coverage


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
      - (NEW) av_utilization, pv_coverage
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

    # NEW: Compute utilization and coverage metrics
    avg_av_util, weighted_av_util, _ = compute_av_utilization(avs, assignments)
    avg_pv_cov, weighted_pv_cov, _ = compute_pv_coverage(pvs, assignments)

    return {
        "baseline_total": baseline_total,
        "greedy_total": greedy_total,
        "total_saving": total_saving,
        "saving_percent": saving_percent,
        "matched_pv": matched_pvs,
        "matched_ratio": matched_ratio,
        "avg_saving_per_pv": avg_saving_per_pv,
        "runtime_sec": runtime_sec,
        # NEW: Utilization metrics
        "av_utilization": avg_av_util,
        "av_utilization_weighted": weighted_av_util,
        "pv_coverage": avg_pv_cov,
        "pv_coverage_weighted": weighted_pv_cov,
    }
