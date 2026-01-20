"""
Greedy Multi-AV Matching Algorithm

Priority 2: Allow ONE PV to be matched to MULTIPLE AVs along different segments.

Key Changes from Original:
1. Track "remaining segments" instead of "used PVs"
2. New SegmentAssignment dataclass for segment-level matching
3. PV can have multiple assignments across different route segments

Step 4 Enhancement: Time-based constraints
- Each vehicle has entry_time and speed
- Matching requires time synchronization at coupling point
- Time tolerance parameter allows for realistic matching windows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from core.models import ActiveVehicle, PassiveVehicle


# =============================================================================
# TIME CONSTRAINT CONFIGURATION (Step 4)
# =============================================================================

# Default time tolerance for coupling (time units)
# AV and PV must arrive at coupling point within this time window
DEFAULT_TIME_TOLERANCE: float = 5.0


# =============================================================================
# NEW DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Segment:
    """
    Represents a segment of a PV's route.

    A PV's full route can be divided into multiple segments,
    each potentially matched to a different AV.
    """
    start: int  # segment start point
    end: int    # segment end point

    @property
    def length(self) -> int:
        return self.end - self.start

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(f"Segment start({self.start}) must be < end({self.end})")


@dataclass(frozen=True)
class SegmentAssignment:
    """
    Assignment of a PV SEGMENT to an AV.

    Unlike original Assignment which represents whole-route matching,
    this represents a single segment of PV's journey being towed by an AV.

    Example:
        PV1 (entry=1, exit=5) could have:
        - SegmentAssignment(pv=PV1, av=AV1, cp=1, dp=3)  # segment 1→3
        - SegmentAssignment(pv=PV1, av=AV2, cp=3, dp=5)  # segment 3→5

    Step 4 Enhancement:
        - coupling_time: Time when AV and PV meet at coupling point
        - decoupling_time: Time when PV is released at decoupling point
    """
    pv: PassiveVehicle
    av: ActiveVehicle
    cp: int  # coupling point (start of this segment)
    dp: int  # decoupling point (end of this segment)
    coupling_time: float = 0.0    # Time at coupling point (Step 4)
    decoupling_time: float = 0.0  # Time at decoupling point (Step 4)

    @property
    def saved_distance(self) -> int:
        """Distance saved by this segment assignment."""
        return self.dp - self.cp

    @property
    def segment(self) -> Segment:
        """The segment this assignment covers."""
        return Segment(self.cp, self.dp)

    @property
    def towing_duration(self) -> float:
        """Duration of towing (Step 4)."""
        return self.decoupling_time - self.coupling_time


@dataclass
class PVRoutingState:
    """
    Tracks the routing state of a PV across the matching process.

    Key concept: Instead of marking PV as "used" after one match,
    we track which segments are still "uncovered" and available for matching.

    Example:
        PV1: entry=0, exit=10
        Initial uncovered: [(0, 10)]
        After matching to AV1 for segment (2, 6):
        Uncovered becomes: [(0, 2), (6, 10)]  # two remaining segments

    Step 4 Enhancement:
        - Each uncovered segment also tracks its entry_time (when PV arrives at segment start)
        - uncovered_segments: List[(start, end, entry_time)]
    """
    pv: PassiveVehicle
    uncovered_segments: List[Tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.uncovered_segments:
            # Initially, entire route is uncovered
            # (start_point, end_point, time_at_start)
            self.uncovered_segments = [
                (self.pv.entry_point, self.pv.exit_point, self.pv.entry_time)
            ]

    @property
    def total_uncovered_distance(self) -> int:
        """Total distance still not matched to any AV."""
        return sum(end - start for start, end, _ in self.uncovered_segments)

    @property
    def is_fully_covered(self) -> bool:
        """Check if PV's entire route is covered by AVs."""
        return len(self.uncovered_segments) == 0

    def get_time_at_point(self, point: int) -> Optional[float]:
        """
        Calculate when PV reaches a given point based on current routing state.

        Step 4: This accounts for previous towing segments where PV traveled
        at AV's speed rather than its own speed.
        """
        for seg_start, seg_end, seg_entry_time in self.uncovered_segments:
            if seg_start <= point <= seg_end:
                # PV is self-driving in this segment
                return seg_entry_time + (point - seg_start) / self.pv.speed
        return None

    def get_overlap_with_av(
        self,
        av: ActiveVehicle,
        time_tolerance: float = DEFAULT_TIME_TOLERANCE,
        enable_time_constraints: bool = False
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Find the best overlapping segment between this PV's uncovered portions
        and the given AV's route.

        Step 4 Enhancement:
        - When enable_time_constraints=True, also checks time feasibility
        - AV and PV must arrive at coupling point within time_tolerance

        Returns:
            Tuple of (cp, dp, coupling_time, decoupling_time) or None
            - cp: coupling point
            - dp: decoupling point
            - coupling_time: when both vehicles meet at cp
            - decoupling_time: when towing ends at dp
        """
        best_overlap = None
        best_length = 0

        for seg_start, seg_end, seg_entry_time in self.uncovered_segments:
            # Compute spatial overlap between uncovered segment and AV route
            cp = max(seg_start, av.entry_point)
            dp = min(seg_end, av.exit_point)

            if dp <= cp:
                continue  # No spatial overlap

            segment_length = dp - cp

            if enable_time_constraints:
                # Step 4: Time-based feasibility check
                # Calculate when PV arrives at potential coupling point
                pv_time_at_cp = seg_entry_time + (cp - seg_start) / self.pv.speed

                # Calculate when AV arrives at potential coupling point
                av_time_at_cp = av.time_at_point(cp)
                if av_time_at_cp is None:
                    continue

                # Check if times are within tolerance
                time_diff = abs(pv_time_at_cp - av_time_at_cp)
                if time_diff > time_tolerance:
                    continue  # Time mismatch - cannot couple

                # Coupling time is when both vehicles can meet (the later arrival)
                coupling_time = max(pv_time_at_cp, av_time_at_cp)

                # After coupling, PV travels at AV's speed
                towing_duration = (dp - cp) / av.speed
                decoupling_time = coupling_time + towing_duration

                # Verify AV can reach decoupling point
                av_time_at_dp = av.time_at_point(dp)
                if av_time_at_dp is None:
                    continue
            else:
                # No time constraints - use default values
                coupling_time = 0.0
                decoupling_time = 0.0

            if segment_length > best_length:
                best_overlap = (cp, dp, coupling_time, decoupling_time)
                best_length = segment_length

        return best_overlap

    def mark_segment_covered(
        self,
        cp: int,
        dp: int,
        coupling_time: float = 0.0,
        decoupling_time: float = 0.0,
        av_speed: float = 1.0
    ) -> None:
        """
        Mark a segment as covered (matched to an AV).
        Updates uncovered_segments by removing the covered portion.

        Step 4 Enhancement:
        - Updates entry times for remaining segments based on towing
        - After being towed, PV resumes self-driving at decoupling point

        Args:
            cp: coupling point
            dp: decoupling point
            coupling_time: when coupling occurs (Step 4)
            decoupling_time: when decoupling occurs (Step 4)
            av_speed: speed of AV during towing (Step 4)
        """
        new_uncovered = []

        for seg_start, seg_end, seg_entry_time in self.uncovered_segments:
            if dp <= seg_start or cp >= seg_end:
                # No overlap, keep segment as is
                new_uncovered.append((seg_start, seg_end, seg_entry_time))
            else:
                # There is overlap, split the segment
                if seg_start < cp:
                    # Left portion remains uncovered (before coupling)
                    new_uncovered.append((seg_start, cp, seg_entry_time))
                if dp < seg_end:
                    # Right portion remains uncovered (after decoupling)
                    # PV resumes self-driving at decoupling_time
                    new_uncovered.append((dp, seg_end, decoupling_time))

        self.uncovered_segments = new_uncovered


@dataclass
class AVCapacityState:
    """
    Tracks AV capacity at each point along the highway.

    Key insight: In multi-segment matching, AV capacity is not just a single number.
    Capacity usage varies along the route as PVs attach and detach.

    Example:
        AV1 (capacity=3, route: 0→10)
        - PV1 attaches at 2, detaches at 5
        - PV2 attaches at 3, detaches at 8

        Capacity usage at different points:
        Point 0-2: 0 PVs
        Point 2-3: 1 PV (PV1)
        Point 3-5: 2 PVs (PV1, PV2)
        Point 5-8: 1 PV (PV2)
        Point 8-10: 0 PVs
    """
    av: ActiveVehicle
    # List of (attach_point, detach_point) for each currently assigned PV
    assigned_segments: List[Tuple[int, int]] = field(default_factory=list)

    def get_capacity_at_point(self, point: int) -> int:
        """Get number of PVs attached at a specific point."""
        count = 0
        for cp, dp in self.assigned_segments:
            if cp <= point < dp:
                count += 1
        return count

    def get_available_capacity_for_segment(self, cp: int, dp: int) -> int:
        """
        Get the minimum available capacity across a segment.

        This is crucial: we need to check capacity at ALL points in the segment,
        not just at one point. The bottleneck determines if we can add a PV.
        """
        # Check capacity at critical points (where capacity changes)
        critical_points = {cp, dp}
        for seg_cp, seg_dp in self.assigned_segments:
            if cp <= seg_cp < dp:
                critical_points.add(seg_cp)
            if cp < seg_dp <= dp:
                critical_points.add(seg_dp)

        max_usage = 0
        for point in critical_points:
            if cp <= point < dp:
                usage = self.get_capacity_at_point(point)
                max_usage = max(max_usage, usage)

        return self.av.capacity - max_usage

    def can_accommodate_segment(self, cp: int, dp: int) -> bool:
        """Check if AV can accommodate a new PV for the given segment."""
        return self.get_available_capacity_for_segment(cp, dp) >= 1

    def add_assignment(self, cp: int, dp: int) -> None:
        """Record a new PV assignment for a segment."""
        self.assigned_segments.append((cp, dp))


# =============================================================================
# MULTI-AV GREEDY ALGORITHM
# =============================================================================

def compute_shared_segment_multi(
    av: ActiveVehicle,
    pv_state: PVRoutingState,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
    enable_time_constraints: bool = False
) -> Optional[Tuple[int, int, float, float]]:
    """
    Compute the best shared segment between AV and PV's uncovered portions.

    Difference from original:
    - Original: looks at PV's full route
    - This: looks at PV's UNCOVERED segments only

    Step 4 Enhancement:
    - Also considers time feasibility when enable_time_constraints=True

    Returns:
        Tuple of (cp, dp, coupling_time, decoupling_time) or None
    """
    return pv_state.get_overlap_with_av(av, time_tolerance, enable_time_constraints)


def greedy_multi_av_matching(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    l_min: int,
    *,
    enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE
) -> Tuple[List[SegmentAssignment], float, Dict[str, List[SegmentAssignment]]]:
    """
    Greedy Multi-AV Platoon Matching

    Key differences from original greedy_platoon_matching:

    1. ITERATION: Instead of single-pass, we iterate until no more valid matches
       - Each iteration finds new opportunities as PV segments get split

    2. STATE TRACKING:
       - PV: Track uncovered segments (not just "used" flag)
       - AV: Track capacity at each point along route (not just total count)

    3. CANDIDATE GENERATION:
       - Regenerate candidates each iteration based on current uncovered segments

    Step 4 Enhancement:
    - enable_time_constraints: If True, considers time feasibility for matching
    - time_tolerance: Maximum time difference allowed for coupling (default: 5.0)

    When time constraints are enabled:
    - AV and PV must arrive at coupling point within time_tolerance
    - This significantly reduces feasible matches, making optimization harder
    - Hungarian algorithm may show more benefit over Greedy in this scenario

    Returns:
        - assignments: List of SegmentAssignment
        - total_saving: Sum of all saved distances
        - pv_assignments: Dict mapping PV id to list of its assignments
    """

    # Initialize state trackers
    pv_states: Dict[str, PVRoutingState] = {
        pv.id: PVRoutingState(pv=pv) for pv in pvs
    }

    av_states: Dict[str, AVCapacityState] = {
        av.id: AVCapacityState(av=av) for av in avs
    }

    all_assignments: List[SegmentAssignment] = []
    total_saving = 0.0

    # Track assignments per PV for analysis
    pv_assignments: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}

    # Iterate until no more valid matches can be made
    iteration = 0
    while True:
        iteration += 1

        # Generate candidates based on CURRENT uncovered segments
        # Step 4: Extended tuple to include time information
        candidates: List[Tuple[int, ActiveVehicle, PassiveVehicle, int, int, float, float]] = []

        for av in avs:
            av_state = av_states[av.id]

            for pv in pvs:
                pv_state = pv_states[pv.id]

                # Skip fully covered PVs
                if pv_state.is_fully_covered:
                    continue

                # Find overlap with UNCOVERED portions (Step 4: with time constraints)
                overlap = compute_shared_segment_multi(
                    av, pv_state, time_tolerance, enable_time_constraints
                )
                if overlap is None:
                    continue

                cp, dp, coupling_time, decoupling_time = overlap
                segment_length = dp - cp

                # Check minimum length constraint
                if segment_length < l_min:
                    continue

                # Check AV capacity for this segment
                if not av_state.can_accommodate_segment(cp, dp):
                    continue

                # Valid candidate (Step 4: include time info)
                candidates.append((
                    segment_length, av, pv, cp, dp, coupling_time, decoupling_time
                ))

        # No more valid candidates - we're done
        if not candidates:
            break

        # Sort by saving (segment length) descending - greedy choice
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Take the best candidate
        saving, av, pv, cp, dp, coupling_time, decoupling_time = candidates[0]

        # Make the assignment (Step 4: include time info)
        assignment = SegmentAssignment(
            pv=pv, av=av, cp=cp, dp=dp,
            coupling_time=coupling_time,
            decoupling_time=decoupling_time
        )
        all_assignments.append(assignment)
        pv_assignments[pv.id].append(assignment)

        # Update states (Step 4: pass time info for proper segment tracking)
        pv_states[pv.id].mark_segment_covered(
            cp, dp, coupling_time, decoupling_time, av.speed
        )
        av_states[av.id].add_assignment(cp, dp)

        total_saving += saving

    return all_assignments, total_saving, pv_assignments


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_multi_matching_results(
    pvs: List[PassiveVehicle],
    pv_assignments: Dict[str, List[SegmentAssignment]]
) -> Dict:
    """
    Analyze the results of multi-AV matching.

    Returns statistics about:
    - How many PVs got matched to multiple AVs
    - Coverage percentage
    - etc.
    """
    stats = {
        "total_pvs": len(pvs),
        "pvs_with_0_avs": 0,
        "pvs_with_1_av": 0,
        "pvs_with_multi_avs": 0,
        "max_avs_per_pv": 0,
        "total_segments_assigned": 0,
        "coverage_ratios": [],
    }

    for pv in pvs:
        assignments = pv_assignments[pv.id]
        num_avs = len(assignments)

        if num_avs == 0:
            stats["pvs_with_0_avs"] += 1
        elif num_avs == 1:
            stats["pvs_with_1_av"] += 1
        else:
            stats["pvs_with_multi_avs"] += 1

        stats["max_avs_per_pv"] = max(stats["max_avs_per_pv"], num_avs)
        stats["total_segments_assigned"] += num_avs

        # Calculate coverage ratio for this PV
        total_route = pv.exit_point - pv.entry_point
        covered = sum(a.saved_distance for a in assignments)
        coverage_ratio = covered / total_route if total_route > 0 else 0
        stats["coverage_ratios"].append(coverage_ratio)

    stats["avg_coverage_ratio"] = (
        sum(stats["coverage_ratios"]) / len(stats["coverage_ratios"])
        if stats["coverage_ratios"] else 0
    )

    return stats


def print_pv_route_breakdown(
    pv: PassiveVehicle,
    assignments: List[SegmentAssignment]
) -> str:
    """
    Generate a visual breakdown of a PV's route and AV assignments.

    Example output:
    PV1 Route: [0 ====== 10]
      Segment 0→3: AV2 (saved: 3)
      Segment 3→7: AV1 (saved: 4)
      Segment 7→10: No AV (self-driving)
    """
    lines = [f"\n{pv.id} Route: [{pv.entry_point} → {pv.exit_point}]"]

    if not assignments:
        lines.append(f"  Entire route: No AV (self-driving)")
        return "\n".join(lines)

    # Sort assignments by coupling point
    sorted_assignments = sorted(assignments, key=lambda a: a.cp)

    current_point = pv.entry_point

    for assignment in sorted_assignments:
        # Gap before this assignment?
        if current_point < assignment.cp:
            lines.append(
                f"  Segment {current_point}→{assignment.cp}: "
                f"No AV (self-driving, distance: {assignment.cp - current_point})"
            )

        lines.append(
            f"  Segment {assignment.cp}→{assignment.dp}: "
            f"{assignment.av.id} (saved: {assignment.saved_distance})"
        )
        current_point = assignment.dp

    # Gap after last assignment?
    if current_point < pv.exit_point:
        lines.append(
            f"  Segment {current_point}→{pv.exit_point}: "
            f"No AV (self-driving, distance: {pv.exit_point - current_point})"
        )

    return "\n".join(lines)
