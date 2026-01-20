"""
Hungarian Multi-AV Matching Algorithm

Priority 2: Allow ONE PV to be matched to MULTIPLE AVs along different segments.

=============================================================================
ALGORITHM DESIGN (Professor's Perspective)
=============================================================================

CHALLENGE:
- Standard Hungarian solves assignment problems where each PV maps to ONE AV
- We need PV → [AV₁(seg1), AV₂(seg2), ...] mapping

SOLUTION APPROACH: Iterative Hungarian with Segment Tracking
- Similar to Greedy Multi but uses Hungarian for OPTIMAL selection each round
- Each iteration: Hungarian finds the best current assignment
- After assignment: Update PV's uncovered segments, AV's capacity at each point
- Repeat until no more valid assignments possible

WHY THIS APPROACH?
1. Pure Hungarian can't handle segment splitting natively
2. Iterative approach maintains optimality within each round
3. Greedy-like iteration but with Hungarian's optimal matching per round
4. Practical balance between optimality and complexity

COMPLEXITY ANALYSIS:
- Each Hungarian call: O(n³) where n = max(slots, PVs)
- Number of iterations: O(total_segments_created)
- Total: O(iterations × n³) - more expensive than Greedy Multi but more optimal

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from core.models import ActiveVehicle, PassiveVehicle

# Step 4: Default time tolerance for coupling
DEFAULT_TIME_TOLERANCE: float = 5.0


# =============================================================================
# REUSE DATA STRUCTURES FROM greedy_multi.py
# =============================================================================

@dataclass(frozen=True)
class SegmentAssignment:
    """
    Assignment of a PV SEGMENT to an AV.
    Same structure as greedy_multi for compatibility.
    """
    pv: PassiveVehicle
    av: ActiveVehicle
    cp: int  # coupling point
    dp: int  # decoupling point

    @property
    def saved_distance(self) -> int:
        return self.dp - self.cp


@dataclass
class PVRoutingState:
    """
    Tracks uncovered segments of a PV's route.
    Reused from greedy_multi concept.

    Step 4 Enhancement:
    - Supports time-based matching constraints
    - Tracks current_time for sequential segment assignments
    """
    pv: PassiveVehicle
    uncovered_segments: List[Tuple[int, int]] = field(default_factory=list)
    # Step 4: Track time state for sequential assignments
    current_time: float = 0.0

    def __post_init__(self):
        if not self.uncovered_segments:
            self.uncovered_segments = [(self.pv.entry_point, self.pv.exit_point)]
        # Step 4: Initialize current_time to PV's entry time
        if self.current_time == 0.0:
            self.current_time = self.pv.entry_time

    @property
    def is_fully_covered(self) -> bool:
        return len(self.uncovered_segments) == 0

    def get_overlap_with_av(
        self,
        av: ActiveVehicle,
        enable_time_constraints: bool = False,
        time_tolerance: float = DEFAULT_TIME_TOLERANCE,
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Find best overlapping segment between uncovered portions and AV route.

        Step 4 Enhancement:
        - If enable_time_constraints is True, checks time compatibility
        - Returns (cp, dp, coupling_time, decoupling_time) or None

        Returns:
            If time constraints enabled: (cp, dp, coupling_time, decoupling_time)
            If time constraints disabled: (cp, dp, 0.0, 0.0) for compatibility
            None if no valid overlap
        """
        best_overlap = None
        best_length = 0

        for seg_start, seg_end in self.uncovered_segments:
            cp = max(seg_start, av.entry_point)
            dp = min(seg_end, av.exit_point)

            if dp <= cp:
                continue

            # Step 4: Time constraint check
            if enable_time_constraints:
                # Calculate time at coupling point for both vehicles
                pv_time_at_cp = self.pv.time_at_point(cp)
                av_time_at_cp = av.time_at_point(cp)

                if pv_time_at_cp is None or av_time_at_cp is None:
                    continue

                # Check if they arrive within time tolerance
                time_diff = abs(pv_time_at_cp - av_time_at_cp)
                if time_diff > time_tolerance:
                    continue

                # Calculate coupling and decoupling times
                coupling_time = max(pv_time_at_cp, av_time_at_cp)
                decoupling_time = coupling_time + (dp - cp) / av.speed

                if (dp - cp) > best_length:
                    best_overlap = (cp, dp, coupling_time, decoupling_time)
                    best_length = dp - cp
            else:
                # No time constraints
                if (dp - cp) > best_length:
                    best_overlap = (cp, dp, 0.0, 0.0)
                    best_length = dp - cp

        return best_overlap

    def mark_segment_covered(self, cp: int, dp: int) -> None:
        """Mark a segment as covered, splitting uncovered segments."""
        new_uncovered = []

        for seg_start, seg_end in self.uncovered_segments:
            if dp <= seg_start or cp >= seg_end:
                new_uncovered.append((seg_start, seg_end))
            else:
                if seg_start < cp:
                    new_uncovered.append((seg_start, cp))
                if dp < seg_end:
                    new_uncovered.append((dp, seg_end))

        self.uncovered_segments = new_uncovered


@dataclass
class AVCapacityState:
    """
    Tracks AV capacity at each point along the highway.
    Capacity varies as PVs attach/detach at different points.
    """
    av: ActiveVehicle
    assigned_segments: List[Tuple[int, int]] = field(default_factory=list)

    def get_capacity_at_point(self, point: int) -> int:
        """Get number of PVs attached at a specific point."""
        count = 0
        for cp, dp in self.assigned_segments:
            if cp <= point < dp:
                count += 1
        return count

    def get_available_capacity_for_segment(self, cp: int, dp: int) -> int:
        """Get minimum available capacity across a segment."""
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
        return self.get_available_capacity_for_segment(cp, dp) >= 1

    def add_assignment(self, cp: int, dp: int) -> None:
        self.assigned_segments.append((cp, dp))


# =============================================================================
# HUNGARIAN ALGORITHM CORE (from original hungarian.py)
# =============================================================================

def _hungarian_min_cost(cost: List[List[float]]) -> List[int]:
    """
    Solve the assignment problem (min-cost) for a square matrix.
    Returns assignment[i] = j means row i is matched to column j.
    """
    n = len(cost)
    if n == 0:
        return []

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = [-1] * n
    for j in range(1, n + 1):
        row = p[j]
        if row != 0:
            row_to_col[row - 1] = j - 1
    return row_to_col


# =============================================================================
# ITERATIVE HUNGARIAN FOR MULTI-AV MATCHING
# =============================================================================

def _build_cost_matrix_for_iteration(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    pv_states: Dict[str, PVRoutingState],
    av_states: Dict[str, AVCapacityState],
    l_min: int,
    enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
) -> Tuple[List[List[float]], Dict[Tuple[int, int], Tuple[int, int, int]], List[Tuple[ActiveVehicle, int]]]:
    """
    Build cost matrix for one iteration of Hungarian algorithm.

    Key difference from original:
    - Uses PV's UNCOVERED segments (not full route)
    - Checks AV's AVAILABLE capacity at segment (not just total count)

    Returns:
        - cost: Cost matrix for Hungarian
        - feasible: Dict mapping (slot_idx, pv_idx) -> (saving, cp, dp)
        - slot_to_av: List mapping slot_idx -> (AV, slot_number_for_this_av)
    """
    # Expand AVs into "virtual slots" based on AVAILABLE capacity
    slot_to_av: List[Tuple[ActiveVehicle, int]] = []

    for av in avs:
        av_state = av_states[av.id]
        # For simplicity, create slots equal to original capacity
        # Feasibility check happens when building cost matrix
        for slot_num in range(av.capacity):
            slot_to_av.append((av, slot_num))

    num_slots = len(slot_to_av)
    num_pv = len(pvs)

    if num_slots == 0 or num_pv == 0:
        return [], {}, []

    # Compute feasible pairs with current state
    feasible: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    max_saving = 0

    for slot_idx, (av, _) in enumerate(slot_to_av):
        av_state = av_states[av.id]

        for pv_idx, pv in enumerate(pvs):
            pv_state = pv_states[pv.id]

            # Skip fully covered PVs
            if pv_state.is_fully_covered:
                continue

            # Get overlap with UNCOVERED segment (Step 4: with time constraints)
            overlap = pv_state.get_overlap_with_av(
                av,
                enable_time_constraints=enable_time_constraints,
                time_tolerance=time_tolerance,
            )
            if overlap is None:
                continue

            cp, dp, _, _ = overlap  # Unpack with time values (ignored here)
            saving = dp - cp

            if saving < l_min:
                continue

            # Check AV capacity for this segment
            if not av_state.can_accommodate_segment(cp, dp):
                continue

            # Valid candidate - but only store once per (av, pv) pair
            # Use the first slot for this AV that can accommodate
            existing_key = None
            for s_idx, (s_av, _) in enumerate(slot_to_av[:slot_idx]):
                if s_av.id == av.id and (s_idx, pv_idx) in feasible:
                    existing_key = (s_idx, pv_idx)
                    break

            if existing_key is None:
                feasible[(slot_idx, pv_idx)] = (saving, cp, dp)
                max_saving = max(max_saving, saving)

    if max_saving == 0:
        return [], {}, []

    # Build cost matrix
    n = num_slots + num_pv
    BASE_UNMATCH_COST = max_saving + 1.0
    BIG_M = max_saving * 1000.0

    cost = [[BASE_UNMATCH_COST for _ in range(n)] for _ in range(n)]

    # Fill slot->PV costs
    for slot_idx in range(num_slots):
        for pv_idx in range(num_pv):
            if (slot_idx, pv_idx) in feasible:
                saving, _, _ = feasible[(slot_idx, pv_idx)]
                cost[slot_idx][pv_idx] = max_saving - saving + 1
            else:
                cost[slot_idx][pv_idx] = BIG_M

    # Dummy assignments
    for slot_idx in range(num_slots):
        for dummy_pv in range(num_pv, n):
            cost[slot_idx][dummy_pv] = BASE_UNMATCH_COST

    for dummy_slot in range(num_slots, n):
        for pv_idx in range(num_pv):
            cost[dummy_slot][pv_idx] = BASE_UNMATCH_COST

    for dummy_slot in range(num_slots, n):
        for dummy_pv in range(num_pv, n):
            cost[dummy_slot][dummy_pv] = 0.0

    return cost, feasible, slot_to_av


def hungarian_multi_av_matching(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    l_min: int,
    *,
    enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
) -> Tuple[List[SegmentAssignment], float, Dict[str, List[SegmentAssignment]]]:
    """
    Hungarian Multi-AV Matching Algorithm.

    Uses iterative Hungarian to optimally match PV segments to AVs.
    Each iteration finds the best assignment given current state,
    then updates PV uncovered segments and AV capacity.

    Step 4 Enhancement:
    - enable_time_constraints: If True, uses time-based matching
    - time_tolerance: Max time difference for coupling (default: 5.0)

    Returns:
        - assignments: List of SegmentAssignment
        - total_saving: Sum of all saved distances
        - pv_assignments: Dict mapping PV id to list of its assignments
    """
    if not avs or not pvs:
        return [], 0.0, {pv.id: [] for pv in pvs}

    # Initialize state trackers
    pv_states: Dict[str, PVRoutingState] = {
        pv.id: PVRoutingState(pv=pv) for pv in pvs
    }
    av_states: Dict[str, AVCapacityState] = {
        av.id: AVCapacityState(av=av) for av in avs
    }

    all_assignments: List[SegmentAssignment] = []
    pv_assignments: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}
    total_saving = 0.0

    iteration = 0
    max_iterations = len(avs) * len(pvs) * 2  # Safety limit

    while iteration < max_iterations:
        iteration += 1

        # Build cost matrix for current state (Step 4: with time constraints)
        cost, feasible, slot_to_av = _build_cost_matrix_for_iteration(
            avs, pvs, pv_states, av_states, l_min,
            enable_time_constraints=enable_time_constraints,
            time_tolerance=time_tolerance,
        )

        if not cost or not feasible:
            break

        num_slots = len(slot_to_av)
        num_pv = len(pvs)

        # Solve Hungarian for this iteration
        row_to_col = _hungarian_min_cost(cost)

        # Extract assignments from this iteration
        assignments_this_round: List[Tuple[SegmentAssignment, int]] = []

        for slot_idx in range(min(len(row_to_col), num_slots)):
            col_idx = row_to_col[slot_idx]

            if col_idx < 0 or col_idx >= num_pv:
                continue

            if (slot_idx, col_idx) not in feasible:
                continue

            saving, cp, dp = feasible[(slot_idx, col_idx)]
            av, _ = slot_to_av[slot_idx]
            pv = pvs[col_idx]

            assignment = SegmentAssignment(pv=pv, av=av, cp=cp, dp=dp)
            assignments_this_round.append((assignment, saving))

        # No assignments found - we're done
        if not assignments_this_round:
            break

        # Apply assignments (take all valid assignments from this round)
        # To maintain Hungarian optimality, we could take just the best one
        # But taking all valid ones from the optimal matching is also valid

        # Strategy: Take assignments that don't conflict
        # (same PV segment can't be assigned twice in same round)
        used_pv_segments: Dict[str, List[Tuple[int, int]]] = {}

        for assignment, saving in assignments_this_round:
            pv_id = assignment.pv.id
            cp, dp = assignment.cp, assignment.dp

            # Check if this segment overlaps with already assigned segment this round
            if pv_id in used_pv_segments:
                overlap = False
                for used_cp, used_dp in used_pv_segments[pv_id]:
                    if not (dp <= used_cp or cp >= used_dp):
                        overlap = True
                        break
                if overlap:
                    continue

            # Apply assignment
            all_assignments.append(assignment)
            pv_assignments[pv_id].append(assignment)
            total_saving += saving

            # Update states
            pv_states[pv_id].mark_segment_covered(cp, dp)
            av_states[assignment.av.id].add_assignment(cp, dp)

            # Track used segments
            if pv_id not in used_pv_segments:
                used_pv_segments[pv_id] = []
            used_pv_segments[pv_id].append((cp, dp))

    return all_assignments, total_saving, pv_assignments
