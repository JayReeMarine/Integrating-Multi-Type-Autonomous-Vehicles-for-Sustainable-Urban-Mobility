from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from core.models import ActiveVehicle, PassiveVehicle


@dataclass(frozen=True)
class Assignment:
    """Result of assigning a PV to an AV by using Hungarian algorithm."""
    pv: PassiveVehicle
    av: ActiveVehicle
    cp: int  # coupling point
    dp: int  # decoupling point

    @property
    def saved_distance(self) -> int:
        return self.dp - self.cp


def compute_shared_segment(av: ActiveVehicle, pv: PassiveVehicle) -> Optional[Tuple[int, int]]:
    """
    Compute shared segment (cp, dp) on a 1D highway.
    Returns None if there is no overlap.
    """
    cp = max(av.entry_point, pv.entry_point)
    dp = min(av.exit_point, pv.exit_point)
    if dp <= cp:
        return None
    return cp, dp


def compute_energy_saving(cp: int, dp: int) -> int:
    """Simple version of energy saving calculation."""
    return dp - cp


# -----------------------------
# Standard Hungarian (Min-Cost)
# -----------------------------
def _hungarian_min_cost(cost: List[List[float]]) -> List[int]:
    """
    Solve the assignment problem (min-cost) for a square matrix using Hungarian algorithm.
    
    Returns:
        assignment: list 'match_col_for_row' where assignment[i] = j
                    means row i is matched to column j.
    """
    n = len(cost)
    if n == 0:
        return []
    
    # Potentials (dual variables) - use float for better precision
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    # p[j] = matched row for column j
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)  # Use float('inf') instead of 10**18
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

        # Augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build row->col assignment from p[col] = row
    row_to_col = [-1] * n
    for j in range(1, n + 1):
        row = p[j]
        if row != 0:
            row_to_col[row - 1] = j - 1
    return row_to_col


# ----------------------------------------
# Capacity-Constrained Hungarian via Slots
# ----------------------------------------
def hungarian_platoon_matching(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    l_min: int,
) -> Tuple[List[Assignment], int]:
    """
    Optimal matching using Hungarian algorithm with AV capacity constraints.
    
    Optimized version that reduces matrix size and improves performance.
    """
    if not avs or not pvs:
        return [], 0
    
    # 1) Expand AVs into slots
    slot_to_av: List[ActiveVehicle] = []
    for av in avs:
        for _ in range(av.capacity):
            slot_to_av.append(av)

    num_slots = len(slot_to_av)
    num_pv = len(pvs)

    if num_slots == 0 or num_pv == 0:
        return [], 0

    # 2) Precompute feasible pairs and savings
    feasible: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    max_saving = 0

    for slot_idx, av in enumerate(slot_to_av):
        for pv_idx, pv in enumerate(pvs):
            shared = compute_shared_segment(av, pv)
            if shared is None:
                continue
            cp, dp = shared
            saving = compute_energy_saving(cp, dp)
            if saving < l_min:
                continue
            feasible[(slot_idx, pv_idx)] = (saving, cp, dp)
            max_saving = max(max_saving, saving)

    # Early exit if no feasible pairs
    if max_saving == 0:
        return [], 0

    # 3) Create balanced square matrix
    # Use total size = slots + pvs to allow unmatched
    n = num_slots + num_pv
    
    # Cost setup with better scaling
    BASE_UNMATCH_COST = max_saving + 1.0  # Cost of not matching
    BIG_M = max_saving * 1000.0  # Large penalty for impossible matches
    
    # Initialize cost matrix
    cost = [[BASE_UNMATCH_COST for _ in range(n)] for _ in range(n)]

    # Fill slot->PV costs (real matching)
    for slot_idx in range(num_slots):
        for pv_idx in range(num_pv):
            if (slot_idx, pv_idx) in feasible:
                saving, _, _ = feasible[(slot_idx, pv_idx)]
                # Convert maximize saving to minimize cost
                cost[slot_idx][pv_idx] = max_saving - saving + 1
            else:
                cost[slot_idx][pv_idx] = BIG_M
    
    # Set up dummy assignments (slot -> dummy PV, dummy slot -> PV)
    # Slots can match to dummy PVs (columns num_pv to n-1)
    for slot_idx in range(num_slots):
        for dummy_pv in range(num_pv, n):
            cost[slot_idx][dummy_pv] = BASE_UNMATCH_COST
    
    # Dummy slots can match to real PVs  
    for dummy_slot in range(num_slots, n):
        for pv_idx in range(num_pv):
            cost[dummy_slot][pv_idx] = BASE_UNMATCH_COST
    
    # Dummy to dummy is free
    for dummy_slot in range(num_slots, n):
        for dummy_pv in range(num_pv, n):
            cost[dummy_slot][dummy_pv] = 0.0

    # 4) Solve with Hungarian algorithm
    row_to_col = _hungarian_min_cost(cost)

    # 5) Extract real assignments
    assignments: List[Assignment] = []
    total_saving = 0

    for slot_idx in range(min(len(row_to_col), num_slots)):
        col_idx = row_to_col[slot_idx]
        
        # Only consider real PV assignments
        if col_idx < 0 or col_idx >= num_pv:
            continue
            
        if (slot_idx, col_idx) not in feasible:
            continue

        saving, cp, dp = feasible[(slot_idx, col_idx)]
        av = slot_to_av[slot_idx]
        pv = pvs[col_idx]

        assignments.append(Assignment(pv=pv, av=av, cp=cp, dp=dp))
        total_saving += saving

    return assignments, total_saving
