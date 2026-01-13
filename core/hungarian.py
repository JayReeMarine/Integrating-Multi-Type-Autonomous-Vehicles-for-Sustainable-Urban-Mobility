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
def _hungarian_min_cost(cost: List[List[int]]) -> List[int]:
    """
    Solve the assignment problem (min-cost) for a square matrix using Hungarian algorithm.

    Returns:
        assignment: list 'match_col_for_row' where assignment[i] = j
                    means row i is matched to column j.
    """
    n = len(cost)
    # Potentials (dual variables)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    # p[j] = matched row for column j
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [10**18] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**18
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

    Approach:
      - Expand each AV i into Ci identical 'slots' (each slot can take 1 PV).
      - Create a square cost matrix with dummy rows/cols to allow "unmatched".
      - Convert "maximize saving" to "minimize cost".

    Returns:
        - assignments: list of Assignment(pv, av, cp, dp)
        - total_saving: sum of saved distances
    """
    # 1) Expand AVs into slots
    slots: List[ActiveVehicle] = []
    slot_to_av: List[ActiveVehicle] = []
    for av in avs:
        for _ in range(av.capacity):
            slot_to_av.append(av)

    num_slots = len(slot_to_av)
    num_pv = len(pvs)

    # If there is no capacity at all, trivially return
    if num_slots == 0 or num_pv == 0:
        return [], 0

    # 2) Precompute savings & shared segments for feasible pairs
    #    key: (slot_idx, pv_idx) -> (saving, cp, dp)
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
            if saving > max_saving:
                max_saving = saving

    # 3) Build square cost matrix
    #    n = max(num_slots, num_pv)
    n = max(num_slots, num_pv)

    # We want costs non-negative, so:
    #   cost = (max_saving - saving) for feasible
    # Unmatched should be allowed at ~ "saving=0" cost -> max_saving
    # Infeasible should be MUCH worse than unmatched
    BASE_UNMATCH_COST = max_saving  # corresponds to 0 saving
    BIG_M = (max_saving + 1) * 10_000  # huge penalty

    cost = [[BASE_UNMATCH_COST for _ in range(n)] for _ in range(n)]

    # Fill real slot -> real pv costs
    for slot_idx in range(num_slots):
        for pv_idx in range(num_pv):
            if (slot_idx, pv_idx) in feasible:
                saving, _, _ = feasible[(slot_idx, pv_idx)]
                cost[slot_idx][pv_idx] = max_saving - saving
            else:
                # Infeasible edges should be avoided;
                # since dummy match exists with BASE_UNMATCH_COST, algorithm won't pick BIG_M.
                cost[slot_idx][pv_idx] = BIG_M

    # Remaining rows/cols are dummy rows/cols already BASE_UNMATCH_COST

    # 4) Run Hungarian (min-cost)
    row_to_col = _hungarian_min_cost(cost)

    # 5) Reconstruct assignments
    assignments: List[Assignment] = []
    total_saving = 0

    for row_idx, col_idx in enumerate(row_to_col):
        # Only interpret real slot rows
        if row_idx >= num_slots:
            continue
        # Only interpret real PV cols
        if col_idx < 0 or col_idx >= num_pv:
            continue

        if (row_idx, col_idx) not in feasible:
            # matched to infeasible (shouldn't happen because BIG_M),
            # but keep it safe.
            continue

        saving, cp, dp = feasible[(row_idx, col_idx)]
        av = slot_to_av[row_idx]
        pv = pvs[col_idx]

        assignments.append(Assignment(pv=pv, av=av, cp=cp, dp=dp))
        total_saving += saving

    return assignments, total_saving
