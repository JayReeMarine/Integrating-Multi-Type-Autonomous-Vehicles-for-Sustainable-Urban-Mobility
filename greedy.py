from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict

from models import ActiveVehicle, PassiveVehicle


@dataclass(frozen=True)
class Assignment:
    """result of assigning a PV to an AV by using greedy algorithm"""
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

def compute_energy_saving(cp:int, dp: int) -> float:
    """simple version of energy saving calculation"""
    return dp - cp

def greedy_platoon_matching(avs: List[ActiveVehicle], pvs: List[PassiveVehicle], l_min: int) -> Tuple[List[Assignment], float]:
    """
    Greedy Maximum-weight platoon Matching

    Returns values:
        - assignments: List of Assignment (pv, av, cp, dp)
        - total energy saving: sum(dp - cp)
    """

    # 1. initialising candidate set
    candidates: List[Tuple[int, ActiveVehicle, PassiveVehicle, int, int]] = []
    # tuple: (saving, av, pv, cp, dp)

    for av in avs:
        for pv in pvs:
            shared = compute_shared_segment(av, pv)
            if shared is None:
                continue
            cp, dp = shared
            if (dp- cp) < l_min:
                continue
            
            saving = compute_energy_saving(cp, dp)
            candidates.append((saving, av, pv, cp, dp))
    
    # 2. sort candidates by energy saving (descending order)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. Assign based on sort order
    assignments: List[Assignment] = []
    used_pvs: Set[str] = set()
    av_load: Dict[str, int] = {av.id:0 for av in avs}

    total_saving = 0.0

    for saving, av, pv, cp, dp in candidates:
        #check condition before loading
        if pv.id in used_pvs:
            continue
        if av_load[av.id] >= av.capacity:
            continue
        
        #assign
        assignments.append(Assignment(pv, av, cp, dp))
        used_pvs.add(pv.id)
        av_load[av.id] += 1
        total_saving += saving
    
    return assignments, total_saving



