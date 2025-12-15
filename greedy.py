from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict

from models import ActiveVehicle, PassiveVehicle


@dataclass(frozen=True, slots=True)
class Assignment:
    """result of assigning a PV to an AV by using greedy algorithm"""
    pv: PassiveVehicle
    av: ActiveVehicle
    cp: int  # coupling point
    dp: int  # decoupling point


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



