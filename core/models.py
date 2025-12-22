from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ActiveVehicle:
    """Active Vehicle (AV): can tow multiple PVs."""
    id: str
    entry_point: int
    exit_point: int
    capacity: int

    @property
    def vehicle_type(self) -> Literal["AV"]:
        return "AV"

    def __post_init__(self) -> None:
        if self.entry_point >= self.exit_point:
            raise ValueError(f"AV({self.id}) entry_point must be < exit_point")
        if self.capacity <= 0:
            raise ValueError(f"AV({self.id}) capacity must be positive")

@dataclass(frozen=True)
class PassiveVehicle:
    """Passive Vehicle (PV): can be towed by an AV."""
    id: str
    entry_point: int
    exit_point: int

    @property
    def vehicle_type(self) -> Literal["PV"]:
        return "PV"

    def __post_init__(self) -> None:
        if self.entry_point >= self.exit_point:
            raise ValueError(f"PV({self.id}) entry_point must be < exit_point")