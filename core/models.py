from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class ActiveVehicle:
    """
    Active Vehicle (AV): can tow multiple PVs.

    Time-based parameters (Step 4):
    - entry_time: Time when AV enters the highway at entry_point
    - speed: Constant speed of AV (distance units per time unit)
    - Time at any point X can be computed as: entry_time + (X - entry_point) / speed
    """
    id: str
    entry_point: int
    exit_point: int
    capacity: int
    entry_time: float = 0.0  # Time when vehicle enters highway
    speed: float = 1.0       # Constant speed (distance/time)

    @property
    def vehicle_type(self) -> Literal["AV"]:
        return "AV"

    @property
    def exit_time(self) -> float:
        """Time when AV exits the highway."""
        return self.entry_time + (self.exit_point - self.entry_point) / self.speed

    def time_at_point(self, point: int) -> Optional[float]:
        """
        Calculate the time when AV reaches a specific point.
        Returns None if point is outside AV's route.
        """
        if point < self.entry_point or point > self.exit_point:
            return None
        return self.entry_time + (point - self.entry_point) / self.speed

    def __post_init__(self) -> None:
        if self.entry_point >= self.exit_point:
            raise ValueError(f"AV({self.id}) entry_point must be < exit_point")
        if self.capacity <= 0:
            raise ValueError(f"AV({self.id}) capacity must be positive")
        if self.speed <= 0:
            raise ValueError(f"AV({self.id}) speed must be positive")

@dataclass(frozen=True)
class PassiveVehicle:
    """
    Passive Vehicle (PV): can be towed by an AV.

    Time-based parameters (Step 4):
    - entry_time: Time when PV enters the highway at entry_point
    - speed: Constant speed when self-driving (distance units per time unit)
    - When towed, PV moves at AV's speed
    """
    id: str
    entry_point: int
    exit_point: int
    entry_time: float = 0.0  # Time when vehicle enters highway
    speed: float = 1.0       # Constant speed when self-driving

    @property
    def vehicle_type(self) -> Literal["PV"]:
        return "PV"

    @property
    def exit_time(self) -> float:
        """Time when PV exits the highway (if self-driving entire route)."""
        return self.entry_time + (self.exit_point - self.entry_point) / self.speed

    def time_at_point(self, point: int) -> Optional[float]:
        """
        Calculate the time when PV reaches a specific point (self-driving).
        Returns None if point is outside PV's route.
        """
        if point < self.entry_point or point > self.exit_point:
            return None
        return self.entry_time + (point - self.entry_point) / self.speed

    def __post_init__(self) -> None:
        if self.entry_point >= self.exit_point:
            raise ValueError(f"PV({self.id}) entry_point must be < exit_point")
        if self.speed <= 0:
            raise ValueError(f"PV({self.id}) speed must be positive")