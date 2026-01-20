from core.models import ActiveVehicle, PassiveVehicle
import random
from typing import List, Tuple, Optional

# -----------------------
# Global configuration
# -----------------------

L_MIN: int = 10  # minimum shared distance for platooning

# Default time parameters (Step 4)
DEFAULT_AV_SPEED: float = 1.0   # AV speed (distance units per time unit)
DEFAULT_PV_SPEED: float = 1.0   # PV speed when self-driving
DEFAULT_TIME_WINDOW: float = 100.0  # Maximum time span for vehicle entry

# -----------------------
# Generated mock data (experiment use)
# -----------------------

def generate_mock_data(
    num_av: int,
    num_pv: int,
    highway_length: int,
    av_capacity_range: Tuple[int, int],
    min_trip_length: int,
    seed: int = 42,
    *,
    # Time-based parameters (Step 4)
    enable_time_constraints: bool = False,
    av_speed_range: Optional[Tuple[float, float]] = None,
    pv_speed_range: Optional[Tuple[float, float]] = None,
    time_window: float = DEFAULT_TIME_WINDOW,
) -> Tuple[List[ActiveVehicle], List[PassiveVehicle], int]:
    """
    Generate reproducible mock data for experiments.

    Args:
        num_av: number of Active Vehicles
        num_pv: number of Passive Vehicles
        highway_length: total length of highway (0 ~ highway_length)
        av_capacity_range: (min_capacity, max_capacity)
        min_trip_length: minimum trip length for any vehicle
        seed: random seed for reproducibility

        # Time constraint parameters (Step 4):
        enable_time_constraints: If True, generate realistic time parameters
        av_speed_range: (min_speed, max_speed) for AVs. Default: (1.0, 1.0)
        pv_speed_range: (min_speed, max_speed) for PVs. Default: (1.0, 1.0)
        time_window: Maximum time span during which vehicles can enter highway

    Returns:
        avs, pvs, l_min

    Time Constraint Model (Step 4):
    ==============================
    When enable_time_constraints=True:
    - Each vehicle gets a random entry_time within [0, time_window]
    - Each vehicle gets a speed from its speed_range
    - Time at any point can be computed as: entry_time + (point - entry_point) / speed

    Matching feasibility with time constraints:
    - AV and PV can couple at point X only if they arrive at similar times
    - Time tolerance is handled in the matching algorithm (greedy_multi.py)
    """
    random.seed(seed)

    avs: List[ActiveVehicle] = []
    pvs: List[PassiveVehicle] = []

    # Set default speed ranges if not provided
    if av_speed_range is None:
        av_speed_range = (DEFAULT_AV_SPEED, DEFAULT_AV_SPEED)
    if pv_speed_range is None:
        pv_speed_range = (DEFAULT_PV_SPEED, DEFAULT_PV_SPEED)

    # Generate Active Vehicles
    for i in range(num_av):
        entry = random.randint(0, highway_length - min_trip_length)
        exit = random.randint(entry + min_trip_length, highway_length)
        capacity = random.randint(*av_capacity_range)

        # Time parameters (Step 4)
        if enable_time_constraints:
            entry_time = random.uniform(0, time_window)
            speed = random.uniform(*av_speed_range)
        else:
            entry_time = 0.0
            speed = DEFAULT_AV_SPEED

        avs.append(
            ActiveVehicle(
                id=f"AV{i+1}",
                entry_point=entry,
                exit_point=exit,
                capacity=capacity,
                entry_time=entry_time,
                speed=speed,
            )
        )

    # Generate Passive Vehicles
    for i in range(num_pv):
        entry = random.randint(0, highway_length - min_trip_length)
        exit = random.randint(entry + min_trip_length, highway_length)

        # Time parameters (Step 4)
        if enable_time_constraints:
            entry_time = random.uniform(0, time_window)
            speed = random.uniform(*pv_speed_range)
        else:
            entry_time = 0.0
            speed = DEFAULT_PV_SPEED

        pvs.append(
            PassiveVehicle(
                id=f"PV{i+1}",
                entry_point=entry,
                exit_point=exit,
                entry_time=entry_time,
                speed=speed,
            )
        )

    return avs, pvs, L_MIN
