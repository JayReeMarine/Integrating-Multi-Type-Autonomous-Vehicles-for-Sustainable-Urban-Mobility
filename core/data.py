from core.models import ActiveVehicle, PassiveVehicle
import random
from typing import List, Tuple

# -----------------------
# Global configuration
# -----------------------

L_MIN: int = 10  # minimum shared distance for platooning


# -----------------------
# Static mock data (debug / sanity check)
# -----------------------

ACTIVE_VEHICLES = [
    ActiveVehicle(id="AV1", entry_point=0, exit_point=80, capacity=2),
    ActiveVehicle(id="AV2", entry_point=20, exit_point=100, capacity=1),
    ActiveVehicle(id="AV3", entry_point=10, exit_point=70, capacity=3),
]

PASSIVE_VEHICLES = [
    PassiveVehicle(id="PV1", entry_point=5, exit_point=60),
    PassiveVehicle(id="PV2", entry_point=15, exit_point=50),
    PassiveVehicle(id="PV3", entry_point=30, exit_point=90),
    PassiveVehicle(id="PV4", entry_point=0, exit_point=40),
    PassiveVehicle(id="PV5", entry_point=45, exit_point=85),
]


def load_mock_data():
    """
    Returns fixed mock data (for debugging / sanity checking).
    """
    return ACTIVE_VEHICLES, PASSIVE_VEHICLES, L_MIN


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

    Returns:
        avs, pvs, l_min
    """
    random.seed(seed)

    avs: List[ActiveVehicle] = []
    pvs: List[PassiveVehicle] = []

    # Generate Active Vehicles
    for i in range(num_av):
        entry = random.randint(0, highway_length - min_trip_length)
        exit = random.randint(entry + min_trip_length, highway_length)
        capacity = random.randint(*av_capacity_range)

        avs.append(
            ActiveVehicle(
                id=f"AV{i+1}",
                entry_point=entry,
                exit_point=exit,
                capacity=capacity,
            )
        )

    # Generate Passive Vehicles
    for i in range(num_pv):
        entry = random.randint(0, highway_length - min_trip_length)
        exit = random.randint(entry + min_trip_length, highway_length)

        pvs.append(
            PassiveVehicle(
                id=f"PV{i+1}",
                entry_point=entry,
                exit_point=exit,
            )
        )

    return avs, pvs, L_MIN
