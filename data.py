from models import ActiveVehicle, PassiveVehicle

# -----------------------
# Global configuration
# -----------------------

L_MIN: int = 10  # minimum shared distance for platooning

# -----------------------
# Mock Active Vehicles
# -----------------------

ACTIVE_VEHICLES = [
    ActiveVehicle(
        id="AV1",
        entry_point=0,
        exit_point=80,
        capacity=2,
    ),
    ActiveVehicle(
        id="AV2",
        entry_point=20,
        exit_point=100,
        capacity=1,
    ),
    ActiveVehicle(
        id="AV3",
        entry_point=10,
        exit_point=70,
        capacity=3,
    ),
]


# -----------------------
# Mock Passive Vehicles
# -----------------------

PASSIVE_VEHICLES = [
    PassiveVehicle(
        id="PV1",
        entry_point=5,
        exit_point=60,
    ),
    PassiveVehicle(
        id="PV2",
        entry_point=15,
        exit_point=50,
    ),
    PassiveVehicle(
        id="PV3",
        entry_point=30,
        exit_point=90,
    ),
    PassiveVehicle(
        id="PV4",
        entry_point=0,
        exit_point=40,
    ),
    PassiveVehicle(
        id="PV5",
        entry_point=45,
        exit_point=85,
    ),
]


# -----------------------
# Helper accessor
# -----------------------

def load_mock_data():
    """
    Returns mock data for greedy / Hungarian algorithms.

    Returns:
        avs (list[ActiveVehicle])
        pvs (list[PassiveVehicle])
        l_min (int)
    """
    return ACTIVE_VEHICLES, PASSIVE_VEHICLES, L_MIN
