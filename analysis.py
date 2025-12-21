# analysis.py
# --------------------------------------------------
# This module contains analysis utilities for:
# 1) Trip length distribution (short / medium / long)
# 2) Feasible AV–PV platooning pair ratio under L_MIN
#
# These analyses are used to validate whether
# the generated dataset is realistic and suitable
# for platoon formation experiments.
# --------------------------------------------------

from typing import List
from models import ActiveVehicle, PassiveVehicle
from greedy import compute_shared_segment


# --------------------------------------------------
# Basic utility
# --------------------------------------------------

def trip_length(entry: int, exit: int) -> int:
    """
    Compute trip length on a 1D highway.

    Args:
        entry: entry point of the vehicle
        exit: exit point of the vehicle

    Returns:
        Trip length (exit - entry)
    """
    return exit - entry


# --------------------------------------------------
# Trip distribution analysis
# --------------------------------------------------

def analyze_trip_distribution(
    vehicles: List,
    highway_length: int,
    label: str,
) -> None:
    """
    Analyse trip length distribution for a set of vehicles.

    Vehicles are categorized into short, medium, and long trips
    based on fractions of the highway length.

    Args:
        vehicles: list of ActiveVehicle or PassiveVehicle
        highway_length: total highway length
        label: string label for printing (e.g., "AV", "PV")
    """
    trips = [trip_length(v.entry_point, v.exit_point) for v in vehicles]

    short_threshold = 0.3 * highway_length
    long_threshold = 0.6 * highway_length

    short_trips = [t for t in trips if t < short_threshold]
    medium_trips = [t for t in trips if short_threshold <= t <= long_threshold]
    long_trips = [t for t in trips if t > long_threshold]

    print(f"\n--- {label} Trip Distribution ---")
    print(f"Total vehicles : {len(trips)}")
    print(f"Min trip       : {min(trips)}")
    print(f"Max trip       : {max(trips)}")
    print(f"Avg trip       : {sum(trips) / len(trips):.2f}")
    print(f"Short trips    : {len(short_trips)}")
    print(f"Medium trips   : {len(medium_trips)}")
    print(f"Long trips     : {len(long_trips)}")


# --------------------------------------------------
# Feasible platooning pair analysis
# --------------------------------------------------

def analyze_feasible_pairs(
    avs: List[ActiveVehicle],
    pvs: List[PassiveVehicle],
    l_min: int,
) -> None:
    """
    Analyse how many AV–PV pairs satisfy the minimum
    shared distance constraint (L_MIN).

    This metric indicates how dense or sparse
    platooning opportunities are in the dataset.

    Args:
        avs: list of ActiveVehicle
        pvs: list of PassiveVehicle
        l_min: minimum shared distance for platooning
    """
    total_pairs = len(avs) * len(pvs)
    feasible_pairs = 0

    for av in avs:
        for pv in pvs:
            shared = compute_shared_segment(av, pv)
            if shared is None:
                continue

            cp, dp = shared
            if (dp - cp) >= l_min:
                feasible_pairs += 1

    ratio = feasible_pairs / total_pairs if total_pairs > 0 else 0.0

    print("\n--- Feasible Platooning Pairs ---")
    print(f"Total AV–PV pairs    : {total_pairs}")
    print(f"Feasible pairs (≥L)  : {feasible_pairs}")
    print(f"Feasible ratio       : {ratio:.2f}")
