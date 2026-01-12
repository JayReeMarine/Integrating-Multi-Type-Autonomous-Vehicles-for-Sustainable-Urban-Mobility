# Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility

# To run main file

python -m experiments.main

50 100 200 400
# To run the Graph

python -m visualization.plot_all


# 1. Greedy Platoon Matching — Parameters and Outputs 

This document defines the input parameters and return values for **Algorithm 1: Greedy Maximum-Weight Platoon Matching** under the **simplified model** (energy saving is proportional to shared distance).

---

## Inputs (Parameters)

### 1. Active Vehicle (AV)
Each **Active Vehicle (AV)** is defined by:

- `entry_point` (int): Entry position on the highway
- `exit_point` (int): Exit position on the highway
- `capacity` (int): Maximum number of PVs the AV can tow simultaneously

> Note: `entry_point < exit_point` must hold.

---

### 2. Passive Vehicle (PV)
Each **Passive Vehicle (PV)** is defined by:

- `entry_point` (int): Entry position on the highway
- `exit_point` (int): Exit position on the highway

> Note: `entry_point < exit_point` must hold.

---

### 3. Minimum Shared Distance (`L_min`)
- `L_min` (int): Minimum shared distance required for platooning (coupling)

A candidate pair `(AV_i, PV_j)` is considered feasible only if:

- `shared_distance(AV_i, PV_j) >= L_min`

---

## Outputs (Return Values)

### 1. Assignment Set
The algorithm returns an **assignment set** `A` consisting of tuples:

- `(pv, av, cp, dp)`

where:
- `pv` is the selected Passive Vehicle
- `av` is the assigned Active Vehicle
- `cp` (int) is the coupling point (start of shared segment)
- `dp` (int) is the decoupling point (end of shared segment)

---

### 2. Total Saved Distance
The algorithm also returns the **total saved distance**, defined as:

- `total_saved_distance = sum(dp - cp for each (pv, av, cp, dp) in A)`

This represents the total shared travel distance during which PVs are coupled (towed) under the simplified distance-based saving model.