# Dynamic Platoon Formation of Multi-Type Autonomous Vehicles for Sustainable Urban Mobility

A simulation framework for optimizing energy-efficient platoon formation between Active Vehicles (AVs) and Passive Vehicles (PVs) on highway systems. This project implements and compares two algorithms for vehicle matching: a Greedy Maximum-Weight Matching algorithm and an Iterative Linear Assignment (ILA) method (assignment solved via SciPy's Jonker-Volgenant backend).

## Overview

This research introduces a novel concept of **active and passive autonomous vehicles**, where smaller passive vehicles (PVs) can temporarily attach to larger active vehicles (AVs) during shared highway segments. Unlike traditional platooning where vehicles maintain virtual formations through coordinated driving, our approach enables **physical attachment** where PV propulsion is offset during attached phases while AVs bear the additional towing load.

### Key Contributions

- **Problem Formulation**: Dynamic platoon formation with multi-segment matching, point-wise AV capacity constraints, and temporal synchronization
- **Algorithms**: A greedy matching algorithm and an assignment-based ILA method with per-iteration optimality
- **Experimental Framework**: Controlled parameter sweeps with reproducible visualization and comparison tools

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/JayReeMarine/Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility.git
cd Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility

# 2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run all experiments
python3 -m experiments.main

# 4. Generate visualization graphs
python3 -m visualization.plot_all
```

## Project Structure

```
.
├── core/                          # Core algorithm implementations
│   ├── models.py                  # Vehicle data models (ActiveVehicle, PassiveVehicle)
│   ├── greedy.py                  # Basic greedy matching algorithm
│   ├── greedy_multi.py            # Extended greedy with time constraints
│   ├── hungarian.py               # Basic Hungarian algorithm
│   ├── hungarian_multi.py         # Extended Hungarian with time constraints
│   ├── metrics.py                 # Performance metrics computation
│   ├── data.py                    # Mock data generation
│   └── analysis.py                # Analysis utilities
│
├── experiments/                   # Experiment runners
│   ├── main.py                    # Run all experiments
│   ├── common.py                  # Shared experiment utilities
│   ├── run_greedy_pv_av_sweep.py  # Greedy: PV/AV count sweep
│   ├── run_greedy_length_sweep.py # Greedy: Highway length sweep
│   ├── run_greedy_capacity_sweep.py # Greedy: AV capacity sweep
│   ├── run_hungarian_pv_av_sweep.py # Hungarian: PV/AV count sweep
│   ├── run_hungarian_length_sweep.py # Hungarian: Highway length sweep
│   └── run_hungarian_capacity_sweep.py # Hungarian: AV capacity sweep
│
├── visualization/                 # Plotting and analysis
│   ├── plot_all.py                # Generate all plots
│   ├── compare_algorithms.py      # Algorithm comparison analysis
│   ├── plot_greedy_*.py           # Greedy-specific plots
│   ├── plot_hungarian_*.py        # Hungarian-specific plots
│   └── figures/                   # Generated figures output
│       ├── greedy/                # Greedy algorithm figures
│       ├── hungarian/             # Hungarian algorithm figures
│       └── comparison/            # Comparison tables and figures
│
├── data/                          # Experiment results
│   └── results/
│       ├── greedy/                # Greedy CSV results
│       └── hungarian/             # Hungarian CSV results
│
├── paper/                         # Academic paper (LaTeX)
│   ├── conference_101719.tex      # Main LaTeX source
│   ├── conference_101719.pdf      # Compiled PDF
│   └── figures/                   # Paper figures
│
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher (tested with Python 3.11)
- pip (Python package manager)
- LaTeX distribution with `latexmk` (optional, for paper compilation)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JayReeMarine/Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility.git
   cd Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python3 -c "import pandas, numpy, matplotlib, scipy; print('All dependencies installed!')"
   ```

> **Note for macOS users**: Use `python3` instead of `python` for all commands.

## Usage

### Running All Experiments

To run the complete experiment suite (both Greedy and Hungarian algorithms with all parameter sweeps):

```bash
python3 -m experiments.main
```

This will execute:
- PV/AV count sweeps (varying number of passive and active vehicles)
- Highway length sweeps (50, 100, 200, 400, 800, 1600 units)
- AV capacity sweeps (capacity ranges 2-16)

Results are saved to `data/results/greedy/` and `data/results/hungarian/`.

### Running Individual Experiments

Run specific experiment sweeps:

```bash
# Greedy algorithm experiments
python3 -m experiments.run_greedy_pv_av_sweep
python3 -m experiments.run_greedy_length_sweep
python3 -m experiments.run_greedy_capacity_sweep

# Hungarian algorithm experiments
python3 -m experiments.run_hungarian_pv_av_sweep
python3 -m experiments.run_hungarian_length_sweep
python3 -m experiments.run_hungarian_capacity_sweep
```

### Generating Visualizations

After running experiments, generate all plots and comparison analysis:

```bash
python3 -m visualization.plot_all
```

This generates:
1. **Greedy algorithm plots** - Saved to `visualization/figures/greedy/`
2. **Hungarian algorithm plots** - Saved to `visualization/figures/hungarian/`
3. **Comparison analysis** - Saved to `visualization/figures/comparison/`

Generated figures include:
- Total energy savings vs. parameters
- Match ratio (percentage of PVs matched)
- Average saving per matched PV
- Runtime performance
- Saving percentage

## Algorithm Details

### Problem Formulation

The platoon formation problem is formulated as an optimization problem:

**Objective**: Maximize total energy savings through strategic platoon formation

**Constraints**:
- Each AV can tow at most `C_i` PVs simultaneously at any highway position (point-wise capacity)
- Each PV can be towed by multiple AVs across route segments, but not by two AVs at the same position/time
- Minimum shared distance `L_min` is required for a feasible towing segment

### Greedy Maximum-Weight Matching

**Worst-Case Time Complexity**: O((NM)^2 log(NM)) for iterative re-generation/re-sorting

**Approach**:
1. Generate all feasible (AV, PV) candidate pairs
2. Sort candidates by energy saving (descending)
3. Assign greedily while respecting capacity constraints

**Role in this project**: A simple heuristic comparator with low implementation overhead

### Iterative Linear Assignment (ILA)

**Per-Iteration Complexity**: LSAP solved in O(max(K, M)^3), where K is total virtual AV slots

**Approach**:
1. Expand AVs into capacity slots
2. Construct bipartite graph with feasibility edges
3. Solve minimum-cost assignment using `scipy.optimize.linear_sum_assignment`
4. Apply accepted matches, update states, and iterate

**Guarantee**: Computes an optimal assignment for each iteration's linear subproblem (per-iteration optimality, not global optimality for the full multi-segment problem)

## Input Parameters

### Active Vehicle (AV)
| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier |
| `entry_point` | int | Highway entry position |
| `exit_point` | int | Highway exit position |
| `capacity` | int | Maximum PVs that can be towed |
| `entry_time` | float | Time entering highway (optional) |
| `speed` | float | Travel speed (optional) |

### Passive Vehicle (PV)
| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier |
| `entry_point` | int | Highway entry position |
| `exit_point` | int | Highway exit position |
| `entry_time` | float | Time entering highway (optional) |
| `speed` | float | Travel speed when self-driving (optional) |

### Algorithm Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `L_min` | Minimum shared distance for platooning | 10 |
| `time_tolerance` | Max time difference for coupling | 5.0 |
| `time_window` | Time span for vehicle entry | 100.0 |

## Output Metrics

| Metric | Description |
|--------|-------------|
| `total_saving` | Total energy saved (distance units) |
| `saving_percent` | Percentage of baseline energy saved |
| `matched_ratio` | Fraction of PVs successfully matched |
| `avg_saving_per_pv` | Average saving per matched PV |
| `runtime_sec` | Algorithm execution time |
| `av_utilization` | AV capacity utilization percentage |
| `pv_coverage` | PV route coverage percentage |

## Experimental Results Summary

Based on experiments comparing Greedy vs ILA:

| Metric | Observed Trend |
|--------|----------------|
| Total covered distance (saving proxy) | Both methods improve with capacity; ILA consistently matches or exceeds Greedy in tested settings |
| Saving percentage | Typically in the 25--52% range depending on scenario (paper setting) |
| ILA vs Greedy saving gap | 0.2--0.9 percentage-point higher saving rates (capacity sweep) |
| Runtime | In this implementation, ILA is often faster (2.8--11.5x in capacity sweep; 5.5--26.9x in length sweep) |

Greedy remains a useful heuristic comparator, while ILA is the primary method used for final analysis in the paper.

## Paper Compilation

To compile the LaTeX paper to PDF:

```bash
cd paper
latexmk -pdf conference_101719.tex
```

The compiled PDF will be generated as `conference_101719.pdf`.

> **Note**: If you see "Nothing to do" or "All targets are up-to-date", this means the PDF already exists and is current - this is not an error!

## Code Availability

Code available at:
`https://github.com/JayReeMarine/Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility`

## Dependencies

- `pandas>=1.5.0` - Data manipulation and CSV handling
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Visualization and plotting
- `scipy>=1.9.0` - Scientific computing utilities

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ree2026dynamic,
  title        = {Dynamic Platoon Formation of Multi-Type Autonomous Vehicles for Sustainable Urban Mobility},
  author       = {Jaeyun Ree and Mohammed Eunus Ali},
  year         = {2026},
  note         = {Manuscript under review}
}

@misc{ree2026code,
  title        = {Codebase for Dynamic Platoon Formation of Multi-Type Autonomous Vehicles},
  author       = {Jaeyun Ree and Mohammed Eunus Ali},
  year         = {2026},
  howpublished = {\url{https://github.com/JayReeMarine/Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
