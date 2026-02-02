# Dynamic Platoon Formation for Multi-Modal Autonomous Vehicle Systems

A simulation framework for optimizing energy-efficient platoon formation between Active Vehicles (AVs) and Passive Vehicles (PVs) on highway systems. This project implements and compares two algorithms for vehicle matching: a Greedy Maximum-Weight Matching algorithm and a Capacity-Constrained Hungarian algorithm.

## Overview

This research introduces a novel concept of **active and passive autonomous vehicles**, where smaller passive vehicles (PVs) can temporarily attach to larger active vehicles (AVs) during shared highway segments. Unlike traditional platooning where vehicles maintain virtual formations through coordinated driving, our approach enables **physical attachment** where PVs are towed by AVs, achieving near-100% energy savings for towed vehicles during attached phases.

### Key Contributions

- **Simplified Problem Formulation**: Deterministic baseline with clean mapping to classic optimization problems
- **Comprehensive Formulation**: Real-world model addressing traffic dynamics and temporal constraints
- **Efficient Algorithms**: Both centralized (optimal) and greedy (near-optimal) solutions for vehicle matching
- **Experimental Framework**: Extensive parameter sweep experiments with visualization tools

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

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility.git
   cd Integrating-Multi-Type-Autonomous-Vehicles-for-Sustainable-Urban-Mobility
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running All Experiments

To run the complete experiment suite (both Greedy and Hungarian algorithms with all parameter sweeps):

```bash
python -m experiments.main
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
python -m experiments.run_greedy_pv_av_sweep
python -m experiments.run_greedy_length_sweep
python -m experiments.run_greedy_capacity_sweep

# Hungarian algorithm experiments
python -m experiments.run_hungarian_pv_av_sweep
python -m experiments.run_hungarian_length_sweep
python -m experiments.run_hungarian_capacity_sweep
```

### Generating Visualizations

After running experiments, generate all plots and comparison analysis:

```bash
python -m visualization.plot_all
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
- Each AV can tow at most `C_i` PVs simultaneously (capacity constraint)
- Each PV can be assigned to at most one AV (assignment constraint)
- Minimum shared distance `L_min` required for platooning (feasibility constraint)

### Greedy Maximum-Weight Matching

**Time Complexity**: O(NM log(NM))

**Approach**:
1. Generate all feasible (AV, PV) candidate pairs
2. Sort candidates by energy saving (descending)
3. Assign greedily while respecting capacity constraints

**Approximation Guarantee**: Achieves 1/2-approximation for submodular energy functions

### Capacity-Constrained Hungarian Algorithm

**Time Complexity**: O((N + M)^3)

**Approach**:
1. Expand AVs into capacity slots
2. Construct bipartite graph with feasibility edges
3. Solve minimum-cost assignment using Hungarian algorithm
4. Extract optimal assignments

**Guarantee**: Computes globally optimal solution

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

Based on experiments comparing Greedy vs Hungarian algorithms:

| Metric | Greedy Performance |
|--------|-------------------|
| Total Saving | ~99% of optimal |
| Match Ratio | ~95% of optimal |
| Runtime | 10-30x faster than Hungarian |

The greedy algorithm provides an excellent balance between performance and computational efficiency, making it suitable for real-time applications.

## Paper Compilation

To compile the LaTeX paper to PDF:

```bash
cd paper
latexmk -pdf conference_101719.tex
```

The compiled PDF will be generated as `conference_101719.pdf`.

## Dependencies

- `pandas>=1.5.0` - Data manipulation and CSV handling
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Visualization and plotting
- `scipy>=1.9.0` - Scientific computing utilities

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{platoon2024,
  title={Dynamic Platoon Formation for Multi-Modal Autonomous Vehicle Systems},
  author={[Author Names]},
  booktitle={[Conference Name]},
  year={2024}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
