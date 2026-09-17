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
├── milp/                          # Revision: exact MILP solver (HiGHS via SciPy)
│   ├── exact.py                   # build() / solve() / extract_segments()
│   ├── run_comparison.py          # greedy & ILA vs optimum on small instances
│   ├── scale_test.py              # how large the MILP stays tractable
│   ├── restricted.py              # suffix-only / full-overlap restricted optima
│   └── sweep_*.py, diagnose.py    # congestion-aware variants, structural diagnosis
│
├── sumo/                          # Revision: realistic scenarios on the M1 (SUMO)
│   ├── PRIMER-sumo.md             # SUMO primer for this repo
│   ├── NOTES-sumo.md              # SUMO track research notes
│   ├── convert.py                 # SUMO output -> ActiveVehicle / PassiveVehicle
│   ├── smoke_match.py             # greedy / ILA on the converted instance
│   ├── ratio_sweep.py             # AV:PV ratio sweep on the converted instance
│   ├── osm/                       # OSM extract + netconvert network (gitignored)
│   └── m1/                        # M1 inbound scenario (corridor, demand, outputs)
│
├── analysis/                      # Revision: re-analysis and figures
│   ├── reproduce_check.py         # verifies stored results reproduce exactly
│   ├── plot_*.py                  # figures for the revision
│   └── figures/
│
├── reports/                       # Interim findings and meeting memos
├── NOTES.md                       # Revision research notes (main track)
├── requirements.txt               # Python dependencies (ranges)
└── requirements-lock.txt          # Pinned versions actually used in the revision
```

## Installation

### Prerequisites

- Python 3.8 or higher (paper results: Python 3.11; revision work: Python 3.14, see `requirements-lock.txt`)
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

## Revision Work (2026-09, after ITSC / SIGSPATIAL reviews)

Everything below runs from the repository root with the project `venv/`
(Python 3.14, versions pinned in `requirements-lock.txt`). The `milp/`,
`sumo/` and `analysis/` scripts import from `core/`, so they need
`PYTHONPATH=.`; either activate the venv or call `venv/bin/python` directly.

```bash
cd /path/to/platoon-formation
python3 -m venv venv
venv/bin/pip install -r requirements-lock.txt      # exact versions used in the revision
venv/bin/pip install -r sumo/requirements-sumo.txt # SUMO 1.27.1 (pip distribution)
```

Research notes: [`NOTES.md`](NOTES.md) (main track: environment, re-analysis,
exact solver, diagnosis) and [`sumo/NOTES-sumo.md`](sumo/NOTES-sumo.md)
(SUMO track: design decisions D1-D6, demand survey, M1 runs).

### 0. Reproduction check

Re-runs a subset of the stored `pv_av_sweep` cells with the current
environment and compares `saving_percent` against the CSVs used in the paper.
Expect `48/48` matches.

```bash
PYTHONPATH=. venv/bin/python analysis/reproduce_check.py
```

### 1. Exact optimum (MILP)

Small instances (up to ~AV 15 / PV 30 within a few minutes) solved to proven
optimality; greedy and ILA are scored as a percentage of the optimum.

```bash
PYTHONPATH=. venv/bin/python milp/run_comparison.py --quick          # a few instances, fast
PYTHONPATH=. venv/bin/python milp/run_comparison.py --time-limit 600 # full set
PYTHONPATH=. venv/bin/python milp/scale_test.py                      # where the MILP stops proving optimality
PYTHONPATH=. venv/bin/python milp/restricted.py                      # suffix-only / full-overlap restricted optima
PYTHONPATH=. venv/bin/python milp/diagnose.py                        # where greedy / ILA choose differently from the optimum
PYTHONPATH=. venv/bin/python milp/collect_results.py                 # -> data/results/milp/*.csv
```

Figures from the stored results:

```bash
PYTHONPATH=. venv/bin/python analysis/plot_gap_vs_ratio.py   # ILA - greedy vs AV:PV ratio (existing sweep)
PYTHONPATH=. venv/bin/python analysis/plot_milp_gap.py       # % of optimum per instance, choice vs shape
PYTHONPATH=. venv/bin/python analysis/plot_time_effect.py    # ILA advantage with time constraints OFF vs ON
```

### 2. SUMO pipeline on the M1 Monash Freeway (inbound, 20.5 km)

```
osm/m1.net.xml -> m1/corridor.py -> randomTrips -> sumo -> convert.py -> greedy / ILA
```

Run the whole chain from scratch (about 5 minutes; the matching step is the
slow part):

```bash
# a. main-line chain and the 16 ramp positions -> sumo/m1/corridor.json
PYTHONPATH=. venv/bin/python sumo/m1/corridor.py

# b. inbound-only origin / destination weights for randomTrips -> sumo/m1/inbound.{src,dst}.xml
PYTHONPATH=. venv/bin/python sumo/m1/make_weights.py

# c. placeholder demand, one hour, 1800 vehicles (uniform over inbound-side fringe edges)
cd sumo/m1 && SUMO_HOME=../../venv/lib/python3.14/site-packages/sumo \
  ../../venv/bin/python $SUMO_HOME/tools/randomTrips.py -n ../osm/m1.net.xml \
  -o trips.xml -r routes.xml -b 0 -e 3600 --period 2 --weights-prefix inbound \
  --vehicle-class passenger --seed 42 --validate && cd ../..

# d. simulate (about 4 s) -> tripinfo.xml, fcd.xml (10 s sampling)
cd sumo/m1 && ../../venv/bin/sumo -c m1.sumocfg && cd ../..

# e. convert and measure instance structure -> sumo/m1/trips.json, stdout summary
PYTHONPATH=. venv/bin/python sumo/convert.py sumo/m1 --summary --dump sumo/m1/trips.json

# f. greedy / ILA on the converted instance -> sumo/m1/smoke_match.json
PYTHONPATH=. venv/bin/python sumo/smoke_match.py

# g. AV:PV ratio sweep -> sumo/m1/ratio_sweep.csv  (about 8 min per seed)
PYTHONPATH=. venv/bin/python -u sumo/ratio_sweep.py --ratios 0.2 0.4 0.6 0.8 --seeds 42

# h. figures -> analysis/figures/
PYTHONPATH=. venv/bin/python analysis/plot_corridor.py            # corridor schematic
PYTHONPATH=. venv/bin/python analysis/plot_entry_dist.py          # synthetic vs M1 distributions
PYTHONPATH=. venv/bin/python analysis/plot_ratio_m1_vs_synth.py   # ratio sweep, M1 vs synthetic
```

Steps a-e are fast; f and g are slow because the greedy baseline is pure
Python (see Revision Plan item #3).

Demand is a `randomTrips` placeholder: only the road network is real. The
structure measurements (9 entry / 9 exit points, 44 OD pairs, share of trips
below `L_min`, constant-speed drift) do not depend on the demand model; the
saving percentages do. Options for calibrating demand against DTP detector
counts are in `sumo/NOTES-sumo.md`.

Watching the simulation (needs XQuartz; the script re-registers the X
authorisation cookie for the current hostname):

```bash
sumo/m1/gui.sh
```

In the GUI, open the visualisation settings (colour-wheel icon), set
*Vehicles -> Exaggerate by* to 20 and *Colour by -> speed*; at the default
zoom a 4.5 m car on a 20 km corridor is smaller than a pixel.

### 3. Where results are stored

| Path | Content |
|---|---|
| `data/results/greedy/`, `data/results/hungarian/` | original paper sweeps (unchanged) |
| `data/results/milp/` | exact-optimum comparison, time-effect, restricted optima |
| `sumo/m1/structure.json` | instance-structure metrics of the M1 scenario |
| `sumo/m1/smoke_match.json`, `sumo/m1/ratio_sweep.csv` | greedy / ILA on M1 |
| `analysis/figures/` | all revision figures |
| `reports/` | interim findings (docx / md) and meeting memos |

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
