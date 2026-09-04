# Working notes — revision of the 1D dynamic platoon formation paper

Context: SIGSPATIAL 2026 short paper (Paper 83) was rejected on 2026-08-12.
The Revision Plan (shared with Aamir and Eunus on 2026-08-21) lists ten reviewer
weaknesses. Aamir asked to front-load items #1 (realistic data via SUMO),
#2 (does ILA actually beat greedy) and, if possible, #3 (fair greedy baseline),
so that we learn early whether this work has potential.

Repository moved out of iCloud Drive on 2026-09-04; it now lives at
`~/dev/platoon-formation`. Use `venv/` (Python 3.14) for everything.

---

## 2026-09-04

### Environment consolidated and reproduction confirmed

There were two virtual environments with different contents, and the package
versions had drifted well away from those reported in the paper:

| | paper | now |
|---|---|---|
| Python | 3.11 | 3.14.2 |
| NumPy | 1.26 | 2.4.0 |
| SciPy | 1.11 | 1.17.0 |

Consolidated to a single `venv`, pinned in `requirements-lock.txt`, and wrote
`analysis/reproduce_check.py`, which re-runs a subset of the stored sweep
configurations and compares them against `data/results/*/pv_av_sweep.csv`
(produced 2026-01-26).

**Result: 48/48 combinations match exactly** (`baseline_total_distance`,
`total_saving`, `saving_percent`, `matched_pv`), for both greedy and ILA.
The published numbers are reproducible in the current environment, so any
change from here on can be compared directly against them.

Note: `numpy==2.4.0` is a *yanked* release (backward-compatibility bug).
Move to a non-yanked version and re-run the reproduction check before the
environment is reported in the paper.

`scipy.optimize.milp` (HiGHS) is available in this environment, so the exact
solver of item #4 needs no new dependency.

### Re-analysis of the existing sweep data

The stored `pv_av_sweep.csv` files for greedy and ILA cover identical instances
(same seeds 42-45, same parameters; `baseline_total_distance` agrees on all 216
paired rows), so they can be differenced per instance.

Grouping the paired difference by the active-to-passive ratio |AV|/|PV| gives an
inverted-U (see `analysis/plot_gap_vs_ratio.py`):

| ratio | 0.01 | 0.05 | 0.10 | 0.20 | 0.40 | **0.80** | 1.60 | 3.20 |
|---|---|---|---|---|---|---|---|---|
| ILA − greedy (pp) | 0.000 | ~0.02 | ~0.10 | 0.17–0.40 | 1.12–1.25 | **1.26** | 0.32–0.44 | 0.05–0.18 |

Two consequences:

1. **The sweep proposed in Revision Plan item 2** (400 PVs against 100, 50, 25
   and 10 AVs, i.e. ratios 0.25 down to 0.025) lies entirely on the left tail,
   where the difference goes to zero. Running it as written would produce a null
   result regardless of how realistic the data is. The sweep should instead
   cover roughly 0.4-1.6. The same applies to "tighter capacities": in
   `capacity_sweep.csv` the gap *shrinks* as capacity tightens (0.84 pp at
   C_max=16 down to 0.35 pp at C_max=2).

2. **Both published figures were drawn at a single ratio each** — the capacity
   sweep at N=50/M=200 (0.25) and the length sweep at N=80/M=400 (0.20), both on
   the flat part of the curve. `compare_algorithms.py` only ever plotted those
   two sweeps; the pv/av sweep was tabulated but never plotted. This is a
   plausible partial explanation for the 0.2-0.9 pp the reviewers objected to,
   alongside the paper's own explanation (that a 1D corridor limits combinatorial
   diversity).

Caveats: four seeds only; one cell at ratio 0.4 (AV 80 / PV 200) sits at
0.001 pp against ~1.1-1.25 pp for other cells at the same ratio, so the curve is
suggestive rather than settled. Item #5 (more seeds) would settle it. All of
this is on the existing uniform 1D synthetic data.

### Open question this does not answer

The gap peaks at 1.26 pp, which is still small. Whether that is because ILA is
weak or because greedy is already near-optimal cannot be told from these numbers,
and the two imply opposite responses (improve the algorithm vs. change the
formulation). Only an exact solver answers it, which argues for pulling a small
MILP (item #4) forward rather than leaving it until after the SUMO work.

Related: item #9 proposes restricting the formal definition to maximal residual
overlap segments, matching what the algorithms do. That removes partial-segment
towing, which is one of the few structural sources of value for coordination over
a myopic choice, and would shrink the achievable gap by construction. Worth
raising before committing to that fix.

### Next

- [ ] Small-instance MILP, starting from the time-free case, to measure how far
      greedy is from optimal (item #4, pulled forward)
- [ ] Install SUMO, import a small OSM network; confirm the 1D projection
      approach with Mushfiq (extract one arterial and keep only vehicles
      traversing it)
- [ ] Re-run the ratio sweep densely over 0.4-1.6 with ~20 seeds, and re-run the
      capacity sweep at the peak ratio (items #2, #5)
- [ ] Vectorise greedy with numpy, with the stored `saving_percent` values as a
      regression oracle (item #3). Expect ILA's current runtime advantage to
      disappear, since it comes from SciPy's compiled LSAP rather than from the
      algorithm.
