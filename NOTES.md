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

### Exact solver (item #4, pulled forward)

`milp/exact.py` formulates the problem as a MILP and solves it with HiGHS via
`scipy.optimize.milp`. Positions are integers, so a towing segment is a run of
unit intervals; `z[i,j,x]` says PV *j* is towed by AV *i* over `[x, x+1)`, start
indicators enforce `L_min`, and the point-wise capacity and PV non-overlap
constraints come straight from Section 2.3 of the paper. Dropping integrality
gives the LP relaxation, which is an upper bound when the MILP does not finish.

**Scope.** The temporal synchronisation constraint is state dependent (towing a
PV changes its arrival time on later segments), which is not linear, so this
covers the spatial problem only — `enable_time_constraints=False`. That is
enough to ask whether the formulation leaves room for coordination at all.

**Tractability.** 21 instances (7 sizes x 3 seeds, AV 4-15 / PV 8-30) solved to
proven optimality; the largest, AV 15 / PV 30, took up to 142 s with ~10,500
binaries. AV 20 / PV 40 did not prove optimality within 60 s, so that is roughly
where the exact comparison has to stop and hand over to the LP bound.

**Result.** Both heuristics sit well short of the optimum, and neither is better
than the other:

| | mean % of optimum | range |
|---|---|---|
| greedy | **93.33%** | 80.95 - 100.00 |
| ILA | **93.09%** | 80.95 - 100.00 |

ILA wins on 9 instances, loses on 8, ties on 4 — a coin flip. About 7% of the
achievable towed distance is left on the table by both.

**Where that 7% lives.** Solve each instance twice: once unrestricted, and once
with every segment forced to run to the end of its overlap (`suffix_only`). The
restricted version is the only shape greedy and ILA ever commit - both set
`s = max(e_i, l_j)`, `e = dp_ij`, so the start can be pushed back by an earlier
tow on the same PV but the end never is. 18 instances, both solves proven
optimal (`milp/restricted.py`, `data/results/milp/restricted.csv`):

| | mean |
|---|---|
| greedy / unrestricted optimum | 93.36% |
| suffix-only optimum / unrestricted optimum | **99.35%** |
| greedy / suffix-only optimum | **93.96%** |
| ILA / suffix-only optimum | **93.64%** |

**Almost none of the shortfall is about expressiveness.** Restricting to the
segment shapes the algorithms can actually produce costs only 0.65% of the
achievable towed distance. The remaining ~6% is lost by choosing badly among
moves that were available all along. ILA is marginally *worse* than greedy even
against the restricted optimum, so its per-round LSAP optimality is not
converting into better choices.

Counting segments gives the opposite impression and should not be used: 33% of
optimal segments end before `dp` and are therefore unreachable, but they are
short and contribute little value. (An earlier version of this note reported
57.6% "partial" segments and concluded that most of the headroom was out of
reach. That figure also counted late-*starting* segments, which the algorithms
can produce through the residual mechanism, and counting segments rather than
distance was the wrong measure regardless. The value-based split above
supersedes it.)

Figure: `analysis/figures/milp_gap.png` (`analysis/plot_milp_gap.py`).

### Item #9 is safe to do as planned, and now defensible

Item #9 proposes restricting the formal definition to the segment shapes the
algorithms handle. The measurement above says that costs 0.65% of the achievable
towed distance, so it is a fair restriction rather than a retreat - and it can
now be justified with a number instead of asserted. Worth reporting that way
in the revision.

### Where ILA's advantage actually comes from

Comparing greedy and ILA at paper scale with the temporal constraint off and on
(`data/results/milp/time_effect.csv`, figure `analysis/figures/time_effect.png`):

| configuration | AV:PV | temporal OFF | temporal ON |
|---|---|---|---|
| capacity sweep (paper default) | 50:200 | +0.26% | +1.84% |
| length sweep (paper default) | 80:400 | +0.11% | +1.38% |
| ratio 0.8 | 160:200 | +1.19% | +1.89% |
| ratio 0.8, larger | 320:400 | +0.23% | +2.47% |

ILA's advantage is larger with the temporal constraint active in all four
configurations. The paper attributes the advantage to globally coordinated
*spatial* assignment and predicts it will widen on 2D road networks, where
overlap patterns are richer. These numbers point elsewhere: with the temporal
constraint off, the advantage nearly disappears even at the ratio where it is
largest.

Caveat: four seeds, and the ON/OFF intervals overlap within each configuration,
so this rests on the pattern holding in all four rather than on any single one.
It is a reason to check the premise of the 2D paper before committing to it, not
yet a refutation.

Correction to the earlier note in this file: an initial run at AV <= 10 suggested
ILA was *worse* than greedy. Over the full 21 instances it is a tie, and at paper
scale ILA is consistently ahead by a small margin. The small-instance result was
a sample-size effect.

### Diagnosis: which decisions the optimum makes differently

`milp/diagnose.py` solves each instance three times, restricting the segment
shapes the solver may use, and also compares the structure of the solutions
rather than only their values. 15 instances, all three solves proven optimal.

Value, as a share of the unrestricted optimum:

| | mean |
|---|---|
| suffix-only optimum (segment may start late, must end at `dp`) | 99.41% |
| full-overlap optimum (segment covers the whole overlap) | 98.03% |
| greedy | 93.79% |
| ILA | 93.43% |
| greedy against the full-overlap optimum | **95.68%** |
| ILA against the full-overlap optimum | **95.32%** |

So of the ~6.2% shortfall, only ~2% needs segment shapes beyond the simplest
one. **The remaining ~4.3% is lost choosing badly among whole-overlap moves that
were available from the start.** ILA is slightly worse than greedy even there.

(ILA exceeds the full-overlap optimum on two instances - 101.5% and 103.3%.
That is not an error: ILA does produce late-starting segments, as a by-product
of `l_j` advancing after an earlier tow, and the full-overlap solve forbids
them. It confirms the mechanism rather than contradicting it.)

Structure of the solutions, averaged over the same instances:

| | segments | mean length | PVs served | AV capacity used |
|---|---|---|---|---|
| optimum | **23.7** | **15.2** | **87.6%** | **75.1%** |
| greedy | 13.6 | 24.1 | 75.5% | 70.7% |
| ILA | 15.3 | 21.2 | 88.3% | 70.3% |

**The optimum uses about 74% more segments, each about 37% shorter.** greedy
sorts candidates by segment length and commits the longest first; a long tow
occupies an AV's capacity over a long stretch of road and blocks other PVs from
attaching anywhere inside it. The optimum instead splits service into shorter
tows, packs more PVs into the same capacity (75.1% vs 70.7% used) and serves
87.6% of PVs against greedy's 75.5%.

Worth noting for the paper: multi-segment service - a PV handed from one AV to
another - is the feature the paper puts forward as novel, and it is precisely
what the optimum exploits and the two algorithms underuse.

### Concrete targets for a better algorithm

- **greedy** ranks candidates by saved distance alone. The quantity that matters
  is closer to saving *per unit of capacity occupied*, since a long tow costs an
  AV a long stretch of exclusivity.
- **ILA** builds its LSAP cost matrix from residual overlap length only. The
  point-wise capacity conflict is not priced in at all - it is handled after the
  solve, by discarding conflicting pairs. So the round is optimal with respect
  to a cost that ignores the constraint that actually binds. Pricing congestion
  into the cost (for instance, penalising segments that cross positions where
  the AV is already near capacity) is the most direct candidate fix, and the one
  that would give ILA a reason to exist.

### What this means for the go/no-go question

Aamir asked for early clarity on whether the work has potential. The answer:

- There **is** headroom - about 7% against the exact optimum, not the sub-1%
  that would have made an algorithmic contribution impossible.
- **It is reachable within the current problem definition.** Only 0.65% of it
  needs segment shapes the algorithms cannot express; the rest is choice
  quality. So the formulation does not have to change.
- **Neither algorithm captures it.** greedy reaches 93.96% of what its own move
  set allows, and ILA 93.64% - ILA is not converting global per-round matching
  into better selections.

So the honest position on item #2 is that ILA, as it stands, does not earn its
place; but the problem does have roughly 6% of headroom for an algorithm that
chooses better, and that is a well-defined target rather than a hope. What such
an algorithm looks like - local search over committed segments, a lookahead on
the capacity-critical positions, or an LSAP whose costs price in the capacity
conflict instead of discarding afterwards - is the open question.

Still to verify: all of this is on the uniform synthetic generator, on small
instances, and with the temporal constraint disabled in the exact comparison.
The SUMO data (item #1) has not entered yet.

### Next

- [x] Small-instance MILP to measure how far greedy is from optimal
      (item #4) - done 2026-09-04; see above
- [x] Find where the ~6% of bad choices occur - done 2026-09-04, see the
      diagnosis section above
- [ ] Price point-wise capacity into ILA's cost matrix and re-measure against
      the exact optimum. This is the route to item #2
- [ ] Try ranking greedy by saving per unit of occupied capacity rather than by
      saving alone, as a second baseline
- [ ] Report item #9 as measured rather than assumed: the restriction costs
      0.65% of achievable distance
- [ ] Install SUMO, import a small OSM network; confirm the 1D projection
      approach with Mushfiq (extract one arterial and keep only vehicles
      traversing it)
- [ ] Re-run the ratio sweep densely over 0.4-1.6 with ~20 seeds, and re-run the
      capacity sweep at the peak ratio (items #2, #5)
- [ ] Vectorise greedy with numpy, with the stored `saving_percent` values as a
      regression oracle (item #3). Expect ILA's current runtime advantage to
      disappear, since it comes from SciPy's compiled LSAP rather than from the
      algorithm.
