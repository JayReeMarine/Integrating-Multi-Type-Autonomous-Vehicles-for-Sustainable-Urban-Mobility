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

### Trying to fix the algorithms: congestion-aware variants

`core/congestion.py` adds three variants. None of them touches the shipped code;
each reproduces its original exactly at the neutral parameter, which the sanity
checks assert.

For AV *i* at position *x*, let `contest(i,x)` be the number of still-uncovered
PVs that AV *i* could tow across *x*, and `free(i,x)` its remaining capacity
there. `scarcity = contest / free`. Then:

- `greedy_congestion_matching(lam)` ranks candidates by
  `saving / (1 + lam * mean scarcity)` instead of by `saving`. `lam=0` is the
  shipped greedy.
- `ila_congestion_matching(lam)` puts the same adjusted value into the LSAP cost
  matrix. `lam=0` is the shipped ILA.
- `ila_filter_matching(theta)` keeps the LSAP objective as raw towed distance and
  instead withholds candidates whose mean scarcity exceeds `theta` from the
  current round; they stay eligible later. A round that would be empty falls
  back to the unfiltered set. `theta=inf` is the shipped ILA.

**Time-free, against the exact optimum** (`milp/sweep_variants.py`, the same 15
instances):

| variant | mean % of optimum | worst |
|---|---|---|
| greedy (shipped) | 93.79% | 87.11% |
| **greedy + lam=2** | **95.99%** | 89.69% |
| ILA (shipped) | 93.43% | 83.51% |
| ILA + lam=1 | 92.43% | 83.51% |
| **ILA + filter theta=2.5** | **94.80%** | 83.51% |

Two things follow. First, congestion is a real signal: it recovers roughly a
third of greedy's shortfall from one cheap statistic. Second, *where* it is
injected matters. Discounting the LSAP cost makes ILA worse at every weight
tried, because the round then maximises adjusted score and trades real towed
distance away to avoid contested road. Using the same signal only to withhold
candidates, leaving the objective as raw distance, improves ILA by 1.37 pp. The
framework is not broken; the first injection point was wrong.

But in this setting greedy still ends up 1.18 pp ahead of the best ILA variant.
On the spatial problem alone, the signal does not need global coordination.

**Time on, at paper scale** (`milp/sweep_time_on.py`, AV 50-320, 4 seeds), as a
percentage of the shipped greedy:

| configuration | greedy | greedy+lam2 | ILA | ILA+lam1 | ILA+filter |
|---|---|---|---|---|---|
| capacity sweep 50:200 | 100% | +2.02% | +1.82% | +1.95% | +0.12% |
| length sweep 80:400 | 100% | +1.62% | +1.34% | +1.50% | +1.22% |
| ratio 0.8, 160:200 | 100% | +1.41% | **+1.91%** | +0.69% | +0.60% |
| ratio 0.8, 320:400 | 100% | **-1.59%** | **+2.47%** | +0.06% | -0.91% |
| **mean** | 100.00% | 100.87% | **101.88%** | 101.05% | 100.26% |

**With the temporal constraint active and at the paper's own scale, the shipped
ILA is the best of the five.** The congestion variants do not help it there, and
the congestion-aware greedy actually turns negative on the largest
configuration while ILA is at its strongest.

### The two regimes disagree, and that is the finding

| setting | best |
|---|---|
| temporal constraint off, small instances | greedy + congestion (96.0% of optimum) |
| temporal constraint on, paper scale | ILA (+1.88% over greedy) |

Taken with the ON/OFF comparison above, the picture is consistent: **ILA earns
its place only where temporal feasibility binds.** On the purely spatial problem
a myopic rule with a capacity-aware ranking does better, and ILA's per-round
optimality buys nothing. Once arrival-time synchronisation starts eliminating
candidate pairs, which pairs are chosen begins to matter and the global
assignment pays for itself.

That is a sharper and better-supported claim than the paper's current
explanation, which attributes the advantage to spatial coordination on a 1D
corridor and predicts it will widen on 2D road networks.

### What to change in the paper

1. **Report the ratio sweep, and report it at the contested end.** The paper
   measures only at |AV|/|PV| of 0.20 and 0.25, where ILA gains 1.3-1.8%. At 0.8
   it gains 1.9-2.5%. Showing the whole sweep and the trend answers item #2 with
   a mechanism rather than a single number - and is not cherry-picking, provided
   the full curve is shown.
2. **Explain the advantage as temporal, not spatial**, and support it with the
   ON/OFF contrast. This also means the premise behind the 2D paper needs
   re-examining before that project starts.
3. **Add congestion-aware greedy as the second baseline** item #4 asks for. It
   beats the shipped greedy in three of four configurations, so showing ILA
   ahead of it is a stronger result than beating plain greedy. Its instability
   at the largest configuration should be reported, not hidden.

Remaining weaknesses, to state plainly in any write-up: ILA's advantage is still
small (~1.9%); four seeds; no exact reference in the time-on setting, since the
MILP cannot express state-dependent arrival times, so only relative comparisons
are available there; and the congestion variants are a first cut with a coarse
parameter grid.

### What this means for the go/no-go question

Aamir asked for early clarity on whether the work has potential. The answer is
**conditional go**.

- There is about 7% of headroom against the exact optimum, so an algorithmic
  contribution is possible in principle. It is not the sub-1% that would have
  ended the work.
- **ILA does earn its place, but only in the regime the paper actually uses**:
  with temporal synchronisation active and at fleet scale it is the best of the
  five variants tried, by 1.88% over greedy on average and 2.47% at the most
  contested configuration. On the purely spatial problem it does not, and a
  capacity-aware greedy beats it.
- The advantage is still modest, and the reviewers rejected 0.2-0.9 pp as too
  small. Measuring at the contested ratio rather than at 0.20-0.25 roughly
  doubles it, which helps but may not be enough on its own.

The decision to put to Aamir and Eunus is therefore not "continue or stop" but
which of these to spend the next weeks on: strengthening ILA's margin by
reporting it where it is real, or accepting that the algorithmic contribution is
thin and rebuilding the paper around the exact solver and the characterisation
of when coordination pays.

### Next

- [x] Small-instance MILP to measure how far greedy is from optimal
      (item #4) - done 2026-09-04; see above
- [x] Find where the ~6% of bad choices occur - done 2026-09-04, see the
      diagnosis section above
- [x] Price point-wise capacity into ILA and into greedy, and measure both
      against the exact optimum - done 2026-09-04, see above
- [ ] Re-run the ratio sweep with many more seeds, temporal constraint on,
      covering 0.2 to 1.6, to establish the trend in ILA's advantage properly
      (items #2 and #5 together)
- [ ] Decide with Aamir and Eunus between strengthening ILA's margin and
      rebuilding the contribution around the exact solver
- [ ] Vectorise greedy (item #3). Quality is unaffected - implementation does
      not change which segments are selected - but the runtime claim in the
      abstract cannot stand until both run at compiled speed
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
