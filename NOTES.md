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

---

## 2026-10-01 — ILA against the exact optimum across the AV:PV ratio

Eunus, 18 Sep: drop greedy as the headline baseline; the paper's question is how
close the scalable heuristic (ILA) gets to the exact optimum, and specifically
whether the gap widens when AVs are scarce (AV:PV = 1, 2, 5, 10, 20 %) because
"if you give one PV the chance, maybe you lose another that would go the longer
distance".

**Scope.** `milp.exact` covers the spatial problem only
(`enable_time_constraints=False`); there is still no optimum baseline on the
temporal axis, which is where ILA's advantage over greedy lives. Everything
below is the spatial problem.

### What was run

| script | instances |
|---|---|
| `milp/probe_lowratio.py` | tractability probe |
| `milp/lowratio_sweep.py` | synthetic uniform, PV = 50 and PV = 100, ratios 1–20 %, 10 seeds each (plus 40 % at PV = 50) |
| `milp/m1_lowratio.py` | M1 inbound corridor, 5-minute windows from `sumo/convert.py` D6, 12 windows per ratio, ratios 1–20 % |

Outputs: `data/results/milp/lowratio_50.csv`, `lowratio_100.csv`,
`m1_lowratio_300.csv`. Figures: `analysis/plot_lowratio.py` →
`analysis/figures/ila_vs_optimum_by_ratio.png`, `analysis/plot_saturation.py` →
`analysis/figures/av_saturation.png`.

**Tractability was the opposite of what was expected.** Low ratios are *cheap*:
the MILP has one binary per (AV, PV, unit interval), so few AVs means few
binaries. PV 200 at 1 % proves optimal in 1.4 s, where the September work
could not get past AV 15 / PV 30. The expensive end is 20 % and above, where
some instances hit the 900 s limit and are recorded against the LP bound
(so their ILA % is a lower bound). One row (PV 100, ratio 20 %, seed 46) had
the LP relaxation itself time out, giving `optimum = inf`; it is dropped in
analysis.

### Result — the gap does **not** widen when AVs are scarce; it narrows

ILA as % of the exact optimum, proven-optimal instances only:

| AV:PV | synthetic PV = 50 | synthetic PV = 100 | M1 5-min windows |
|---|---|---|---|
| 1 % | 98.7 % (8/10 exactly optimal) | 99.2 % (8/10) | **100.00 % (12/12)** |
| 2 % | 98.7 % (8/10) | 99.5 % (8/10) | 99.7 % (11/12) |
| 5 % | 98.8 % (7/10) | 99.4 % (7/10) | 99.7 % (11/12) |
| 10 % | 98.0 % (5/10) | 99.1 % (4/10) | 99.4 % |
| 20 % | 95.9 % (1/10) | 96.8 % | — (LP bound) |
| 40 % | 96.4 % (n = 1) | — | — |

For reference, the September instances (AV 15 / PV 30, ratio ≈ 50 %) gave 93 %.
The three data sets agree, and the real corridor is the most extreme: at 1 % ILA
is exactly optimal in every one of the twelve windows.

So Eunus's hypothesis is not supported. Contention in the sense of "many PVs per
AV" does not hurt the heuristic.

### Correction to a first explanation

The first reading recorded in chat — "at low ratios the AVs saturate, so the
optimum has no freedom either" — is **wrong**. Measured: the optimum fills
99–100 % of the AV capacity-distance at *every* ratio from 1 % to 40 %.
Saturation does not vary with the ratio.

Towed distance *is* capacity-distance used, so "% of optimum" is exactly "how
well the AVs were packed". Splitting the two:

| AV:PV | optimum fills | ILA fills (PV = 100) |
|---|---|---|
| 1 % | 100.0 % | 99.2 % |
| 5 % | 100.0 % | 99.4 % |
| 10 % | 100.0 % | 99.1 % |
| 20 % | 100.0 % | 97.6 % |

**The AV fleet is the binding resource at every ratio; the optimum always keeps
it essentially full. The gap is entirely ILA's packing.** With one or two AVs
there is nothing to coordinate — tow the longest overlap available. With ten or
twenty AVs, which PV goes to which AV becomes combinatorial, and ILA's
discard-on-conflict step loses value it cannot recover.

### What this means for the paper

1. **Where the heuristic matters is where AVs are plentiful.** The headroom
   grows with the ratio: ~0.5 pp at 1–5 %, ~2.4 pp at 20 %, ~3.6 pp at 40 %,
   7 pp at ≈ 50 %.
2. **The amount of energy at stake grows the same way.** The optimum tows 2 %
   of total PV distance at ratio 1 %, 14 % at 5 %, 42–44 % at 20 %, 60 % at
   40 %. Both the prize and the difficulty increase with the ratio.
3. **So "what is a realistic AV:PV ratio" now directly decides the paper's
   contribution.** If the realistic regime is 1–5 %, ILA is already optimal and
   there is no algorithmic contribution to make — the paper has to stand on the
   problem, the exact benchmark and the realistic evaluation. If it is 20 % or
   more, the 2–7 pp headroom is real and a better packing step is worth
   building. This is the question to put to Eunus and Aamir.
4. **ILA's measured weakness is packing, not expressiveness** — consistent with
   the September finding that restricting the model to algorithm-reachable
   segments costs only 0.65 %. A refinement step should target reassignment
   among AVs, not segment shape.

### Caveats

- Spatial problem only; no optimum baseline exists with time constraints on.
- M1 demand is still the `randomTrips` placeholder (`sumo/NOTES-sumo.md`).
- M1 sub-instances are a thinned sample of the hour restricted to a 5-minute
  entry band (~2.3× fewer candidate partners than reality); the comparison
  should survive this but absolute saving levels should not be quoted from them.
- 20 % rows partly rest on the LP bound, which understates ILA's true ratio.

### Next

- [ ] Put the ratio question to Eunus and Aamir with these figures
- [ ] If the high-ratio regime is the one that matters: prototype a reassignment
      /local-search step after ILA and score it on these same instances
- [ ] Extend the high-ratio synthetic sweep (40 %, 60 %, 100 %) beyond one seed
- [ ] A temporal optimum baseline remains the open methodological gap

---

## 2026-10-01 (later) — A refinement step that recovers the gap

Having measured *where* ILA loses (above), the obvious next question is whether
it can be fixed. It can, and the route to the fix came from two measurements
rather than from guessing.

### Diagnosis: the lost capacity is unreachable, not unused

For AV 10 / PV 50 seed 43 (ILA at 92.2 % of the optimum):

| | ILA | optimum |
|---|---|---|
| towing segments | 21 | **53** |
| mean segment length | 36.0 | **15.5** |
| PVs served | 21 | **37** |
| segments strictly inside the pair's maximal overlap | 0 by construction | 43 of 53 |

ILA leaves 9.8 % of the AV capacity-distance idle, but **every idle run is
shorter than `L_min = 10`** (longest observed: 8). Nothing can be inserted into
it. ILA always commits the whole residual overlap, so it fills the AVs with a
few long tows; the optimum cuts tows short and hands the capacity to other PVs.

This also explains why the first refinement attempt did nothing. A pass that
only inserts, extends and single-ejects gained 5 units on one instance and 0 on
others, finishing in 0.00 s — by construction it cannot find anything, because
ILA stops exactly when no insertable segment remains.

### The fix: ruin and recreate, with a *randomised* rebuild

`core/refine.py::refine_ils`. Each iteration removes every segment that an
AV subset holds inside a random road window, rebuilds greedily, and keeps the
result only if the total improved.

The rebuild rule is what matters. Inserting the longest feasible run first
reproduces ILA's own bias and gains almost nothing. Sampling a run with
probability proportional to its length (`alpha = 1`) finds the
many-short-segments structure the optimum uses:

| AV 10 / PV 50, seed 43 | % of optimum |
|---|---|
| ILA | 92.20 |
| + rebuild longest-first (`alpha = inf`) | 92.93 |
| + rebuild sampled, `alpha = 2` | 98.41 |
| + rebuild sampled, **`alpha = 1`** | **98.90** |
| + rebuild uniform (`alpha = 0`) | 98.17 |

Rebuilt segments may be partial (a run is limited by capacity and by what the PV
already has), which is the degree of freedom ILA never uses.

### Result — 102 synthetic + 60 M1 instances, ~2 s per instance

Synthetic, mean over 10 seeds per cell (proven-optimal instances):

| PV | 1 % | 2 % | 5 % | 10 % | 20 % | 40 % |
|---|---|---|---|---|---|---|
| 50, ILA | 98.73 | 98.73 | 98.83 | 98.04 | 95.90 | 94.02 |
| 50, **+ refinement** | **99.86** | **99.86** | **99.92** | **99.80** | **99.48** | **97.57** |
| 100, ILA | 99.17 | 99.53 | 99.43 | 99.12 | 97.80 | — |
| 100, **+ refinement** | **100.00** | **100.00** | **99.85** | **99.80** | **98.68** | — |

M1 corridor, 12 five-minute windows per ratio:

| | 1 % | 2 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|---|
| ILA | 100.00 | 99.73 | 99.68 | 99.01 | 99.25 |
| **+ refinement** | **100.00** | **100.00** | **100.00** | **100.00** | **99.78** |

**Over all 50 proven-optimal M1 instances the refined solution averages
100.00 % of the optimum.** On synthetic instances the gain is largest exactly
where ILA is weakest: +3.6 pp at 20 % and at 40 %.

Figure: `analysis/plot_refine.py` → `analysis/figures/refinement_vs_optimum.png`.
Evaluation: `milp/refine_eval.py` → `data/results/milp/refine_synthetic.csv`,
`refine_m1.csv`.

### The refinement is not specific to ILA — tested

Starting the same refinement from greedy instead of ILA (5 seeds per cell,
proven-optimal instances, PV = 50):

| ratio | greedy | greedy + refinement | ILA | ILA + refinement |
|---|---|---|---|---|
| 10 % | 97.47 | **99.93** | 97.66 | 99.81 |
| 20 % | 93.96 | **98.91** | 94.50 | 99.16 |
| 40 % | 93.56 | **98.49** | 93.92 | 97.74 |

Both starting points land in the same place, and greedy + refinement is ahead at
two of the three ratios. **So the contribution is the refinement step, not ILA.**
Claiming "ILA + refinement" as the method would overstate ILA's role; the honest
statement is that a ruin-and-recreate post-processor takes *either* heuristic
from 94-98 % to 98-100 % of the optimum. This is also a cleaner story than the
one the paper had: the starting heuristic stops being the contribution.

### What this gives the paper

A contribution that is not "ILA beats greedy":

> the problem; an exact formulation; a measurement of how far the existing
> heuristics sit from the optimum and *why*; and a refinement step that closes
> that gap to ~0 on real-corridor instances at ~2 s per instance, from either
> starting heuristic.

The diagnosis and the fix are linked: the fix follows from the measurement
(long committed overlaps leave sub-`L_min` fragments), not from trial and error.

### Difficulty is an inverted U in the ratio, and the refinement tracks it

Extending the synthetic sweep to 40, 60 and 100 % (PV = 50, proven-optimal
instances; n is small at the top end because some did not prove within 600 s):

| ratio | AV | greedy | ILA | optimum tows this share of PV distance | ILA + refinement |
|---|---|---|---|---|---|
| 1-5 % | 1-2 | 98.7 | 98.8 | 5-9 % | 99.9 |
| 10 % | 5 | 97.9 | 98.0 | 22 % | 99.8 |
| 20 % | 10 | 95.3 | 95.9 | 44 % | 99.5 |
| **40 %** | 20 | **93.6** | **93.9** | 67 % | **97.7** |
| 60 % | 30 | 95.0 | 96.8 | 98 % | 97.8 |
| 100 % | 50 | 98.4 | 99.1 | 99.9 % | 99.1 |

Both ends are easy for a different reason. At a low ratio there is little to
tow, so there is nothing to choose between. At a high ratio the AVs are so
plentiful that essentially all PV distance gets towed anyway (99.9 % at
ratio 1.0), so again the choice hardly matters. **The hard regime is the middle,
around 20-40 %, where there is a lot to tow and not quite enough capacity** —
and that is exactly where the refinement gains most (+3.6 to +3.8 pp).

**Budget limitation, not a method limitation.** The gain falls away above 40 %
because the prototype runs out of time, not because there is nothing to find:
`_State.candidates` rescans every (AV, PV) pair over the grid for every
insertion, so one iteration at 50 AVs costs ~100x one at 5 AVs. At ratio 1.0
the 15 s budget buys too few iterations to accept anything (gain exactly 0.00
in all 4 instances, all hitting the limit). Incremental candidate updates would
fix this; it has not been done.

### Caveats

- Spatial problem only. The refinement does not yet handle time constraints,
  and no exact baseline exists on the temporal axis.
- Scored against stored optima; instances where optimality was not proven are
  excluded from the tables above.
- Runtime is not optimised: `_State.runs` rescans the grid for every candidate.
  The 2 s figure is a Python prototype, not a complexity claim.
- Greedy was not re-run with refinement; whether the same step helps greedy by
  as much is untested, and worth knowing before claiming the step is specific
  to ILA.

### Next

- [x] ~~Run the refinement on greedy too~~ — it lifts greedy equally; the
      contribution is the refinement, not ILA (section above)
- [ ] Sensitivity: iterations / time budget vs quality
- [ ] Extend to the temporal problem
- [ ] Larger instances where the MILP cannot reach, using the LP bound

---

## 2026-10-01 (late) — An exact optimum WITH time constraints

The open methodological gap in every section above was that `milp.exact` covers
the spatial problem only, while ILA's advantage over greedy lives entirely on
the temporal axis. The September note recorded this as "non-linear, because
towing changes later arrival times". That is true of the schedule but **not of
the model**: the PV's clock is a linear function of the towing decisions.

`milp/exact_time.py` adds `build_time()` / `solve_time()`.

### Formulation

AV clocks are unaffected by towing, so `t_i(x)` is a constant. The PV clock
`t[j,x]` is a variable, and over one unit interval:

```
z[i,j,x] = 1        ->  t[j,x+1] = t_i(x+1)        (a constant)
sum_i z[i,j,x] = 0  ->  t[j,x+1] = t[j,x] + 1/v_j
```

Both written with an indicator (big-M). The tolerance is checked only where a
tow starts, which is what `core/hungarian_multi.py` does
(`abs(pv_time_at_cp - av_time_at_cp) > time_tolerance` rejects a pair):

```
|t[j,x] - t_i(x)| <= tau + M (1 - s[i,j,x])
```

Big-M is sized from the instance (entry-time horizon + span / slowest speed +
tau), not guessed.

### A modelling error the validation caught

The first version advanced the PV clock at the *AV's rate* while carrying the
coupling discrepancy forward, instead of snapping the PV's clock to the AV's.
On 1 of 12 validation instances the heuristics then beat the "optimum" — which
is impossible for a correct model. The heuristics are right: a towed PV rides
the AV, and `hungarian_multi` sets the downstream segment start to
`decoupling_time`, the AV's time at the decoupling point. After the fix, 9 of 9
re-run instances satisfy `heuristic <= optimum`, and the optima rose (82 -> 103,
185 -> 196, 120 -> 123), confirming the first model was wrongly restrictive.

**The invariant `no heuristic may exceed the optimum` is the cheap check for
this class of error, and `milp/time_sweep.py` records it per row
(`invariant_ok`) rather than averaging it away.**

### First numbers — the temporal gap is much larger than the spatial one

| AV / PV / seed | time optimum | greedy | ILA | ILA % of optimum |
|---|---|---|---|---|
| 8 / 16 / 42 | 113 | 103 | 103 | 91.2 |
| 8 / 16 / 44 | 44 | 34 | 34 | 77.3 |
| 10 / 25 / 42 | 123 | 89 | 89 | 72.4 |
| 10 / 25 / 43 | 121 | 121 | 121 | 100.0 |
| 15 / 30 / 42 | 316 | 272 | 272 | 86.1 |
| 15 / 30 / 44 | 196 | 144 | 144 | 73.5 |

Against the spatial optimum the heuristics sit at 93-99 %. **With time
constraints on they range from 72 % to 100 %.** If this holds across a proper
sweep it is a far bigger result than anything in the spatial sections: the
headroom the paper should be chasing is temporal, not spatial, and it is 10-28
pp rather than 2-7 pp.

Treat the table as preliminary — six instances, one tau, one time window.
`milp/time_sweep.py` is running PV in {20, 30, 50} x ratio in {0.1, 0.2, 0.4,
0.6} x 5 seeds, recording greedy, ILA, the temporal optimum and the spatial
optimum side by side.

### What this changes

1. The refinement in `core/refine.py` is **spatial only**. If the temporal gap
   is confirmed, the refinement has to be extended to the temporal problem
   before it can be the paper's method — right now it improves the easier axis.
2. ILA vs greedy can finally be scored on the axis where they differ. In the
   six instances above they are identical in five and differ by 3 units in one,
   which is consistent with every earlier finding.
3. The paper gains the thing reviewers asked for and the project did not have:
   an exact optimum for the problem as actually defined, time constraints
   included.

### Caveats

- Big-M weakens the LP relaxation; expect this to scale worse than the spatial
  MILP. AV 20 / PV 40 took 105 s under the first (wrong) model and has not been
  re-timed under the corrected one.
- Only tau = 5 model units and time window 100 are tested so far. Both are the
  paper's defaults; neither has been varied.
- The semantics modelled are the ones the heuristics implement. If the paper
  changes the waiting model (Revision Plan item 10), this MILP changes with it.

---

## 2026-10-02 — Temporal semantics: one false alarm, one real bug, one design hole

Three separate things turned up while building an exact optimum with time
constraints. They are easy to confuse, so they are separated here.

### (a) A false alarm — retracted

Earlier today I claimed both heuristics return time-infeasible solutions, based
on AV 12 / PV 20 seed 46 where they reported 187 against a "proven optimum" of
179. **That claim was wrong.** My checker assumed the PV's clock snaps to the
AV's at coupling; the implementation instead models *waiting*
(`coupling_time = max(pv_time, av_time)`). Under its own convention that
instance is fine:

```
PV1 reaches 67 at 56.861; AV2 is at 67 at 52.627  -> |diff| 4.23 <= tau   OK
coupling = max(...) = 56.861                      <- the AV waits
decoupling = 56.861 + 11/1.015 = 67.696
AV11 is at 78 at 70.518 -> |67.696 - 70.518| = 2.82 <= tau                OK
```

The MILP was solving a *narrower* problem, so its optimum was legitimately
below the heuristic value. `milp/exact_time.py` now takes
`waiting={"either","pv_only","none"}`; `"either"` matches the implementation and
is the only setting under which heuristic-vs-optimum is apples to apples.

### (b) A real bug — a second tow is checked against the pre-tow schedule

Re-auditing under the correct convention still leaves violations, and a small
instance makes the cause unambiguous (AV 5 / PV 10, seed 43, greedy):

```
PV3 towed by AV5 [13,38)      -> decoupling 44.27
free-runs 38 -> 63            -> 44.27 + 25/1.177 = 65.52   (true clock)
AV3 is at 63 at 51.41         -> |65.52 - 51.41| = 14.11 > tau = 5
but the recorded coupling is 54.56
  and PV3's *un-towed* time at 63 is 6.12 + 57/1.177 = 54.55
```

The second coupling was evaluated against PV3's **original** schedule, ignoring
that the first tow had delayed it by about 11. `PVRoutingState` stores the
updated time correctly; something upstream decides before applying.

**Rate, 60 instances x 2 algorithms (PV 20/50/100, ratio 10-80 %, 5 seeds,
tau = 5), checked under `waiting="either"`:**

| | instances affected | infeasible couplings | towed distance in affected PVs |
|---|---|---|---|
| greedy | 8 / 60 | 15 of 1370 segments (1.09 %) | 2.08 % |
| ILA | 10 / 60 | 17 of 1457 segments (1.17 %) | 2.37 % |

By ratio the rate climbs with the number of AVs (greedy 1/15 at 10 %, 5/15 at
80 %; ILA 0/15 at 10 %, 5/15 at 80 %) — more AVs, more hand-overs, more chances
to use a stale clock.

**Why ~1 % matters here:** the ILA-greedy differences this project reports are
1-2.5 pp, the same order as the contamination. Any time-constrained comparison
of the two is unsafe until this is fixed, including the 18 Sep update to Eunus
and the September "two regimes" argument for keeping ILA. The spatial results
are untouched (no clocks, no hand-overs), which covers the exact-optimum work,
the low-ratio sweep and the refinement.

### (c) A design hole — waiting is not propagated to the AV

Separately from the bug: when an AV waits, **its own timetable is never
updated**. `av.time_at_point(x)` keeps returning the un-waited time for every
other pairing, so an AV that waited 4 units at one point still claims its
original arrival everywhere downstream.

Over 948 segments (greedy and ILA, AV/PV from 12/20 to 40/100, 5 seeds):

| | |
|---|---|
| segments where the AV waits for the PV | **457 (48.2 %)** |
| mean wait | **2.54 time units** (about 21 s at the M1 scale) |
| AVs waiting for one PV while carrying others on the un-shifted timetable | **180** |

Waiting happens in about half of all tows, so this is not an edge case. It is
reviewer item 10 — "the tolerance permits different arrival times, but waiting
and its effect on subsequent route-state updates are not modelled" — with a
number attached.

### What has to happen

1. **Fix (b).** Evaluate a candidate coupling against the clock the PV actually
   has at that moment. Add `milp/check_feasible.py` to the experiment harness so
   it cannot regress.
2. **Decide the convention for (c)**, then make the model and the implementation
   agree. Options: no waiting; PV waits only (keeps the AV timetable exact);
   or either waits with the AV's schedule propagated to all its passengers (most
   faithful, most work).
3. **Re-run everything time-constrained** and re-state the 18 Sep numbers.
4. The time-constrained optimum numbers recorded earlier today (greedy 85.2 %,
   ILA 85.7 %) were computed under `waiting="none"` and compare against the
   wrong problem. **Discard them.**

### Solver note

Under `waiting="either"` the LP relaxation is weak enough to be useless unless
the clock variables are given per-variable windows (a PV cannot arrive before
travelling at the fastest speed available, nor later than riding the slowest,
give or take one tolerance per coupling). Without them an LP bound of 240.9
appeared against a true optimum of 63. With them, 18 of 20 validation instances
prove optimality in under 5 s; the two that do not are at AV 12 / PV 20.

### Lesson

The invariant `heuristic <= optimum` fired three times today: on a genuine MILP
error (the clock recursion), on my own wrong assumption (the waiting
convention), and on a genuine implementation bug (the stale clock). It tells you
something is wrong, never which side. Each time the answer came from tracing one
small instance by hand.

---

## 2026-10-02 (later) — The stale-clock bug: fixed, and its effect measured

### The fix

`core/timing.py` is now the single place that knows how a towed PV's clock
behaves. `simulate()` replays a PV's whole journey under the "either side waits"
convention; `is_feasible()` answers whether every coupling in a proposed set of
tows still satisfies the tolerance. Both matchers call it before committing a
candidate, instead of checking only the candidate's own coupling point.

Verification: 30 instances (AV 5-80, PV 10-200, 5 seeds, tau = 5) checked with
`milp/check_feasible.py` — **0 violations for greedy and 0 for ILA**, against
8/60 and 10/60 before.

### Effect on the results — much smaller than I warned

Re-running the stored `pv_av_sweep` (128 comparable cells, both algorithms,
`analysis/rerun_gap_after_fix.py` → `data/results/gap_after_fix.csv`):

| | mean change | worst cell |
|---|---|---|
| greedy saving | **-0.034 pp** | -0.640 pp |
| ILA saving | **-0.039 pp** | -0.650 pp |

| | before fix | after fix |
|---|---|---|
| mean ILA - greedy gap | +0.263 pp | **+0.259 pp** |

Per ratio, the curve is unchanged in shape and almost unchanged in value:

| ratio | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 | 1.6 | 3.2 |
|---|---|---|---|---|---|---|---|
| gap before | +0.030 | +0.075 | +0.197 | +0.712 | **+0.933** | +0.389 | +0.115 |
| gap after | +0.030 | +0.071 | +0.193 | +0.692 | **+0.918** | +0.391 | +0.115 |

**Correction to what I wrote earlier today.** I said every time-constrained
comparison was "unsafe until this is fixed", including the 18 Sep update to
Eunus and the September two-regimes argument. Measured, the effect on the
*difference* between the two heuristics is 0.004 pp — negligible. The reason is
that the infeasible tows were rare (~1 % of segments) and occurred at
essentially the same rate in both algorithms, so they cancel in the comparison.

So: the bug was real, the fix is right, and **no earlier conclusion changes**.
The inverted U, the peak at ratio 0.8, and ILA's small advantage all survive.
Nothing needs retracting to the supervisors; the honest framing is "found a
correctness bug, fixed it, measured the impact, conclusions unchanged".

### What is still open

- The *design* hole (c) stands: an AV that waits does not have its own timetable
  updated, and waiting happens in 48 % of tows. That is a modelling decision for
  the paper, not a bug, and it is reviewer item 10.
- `milp/exact_time.py` now has the correct convention (`waiting="either"`), but
  the time-constrained optimum numbers have to be recomputed with it; the ones
  recorded earlier today used `waiting="none"` and compare against a different
  problem.
- `core/refine.py` is still spatial only.
