# Interim Findings: Exact Optimum and Algorithm Diagnosis

**Purpose of this document**

Aamir asked that items #1, #2 and, if possible, #3 be done as early as possible,
so that we would know sooner whether ILA has the potential to justify the work.
This reports what came out of that, and the one decision I would like your view
on before committing the next few weeks.

On the question that has been raised, whether ILA has the potential to justify
this work, the measurements point to a qualified yes. There is real space for an
algorithmic contribution, and ILA does earn its place, though only in the regime
where temporal synchronisation binds, and by a margin that is larger than the
paper currently reports yet still modest. Whether that margin is large enough is
the judgement I would like your view on, and Section 7 sets out the two ways
forward I can see.

---

## Summary

**What I did**

1. Pinned the environment and verified the published numbers still reproduce.
2. Re-analysed the existing sweep data along the axis that controls contention.
3. Built an exact MILP solver, bringing item #4 forward, because item #2 cannot
   be interpreted without it.
4. Diagnosed which decisions the optimum makes differently.
5. Implemented and measured congestion-aware variants of both algorithms.

**What came out**

6. Both heuristics reach about 93% of the exact optimum, and neither beats the
   other. The 7% shortfall is not a limit of the problem definition.
7. The shortfall is choice quality, not expressiveness, only about 2% of it
   needs segment shapes the algorithms cannot produce.
8. The optimum uses 74% more towing segments, each 37% shorter, and serves more
   PVs from the same capacity. Greedy's "longest first" rule buys distance at
   the cost of monopolising an AV over a long stretch of road.
9. ILA's advantage is temporal, not spatial. With the synchronisation constraint
   switched off it nearly disappears; with it on, ILA is the best of five
   variants at the paper's own scale.
10. The paper measures at |AV|/|PV| ratios of 0.20 and 0.25, where the advantage
    is smallest. At 0.8 it roughly doubles.

---

## 1. The sweep range proposed in item #2 was the wrong range

Item #2 proposed sweeping the active-to-passive ratio at 400 PVs against 100,
50, 25 and 10 AVs, and tightening capacities, on the reasoning that scarcer AVs
mean more contention.

Re-aggregating the PV/AV count sweep by that ratio shows the opposite. The
ILA-greedy gap follows an inverted U: it peaks at a ratio near 0.8 (1.26 pp) and
falls to zero as active vehicles become scarce. The proposed range corresponds
to ratios of 0.25 down to 0.025, the entire flat left tail. The same applies to
tightening capacity: in the capacity sweep of Figure 2(a) of the submission, the
gap shrinks as capacity tightens, from 0.84 pp at C_max = 16 down to 0.35 pp at
C_max = 2.

![**Figure 1.** ILA's advantage over greedy against the active-to-passive ratio, from the existing sweep results. The shaded band is the range proposed in item #2 of the Revision Plan.](analysis/figures/gap_vs_ratio.png){width=6.2in}

The mechanism is that contention is not AV scarcity. When active vehicles are
few, each passive vehicle has almost no feasible partners, the bipartite graph
is sparse and the assignment is close to forced, so there is nothing for
coordination to exploit. The gap appears where many comparable options compete
for the same point-wise capacity.

This matters for item #1 as well, since it determines which SUMO scenarios are
worth generating. Running the sweep as originally written would have produced a
null result regardless of how realistic the data was, and we would not have been
able to tell that null result apart from a genuine finding about realistic
demand.

---

## 2. Exact solver (item #4, brought forward)

Before any of this I consolidated the two diverged Python environments, pinned
the versions and re-ran a subset of the stored experiments against the committed
results. All 48 checked combinations reproduce exactly, for both algorithms, so
the figures in the submitted paper remain a valid baseline for anything measured
from here on.

Item #2 asks whether ILA's small margin over greedy is a weakness of ILA. That
question cannot be answered from the margin alone: a small margin is equally
consistent with greedy already being near-optimal, and the two imply opposite
responses. The Revision Plan already noted this dependency; it simply placed
item #4 later.

I formulated the problem as a mixed-integer linear program and solved it with an
off-the-shelf solver (HiGHS, reached through SciPy, so no new dependency).
Positions along the corridor are integers, so a towing segment is a run of unit
intervals; auxiliary indicators enforce the minimum towing length L_min, and the
point-wise capacity and PV non-overlap constraints are taken directly from
Section 2.3 of the submission.

**Scope.** The temporal synchronisation constraint is state-dependent, since
towing a PV changes its arrival times downstream, and is therefore not
expressible as a linear constraint. The exact solver covers the spatial problem
only. This is a deliberate scope choice rather than an omission: it is enough to
establish whether the formulation leaves room for coordination at all.

**Tractability.** 21 instances (4-15 AVs against 8-30 PVs, three seeds each)
solved to proven optimality. The largest, 15 AVs and 30 PVs, took up to 142
seconds with about 10,500 binary variables. At 20 AVs and 40 PVs optimality was
no longer proven within a minute; beyond that point the LP relaxation still
gives an upper bound, so the comparison degrades rather than stopping.

**Result.**

| | mean, as % of the exact optimum |
| --- | --- |
| greedy | 93.33% |
| ILA | 93.09% |

ILA wins on 9 instances, loses on 8 and ties on 4. About 7% of the achievable
towed distance is left on the table by both.

The first thing this settles: greedy is not near-optimal. Had it been at 99%, no
algorithm could have improved on it and the work would have had to change
direction. There is room.

---

## 3. The shortfall is choice, not expressiveness

Both algorithms always commit the entire remaining overlap of a pair, never a
part of it. The start of a segment can be pushed back by an earlier tow on the
same PV, but the end is always the end of the overlap. Solving each instance
again under exactly that restriction separates the two possible causes of the
shortfall.

| | mean, as % of the unrestricted optimum |
| --- | --- |
| unrestricted optimum | 100% |
| optimum with segments allowed to start late only | 99.41% |
| optimum with whole-overlap segments only | 98.03% |
| greedy | 93.79% |
| ILA | 93.43% |

Restricting the solver to the segment shapes the algorithms can produce costs
about 2%. The remaining 4-6% is lost choosing badly among moves that were
available all along.

This resolves item #9 favourably. Narrowing the formal definition to the
segments the algorithms actually handle costs under 1% of achievable distance,
so it is a fair narrowing rather than a retreat, and we can now justify it with
a measurement instead of asserting it.

---

## 4. What the optimum does differently

Comparing the solutions themselves, averaged over 15 instances:

| | segments | mean length | PVs served | AV capacity used |
| --- | --- | --- | --- | --- |
| optimum | 23.7 | 15.2 | 87.6% | 75.1% |
| greedy | 13.6 | 24.1 | 75.5% | 70.7% |
| ILA | 15.3 | 21.2 | 88.3% | 70.3% |

The optimum uses about 74% more segments, each about 37% shorter. Greedy ranks
candidates by saved distance and commits the longest first. A long tow occupies
an AV's capacity across a long stretch of the corridor and shuts every other PV
out of it for that distance. The optimum instead splits service into shorter
tows, packs more vehicles into the same capacity (75.1% against 70.7%) and
serves 87.6% of PVs against greedy's 75.5%.

Worth noting for the paper: multi-segment service, a PV handed from one AV to
another, is the feature we put forward as novel in Section 1 of the submission,
and it is precisely what the optimum exploits and what both of our algorithms
underuse.

![**Figure 2.** Left: how far each heuristic sits from the exact optimum, per instance. Right: greedy's shortfall split by value into choosing badly among available moves and segment shapes it cannot express.](analysis/figures/milp_gap.png){width=6.5in}

---

## 5. Attempts to close the gap, and where ILA actually earns its place

I implemented congestion-aware variants of both algorithms. For an AV at a given
position along the corridor, let *contest* be the number of still-uncovered PVs
that it could tow across that position, and *free* its remaining capacity there;
*scarcity* is their ratio. A candidate segment is then discounted by the mean
scarcity of the road it would occupy. Each variant reproduces its original
exactly at the neutral parameter, which is asserted as a test.

**Against the exact optimum, temporal constraint off:**

| | mean, as % of the optimum |
| --- | --- |
| greedy, as published | 93.79% |
| greedy + congestion | 95.99% |
| ILA, as published | 93.43% |
| ILA + congestion in the assignment cost | 92.43% |
| ILA + congestion as a candidate filter | 94.80% |

Two findings. Congestion is a genuine signal: one cheap statistic recovers about
a third of greedy's shortfall. And where it is injected decides whether it helps.
Discounting the assignment cost makes ILA worse, because each round then
maximises the adjusted score and trades real towed distance away to avoid
contested road. Using the same signal only to withhold contested candidates from
a round, with the objective left as raw towed distance, improves ILA by 1.4
points.

**With the temporal constraint on, at the paper's own scale** (50-320 AVs, four
seeds, as a percentage of published greedy):

| configuration | greedy + congestion | ILA |
| --- | --- | --- |
| capacity sweep, 50 AVs : 200 PVs | +2.02% | +1.82% |
| length sweep, 80 AVs : 400 PVs | +1.62% | +1.34% |
| ratio 0.8, 160 AVs : 200 PVs | +1.41% | **+1.91%** |
| ratio 0.8, 320 AVs : 400 PVs | **-1.59%** | **+2.47%** |
| **mean** | +0.87% | **+1.88%** |

Here the published ILA is the best of the five variants. The congestion-aware
greedy turns negative on the largest configuration, exactly where ILA is
strongest.

This is consistent with a separate contrast: ILA's advantage over greedy is
larger with the temporal constraint active in all four configurations
(+0.11-1.19% with it off, +1.34-2.47% with it on).

![**Figure 3.** ILA's advantage over greedy with the temporal synchronisation constraint switched off and on, at the paper's own configurations and at the most contested ratio.](analysis/figures/time_effect.png){width=5.8in}

The conclusion I draw: ILA earns its place where temporal feasibility binds. On
the purely spatial problem a myopic rule with a capacity-aware ranking does
better and ILA's per-round optimality buys nothing. Once arrival-time
synchronisation begins eliminating candidate pairs, which pairs are chosen
starts to matter and the global assignment pays for itself.

This is a sharper and better-supported explanation than the one currently in
Section 4 of the submission, which attributes the advantage to spatial
coordination being limited on a 1D corridor and predicts it will widen on 2D
road networks. It also means the premise behind the second (2D) paper should be
re-examined before that project starts, since 2D adds spatial richness, and
spatial richness is not where the advantage comes from.

---

## 6. What I propose to change in the paper

1. **Report the ratio sweep in full, and measure where contention is real.** The
   submission reports two configurations, at ratios 0.20 and 0.25, where ILA
   gains 1.3-1.8%. At 0.8 it gains 1.9-2.5%. Showing the whole curve and the
   trend answers item #2 with a mechanism rather than a single number, and is
   not selective reporting provided the full sweep is shown.
2. **Explain the advantage as temporal rather than spatial**, supported by the
   on/off contrast above. This replaces a prediction with a measurement.
3. **Add the congestion-aware greedy as the second baseline item #4 calls for.**
   It beats published greedy in three of four configurations, so showing ILA
   ahead of it is a stronger claim than beating plain greedy alone. Its
   instability at the largest configuration should be reported, not hidden.
4. **Report the exact comparison itself.** "Our heuristics reach 93% of the
   exact optimum on instances up to 15 AVs and 30 PVs, and a capacity-aware
   variant reaches 96%" answers item #4 directly, and item #9 can now proceed as
   planned with a measurement behind it.

---

## 7. The decision I would like your view on

The question is no longer whether to continue, but where the next few weeks
should go.

**Option A, strengthen ILA's margin.** Re-run the ratio sweep with many more
seeds and the temporal constraint active, across ratios 0.2 to 1.6, and
establish the trend properly. Keep ILA as the paper's algorithmic contribution,
now measured where its advantage is real and explained by a mechanism we can
demonstrate.

**Option B, rebuild the contribution around the exact solver.** Treat the
algorithmic margin as too thin to carry the paper, and lead instead with the
formalisation, the exact optimum, and the characterisation of when coordination
pays and when it does not.

My own reading is that A is worth attempting first, because the margin roughly
doubles once measured at the contested ratio and we now have an explanation for
it rather than a hope. But whether about 2% clears the bar that 0.2-0.9 pp
failed is a judgement I would rather not make alone.

A second question: given Section 5, should the 2D paper still be planned on the
assumption that richer spatial structure widens ILA's margin?

---

## 8. Limitations

Stated plainly, since they bound everything above.

- **ILA's advantage remains small**, 1.9% on average and 2.5% at best. The
  reviewers rejected 0.2-0.9 pp as too small; whether 2% is enough is precisely
  the judgement in Section 7.
- **Three to four seeds.** Item #5 has not been done yet, and the temporal
  on/off intervals overlap within each configuration, so that pattern rests on
  holding in all four configurations rather than on any single one.
- **No exact reference in the temporal setting.** The MILP cannot express
  state-dependent arrival times, so where the temporal constraint is active only
  relative comparisons are available.
- **Exact solving stops at 15 AVs and 30 PVs.**
- **All of this is on the uniform synthetic generator.** SUMO data (item #1) has
  not entered yet; that work is progressing in parallel and the corridor
  extraction is done.
- **The congestion variants are a first cut** with a coarse parameter grid.

---

All code, result files and working notes are in the project repository.
