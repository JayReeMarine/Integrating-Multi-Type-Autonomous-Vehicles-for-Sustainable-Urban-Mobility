# Meeting with Eunus — Fri 18 Sep 2026, 11:00

## Status in one line
Exact-optimum analysis done (reported to Aamir 7 Sep). SUMO pipeline now runs
end-to-end on the real M1 corridor; three design decisions are needed before
the numbers it produces mean anything.

## What exists
| | |
|---|---|
| Exact MILP solver | greedy 93.3%, ILA 93.1% of optimum (21 instances, AV ≤15 / PV ≤30). Shortfall is *choice quality*, not model expressiveness (restricting to full-overlap segments costs only 0.65%). |
| Time constraints | ILA's advantage over greedy is +0.1–1.2 pp with time OFF, +1.3–2.5 pp with time ON. The advantage is temporal. |
| Ratio sweep (old data) | ILA–greedy gap is inverted-U in AV:PV ratio: ≈0 at ≤0.1, peak 1.26 pp at 0.8, ≈0 at ≥3.2. The paper measured at 0.20–0.25. |
| SUMO | M1 Monash Fwy inbound, Springvale Rd → CityLink, 20.5 km, 31 edges, 16 ramps. Converter SUMO → our vehicle format. First 1-h scenario runs in 4 s. Demand = `randomTrips` **placeholder, not calibrated**. |

## What the real corridor changes — independent of the demand model
| | synthetic | M1 |
|---|---|---|
| possible entry / exit points | 101 / 101 | **9 / 9** |
| OD pairs that occur | thousands | **44** |
| trips shorter than L_min (2 km) | 0 by design | **18 %** |
| constant-speed error vs τ = 5 s | 0 by assumption | 0.6 s for < 4 km trips, **22 s for > 15 km trips** |

Figures: `analysis/figures/m1_corridor.png` (the corridor), `analysis/figures/entry_dist_synth_vs_m1.png` (distributions), `analysis/figures/ratio_m1_vs_synth.png` (ratio sweep, M1 vs synthetic).
First ratio sweep on M1 (placeholder demand, random AV/PV labels, 5 seeds —
a first look, not a result):

| AV:PV | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|
| ILA − greedy, time OFF | 0.00 | 0.00 | 0.00 | 0.00 |
| ILA − greedy, time ON (mean ± sd) | +0.77 ± 0.33 | **+1.40 ± 0.40** | +1.08 ± 0.20 | +0.41 ± 0.17 |
| PV distance covered | 53 % | 85 % | 96 % | 98 % |

Same shape as the synthetic sweep (advantage only with time constraints,
inverted U), but the peak is at 0.4 rather than 0.8: on the M1, PVs are almost
all served by ratio 0.6, so there is nothing left to compete for.

## Decisions I need
0. **The claim.** SUMO did not widen the ILA–greedy gap: +0.4 to +1.4 pp (5-seed means) on M1
   versus +0.2 to +1.3 pp on synthetic data, and exactly 0 without time
   constraints. Meanwhile both heuristics reach only ~93 % of the exact optimum.
   The 7 % headroom is seven times the ILA–greedy difference. My reading: the
   Revision Plan's Phase-2 decision point ("if the gap does not widen, the
   contribution needs rethinking") has arrived early. Options:
   (A) design a method that closes part of the 7 % — the exact solutions show
   how: shorter segments, more hand-offs, 88 % of PVs served vs 75 %;
   (B) reposition the paper as problem + exact benchmark + realistic evaluation,
   with ILA as "greedy-quality, faster"; (C) 2D — but the exact-solver diagnosis
   says the loss is choice quality, not model expressiveness, so 2D alone
   would not change it. I would test (A) for two weeks with a local-search
   post-processor against the 21 exact instances; if it cannot reach ~96 %,
   go with (B). Do you agree with this reading, and with the test?
1. **AV:PV ratio.** Your note "specialized → quite high": did you mean the ratio
   is high (many AVs) or the gap should be high? My framing: ratio =
   (share of heavy vehicles that can tow) ÷ (share of cars opting in). What range
   should the sweep cover — 0.2–1.6? And which metric: the paper's saving over
   AV+PV distance (falls past 0.4 because AV distance can never be saved) or
   PV distance covered (monotone)?
2. **AV / PV labelling in SUMO.** SUMO has no such concept. Random fraction
   (current placeholder), by vehicle class (trucks/buses = AV), or something else?
3. **Demand.** randomTrips keeps only the network real (positions and speeds
   realistic; times and OD still uniform). Mushfiq gave no specific guidance.
   Survey of DTP open data (CC BY 4.0, all point counts, no OD):

   | | cost | buys |
   |---|---|---|
   | (i) randomTrips placeholder | done | pipe works; "times" criticism remains |
   | **(ii) partial calibration** — TIRTL 15-min detector counts via `routeSampler` | 2–3 days | eastern 40 % of corridor (16 mainline + High St ramp sites) calibrated to government counts; western 60 % (Burke Rd → CityLink) has no detectors, OD still a method product — must be labelled "partial" |
   | (iii) full calibration | weeks + data we lack (CityLink is Transurban's) | not possible with open data |

   My recommendation is (ii) for this paper, (iii) belongs to the 2D paper.
   Agree?
4. **τ's physical unit.** The paper says τ = 5 s but never fixes the grid scale.
   With 1 grid unit = 205 m, one model time unit ≈ 8 s, so "τ = 5" is either 5 s
   (saving halves on M1: 24 %) or ≈ 42 s (50–52 %). Which reading,
   and should long trips use piecewise speeds given the 22 s drift?
   (Linked to 3: under (ii) entry times come from real-second counts, so real
   seconds is the natural reading; under (i) model units is easier to defend.)
5. **Scale.** Keep 400 PV / 100 AV as the headline, or a full peak hour (~1750
   vehicles on the main line)?

## What I would like to leave with
Ratio range · labelling rule · demand approach for iteration 1 · τ convention.
Next check-in: ratio sweep on M1 with the agreed settings, plus MILP baseline on
5-minute sub-windows (AV ~10 / PV ~25).
