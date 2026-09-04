"""How large an instance can the exact solver still handle?

Reviewer item #4 needs a statement of the form "we solve instances up to N AVs
and M PVs exactly". This grows the instance until the MILP stops proving
optimality within the time limit; from that point the LP relaxation still gives
an upper bound, so the comparison degrades gracefully rather than stopping.
"""
from __future__ import annotations

import sys

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve, extract_segments

SIZES = [(12, 25, 100), (15, 30, 100), (20, 40, 100), (25, 50, 150), (30, 60, 150)]
TIME_LIMIT = 60.0


def towed(a):
    return float(sum(x.dp - x.cp for x in a))


print(f"{'AV':>4}{'PV':>5}{'L':>5}{'ref':>7}{'greedy%':>9}{'ILA%':>8}"
      f"{'partial%':>10}{'vars':>8}{'rows':>8}{'sec':>8}  status", flush=True)

for na, npv, L in SIZES:
    avs, pvs, lmin = generate_mock_data(
        num_av=na, num_pv=npv, highway_length=L, av_capacity_range=(1, 3),
        min_trip_length=10, seed=42, enable_time_constraints=False)
    g, _, _ = greedy_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)
    h, _, _ = hungarian_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)

    r = solve(avs, pvs, lmin, time_limit=TIME_LIMIT)
    ref = r.reference
    if not ref:
        print(f"{na:>4}{npv:>5}{L:>5}  no bound", flush=True)
        continue

    pct = "-"
    if r.proven_optimal:
        segs, npart = extract_segments(avs, pvs, lmin, time_limit=TIME_LIMIT)
        if segs:
            pct = f"{100 * npart / len(segs):.0f}%"

    print(f"{na:>4}{npv:>5}{L:>5}{ref:>7.0f}{100 * towed(g) / ref:>8.2f}%"
          f"{100 * towed(h) / ref:>7.2f}%{pct:>10}{r.n_binary:>8}{r.n_constraints:>8}"
          f"{r.seconds:>8.1f}  {'optimal' if r.proven_optimal else 'LP bound only'}",
          flush=True)
