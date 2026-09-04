"""Split the optimality gap into "bad choices" and "shapes it cannot express".

Two exact solves per instance:

  unrestricted   any contiguous sub-interval of the overlap may be towed
  suffix-only    every segment must run to the end of the overlap, which is the
                 only shape greedy and ILA ever commit

greedy measured against the suffix-only optimum shows how much it loses by
choosing badly among the moves available to it. The distance between the two
optima shows how much is unreachable without widening what the algorithms can
express - the question behind reviewer item #9.
"""
from __future__ import annotations

import csv
import os
import statistics as st

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve

SIZES = [(4, 8, 50), (5, 10, 60), (6, 12, 80), (8, 15, 100), (10, 20, 100), (12, 25, 100)]
SEEDS = [42, 43, 44]
OUT = "data/results/milp/restricted.csv"


def towed(a):
    return float(sum(x.dp - x.cp for x in a))


rows = []
print(f"{'AV':>3}{'PV':>4}{'seed':>6}{'OPT_any':>9}{'OPT_suffix':>12}"
      f"{'greedy':>8}{'ILA':>7}  {'g/suffix':>9}{'suffix/any':>11}", flush=True)

for na, npv, L in SIZES:
    for seed in SEEDS:
        avs, pvs, lmin = generate_mock_data(
            num_av=na, num_pv=npv, highway_length=L, av_capacity_range=(1, 3),
            min_trip_length=10, seed=seed, enable_time_constraints=False)
        g = towed(greedy_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)[0])
        h = towed(hungarian_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)[0])
        a = solve(avs, pvs, lmin, time_limit=180)
        sfx = solve(avs, pvs, lmin, time_limit=180, suffix_only=True)
        if not (a.proven_optimal and sfx.proven_optimal):
            print(f"{na:>3}{npv:>4}{seed:>6}  not proven, skipped", flush=True)
            continue
        rows.append(dict(num_av=na, num_pv=npv, seed=seed, opt_any=a.optimal,
                         opt_suffix=sfx.optimal, greedy=g, ila=h))
        print(f"{na:>3}{npv:>4}{seed:>6}{a.optimal:>9.0f}{sfx.optimal:>12.0f}"
              f"{g:>8.0f}{h:>7.0f}  {100 * g / sfx.optimal:>8.2f}%"
              f"{100 * sfx.optimal / a.optimal:>10.2f}%", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

g_any = st.mean(100 * r["greedy"] / r["opt_any"] for r in rows)
g_sfx = st.mean(100 * r["greedy"] / r["opt_suffix"] for r in rows)
h_sfx = st.mean(100 * r["ila"] / r["opt_suffix"] for r in rows)
sfx_any = st.mean(100 * r["opt_suffix"] / r["opt_any"] for r in rows)
print(f"\n{len(rows)} instances")
print(f"  greedy / unrestricted optimum : {g_any:6.2f}%")
print(f"  suffix-only optimum / unrestricted optimum : {sfx_any:6.2f}%")
print(f"  greedy / suffix-only optimum  : {g_sfx:6.2f}%")
print(f"  ILA    / suffix-only optimum  : {h_sfx:6.2f}%")
print(f"\nwrote {OUT}")
