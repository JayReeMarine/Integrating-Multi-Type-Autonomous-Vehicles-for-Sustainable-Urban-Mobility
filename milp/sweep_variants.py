"""All four algorithm variants against the exact optimum.

  greedy            as shipped
  greedy + lam      congestion discounts the ranking value
  ILA               as shipped
  ILA + lam         congestion discounts the LSAP cost (measured: makes it worse)
  ILA + theta       congestion withholds contested candidates for a round, and
                    the LSAP still maximises raw towed distance
"""
from __future__ import annotations

import math
import statistics as st

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.congestion import (greedy_congestion_matching, ila_congestion_matching,
                             ila_filter_matching)
from milp.exact import solve

SIZES = [(5, 10, 60), (6, 12, 80), (8, 15, 100), (10, 20, 100), (12, 25, 100)]
SEEDS = [42, 43, 44]
LAMBDAS = [0.5, 1.0, 2.0]
THETAS = [4.0, 3.0, 2.5, 2.0, 1.5]


def towed(a):
    return float(sum(x.dp - x.cp for x in a))


inst = []
for na, npv, L in SIZES:
    for seed in SEEDS:
        avs, pvs, lmin = generate_mock_data(
            num_av=na, num_pv=npv, highway_length=L, av_capacity_range=(1, 3),
            min_trip_length=10, seed=seed, enable_time_constraints=False)
        r = solve(avs, pvs, lmin, time_limit=180)
        if r.proven_optimal:
            inst.append((avs, pvs, lmin, r.optimal))
print(f"{len(inst)} instances with a proven optimum\n", flush=True)

rows = []


def record(label, fn):
    pct = [100 * towed(fn(a, p, l)[0]) / o for a, p, l, o in inst]
    rows.append((label, st.mean(pct), min(pct), max(pct)))
    print(f"  {label:<24}{st.mean(pct):6.2f}%", flush=True)


record("greedy (shipped)",
       lambda a, p, l: greedy_multi_av_matching(a, p, l, enable_time_constraints=False))
for lam in LAMBDAS:
    record(f"greedy + lam={lam}",
           lambda a, p, l, lam=lam: greedy_congestion_matching(a, p, l, lam=lam))
record("ILA (shipped)",
       lambda a, p, l: hungarian_multi_av_matching(a, p, l, enable_time_constraints=False))
for lam in LAMBDAS:
    record(f"ILA + lam={lam}",
           lambda a, p, l, lam=lam: ila_congestion_matching(a, p, l, lam=lam))
for th in THETAS:
    record(f"ILA + filter theta={th}",
           lambda a, p, l, th=th: ila_filter_matching(a, p, l, theta=th))

print(f"\n{'variant':<26}{'mean':>8}{'min':>8}{'max':>8}")
print("-" * 50)
for label, m, lo, hi in rows:
    print(f"{label:<26}{m:>7.2f}%{lo:>7.2f}%{hi:>7.2f}%")

best_g = max((r for r in rows if r[0].startswith("greedy")), key=lambda r: r[1])
best_i = max((r for r in rows if r[0].startswith("ILA")), key=lambda r: r[1])
print(f"\nbest greedy variant : {best_g[0]}  {best_g[1]:.2f}%")
print(f"best ILA variant    : {best_i[0]}  {best_i[1]:.2f}%")
print(f"ILA - greedy        : {best_i[1] - best_g[1]:+.2f} pp")
