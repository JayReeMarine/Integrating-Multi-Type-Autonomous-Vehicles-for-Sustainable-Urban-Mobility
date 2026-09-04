"""Does pricing congestion into the value close the gap to the optimum?

Sweeps the weight `lam` for both congestion-aware variants and measures each
against the exact optimum on the instances the MILP can solve. `lam=0` is the
shipped algorithm, so the first row of each block is the current baseline.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.congestion import greedy_congestion_matching, ila_congestion_matching
from milp.exact import solve

SIZES = [(5, 10, 60), (6, 12, 80), (8, 15, 100), (10, 20, 100), (12, 25, 100)]
SEEDS = [42, 43, 44]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def towed(a):
    return float(sum(x.dp - x.cp for x in a))


opt = {}
inst = {}
for na, npv, L in SIZES:
    for seed in SEEDS:
        avs, pvs, lmin = generate_mock_data(
            num_av=na, num_pv=npv, highway_length=L, av_capacity_range=(1, 3),
            min_trip_length=10, seed=seed, enable_time_constraints=False)
        r = solve(avs, pvs, lmin, time_limit=180)
        if not r.proven_optimal:
            continue
        inst[(na, npv, seed)] = (avs, pvs, lmin)
        opt[(na, npv, seed)] = r.optimal
print(f"{len(inst)} instances with a proven optimum", flush=True)

res = defaultdict(list)
for key, (avs, pvs, lmin) in inst.items():
    o = opt[key]
    for lam in LAMBDAS:
        res[("greedy", lam)].append(
            100 * towed(greedy_congestion_matching(avs, pvs, lmin, lam=lam)[0]) / o)
        res[("ILA", lam)].append(
            100 * towed(ila_congestion_matching(avs, pvs, lmin, lam=lam)[0]) / o)

print(f"\n{'lam':>6}{'greedy':>10}{'ILA':>10}{'ILA - greedy':>15}")
print("-" * 41)
for lam in LAMBDAS:
    g = st.mean(res[("greedy", lam)])
    h = st.mean(res[("ILA", lam)])
    tag = "   <- shipped" if lam == 0 else ""
    print(f"{lam:>6}{g:>9.2f}%{h:>9.2f}%{h - g:>14.2f}pp{tag}")

bg = max(LAMBDAS, key=lambda l: st.mean(res[("greedy", l)]))
bh = max(LAMBDAS, key=lambda l: st.mean(res[("ILA", l)]))
print(f"\nbest greedy: lam={bg}  {st.mean(res[('greedy', bg)]):.2f}% of optimum")
print(f"best ILA   : lam={bh}  {st.mean(res[('ILA', bh)]):.2f}% of optimum")
