"""How large can the exact solver go when AVs are scarce?

Eunus (18 Sep): measure how close ILA gets to the optimum at AV:PV = 1, 2, 5,
10, 20 %. The MILP has one binary per (AV, PV, unit interval), so a low ratio is
cheap in AVs but expensive in PVs. This probe finds, for each ratio, the largest
PV count the solver still proves optimal within the time limit.

Run: PYTHONPATH=. venv/bin/python -u milp/probe_lowratio.py
"""
from __future__ import annotations

import argparse
import time

from core.data import generate_mock_data
from milp.exact import solve

ap = argparse.ArgumentParser()
ap.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
ap.add_argument("--pvs", type=int, nargs="+", default=[50, 100, 150, 200])
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--time-limit", type=float, default=120.0)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

print(f"{'ratio':>6}{'AV':>4}{'PV':>5} {'vars':>9}{'rows':>9}{'sec':>8}  status")
print("-" * 52)
for r in a.ratios:
    for npv in a.pvs:
        nav = max(1, round(npv * r))
        avs, pvs, l_min = generate_mock_data(
            num_av=nav, num_pv=npv, highway_length=a.length,
            av_capacity_range=(1, 3), min_trip_length=10, seed=a.seed,
            enable_time_constraints=False)
        t0 = time.perf_counter()
        res = solve(avs, pvs, l_min, time_limit=a.time_limit)
        dt = time.perf_counter() - t0
        status = "proven optimal" if res.proven_optimal else f"NOT proven ({res.status})"
        print(f"{r:6.2f}{nav:4d}{npv:5d} {res.n_binary:9d}{res.n_constraints:9d}{dt:8.1f}  {status}", flush=True)
        if not res.proven_optimal:
            break   # larger PV counts at this ratio will not be easier
