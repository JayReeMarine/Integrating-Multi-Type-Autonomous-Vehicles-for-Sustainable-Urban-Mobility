"""greedy and ILA against the exact optimum WITH time constraints.

This is the baseline the project did not have: milp.exact covers the spatial
problem only, and ILA's advantage over greedy lives on the temporal axis.
milp.exact_time supplies the missing optimum.

Validation invariant: no heuristic may exceed the optimum. Rows that violate it
are flagged in the CSV (`invariant_ok`) rather than silently averaged.

Run:  PYTHONPATH=. venv/bin/python -u milp/time_sweep.py
Writes data/results/milp/time_optimum.csv  (resumable)
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact_time import solve_time
from milp.exact import solve as solve_spatial

ap = argparse.ArgumentParser()
ap.add_argument("--pvs", type=int, nargs="+", default=[20, 30, 50])
ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
ap.add_argument("--tau", type=float, default=5.0)
ap.add_argument("--time-window", type=float, default=100.0)
ap.add_argument("--time-limit", type=float, default=300.0)
a = ap.parse_args()

OUT = Path("data/results/milp/time_optimum.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["pv", "ratio", "av", "seed", "tau", "time_optimum", "proven", "greedy", "ila",
          "greedy_pct", "ila_pct", "spatial_optimum", "invariant_ok", "seconds"]
rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
done = {(int(r["pv"]), float(r["ratio"]), int(r["seed"])) for r in rows}
tw = lambda asg: float(sum(s.dp - s.cp for s in asg))

print(f"tau={a.tau}, time window={a.time_window}, speeds 0.8-1.2")
print(f"{'PV':>4}{'ratio':>7}{'AV':>4}{'seed':>5} | {'time-opt':>9}{'proven':>7} | "
      f"{'greedy':>7}{'ILA':>6} | {'grd/opt':>8}{'ILA/opt':>8} | {'spatial':>8} {'sec':>7}", flush=True)

for npv in a.pvs:
    for r in a.ratios:
        nav = max(1, round(npv * r))
        for seed in a.seeds:
            if (npv, r, seed) in done:
                continue
            avs, pvs, l_min = generate_mock_data(
                num_av=nav, num_pv=npv, highway_length=100, av_capacity_range=(1, 3),
                min_trip_length=10, seed=seed, enable_time_constraints=True,
                av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2),
                time_window=a.time_window)
            g, _, _ = greedy_multi_av_matching(avs, pvs, l_min,
                                               enable_time_constraints=True, time_tolerance=a.tau)
            h, _, _ = hungarian_multi_av_matching(avs, pvs, l_min,
                                                  enable_time_constraints=True, time_tolerance=a.tau)
            res = solve_time(avs, pvs, l_min, tau=a.tau, time_limit=a.time_limit)
            sp = solve_spatial(avs, pvs, l_min, time_limit=a.time_limit)
            ref = res.reference
            ok = bool(ref) and tw(g) <= ref + 1e-6 and tw(h) <= ref + 1e-6
            row = dict(pv=npv, ratio=r, av=nav, seed=seed, tau=a.tau,
                       time_optimum=round(ref, 3) if ref else "", proven=res.proven_optimal,
                       greedy=tw(g), ila=tw(h),
                       greedy_pct=round(100 * tw(g) / ref, 3) if ref else "",
                       ila_pct=round(100 * tw(h) / ref, 3) if ref else "",
                       spatial_optimum=round(sp.reference, 3) if sp.reference else "",
                       invariant_ok=ok, seconds=round(res.seconds, 2))
            rows.append(row)
            print(f"{npv:4d}{r:7.2f}{nav:4d}{seed:5d} | {ref:9.1f}{str(res.proven_optimal):>7} | "
                  f"{tw(g):7.0f}{tw(h):6.0f} | {row['greedy_pct'] or 0:7.2f}%{row['ila_pct'] or 0:7.2f}% | "
                  f"{row['spatial_optimum'] or 0:8.0f} {res.seconds:7.1f}"
                  f"{'' if ok else '  <-- INVARIANT VIOLATED'}", flush=True)
            with OUT.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
