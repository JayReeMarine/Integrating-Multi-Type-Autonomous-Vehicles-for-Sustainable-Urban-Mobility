"""ILA against the exact optimum when active vehicles are scarce.

Eunus, 18 Sep 2026: drop greedy as the headline baseline and ask how close the
scalable heuristic gets to the optimum, specifically at low AV:PV ratios where
contention is high ("if you give one PV the chance, maybe you lose another that
would go the longer distance").

Scope: spatial problem only (enable_time_constraints=False) -- milp.exact does
not model the temporal coupling. Greedy is still recorded, as a footnote.

Run:  PYTHONPATH=. venv/bin/python -u milp/lowratio_sweep.py --pv 50 --seeds 10
Writes data/results/milp/lowratio_<PV>.csv  (resumable: existing rows are kept)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve

ap = argparse.ArgumentParser()
ap.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
ap.add_argument("--pv", type=int, default=50)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--first-seed", type=int, default=42)
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--capacity", type=int, nargs=2, default=[1, 3])
ap.add_argument("--time-limit", type=float, default=900.0)
a = ap.parse_args()

OUT = Path("data/results/milp") / f"lowratio_{a.pv}.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["pv", "ratio", "av", "seed", "optimum", "proven", "ila", "greedy",
          "ila_pct", "greedy_pct", "pv_total_distance", "ila_coverage",
          "opt_coverage", "n_binary", "seconds"]

rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
done = {(float(r["ratio"]), int(r["seed"])) for r in rows}

towed = lambda asg: float(sum(x.dp - x.cp for x in asg))
trip = lambda v: v.exit_point - v.entry_point

print(f"PV={a.pv} capacity={tuple(a.capacity)} length={a.length} "
      f"seeds={a.first_seed}..{a.first_seed + a.seeds - 1}  (time constraints OFF)")
print(f"{'ratio':>6}{'AV':>4}{'seed':>6} {'optimum':>9}{'ILA':>8}{'greedy':>8} "
      f"{'ILA/opt':>9}{'grd/opt':>9} {'sec':>7}  status", flush=True)

for r in a.ratios:
    nav = max(1, round(a.pv * r))
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        if (r, seed) in done:
            continue
        avs, pvs, l_min = generate_mock_data(
            num_av=nav, num_pv=a.pv, highway_length=a.length,
            av_capacity_range=tuple(a.capacity), min_trip_length=10, seed=seed,
            enable_time_constraints=False)
        h, _, _ = hungarian_multi_av_matching(avs, pvs, l_min, enable_time_constraints=False)
        g, _, _ = greedy_multi_av_matching(avs, pvs, l_min, enable_time_constraints=False)
        ila, grd = towed(h), towed(g)
        t0 = time.perf_counter()
        res = solve(avs, pvs, l_min, time_limit=a.time_limit)
        dt = time.perf_counter() - t0
        ref = res.reference
        pv_dist = float(sum(trip(p) for p in pvs))
        row = dict(pv=a.pv, ratio=r, av=nav, seed=seed,
                   optimum=round(ref, 3), proven=res.proven_optimal,
                   ila=round(ila, 3), greedy=round(grd, 3),
                   ila_pct=round(100 * ila / ref, 3) if ref else "",
                   greedy_pct=round(100 * grd / ref, 3) if ref else "",
                   pv_total_distance=pv_dist,
                   ila_coverage=round(100 * ila / pv_dist, 3),
                   opt_coverage=round(100 * ref / pv_dist, 3),
                   n_binary=res.n_binary, seconds=round(dt, 2))
        rows.append(row)
        print(f"{r:6.2f}{nav:4d}{seed:6d} {ref:9.0f}{ila:8.0f}{grd:8.0f} "
              f"{row['ila_pct']:8.2f}%{row['greedy_pct']:8.2f}% {dt:7.1f}  "
              f"{'optimal' if res.proven_optimal else 'LP bound (' + res.status + ')'}", flush=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
