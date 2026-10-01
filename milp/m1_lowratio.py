"""ILA against the exact optimum on real M1 corridor sub-instances.

The synthetic version of this measurement is milp/lowratio_sweep.py.  Here the
instances come from the M1 inbound corridor via sumo/convert.py's D6 window
extraction (see sumo/NOTES-sumo.md, 2026-10-01), so the ramp geometry is real
even though the demand is still the randomTrips placeholder.

One time unit for all windows (auto_time_unit on the full trip list).
Spatial problem only: milp.exact does not model temporal coupling.
Where optimality is not proven within the limit, the LP bound is recorded and
the row is marked proven=False; those rows give a lower bound on ILA/optimum.

Run:  PYTHONPATH=. venv/bin/python -u milp/m1_lowratio.py --window 300
Writes data/results/milp/m1_lowratio_<W>.csv  (resumable)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from core.hungarian_multi import hungarian_multi_av_matching
from core.greedy_multi import greedy_multi_av_matching
from milp.exact import solve
from sumo.convert import load_corridor_trips, auto_time_unit, extract_instance

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="sumo/m1")
ap.add_argument("--window", type=float, default=300.0)
ap.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
ap.add_argument("--time-limit", type=float, default=900.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-windows", type=int, default=12)
a = ap.parse_args()

corr, trips = load_corridor_trips(a.scenario)
tu = auto_time_unit(corr, trips)
OUT = Path("data/results/milp") / f"m1_lowratio_{int(a.window)}.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["window_s", "t0", "ratio", "av", "pv", "optimum", "proven", "ila", "greedy",
          "ila_pct", "greedy_pct", "pv_total_distance", "ila_coverage", "opt_coverage",
          "n_binary", "seconds"]
rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
done = {(float(r["t0"]), float(r["ratio"])) for r in rows}

towed = lambda asg: float(sum(x.dp - x.cp for x in asg))
starts = [i * a.window for i in range(a.max_windows)]

print(f"M1 sub-instances: window {a.window:.0f}s, {len(starts)} windows, "
      f"time unit {tu:.3f}s, time constraints OFF")
print(f"{'t0':>6}{'ratio':>7}{'AV':>4}{'PV':>5} {'optimum':>9}{'ILA':>8} {'ILA/opt':>9}"
      f"{'cov(opt)':>10}{'cov(ILA)':>10} {'sec':>7}  status", flush=True)

for r in a.ratios:
    for t0 in starts:
        if (t0, r) in done:
            continue
        try:
            avs, pvs, l_min, info = extract_instance(
                corr, trips, t0=t0, window_s=a.window, ratio=r,
                seed=a.seed, time_unit_s=tu)
        except Exception as e:
            print(f"{t0:6.0f}{r:7.2f}  skipped: {e}", flush=True)
            continue
        h, _, _ = hungarian_multi_av_matching(avs, pvs, l_min, enable_time_constraints=False)
        g, _, _ = greedy_multi_av_matching(avs, pvs, l_min, enable_time_constraints=False)
        ila, grd = towed(h), towed(g)
        t_start = time.perf_counter()
        res = solve(avs, pvs, l_min, time_limit=a.time_limit)
        dt = time.perf_counter() - t_start
        ref = res.reference
        pv_dist = float(sum(p.exit_point - p.entry_point for p in pvs))
        row = dict(window_s=a.window, t0=t0, ratio=r, av=len(avs), pv=len(pvs),
                   optimum=round(ref, 3), proven=res.proven_optimal,
                   ila=round(ila, 3), greedy=round(grd, 3),
                   ila_pct=round(100 * ila / ref, 3) if ref else "",
                   greedy_pct=round(100 * grd / ref, 3) if ref else "",
                   pv_total_distance=pv_dist,
                   ila_coverage=round(100 * ila / pv_dist, 3),
                   opt_coverage=round(100 * ref / pv_dist, 3),
                   n_binary=res.n_binary, seconds=round(dt, 2))
        rows.append(row)
        print(f"{t0:6.0f}{r:7.2f}{len(avs):4d}{len(pvs):5d} {ref:9.0f}{ila:8.0f} "
              f"{row['ila_pct']:8.2f}%{row['opt_coverage']:9.1f}%{row['ila_coverage']:9.1f}% "
              f"{dt:7.1f}  {'optimal' if res.proven_optimal else 'LP bound'}", flush=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
