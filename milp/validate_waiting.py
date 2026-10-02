"""Check the time MILP against the heuristics under each waiting convention.

The invariant is `heuristic <= optimum` *under the same semantics*. It only
holds for the convention the heuristics actually implement, which is "either
side may wait" (NOTES.md, 2026-10-02). Running all conventions side by side
shows how much the convention is worth, and catches the case where the model
and the implementation have drifted apart again.

Run:  PYTHONPATH=. venv/bin/python -u milp/validate_waiting.py
Writes data/results/milp/waiting_validation.csv  (resumable)
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact_time import solve_time

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=str, nargs="+", default=["5,10", "8,16", "10,20", "12,20", "10,25"])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
ap.add_argument("--tau", type=float, default=5.0)
ap.add_argument("--time-limit", type=float, default=600.0)
ap.add_argument("--modes", type=str, nargs="+", default=["either", "none"])
a = ap.parse_args()

OUT = Path("data/results/milp/waiting_validation.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["av", "pv", "seed", "tau", "greedy", "ila"] + \
         [f"{m}_{k}" for m in a.modes for k in ("opt", "proven")] + ["invariant_ok"]
rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
done = {(int(r["av"]), int(r["pv"]), int(r["seed"])) for r in rows}
tw = lambda asg: float(sum(s.dp - s.cp for s in asg))

hdr = f"{'AV':>4}{'PV':>5}{'seed':>5} | {'greedy':>7}{'ILA':>6} | " + \
      "".join(f"{m:>9}{'prov':>6} | " for m in a.modes) + "check"
print(hdr, flush=True)
for size in a.sizes:
    nav, npv = (int(v) for v in size.split(","))
    for seed in a.seeds:
        if (nav, npv, seed) in done:
            continue
        avs, pvs, l_min = generate_mock_data(
            num_av=nav, num_pv=npv, highway_length=100, av_capacity_range=(1, 3),
            min_trip_length=10, seed=seed, enable_time_constraints=True,
            av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2), time_window=100.0)
        g, _, _ = greedy_multi_av_matching(avs, pvs, l_min, enable_time_constraints=True, time_tolerance=a.tau)
        h, _, _ = hungarian_multi_av_matching(avs, pvs, l_min, enable_time_constraints=True, time_tolerance=a.tau)
        row = dict(av=nav, pv=npv, seed=seed, tau=a.tau, greedy=tw(g), ila=tw(h))
        line = f"{nav:4d}{npv:5d}{seed:5d} | {tw(g):7.0f}{tw(h):6.0f} | "
        check = "-"
        for m in a.modes:
            r = solve_time(avs, pvs, l_min, tau=a.tau, time_limit=a.time_limit, waiting=m)
            row[f"{m}_opt"] = round(r.reference, 3) if r.reference else ""
            row[f"{m}_proven"] = r.proven_optimal
            line += f"{r.reference:9.0f}{str(r.proven_optimal)[:5]:>6} | "
            if m == "either":
                check = ("OK" if max(tw(g), tw(h)) <= r.reference + 1e-6 else "VIOLATED") \
                        if r.proven_optimal else "not proven"
        row["invariant_ok"] = check
        rows.append(row)
        print(line + check, flush=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
