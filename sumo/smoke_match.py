"""Smoke run: greedy and ILA on the converted M1 instance.

Evidence that the pipe works end to end, NOT a result -- demand is a
randomTrips placeholder and AV/PV labels are a random split.

Run:  PYTHONPATH=. venv/bin/python sumo/smoke_match.py [--av-fraction 0.2] [--seed 42]
Writes sumo/m1/smoke_match.json
"""
import argparse
import json
import time
from pathlib import Path

from sumo.convert import load_corridor_trips, label
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.metrics import baseline_total_powered_distance, greedy_total_powered_distance

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="sumo/m1")
ap.add_argument("--av-fraction", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tau", type=float, default=5.0)
a = ap.parse_args()

corr, trips = load_corridor_trips(a.scenario)
rows = []
for tu_label, tu in (("model units (mean speed = 1)", 0), ("real seconds", 1.0)):
    avs, pvs, l_min, info = label(corr, trips, av_fraction=a.av_fraction,
                                  seed=a.seed, time_unit_s=tu)
    base = baseline_total_powered_distance(avs, pvs)
    print(f"\n[{tu_label}]  AV={info['n_av']}  PV={info['n_pv']}  "
          f"dropped<L_min={info['dropped_short']}  1 time unit = {info['time_unit_s']:.2f} s")
    for name, fn in (("greedy", greedy_multi_av_matching), ("ILA", hungarian_multi_av_matching)):
        for ton in (False, True):
            t0 = time.perf_counter()
            asg, _, _ = fn(avs, pvs, l_min, enable_time_constraints=ton, time_tolerance=a.tau)
            dt = time.perf_counter() - t0
            sav = (base - greedy_total_powered_distance(base, asg)) / base * 100
            print(f"  {name:6} time={'ON ' if ton else 'OFF'}  saving={sav:6.2f}%  "
                  f"segments={len(asg):4d}  {dt:5.2f}s")
            rows.append(dict(time_unit=tu_label, algorithm=name, time_on=ton,
                             saving_percent=round(sav, 3), segments=len(asg),
                             runtime_s=round(dt, 3), n_av=info["n_av"], n_pv=info["n_pv"]))

out = Path(a.scenario) / "smoke_match.json"
out.write_text(json.dumps(dict(av_fraction=a.av_fraction, seed=a.seed, tau=a.tau,
                               note="placeholder demand, random labels -- not a result",
                               rows=rows), indent=1))
print("\nwrote", out)
