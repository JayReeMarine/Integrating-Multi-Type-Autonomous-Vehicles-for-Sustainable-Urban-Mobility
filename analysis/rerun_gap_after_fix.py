"""Re-run the stored pv_av sweep after the stale-clock fix and compare gaps.

The fix (core/timing.py, wired into both matchers on 2026-10-02) removes tows
whose coupling the implementation had accepted against a pre-tow clock. Savings
drop slightly as a result. Because the ILA-greedy differences this project
reports are 0.2-2.5 pp, the correction has to be measured, not assumed small.

Run:  PYTHONPATH=. venv/bin/python -u analysis/rerun_gap_after_fix.py
Writes data/results/gap_after_fix.csv
"""
from __future__ import annotations

import csv, statistics as st, collections
from pathlib import Path

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.metrics import baseline_total_powered_distance, greedy_total_powered_distance

stored = {}
for alg, path in (("greedy", "data/results/greedy/pv_av_sweep.csv"),
                  ("hungarian", "data/results/hungarian/pv_av_sweep.csv")):
    for r in csv.DictReader(open(path)):
        key = (int(r["num_av"]), int(r["num_pv"]), int(r["seed"]))
        stored.setdefault(key, {})[alg] = float(r["saving_percent"])
        stored[key]["params"] = (int(r["highway_length"]), int(r["min_trip_length"]),
                                 int(r["capacity_min"]), int(r["capacity_max"]),
                                 float(r["time_tolerance"]), float(r["time_window"]))

OUT = Path("data/results/gap_after_fix.csv")
rows = []
print(f"{'AV':>5}{'PV':>6}{'seed':>5} | {'greedy old':>11}{'new':>8} | {'ILA old':>9}{'new':>8} | "
      f"{'gap old':>9}{'gap new':>9}", flush=True)
for key in sorted(stored, key=lambda k: (k[0] / k[1], k[0])):
    nav, npv, seed = key
    e = stored[key]
    if "greedy" not in e or "hungarian" not in e:
        continue
    L, lmin, cmin, cmax, tol, win = e["params"]
    avs, pvs, l_min = generate_mock_data(
        num_av=nav, num_pv=npv, highway_length=L, av_capacity_range=(cmin, cmax),
        min_trip_length=lmin, seed=seed, enable_time_constraints=True,
        av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2), time_window=win)
    base = baseline_total_powered_distance(avs, pvs)
    out = {}
    for name, fn in (("greedy", greedy_multi_av_matching), ("hungarian", hungarian_multi_av_matching)):
        asg, _, _ = fn(avs, pvs, l_min, enable_time_constraints=True, time_tolerance=tol)
        out[name] = (base - greedy_total_powered_distance(base, asg)) / base * 100
    rows.append(dict(num_av=nav, num_pv=npv, seed=seed, ratio=nav / npv,
                     greedy_old=e["greedy"], greedy_new=round(out["greedy"], 4),
                     ila_old=e["hungarian"], ila_new=round(out["hungarian"], 4),
                     gap_old=round(e["hungarian"] - e["greedy"], 4),
                     gap_new=round(out["hungarian"] - out["greedy"], 4)))
    r = rows[-1]
    print(f"{nav:5d}{npv:6d}{seed:5d} | {r['greedy_old']:11.3f}{r['greedy_new']:8.3f} | "
          f"{r['ila_old']:9.3f}{r['ila_new']:8.3f} | {r['gap_old']:+9.3f}{r['gap_new']:+9.3f}", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n=== summary")
print(f"cells: {len(rows)}")
for tag in ("greedy", "ila"):
    d = [r[f"{tag}_new"] - r[f"{tag}_old"] for r in rows]
    print(f"{tag:7} saving change: mean {st.mean(d):+.3f} pp, worst {min(d):+.3f} pp, "
          f"changed in {sum(1 for x in d if abs(x) > 1e-9)} cells")
go = st.mean(r["gap_old"] for r in rows); gn = st.mean(r["gap_new"] for r in rows)
print(f"ILA - greedy gap: before {go:+.3f} pp, after {gn:+.3f} pp")
by = collections.defaultdict(list)
for r in rows:
    by[round(r["ratio"], 4)].append(r)
print(f"\n{'ratio':>7}{'n':>4} | {'gap before':>11}{'gap after':>11}")
for k in sorted(by):
    g = by[k]
    print(f"{k:7.3f}{len(g):4d} | {st.mean(x['gap_old'] for x in g):+11.3f}{st.mean(x['gap_new'] for x in g):+11.3f}")
print("wrote", OUT)
